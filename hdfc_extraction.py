# financial_flask_hdfc.py
from flask import Flask, request, render_template_string, session, redirect, url_for, jsonify
import fitz, tempfile, re, os, io, sys
from dotenv import load_dotenv
from openai import OpenAI

# ---- OCR deps
from PIL import Image, ImageFilter, ImageOps
import pytesseract
from pytesseract import TesseractNotFoundError

# ---- Extras for robust Tesseract detection + logging
import platform, shutil, logging

# --- API + Flask setup ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# If Tesseract is not on PATH (Windows), set it here (or via env var TESSERACT_CMD)
if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")


# ===================== Tesseract Locator =====================
def ensure_tesseract_available():
    """
    Locate the Tesseract binary and wire it to pytesseract.
    Uses TESSERACT_CMD env, PATH, or common install locations.
    Returns (ok: bool, msg: str, path: str|None)
    """
    # 1) Respect env var if set
    env_path = os.getenv("TESSERACT_CMD")
    if env_path and os.path.isfile(env_path):
        pytesseract.pytesseract.tesseract_cmd = env_path
        try:
            _ = pytesseract.get_tesseract_version()
            return True, f"Tesseract via TESSERACT_CMD: {env_path}", env_path
        except Exception as e:
            return False, f"Tesseract at TESSERACT_CMD failed: {e}", env_path

    # 2) PATH
    which = shutil.which("tesseract")
    if which:
        pytesseract.pytesseract.tesseract_cmd = which
        try:
            _ = pytesseract.get_tesseract_version()
            return True, f"Tesseract on PATH: {which}", which
        except Exception as e:
            return False, f"Tesseract on PATH failed: {e}", which

    # 3) Common install paths
    candidates = []
    sysname = platform.system().lower()
    if "windows" in sysname:
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
    elif "darwin" in sysname:  # macOS
        candidates = ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"]
    else:  # Linux
        candidates = ["/usr/bin/tesseract", "/usr/local/bin/tesseract"]

    for path in candidates:
        if os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            try:
                _ = pytesseract.get_tesseract_version()
                return True, f"Tesseract found at: {path}", path
            except Exception as e:
                return False, f"Tesseract found but failed to run: {e}", path

    return False, (
        "Tesseract not found. Install it and/or set TESSERACT_CMD to the full path, "
        "e.g. C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    ), None


def check_ocr_dependencies():
    """Check if OCR dependencies are available and log status."""
    try:
        logging.basicConfig(level=logging.INFO)
    except Exception:
        pass
    ok, msg, path = ensure_tesseract_available()
    logging.info(msg)
    return ok, ("OCR available" if ok else f"OCR not available: {msg}")


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-production")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB


# ===================== Utilities =====================
def to_float(s):
    if s is None:
        return None
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return None


def safe_div(a, b):
    return round(a / b, 4) if (a is not None and b not in (None, 0)) else None


def fmt_pct(x):
    return f"{x*100:.2f}%" if x is not None else "N/A"


def normalize_unit_label(label_raw):
    if not label_raw:
        return None
    s = str(label_raw).lower()
    if "billion" in s:  return "billions"
    if "million" in s:  return "millions"
    if "thousand" in s: return "thousands"
    if "crore" in s or "crores" in s: return "crores"
    if "lakh" in s or "lakhs" in s:   return "lakhs"
    return None


def detect_units(text):
    """
    Detect currency + magnitude. Handles '₹ in crore', 'Rs. in crores', etc.
    Falls back to AED phrasing if present (legacy).
    """
    currency = None
    units_label = None

    pats_inr = [
        r"(₹|INR|Rs\.?|Rupees)[^\n]{0,30}(in\s+)?(lakh|lakhs|crore|crores|million|millions|billion|billions)",
        r"(₹)\s*in\s*(crore|crores|lakh|lakhs|million|millions|billion|billions)",
        r"₹\s*in\s*crore"
    ]
    for p in pats_inr:
        m = re.search(p, text, re.I)
        if m:
            currency = "INR"
            groups = [g for g in m.groups() if g]
            for g in groups:
                if re.search(r"lakh|crore|million|billion", g, re.I):
                    units_label = normalize_unit_label(g)
            break

    if not currency:
        pats_aed = [
            r"(?:all amounts|figures)[^.\n]{0,80}(?:in|expressed in)[^.\n]{0,80}(AED|UAE\s*Dirhams)[^.\n]{0,40}(thousand|thousands|million|millions|billion|billions)",
            r"(AED|UAE\s*Dirhams)[^.\n]{0,40}(thousand|thousands|million|millions|billion|billions)",
            r"(AED)[^\n]{0,10}\((?:in\s+)?(thousand|thousands|million|millions|billion|billions)\)",
        ]
        for p in pats_aed:
            m = re.search(p, text, re.I)
            if m:
                currency = "AED"
                groups = [g for g in m.groups() if g]
                for g in groups:
                    if re.search(r"thousand|million|billion", g, re.I):
                        units_label = normalize_unit_label(g)
                break

    return {"currency": currency or "INR", "units_label": units_label or "crores"}


# ===================== OCR PIPELINE =====================
def preprocess_for_ocr(pix):
    """
    Convert a PyMuPDF pixmap to a denoised, high-contrast PIL image for OCR.
    """
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    img = img.point(lambda x: 255 if x > 180 else 0, mode='1')
    return img


def ocr_pdf_to_text(path, dpi=300, lang="eng"):
    ocr_available, _ = check_ocr_dependencies()
    if not ocr_available:
        return ""
    out = []
    with fitz.open(path) as doc:
        for pg in doc:
            mat = fitz.Matrix(dpi/72.0, dpi/72.0)
            pix = pg.get_pixmap(matrix=mat, alpha=False)
            img = preprocess_for_ocr(pix)
            txt = pytesseract.image_to_string(
                img, lang=lang,
                config="--oem 3 --psm 6 -c preserve_interword_spaces=1"
            )
            out.append(txt)
    return "\n".join(out)


def extract_text_embedded(path):
    """Pull embedded text using PyMuPDF."""
    with fitz.open(path) as doc:
        return "\n".join(pg.get_text("text") for pg in doc)


def extract_text_with_ocr_fallback(path):
    """
    Try native text extraction; if weak (scanned), OCR it; else merge if helpful.
    """
    native = extract_text_embedded(path)
    if len(native.strip()) >= 200:
        # Still try OCR; if OCR adds more digits/length, merge
        ocr = ocr_pdf_to_text(path) or ""
        d_native = sum(ch.isdigit() for ch in native)
        d_ocr = sum(ch.isdigit() for ch in ocr)
        if d_ocr > d_native * 1.10 or len(ocr) > len(native) * 1.10:
            base = ocr.splitlines()
            extra = [ln for ln in native.splitlines() if ln.strip() and ln not in base]
            return "\n".join(base + extra)
        return native
    # Likely scanned → OCR
    ocr = ocr_pdf_to_text(path) or ""
    if not ocr:
        # Final fallback: block/span crawl
        try:
            with fitz.open(path) as doc:
                text_blocks = []
                for page in doc:
                    blocks = page.get_text("dict")
                    for block in blocks.get("blocks", []):
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line.get("spans", []):
                                    text_blocks.append(span.get("text", ""))
                return " ".join(text_blocks)
        except Exception:
            pass
    return ocr


# ===================== HDFC-specific extraction =====================
NUM = r"([\d,]+(?:\.\d+)?)"

# Two-column (Current vs Prior)
PATTERNS_DUAL = {
    # P&L
    "Interest earned": rf"Interest\s+earned.*?{NUM}\s+{NUM}",
    "Other income": rf"Other\s+Income.*?{NUM}\s+{NUM}",
    "Total income": rf"Total\s+Income.*?{NUM}\s+{NUM}",
    "Interest expended": rf"Interest\s+expended.*?{NUM}\s+{NUM}",
    "Operating expenses": rf"Operating\s+expenses.*?{NUM}\s+{NUM}",
    "Total expenditure": rf"Total\s+Expenditure.*?{NUM}\s+{NUM}",
    "Operating Profit before provisions and contingencies": rf"Operating\s+Profit\s+before\s+provisions\s+and\s+contingencies.*?{NUM}\s+{NUM}",
    "Provisions (other than tax) and Contingencies": rf"Provisions\s*\(other\s+than\s+tax\)\s*and\s*Contingencies.*?{NUM}\s+{NUM}",
    "Profit before tax": rf"Profit\s+from\s+ordinary\s+activities\s+before\s+tax.*?{NUM}\s+{NUM}",
    "Tax Expense": rf"Tax\s+Expense.*?{NUM}\s+{NUM}",
    "Net Profit for the period": rf"Net\s+Profit\s+(?:from\s+ordinary\s+activities\s+)?after\s+tax.*?{NUM}\s+{NUM}",
    # Balance (first two numeric cols if present twice)
    "Deposits (dual)": rf"Deposits\s+{NUM}\s+{NUM}",
    "Borrowings (dual)": rf"Borrowings\s+{NUM}\s+{NUM}",
    "Investments (dual)": rf"Investments\s+{NUM}\s+{NUM}",
    "Advances (dual)": rf"Advances\s+{NUM}\s+{NUM}",
}

# Single-column (Analytical ratios / BS)
PATTERNS_SINGLE = {
    "Gross NPAs": rf"Gross\s+NPAs\s+{NUM}",
    "Net NPAs": rf"Net\s+NPAs\s+{NUM}",
    "% of Gross NPAs to Gross Advances": r"%\s*of\s*Gross\s*NPAs\s*to\s*Gross\s*Advances\s+(1?\d+(?:\.\d+)?%)",
    "% of Net NPAs to Net Advances": r"%\s*of\s*Net\s*NPAs\s*to\s*Net\s*Advances\s+(0?\d+(?:\.\d+)?%)",
    "Return on assets (reported)": r"Return\s+on\s+assets\s*\(average\).*?(\d+(?:\.\d+)?%)",
    "Net worth": rf"Net\s+worth\s+{NUM}",
    # Statement of Assets & Liabilities (current)
    "Deposits": rf"Deposits\s+{NUM}",
    "Borrowings": rf"Borrowings\s+{NUM}",
    "Investments": rf"Investments\s+{NUM}",
    "Advances": rf"Advances\s+{NUM}",
    "Cash and balances with RBI": rf"Cash\s+and\s+balances\s+with\s+Reserve\s+Bank\s+of\s+India\s+{NUM}",
    "Balances with banks": rf"Balances\s+with\s+banks.*?{NUM}",
    "Other assets": rf"Other\s+assets\s+{NUM}",
    "Total assets (BS)": rf"Total\s+{NUM}\s+\d",
}


def _to_pct_str_or_float(maybe_str):
    if maybe_str is None:
        return None
    if isinstance(maybe_str, str) and maybe_str.strip().endswith("%"):
        try:
            return float(maybe_str.strip().strip("%")) / 100.0
        except Exception:
            return None
    return to_float(maybe_str)


def extract_dual(text):
    out = {}
    for k, p in PATTERNS_DUAL.items():
        m = re.search(p, text, re.I | re.S)
        out[k] = {
            "current": to_float(m.group(1)) if m else None,
            "prior": to_float(m.group(2)) if (m and m.lastindex and m.lastindex >= 2) else None,
        }
    return out


def extract_single(text):
    out = {}
    for k, p in PATTERNS_SINGLE.items():
        m = re.search(p, text, re.I | re.S)
        if m:
            g1 = m.group(1)
            if isinstance(g1, str) and g1.strip().endswith("%"):
                out[k] = _to_pct_str_or_float(g1.strip())
            else:
                out[k] = to_float(g1)
        else:
            out[k] = None
    return out


def parse_pdf(path):
    """Parse HDFC PDF with OCR fallback and unit detection."""
    print(f"[parse_pdf] path={path}")
    text = extract_text_with_ocr_fallback(path)
    print(f"[parse_pdf] extracted text len={len(text)}")
    if text:
        print("[parse_pdf] text sample:", repr(text[:300]))
    units = detect_units(text)
    dual_raw = extract_dual(text)
    single_raw = extract_single(text)

    # Filter for display context
    dual = {k: v for k, v in dual_raw.items() if (v.get("current") is not None or v.get("prior") is not None)}
    single = {k: v for k, v in single_raw.items() if v is not None}

    return dual_raw, single_raw, dual, single, units


# ===================== Ratios & Recs =====================
def compute_ratios(dual_raw, single_raw):
    interest_earned = (dual_raw.get("Interest earned") or {}).get("current")
    interest_exp    = (dual_raw.get("Interest expended") or {}).get("current")
    other_inc       = (dual_raw.get("Other income") or {}).get("current")
    oper_exp        = (dual_raw.get("Operating expenses") or {}).get("current")
    op_profit       = (dual_raw.get("Operating Profit before provisions and contingencies") or {}).get("current")
    pbt             = (dual_raw.get("Profit before tax") or {}).get("current")
    tax             = (dual_raw.get("Tax Expense") or {}).get("current")
    pat             = (dual_raw.get("Net Profit for the period") or {}).get("current")

    deposits        = single_raw.get("Deposits") or (dual_raw.get("Deposits (dual)") or {}).get("current")
    advances        = single_raw.get("Advances") or (dual_raw.get("Advances (dual)") or {}).get("current")
    investments     = single_raw.get("Investments") or (dual_raw.get("Investments (dual)") or {}).get("current")
    cash_rbi        = single_raw.get("Cash and balances with RBI")
    bal_banks       = single_raw.get("Balances with banks")
    total_assets    = single_raw.get("Total assets (BS)")

    gross_npa       = single_raw.get("Gross NPAs")
    net_npa         = single_raw.get("Net NPAs")
    roa_reported    = single_raw.get("Return on assets (reported)")

    nii = None
    if interest_earned is not None and interest_exp is not None:
        nii = interest_earned - interest_exp

    net_revenue = None
    if nii is not None:
        net_revenue = nii + (other_inc or 0)

    ratios = [
        ("Cost-to-Income",        safe_div(oper_exp, net_revenue)),
        ("Pre-provision Operating Margin", safe_div(op_profit, net_revenue)),
        ("Tax Rate",              safe_div(tax, pbt)),
        ("Net Profit Margin (on net revenue)", safe_div(pat, net_revenue)),
        ("Loan-to-Deposit (LDR)", safe_div(advances, deposits)),
        ("Liquid Assets % (Cash+Banks+Inv)/Assets", safe_div(
            (cash_rbi or 0) + (bal_banks or 0) + (investments or 0)
            if any(x is not None for x in [cash_rbi, bal_banks, investments]) else None,
            total_assets
        )),
        ("ROA (reported)", roa_reported),  # already converted to fraction if % found
        ("Gross NPA / Advances",  safe_div(gross_npa, advances)),
        ("Net NPA / Advances",    safe_div(net_npa, advances)),
    ]
    return [(n, v) for n, v in ratios if v is not None]


def recommendations(ratios):
    recs = []
    d = dict(ratios)
    if d.get("Cost-to-Income") is not None and d["Cost-to-Income"] > 0.50:
        recs.append("High cost-to-income; review operating expenses.")
    if d.get("Gross NPA / Advances") is not None and d["Gross NPA / Advances"] > 0.035:
        recs.append("Gross NPA ratio looks elevated; strengthen recoveries and provisioning.")
    if d.get("Loan-to-Deposit (LDR)") is not None and d["Loan-to-Deposit (LDR)"] > 0.95:
        recs.append("LDR is high; consider deposit growth or term funding.")
    if d.get("Liquid Assets % (Cash+Banks+Inv)/Assets") is not None and d["Liquid Assets % (Cash+Banks+Inv)/Assets"] < 0.25:
        recs.append("Low liquid-asset buffer; monitor LCR/NSFR and short-term gaps.")
    return recs


def metrics_to_context(dual, single, ratios, units):
    lines = ["Key metrics & ratios (HDFC):"]
    if units and units.get("units_label"):
        lines.append(f"Units detected: {units.get('currency','INR')} {units['units_label']}")
    if dual:
        lines.append("\nP&L / Dual-column lines:")
        for k, v in dual.items():
            lines.append(f"  {k}: current={v['current']}, prior={v['prior']}")
    if single:
        lines.append("\nBalance / Single-column lines:")
        for k, v in single.items():
            lines.append(f"  {k}: {v}")
    if ratios:
        lines.append("\nRatios:")
        for name, val in ratios:
            lines.append(f"  {name}: {fmt_pct(val)}")
    return "\n".join(lines)


# --- Jinja filters ---
@app.template_filter("pct")
def pct(v): return fmt_pct(v)

@app.template_filter("fmt_num")
def jinja_fmt_num(v):
    if v is None: return "N/A"
    try: return f"{float(v):,.2f}"
    except Exception: return str(v)


# ---------- Template ----------
TEMPLATE = """
<!doctype html>
<title>HDFC Financial Analyzer (with OCR)</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
<style>
  .card { margin-bottom: 20px; }
  .badge { font-size:12px; }
  .monospace { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; }
</style>

<div class="container my-4">
  <div class="card">
    <div class="card-body">
      <h4 class="card-title">1) Upload Financial Statement PDF
        {% if has_context %}
          <span class="badge text-bg-success ms-2">Context: True</span>
        {% else %}
          <span class="badge text-bg-secondary ms-2">Context: False</span>
        {% endif %}
        {% if units_label %}
          <span class="badge text-bg-info ms-2">Units: {{ currency }} {{ units_label }}</span>
        {% endif %}
      </h4>
      <form method="post" action="{{ url_for('upload') }}" enctype="multipart/form-data">
        <div class="row g-2 align-items-center">
          <div class="col-auto"><input class="form-control" type="file" name="pdf_file" required></div>
          <div class="col-auto"><button class="btn btn-primary" type="submit">Analyze</button></div>
          <div class="col-auto"><a class="btn btn-outline-secondary" href="{{ url_for('clear') }}">Clear context</a></div>
          <div class="col-auto"><a class="btn btn-outline-dark" href="{{ url_for('debug') }}">/debug</a></div>
          <div class="col-auto"><a class="btn btn-outline-info" href="{{ url_for('test_pdf') }}">/test-pdf</a></div>
        </div>
      </form>
      {% if upload_error %}<div class="text-danger mt-2">{{ upload_error }}</div>{% endif %}
      {% if not ocr_available %}
        <div class="alert alert-warning mt-2">
          OCR is disabled (Tesseract not found). Using embedded PDF text only; scanned tables may parse less accurately.
        </div>
      {% endif %}
    </div>
  </div>

  {% if dual or single %}
  <div class="card">
    <div class="card-body">
      <h3 class="card-title">Extracted Metrics {% if units_label %}<small class="text-muted">(Values in {{ currency }} {{ units_label }})</small>{% endif %}</h3>
      <div class="row">
        {% if dual %}
        <div class="col-md-7">
          <h5>Income Statement (Current vs Prior)</h5>
          <table class="table table-sm table-striped align-middle">
            <thead><tr><th>Line</th><th class="text-end">Current</th><th class="text-end">Prior</th></tr></thead>
            <tbody>
              {% for k,v in dual.items() %}
                <tr>
                  <td><b>{{ k }}</b></td>
                  <td class="text-end">{{ v.current|fmt_num }}</td>
                  <td class="text-end">{{ v.prior|fmt_num }}</td>
                </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        {% endif %}
        {% if single %}
        <div class="col-md-5">
          <h5>Other Key Balances</h5>
          <table class="table table-sm table-striped">
            <thead><tr><th>Item</th><th class="text-end">Value</th></tr></thead>
            <tbody>
              {% for k,v in single.items() %}
                <tr><td><b>{{ k }}</b></td><td class="text-end">{{ v|fmt_num }}</td></tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        {% endif %}
      </div>
    </div>
  </div>
  {% endif %}

  {% if ratios %}
  <div class="card">
    <div class="card-body">
      <h3 class="card-title">Ratios</h3>
      <ul>
        {% for name,val in ratios %}
          <li><b>{{ name }}</b>: {{ val|pct }}</li>
        {% endfor %}
      </ul>
      {% if recs %}
        <h5>Recommendations</h5>
        <ul>{% for r in recs %}<li>{{ r }}</li>{% endfor %}</ul>
      {% endif %}
    </div>
  </div>
  {% endif %}

  <div class="card">
    <div class="card-body">
      <h3 class="card-title">2) Chat with OpenAI about this PDF</h3>
      {% if not has_context %}
        <div class="text-secondary">Upload a financial statement first to give the assistant context. You can still type a question—I'll remind you.</div>
      {% endif %}
      <form method="post" action="{{ url_for('ask') }}">
        <textarea class="form-control" name="prompt" rows="5" placeholder="e.g., Profitability drivers, cost efficiency, liquidity, credit quality...">{{ prompt or '' }}</textarea>
        <div class="mt-3">
          <button class="btn btn-primary" type="submit">Ask</button>
        </div>
      </form>
      {% if answer %}
        <hr>
        <div><b>Assistant:</b></div>
        <div class="monospace">{{ answer }}</div>
      {% endif %}
      {% if error %}<div class="text-danger mt-2">{{ error }}</div>{% endif %}
    </div>
  </div>
</div>
"""


# ---------- Routes ----------
@app.route("/", methods=["GET"])
def home():
    ocr_available, ocr_status = check_ocr_dependencies()
    units = session.get("hdfc_units") or {"currency": "INR", "units_label": "crores"}
    return render_template_string(
        TEMPLATE,
        has_context=bool(session.get("hdfc_context")),
        dual=session.get("hdfc_dual") or {},
        single=session.get("hdfc_single") or {},
        ratios=session.get("hdfc_ratios") or [],
        recs=session.get("hdfc_recs") or [],
        currency=units.get("currency"),
        units_label=units.get("units_label"),
        ocr_available=ocr_available,
        prompt=None,
        answer=None,
        error=None,
        upload_error=None
    )


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("pdf_file")
    if not f or f.filename == "":
        return render_template_string(
            TEMPLATE, has_context=False, dual={}, single={}, ratios=[], recs=[],
            currency="INR", units_label="crores", prompt=None, answer=None, error=None,
            upload_error="Please select an HDFC PDF file.", ocr_available=check_ocr_dependencies()[0]
        )
    if not f.filename.lower().endswith(".pdf"):
        return render_template_string(
            TEMPLATE, has_context=False, dual={}, single={}, ratios=[], recs=[],
            currency="INR", units_label="crores", prompt=None, answer=None, error=None,
            upload_error="Please upload a PDF file only.", ocr_available=check_ocr_dependencies()[0]
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        f.save(tmp.name)
        dual_raw, single_raw, dual, single, units = parse_pdf(tmp.name)
    except Exception as e:
        print("[upload] error:", e)
        return render_template_string(
            TEMPLATE, has_context=False, dual={}, single={}, ratios=[], recs=[],
            currency="INR", units_label="crores", prompt=None, answer=None,
            error=None, upload_error=f"Error processing PDF: {e}", ocr_available=check_ocr_dependencies()[0]
        )
    finally:
        try:
            tmp.close(); os.unlink(tmp.name)
        except Exception:
            pass

    ratios = compute_ratios(dual_raw, single_raw)
    recs   = recommendations(ratios)

    # Persist
    session["hdfc_units"]  = units
    session["hdfc_dual"]   = dual
    session["hdfc_single"] = single
    session["hdfc_ratios"] = ratios
    session["hdfc_recs"]   = recs
    session["hdfc_context"] = metrics_to_context(dual, single, ratios, units)

    return render_template_string(
        TEMPLATE, has_context=True,
        dual=dual, single=single, ratios=ratios, recs=recs,
        currency=units.get("currency"), units_label=units.get("units_label"),
        prompt=None, answer=None, error=None, upload_error=None,
        ocr_available=check_ocr_dependencies()[0]
    )


@app.route("/ask", methods=["POST"])
def ask():
    prompt = (request.form.get("prompt") or "").strip()
    context = session.get("hdfc_context")
    dual    = session.get("hdfc_dual") or {}
    single  = session.get("hdfc_single") or {}
    ratios  = session.get("hdfc_ratios") or []
    recs    = session.get("hdfc_recs") or []
    units   = session.get("hdfc_units") or {"currency": "INR", "units_label": "crores"}

    if not context:
        return render_template_string(
            TEMPLATE, has_context=False, dual={}, single={}, ratios=[], recs=[],
            currency=units.get("currency"), units_label=units.get("units_label"),
            prompt=prompt, answer=None, error="Please upload a financial statement PDF first.",
            upload_error=None, ocr_available=check_ocr_dependencies()[0]
        )

    answer = None
    error_msg = None
    if prompt and client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a bank financial analyst. Be concise and numeric."},
                    {"role": "user", "content": f"{context}\n\nUser prompt: {prompt}"},
                ],
                temperature=0.2,
                max_tokens=600
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            error_msg = f"Error getting AI response: {str(e)}"
    elif prompt and not client:
        error_msg = "OpenAI client not available. Please check your API key configuration."
    elif not prompt:
        error_msg = "Please enter a question."

    return render_template_string(
        TEMPLATE, has_context=True,
        dual=dual, single=single, ratios=ratios, recs=recs,
        currency=units.get("currency"), units_label=units.get("units_label"),
        prompt=prompt, answer=answer, error=error_msg, upload_error=None,
        ocr_available=check_ocr_dependencies()[0]
    )


@app.route("/clear")
def clear():
    for k in ["hdfc_context", "hdfc_units", "hdfc_dual", "hdfc_single", "hdfc_ratios", "hdfc_recs"]:
        session.pop(k, None)
    return redirect(url_for("home"))


@app.route("/debug")
def debug():
    ocr_available, ocr_status = check_ocr_dependencies()
    units = session.get("hdfc_units") or {}
    return jsonify({
        "has_context": bool(session.get("hdfc_context")),
        "dual_keys": list((session.get("hdfc_dual") or {}).keys()),
        "single_keys": list((session.get("hdfc_single") or {}).keys()),
        "ratios": session.get("hdfc_ratios"),
        "recs": session.get("hdfc_recs"),
        "ocr_available": ocr_available,
        "ocr_status": ocr_status,
        "tesseract_cmd": getattr(pytesseract.pytesseract, "tesseract_cmd", None),
        "openai_client": "Available" if client else "Not Available (check API key)",
        "api_key": "Set" if OPENAI_API_KEY else "Missing",
        "units": units,
        "context_length": len(session.get("hdfc_context", "")),
        "session_keys": list(session.keys())
    })


@app.route("/test-pdf")
def test_pdf():
    """
    Quick test route for debugging a fixed local PDF path (update path below).
    """
    pdf_path = os.environ.get("HDFC_TEST_PDF", r"C:\HDFC\HDFC_sample.pdf")
    if not os.path.exists(pdf_path):
        return jsonify({"error": f"Test PDF not found at: {pdf_path} (set HDFC_TEST_PDF env var)"}), 404

    try:
        # Native text
        with fitz.open(pdf_path) as doc:
            native_text = "\n".join(pg.get_text() for pg in doc)

        # OCR text (if available)
        ocr_available, ocr_status = check_ocr_dependencies()
        ocr_text = ""
        if ocr_available:
            try:
                ocr_text = ocr_pdf_to_text(pdf_path)
            except Exception as e:
                ocr_text = f"OCR Error: {e}"

        # Final chosen
        final_text = extract_text_with_ocr_fallback(pdf_path)
        dual_raw, single_raw, dual, single, units = parse_pdf(pdf_path)
        ratios = compute_ratios(dual_raw, single_raw)
        recs   = recommendations(ratios)

        return jsonify({
            "pdf_file": pdf_path,
            "native_text_length": len(native_text),
            "native_text_sample": native_text[:1000],
            "ocr_available": ocr_available,
            "ocr_status": ocr_status,
            "ocr_text_length": len(ocr_text),
            "ocr_text_sample": ocr_text[:1000] if ocr_text else "No OCR text",
            "final_text_length": len(final_text),
            "final_text_sample": final_text[:1000],
            "units": units,
            "dual_keys": list(dual.keys()),
            "single_keys": list(single.keys()),
            "ratios": ratios,
            "recs": recs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- Main ----------
if __name__ == "__main__":
    ok, msg = check_ocr_dependencies()
    print(("✅ " if ok else "⚠️ ") + msg)
    print("HDFC Financial Analyzer on http://127.0.0.1:5075")
    app.run(host="127.0.0.1", port=5075, debug=True, use_reloader=False)
