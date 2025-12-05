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


# ---------- Enhanced ChatGPT-Style Template with Simple Form ----------
TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HDFC Financial Analyzer - AI Assistant</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
      :root {
        --primary-color: #10a37f;
        --secondary-color: #0066cc;
        --bg-light: #f7f7f8;
        --border-color: #e5e5e5;
        --text-muted: #6b7280;
        --shadow: 0 2px 4px rgba(0,0,0,0.1);
      }

      body {
        background-color: var(--bg-light);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      }

      .card { 
        margin-bottom: 24px; 
        border: none;
        box-shadow: var(--shadow);
        border-radius: 12px;
      }
      
      .badge { font-size: 12px; }
      .monospace { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; }
      
      /* ChatGPT-style chat interface */
      .chat-container {
        height: 600px;
        border: 1px solid var(--border-color);
        border-radius: 16px;
        display: flex;
        flex-direction: column;
        background: white;
        box-shadow: var(--shadow);
        overflow: hidden;
      }
      
      .chat-header {
        padding: 16px 20px;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      
      .chat-header h5 {
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      
      .model-badge {
        background: rgba(255,255,255,0.2);
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
      }
      
      .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 20px;
        background: #fafafa;
        scroll-behavior: smooth;
      }
      
      .message {
        margin-bottom: 24px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
      }
      
      .message.user {
        flex-direction: row-reverse;
      }
      
      .message-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 14px;
        flex-shrink: 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
      }
      
      .message.user .message-avatar {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
      }
      
      .message.assistant .message-avatar {
        background: linear-gradient(135deg, var(--primary-color), #0d9488);
        color: white;
      }
      
      .message-content {
        padding: 16px 20px;
        border-radius: 20px;
        max-width: 75%;
        word-wrap: break-word;
        line-height: 1.5;
        position: relative;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
      }
      
      .message.user .message-content {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
      }
      
      .message.assistant .message-content {
        background: white;
        color: #374151;
        border: 1px solid var(--border-color);
      }
      
      .chat-input-container {
        padding: 20px;
        border-top: 1px solid var(--border-color);
        background: white;
      }
      
      .chat-input-form {
        display: flex;
        gap: 12px;
        align-items: flex-end;
      }
      
      .chat-input-wrapper {
        flex: 1;
      }
      
      .chat-input {
        width: 100%;
        resize: vertical;
        min-height: 48px;
        max-height: 120px;
        border: 2px solid var(--border-color);
        border-radius: 24px;
        padding: 14px 20px;
        font-size: 15px;
        line-height: 1.4;
        transition: all 0.2s ease;
        background: #f8f9fa;
      }
      
      .chat-input:focus {
        outline: none;
        border-color: var(--primary-color);
        background: white;
        box-shadow: 0 0 0 3px rgba(16, 163, 127, 0.1);
      }
      
      .chat-controls {
        display: flex;
        gap: 8px;
        flex-direction: column;
      }
      
      .btn-chat {
        border-radius: 24px;
        padding: 10px 20px;
        font-size: 14px;
        font-weight: 500;
        border: none;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        min-width: 100px;
      }
      
      .btn-chat:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      }
      
      .btn-primary-chat {
        background: var(--primary-color);
        color: white;
      }
      
      .btn-reset {
        background: #f59e0b;
        color: white;
      }
      
      .no-messages {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: var(--text-muted);
        text-align: center;
      }
      
      .no-messages i {
        font-size: 48px;
        margin-bottom: 16px;
        opacity: 0.5;
      }
      
      /* Scrollbar styling */
      .chat-messages::-webkit-scrollbar {
        width: 6px;
      }
      
      .chat-messages::-webkit-scrollbar-track {
        background: #f1f1f1;
      }
      
      .chat-messages::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 3px;
      }
      
      .chat-messages::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
      }
    </style>
</head>
<body>

<div class="container my-4">
  <div class="card">
    <div class="card-body">
      <h4 class="card-title">
        <i class="fas fa-file-upload me-2"></i>Upload Financial Statement PDF
        {% if has_context %}
          <span class="badge text-bg-success ms-2"><i class="fas fa-check"></i> Context: Active</span>
        {% else %}
          <span class="badge text-bg-secondary ms-2"><i class="fas fa-times"></i> Context: None</span>
        {% endif %}
        {% if units_label %}
          <span class="badge text-bg-info ms-2"><i class="fas fa-coins"></i> {{ currency }} {{ units_label }}</span>
        {% endif %}
      </h4>
      <form method="post" action="{{ url_for('upload') }}" enctype="multipart/form-data">
        <div class="row g-3 align-items-center">
          <div class="col-auto">
            <input class="form-control" type="file" name="pdf_file" accept=".pdf" required>
          </div>
          <div class="col-auto">
            <button class="btn btn-primary" type="submit">
              <i class="fas fa-chart-line me-2"></i>Analyze
            </button>
          </div>
          <div class="col-auto">
            <a class="btn btn-outline-secondary" href="{{ url_for('clear') }}">
              <i class="fas fa-eraser me-2"></i>Clear Context
            </a>
          </div>
          <div class="col-auto">
            <a class="btn btn-outline-danger" href="{{ url_for('reset_all') }}">
              <i class="fas fa-trash-alt me-2"></i>Reset All
            </a>
          </div>
          <div class="col-auto">
            <a class="btn btn-outline-dark" href="{{ url_for('debug') }}">
              <i class="fas fa-bug me-2"></i>Debug
            </a>
          </div>
          <div class="col-auto">
            <a class="btn btn-outline-info" href="{{ url_for('test_pdf') }}">
              <i class="fas fa-vial me-2"></i>Test PDF
            </a>
          </div>
        </div>
      </form>
      {% if upload_error %}<div class="alert alert-danger mt-3"><i class="fas fa-exclamation-triangle me-2"></i>{{ upload_error }}</div>{% endif %}
      {% if not ocr_available %}
        <div class="alert alert-warning mt-3">
          <i class="fas fa-exclamation-triangle me-2"></i>
          OCR is disabled (Tesseract not found). Using embedded PDF text only; scanned documents may not parse optimally.
        </div>
      {% endif %}
    </div>
  </div>

  {% if dual or single %}
  <div class="card">
    <div class="card-body">
      <h3 class="card-title">
        <i class="fas fa-analytics me-2"></i>Extracted Financial Metrics 
        {% if units_label %}<small class="text-muted">(Values in {{ currency }} {{ units_label }})</small>{% endif %}
      </h3>
      <div class="row">
        {% if dual %}
        <div class="col-md-7">
          <h5><i class="fas fa-chart-bar me-2"></i>Income Statement (Current vs Prior)</h5>
          <div class="table-responsive">
            <table class="table table-sm table-striped align-middle">
              <thead class="table-dark"><tr><th>Line Item</th><th class="text-end">Current</th><th class="text-end">Prior</th></tr></thead>
              <tbody>
                {% for k,v in dual.items() %}
                  <tr>
                    <td><strong>{{ k }}</strong></td>
                    <td class="text-end">{{ v.current|fmt_num }}</td>
                    <td class="text-end">{{ v.prior|fmt_num }}</td>
                  </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        </div>
        {% endif %}
        {% if single %}
        <div class="col-md-5">
          <h5><i class="fas fa-balance-scale me-2"></i>Balance Sheet Items</h5>
          <div class="table-responsive">
            <table class="table table-sm table-striped">
              <thead class="table-dark"><tr><th>Item</th><th class="text-end">Value</th></tr></thead>
              <tbody>
                {% for k,v in single.items() %}
                  <tr><td><strong>{{ k }}</strong></td><td class="text-end">{{ v|fmt_num }}</td></tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        </div>
        {% endif %}
      </div>
    </div>
  </div>
  {% endif %}

  {% if ratios %}
  <div class="card">
    <div class="card-body">
      <h3 class="card-title"><i class="fas fa-calculator me-2"></i>Financial Ratios</h3>
      <div class="row">
        {% for name,val in ratios %}
          <div class="col-md-6 col-lg-4 mb-3">
            <div class="p-3 bg-light rounded">
              <div class="fw-bold text-primary">{{ name }}</div>
              <div class="fs-4 fw-bold">{{ val|pct }}</div>
            </div>
          </div>
        {% endfor %}
      </div>
      {% if recs %}
        <hr>
        <h5><i class="fas fa-lightbulb me-2"></i>Recommendations</h5>
        <div class="alert alert-info">
          <ul class="mb-0">{% for r in recs %}<li>{{ r }}</li>{% endfor %}</ul>
        </div>
      {% endif %}
    </div>
  </div>
  {% endif %}

  <div class="card">
    <div class="chat-container">
      <div class="chat-header">
        <h5><i class="fas fa-robot me-2"></i>AI Financial Assistant</h5>
        <div class="model-badge">GPT-4 Turbo</div>
      </div>
      
      <div class="chat-messages" id="chatMessages">
        {% if chat_history %}
          {% for msg in chat_history %}
            <div class="message {{ msg.role }}">
              <div class="message-avatar">
                {% if msg.role == 'user' %}<i class="fas fa-user"></i>{% else %}<i class="fas fa-robot"></i>{% endif %}
              </div>
              <div class="message-content">{{ msg.content }}</div>
            </div>
          {% endfor %}
          <div id="chat-bottom-anchor"></div>
        {% else %}
          <div class="no-messages">
            <i class="fas fa-comments"></i>
            <h5>Ready to analyze your financial data!</h5>
            <p>Upload a PDF above, then ask me about profitability, efficiency, risk metrics, or any financial insights.</p>
          </div>
        {% endif %}
      </div>
      
      <div class="chat-input-container">
        {% if not has_context %}
          <div class="alert alert-info mb-3">
            <i class="fas fa-info-circle me-2"></i>
            Upload a financial statement first for contextual analysis, or ask general questions.
          </div>
        {% endif %}
        
        <form method="post" action="{{ url_for('ask') }}" class="chat-input-form">
          <div class="chat-input-wrapper">
            <textarea 
              class="chat-input" 
              name="prompt" 
              rows="2" 
              placeholder="Ask about profitability, efficiency, risk analysis, trends..."
              required
            ></textarea>
          </div>
          <div class="chat-controls">
            <button type="submit" class="btn btn-chat btn-primary-chat">
              <i class="fas fa-paper-plane me-2"></i>Send
            </button>
            <a href="{{ url_for('reset_chat') }}" class="btn btn-chat btn-reset">
              <i class="fas fa-refresh me-2"></i>Reset
            </a>
          </div>
        </form>
      </div>
    </div>
  </div>
</div>

<script>
// Improved auto-scroll to bottom of chat messages
document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chatMessages');
    const chatContainer = document.querySelector('.chat-container');
    
    // Function to scroll to bottom
    function scrollToBottom() {
        const anchor = document.getElementById('chat-bottom-anchor');
        if (anchor) {
            anchor.scrollIntoView({ behavior: 'smooth', block: 'end' });
        } else if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        // Also scroll the page to keep chat visible
        setTimeout(() => {
            if (chatContainer) {
                chatContainer.scrollIntoView({ behavior: 'smooth', block: 'end' });
            }
        }, 300);
    }
    
    // Scroll on page load if there are messages
    if (!chatMessages.querySelector('.no-messages')) {
        scrollToBottom();
    }
    
    // Aggressively clear all form inputs to prevent browser prefill
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) {
        fileInput.value = '';
        if (fileInput.files) {
            fileInput.files = null;
        }
    }
    
    // Clear textarea completely
    const textarea = document.querySelector('textarea[name="prompt"]');
    if (textarea) {
        textarea.value = '';
        textarea.innerHTML = '';
        textarea.textContent = '';
    }
    
    // Clear any other form elements
    const allInputs = document.querySelectorAll('input, textarea, select');
    allInputs.forEach(input => {
        if (input.type !== 'submit' && input.type !== 'button') {
            if (input.type === 'file') {
                input.value = '';
            } else if (input.tagName.toLowerCase() === 'textarea') {
                input.value = '';
            }
        }
    });
});

// Auto-resize textarea with error protection
const textarea = document.querySelector('textarea[name="prompt"]');
if (textarea) {
    // Clear on focus to ensure no prefill
    textarea.addEventListener('focus', function() {
        if (!this.value.trim()) {
            this.value = '';
        }
    });
    
    textarea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
}
</script>

</body>
</html>
"""


# ---------- Routes ----------
@app.route("/", methods=["GET"])
def home():
    # Clear ALL session data on fresh home load to prevent any prefill
    session.clear()
    
    ocr_available, ocr_status = check_ocr_dependencies()
    units = {"currency": "INR", "units_label": "crores"}  # Default values
    
    return render_template_string(
        TEMPLATE,
        has_context=False,  # Always false on fresh load
        dual={},
        single={},
        ratios=[],
        recs=[],
        currency=units.get("currency"),
        units_label=units.get("units_label"),
        ocr_available=ocr_available,
        chat_history=[],  # Always empty on fresh load
        prompt=None,  # Always None on home load
        upload_error=None
    )


@app.route("/upload", methods=["POST"])
def upload():
    # Clear all previous data on new upload
    for k in ["hdfc_context", "hdfc_dual", "hdfc_single", "hdfc_ratios", "hdfc_recs", "chat_history"]:
        session.pop(k, None)
        
    f = request.files.get("pdf_file")
    if not f or f.filename == "":
        return render_template_string(
            TEMPLATE, has_context=False, dual={}, single={}, ratios=[], recs=[],
            currency="INR", units_label="crores", chat_history=[], 
            upload_error="Please select an HDFC PDF file.", ocr_available=check_ocr_dependencies()[0],
            prompt=None
        )
    if not f.filename.lower().endswith(".pdf"):
        return render_template_string(
            TEMPLATE, has_context=False, dual={}, single={}, ratios=[], recs=[],
            currency="INR", units_label="crores", chat_history=[],
            upload_error="Please upload a PDF file only.", ocr_available=check_ocr_dependencies()[0],
            prompt=None
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        f.save(tmp.name)
        dual_raw, single_raw, dual, single, units = parse_pdf(tmp.name)
    except Exception as e:
        print("[upload] error:", e)
        return render_template_string(
            TEMPLATE, has_context=False, dual={}, single={}, ratios=[], recs=[],
            currency="INR", units_label="crores", chat_history=[],
            upload_error=f"Error processing PDF: {e}", ocr_available=check_ocr_dependencies()[0],
            prompt=None
        )
    finally:
        try:
            tmp.close(); os.unlink(tmp.name)
        except Exception:
            pass

    ratios = compute_ratios(dual_raw, single_raw)
    recs   = recommendations(ratios)

    # Persist new data with fresh chat
    session["hdfc_units"]  = units
    session["hdfc_dual"]   = dual
    session["hdfc_single"] = single
    session["hdfc_ratios"] = ratios
    session["hdfc_recs"]   = recs
    session["hdfc_context"] = metrics_to_context(dual, single, ratios, units)
    session["chat_history"] = []

    return render_template_string(
        TEMPLATE, has_context=True,
        dual=dual, single=single, ratios=ratios, recs=recs,
        currency=units.get("currency"), units_label=units.get("units_label"),
        chat_history=[], upload_error=None, ocr_available=check_ocr_dependencies()[0],
        prompt=None
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

    # Get or initialize chat history
    chat_history = session.get("chat_history", [])
    
    if prompt:
        # Add user message to history
        chat_history.append({"role": "user", "content": prompt})

    answer = None
    error_msg = None
    
    if prompt and client:
        try:
            # Build messages for API
            messages = []
            
            if context:
                messages.append({
                    "role": "system",
                    "content": f"You are a financial analyst. Be concise and direct. Focus on key insights only.\n\n{context}\n\nProvide brief analysis with specific numbers."
                })
            else:
                messages.append({
                    "role": "system",
                    "content": "You are a financial analyst. Provide brief, actionable insights. Be concise."
                })

            # Add recent chat history (last 5 messages)
            for msg in chat_history[-5:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

            resp = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages,
                temperature=0.1,
                max_tokens=300,
                top_p=0.95
            )
            answer = resp.choices[0].message.content
            
            # Add assistant response to history
            chat_history.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            error_msg = f"AI service error: {str(e)}"
            print(f"[ask] OpenAI error: {e}")
    elif prompt and not client:
        error_msg = "OpenAI client not available. Please check your API key configuration."
    elif not prompt:
        error_msg = "Please enter a question."

    # Keep only last 20 messages
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]

    # Save updated chat history
    session["chat_history"] = chat_history

    return render_template_string(
        TEMPLATE, has_context=bool(context),
        dual=dual, single=single, ratios=ratios, recs=recs,
        currency=units.get("currency"), units_label=units.get("units_label"),
        chat_history=chat_history, upload_error=None, ocr_available=check_ocr_dependencies()[0],
        prompt=None  # Don't prefill the form after submission
    )


@app.route("/reset_chat")
def reset_chat():
    session.pop("chat_history", None)
    return redirect(url_for("home"))


@app.route("/reset_all")
def reset_all():
    """Reset everything - clear all session data for fresh start"""
    session.clear()
    return redirect(url_for("home"))


@app.route("/clear")
def clear():
    for k in ["hdfc_context", "hdfc_units", "hdfc_dual", "hdfc_single", "hdfc_ratios", "hdfc_recs", "chat_history"]:
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
        "session_keys": list(session.keys()),
        "chat_history_length": len(session.get("chat_history", []))
    })


@app.route("/test-pdf")
def test_pdf():
    """
    Quick test route for debugging a fixed local PDF path (update path below).
    """
    pdf_path = os.environ.get("HDFC_TEST_PDF", r"C:\Financial\sample_hdfc.pdf")
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
            "native_sample": native_text[:500],
            "ocr_available": ocr_available,
            "ocr_status": ocr_status,
            "ocr_text_length": len(ocr_text),
            "ocr_sample": ocr_text[:500] if ocr_text else "No OCR text",
            "final_text_length": len(final_text),
            "final_sample": final_text[:500],
            "units": units,
            "dual_extracted": list(dual.keys()),
            "single_extracted": list(single.keys()),
            "computed_ratios": ratios,
            "recommendations": recs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- Main ----------
if __name__ == "__main__":
    ok, msg = check_ocr_dependencies()
    print(("✅ " if ok else "⚠️ ") + msg)
    print("🏦 HDFC Financial Analyzer with ChatGPT-style AI Assistant")
    print("🌐 Running on http://127.0.0.1:5075")
    print("🤖 Powered by GPT-4 Turbo")
    app.run(host="127.0.0.1", port=5075, debug=True, use_reloader=False)