# financial_flask_genai.py
from flask import Flask, request, render_template_string, session, redirect, url_for
import fitz, tempfile, re, os, sys
from dotenv import load_dotenv
from openai import OpenAI

# --- API + Flask setup ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False

# ---------- Helpers ----------
def to_float(s):
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None

def safe_div(a, b):
    return round(a / b, 4) if a is not None and b not in (None, 0) else None

def fmt_pct(x):
    return f"{x*100:.2f}%" if x is not None else "N/A"

def filter_ratios(ratio_list):
    """Remove ratios with None values (which would show as N/A)."""
    return [(name, val) for (name, val) in ratio_list if val is not None]

def filter_dual_metrics(dual_dict):
    """Keep only lines where at least one of current/prior is not None."""
    out = {}
    for k, v in (dual_dict or {}).items():
        cur = v.get("current")
        pri = v.get("prior")
        if cur is not None or pri is not None:
            out[k] = v
    return out

def filter_single_metrics(single_dict):
    """Keep only lines where value is not None."""
    return {k: v for k, v in (single_dict or {}).items() if v is not None}

def normalize_unit_label(label_raw):
    if not label_raw:
        return None
    s = label_raw.lower()
    if "billion" in s:
        return "billions"
    if "million" in s:
        return "millions"
    if "thousand" in s:
        return "thousands"
    return None

def detect_units(text):
    """
    Best-effort scan for currency + magnitude like 'AED millions' / 'Amounts in thousands of UAE dirhams'.
    Returns dict: {"currency": "AED", "units_label": "millions"|"thousands"|"billions"|None}
    """
    currency = None
    units_label = None

    # Common patterns across front matter / notes
    pats = [
        r"(?:all amounts|figures)[^.\n]{0,80}(?:in|expressed in)[^.\n]{0,80}(AED|UAE\s*Dirhams|UAE\s*Dhirhams)[^.\n]{0,40}(thousand|thousands|million|millions|billion|billions)",
        r"(AED|UAE\s*Dirhams|UAE\s*Dhirhams)[^.\n]{0,40}(thousand|thousands|million|millions|billion|billions)",
        r"(Amounts? in)[^.\n]{0,40}(AED|UAE\s*Dirhams)[^.\n]{0,40}(thousand|thousands|million|millions|billion|billions)",
        r"(AED)[^\n]{0,10}\((?:in\s+)?(thousand|thousands|million|millions|billion|billions)\)",
    ]
    for p in pats:
        m = re.search(p, text, re.I)
        if m:
            # Heuristic: last group is the units word
            groups = [g for g in m.groups() if g]
            # Find currency and unit among captured groups
            for g in groups:
                if re.search(r"AED|Dirhams", g, re.I):
                    currency = "AED"
                if re.search(r"thousand|million|billion", g, re.I):
                    units_label = normalize_unit_label(g)
            break

    return {"currency": currency or "AED", "units_label": units_label}

# Regexes for ENBD-style statements (tweak as needed)

# Dual-line patterns (Current vs Prior)
patterns_dual = {
    "Total Operating Income": r"Total operating income\s+([\d,]+)\s+([\d,]+)",
    "General and Administrative Expenses": r"General and administrative expenses\s+\(([\d,]+)\)\s+\(([\d,]+)\)",
    "Operating Profit Before Impairment": r"Operating profit before impairment\s+([\d,]+)\s+([\d,]+)",
    "Profit Before Tax": r"Profit for the period before taxation\s+([\d,]+)\s+([\d,]+)",
    "Taxation Charge": r"Taxation charge\s+\(([\d,]+)\)\s+\(([\d,]+)\)",
    "Profit for the Period": r"Profit for the period\s+([\d,]+)\s+([\d,]+)",
    "Earnings Per Share (AED)": r"Earnings per share\s*\(AED\)\s+([\d\.]+)\s+([\d\.]+)",

    # Balance sheet / interest lines (Current vs Prior when present)
    "Customer and Islamic Deposits": r"Customer(?:\s+and)?\s+Islamic deposits\s+([\d,]+)\s+([\d,]+)",
    "Gross Loans and Receivables (dual)": r"Gross loans and receivables\s+([\d,]+)\s+([\d,]+)",
    "Net Interest + Islamic Income": r"Net interest income and net income from Islamic financing and\s*investment products\s+([\d,]+)\s+([\d,]+)",
}

# Single-line patterns (Current only)
patterns_single = {
    "Gross Loans": r"Gross loans and receivables\s+([\d,]+)\s+[\d,]+",
    "ECL": r"Less:\s*Expected credit losses\s+\(([\d,]+)\)\s+\([\d,]+\)",
    "NPLs": r"Total of credit impaired loans and receivables\s+([\d,]+)\s+[\d,]+",
    "Total Assets": r"Segment Assets[\s\S]*?(\d{1,3}(?:,\d{3})+)\s*\n\s*Segment Liabilities",

    # Liquidity line items (Current only from notes)
    "Cash & CB": r"Cash and deposits with Central Banks\s+([\d,]+)\s+[\d,]+",
    "Due from Banks": r"Due from banks\s+([\d,]+)\s+[\d,]+",
    "Investment Securities": r"Net Investment securities\s+([\d,]+)",
    # Best-effort equity capture from Statement of Changes in Equity
    "Equity (Group Total)": r"Balance as at .*?Group\s+Total\s+(\d{1,3}(?:,\d{3})+)",
}

def extract_dual(text):
    out = {}
    for k, p in patterns_dual.items():
        m = re.search(p, text, re.I)
        out[k] = {
            "current": to_float(m.group(1)) if m else None,
            "prior": to_float(m.group(2)) if (m and m.lastindex and m.lastindex >= 2) else None,
        }
    return out

def extract_single(text):
    out = {}
    for k, p in patterns_single.items():
        m = re.search(p, text, re.I)
        out[k] = to_float(m.group(1)) if m else None
    return out

def parse_pdf(path):
    with fitz.open(path) as doc:
        txt = "\n".join(pg.get_text() for pg in doc)
    units = detect_units(txt)
    return extract_dual(txt), extract_single(txt), units

def compute_ratios(dual, single):
    # Income statement lines
    toi = dual.get("Total Operating Income", {}).get("current")
    ga  = dual.get("General and Administrative Expenses", {}).get("current")
    opb = dual.get("Operating Profit Before Impairment", {}).get("current")
    pat = dual.get("Profit for the Period", {}).get("current")
    pbt = dual.get("Profit Before Tax", {}).get("current")
    tax = dual.get("Taxation Charge", {}).get("current")
    eps_c = dual.get("Earnings Per Share (AED)", {}).get("current")
    eps_p = dual.get("Earnings Per Share (AED)", {}).get("prior")
    nii_islamic = dual.get("Net Interest + Islamic Income", {}).get("current")

    # Balances (current)
    gross  = single.get("Gross Loans")
    ecl    = single.get("ECL")
    npl    = single.get("NPLs")
    assets = single.get("Total Assets")
    cash_cb = single.get("Cash & CB")
    due_banks = single.get("Due from Banks")
    inv_sec = single.get("Investment Securities")
    equity = single.get("Equity (Group Total)")

    # Balances (prior where available in dual)
    gross_prior = dual.get("Gross Loans and Receivables (dual)", {}).get("prior")
    depo_cur    = dual.get("Customer and Islamic Deposits", {}).get("current")
    depo_prior  = dual.get("Customer and Islamic Deposits", {}).get("prior")

    # --- Core set ---
    ratios = [
        ("Cost-to-Income",        safe_div(ga, toi)),
        ("Net Profit Margin",     safe_div(pat, toi)),
        ("Pre-impairment Margin", safe_div(opb, toi)),
        ("NPL Ratio",             safe_div(npl, gross)),
        ("Coverage Ratio",        safe_div(ecl, npl)),
        ("ECL/Gross Loans",       safe_div(ecl, gross)),
        ("Tax Rate",              safe_div(tax, pbt)),
        ("ROA",                   safe_div(pat, assets)),
        ("EPS YoY",               safe_div((eps_c - eps_p) if (eps_c is not None and eps_p is not None) else None, eps_p)),
    ]

    # --- Liquidity ---
    ratios += [
        ("Loan-to-Deposit (LDR)", safe_div(gross, depo_cur)),
        ("Liquid Assets % (Cash+Due+Inv)/Assets", safe_div(
            (cash_cb or 0) + (due_banks or 0) + (inv_sec or 0) if any(x is not None for x in [cash_cb, due_banks, inv_sec]) else None,
            assets
        )),
        # Placeholders (filtered out if None)
        ("NSFR (Basel III)", None),
        ("LCR (Basel III)", None),
    ]

    # --- Profitability / Efficiency additions ---
    earning_assets_proxy = None
    if any(v is not None for v in [gross, inv_sec, due_banks, cash_cb]):
        earning_assets_proxy = (gross or 0) + (inv_sec or 0) + (due_banks or 0) + (cash_cb or 0)
    ratios += [
        ("Operating Expense / Operating Income", safe_div(ga, toi)),
        ("Net Interest Margin (approx.)", safe_div(nii_islamic, earning_assets_proxy)),
        ("ROE", safe_div(pat, equity)),
        ("Interest Spread (placeholder)", None),
    ]

    # --- Asset quality enhancements ---
    net_npl_ratio = None
    if npl is not None and ecl is not None and gross is not None:
        numerator = max(npl - ecl, 0)
        denominator = gross - ecl if (gross - ecl) > 0 else None
        net_npl_ratio = safe_div(numerator, denominator)
    ratios += [
        ("Net NPL Ratio", net_npl_ratio),
        ("Stage 1 ECL / Gross Loans", None),
        ("Stage 2 ECL / Gross Loans", None),
        ("Stage 3 Coverage (ECL/Stage3)", None),
        ("Credit Cost Ratio", safe_div(None, gross)),
    ]

    # --- Growth (if prior values present) ---
    ratios += [
        ("Loan Growth YoY", safe_div((gross - gross_prior) if (gross is not None and gross_prior is not None) else None, gross_prior)),
        ("Deposit Growth YoY", safe_div((depo_cur - depo_prior) if (depo_cur is not None and depo_prior is not None) else None, depo_prior)),
    ]

    return filter_ratios(ratios)

def metrics_to_context(dual, single, ratios, units):
    lines = ["Key metrics & ratios:"]
    if units and units.get("units_label"):
        lines.append(f"Units detected: {units.get('currency','AED')} {units['units_label']}")
    # Only include already-filtered metrics
    for k, v in dual.items():
        lines.append(f"{k}: current={v['current']}, prior={v['prior']}")
    for k, v in single.items():
        lines.append(f"{k}: {v}")
    lines.append("Ratios:")
    for name, val in ratios:
        lines.append(f"{name}: {fmt_pct(val)}")
    return "\n".join(lines)

# --- Jinja filters (fixed) ---
@app.template_filter("pct")
def pct(v):
    return fmt_pct(v)

@app.template_filter("fmt_num")
def jinja_fmt_num(v):
    if v is None:
        return "N/A"
    try:
        return f"{float(v):,.2f}"
    except Exception:
        return str(v)

# ---------- Template (shows full data + chat) ----------
TEMPLATE = """
<!doctype html>
<title>Financial Analyzer</title>
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
        </div>
      </form>
      {% if upload_error %}<div class="text-danger mt-2">{{ upload_error }}</div>{% endif %}
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
        <div class="text-secondary">Upload a PDF first to give the assistant context. You can still type a question—I'll remind you.</div>
      {% endif %}
      <form method="post" action="{{ url_for('ask') }}">
        <textarea class="form-control" name="prompt" rows="5" placeholder="Ask about profitability, cost efficiency, credit quality, etc.">{{ prompt or '' }}</textarea>
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
    has_context = bool(session.get("financial_context"))
    ratios = session.get("financial_ratios")
    dual = session.get("financial_dual") or {}
    single = session.get("financial_single") or {}
    units = session.get("financial_units") or {"currency": "AED", "units_label": None}

    recs = []
    if ratios:
        d = dict(ratios)
        if d.get("Cost-to-Income") is not None and d["Cost-to-Income"] > 0.50:
            recs.append("High cost-to-income; review operating expenses.")
        if d.get("NPL Ratio") is not None and d["NPL Ratio"] > 0.06:
            recs.append("NPL ratio elevated; examine credit concentrations.")
        if d.get("Loan-to-Deposit (LDR)") is not None and d["Loan-to-Deposit (LDR)"] > 0.95:
            recs.append("LDR is high; consider deposit growth or term funding.")
        if d.get("Liquid Assets % (Cash+Due+Inv)/Assets") is not None and d["Liquid Assets % (Cash+Due+Inv)/Assets"] < 0.25:
            recs.append("Low liquid-asset buffer; monitor LCR/NSFR and short-term gaps.")

    return render_template_string(
        TEMPLATE,
        has_context=has_context,
        ratios=ratios,
        recs=recs,
        dual=dual,
        single=single,
        currency=units.get("currency"),
        units_label=units.get("units_label"),
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
            TEMPLATE,
            has_context=False, ratios=None, recs=None,
            dual={}, single={}, prompt=None, answer=None,
            currency="AED", units_label=None,
            error=None, upload_error="Please select a PDF."
        )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        f.save(tmp.name)
        dual_raw, single_raw, units = parse_pdf(tmp.name)
    finally:
        try:
            tmp.close(); os.unlink(tmp.name)
        except Exception:
            pass

    # Filter out N/A metrics BEFORE display/context
    dual = filter_dual_metrics(dual_raw)
    single = filter_single_metrics(single_raw)

    # Compute & FILTER ratios (remove None)
    ratios = compute_ratios(dual_raw, single_raw)  # compute using raw dicts (full data)
    ratios = filter_ratios(ratios)

    # store everything for re-display + chat
    session["financial_units"]  = units
    session["financial_context"] = metrics_to_context(dual, single, ratios, units)
    session["financial_ratios"]  = ratios
    session["financial_dual"]    = dual
    session["financial_single"]  = single

    # Recommendations using filtered ratios
    recs = []
    d = dict(ratios)
    if d.get("Cost-to-Income") is not None and d["Cost-to-Income"] > 0.50:
        recs.append("High cost-to-income; review operating expenses.")
    if d.get("NPL Ratio") is not None and d["NPL Ratio"] > 0.06:
        recs.append("NPL ratio elevated; examine credit concentrations.")
    if d.get("Loan-to-Deposit (LDR)") is not None and d["Loan-to-Deposit (LDR)"] > 0.95:
        recs.append("LDR is high; consider deposit growth or term funding.")
    if d.get("Liquid Assets % (Cash+Due+Inv)/Assets") is not None and d["Liquid Assets % (Cash+Due+Inv)/Assets"] < 0.25:
        recs.append("Low liquid-asset buffer; monitor LCR/NSFR and short-term gaps.")

    return render_template_string(
        TEMPLATE,
        has_context=True, ratios=ratios, recs=recs,
        dual=dual, single=single,
        currency=units.get("currency"),
        units_label=units.get("units_label"),
        prompt=None, answer=None,
        error=None, upload_error=None
    )

@app.route("/ask", methods=["POST"])
def ask():
    prompt = (request.form.get("prompt") or "").strip()
    context = session.get("financial_context")
    ratios = session.get("financial_ratios")
    dual   = session.get("financial_dual") or {}
    single = session.get("financial_single") or {}
    units  = session.get("financial_units") or {"currency": "AED", "units_label": None}

    if not context or not ratios:
        return render_template_string(
            TEMPLATE,
            has_context=False, ratios=None, recs=None,
            dual={}, single={}, prompt=prompt, answer=None,
            currency=units.get("currency"), units_label=units.get("units_label"),
            error="Please upload a PDF first.", upload_error=None
        )

    answer = None
    if prompt and client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a bank financial analyst. Be concise and numeric."},
                    {"role": "user", "content": f"{context}\n\nUser prompt: {prompt}"},
                ],
                temperature=0.2,
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"[OpenAI error] {e}"

    # rebuild recs using filtered ratios already in session
    recs = []
    d = dict(ratios)
    if d.get("Cost-to-Income") is not None and d["Cost-to-Income"] > 0.50:
        recs.append("High cost-to-income; review operating expenses.")
    if d.get("NPL Ratio") is not None and d["NPL Ratio"] > 0.06:
        recs.append("NPL ratio elevated; examine credit concentrations.")
    if d.get("Loan-to-Deposit (LDR)") is not None and d["Loan-to-Deposit (LDR)"] > 0.95:
        recs.append("LDR is high; consider deposit growth or term funding.")
    if d.get("Liquid Assets % (Cash+Due+Inv)/Assets") is not None and d["Liquid Assets % (Cash+Due+Inv)/Assets"] < 0.25:
        recs.append("Low liquid-asset buffer; monitor LCR/NSFR and short-term gaps.")

    return render_template_string(
        TEMPLATE,
        has_context=True, ratios=ratios, recs=recs,
        dual=dual, single=single,
        currency=units.get("currency"), units_label=units.get("units_label"),
        prompt=prompt, answer=answer,
        error=None, upload_error=None
    )

@app.route("/clear")
def clear():
    for k in ["financial_context", "financial_ratios", "financial_dual", "financial_single", "financial_units"]:
        session.pop(k, None)
    return redirect(url_for("home"))

@app.route("/debug")
def debug():
    u = session.get("financial_units") or {}
    return {
        "has_context": bool(session.get("financial_context")),
        "has_ratios": bool(session.get("financial_ratios")),
        "dual_keys": list((session.get("financial_dual") or {}).keys()),
        "single_keys": list((session.get("financial_single") or {}).keys()),
        "units": u,
    }

# Optional CLI mode
def cli_chat():
    if not client:
        print("OPENAI_API_KEY not configured in C:\\EUacademy\\.env")
        return
    context = ""
    try:
        pdf_path = input("PDF path for context (Enter to skip): ").strip()
        if pdf_path:
            dual_raw, single_raw, units = parse_pdf(pdf_path)
            dual = filter_dual_metrics(dual_raw)
            single = filter_single_metrics(single_raw)
            ratios_local = filter_ratios(compute_ratios(dual_raw, single_raw))
            context = metrics_to_context(dual, single, ratios_local, units)
            print("Parsed PDF. Context prepared.")
        else:
            print("No PDF context. Chatting without financial context.")
    except Exception as e:
        print(f"[PDF parse skipped] {e}")

    print("\n💬 Chat mode (type 'q' to quit)")
    while True:
        q = input("\nYour prompt> ").strip()
        if q.lower() == "q":
            print("Bye!")
            break
        if not q:
            continue
        try:
            msg = (f"{context}\n\nUser prompt: {q}") if context else q
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a bank financial analyst. Be concise and numeric."},
                    {"role": "user", "content": msg},
                ],
                temperature=0.2,
            )
            print("\nAssistant:", resp.choices[0].message.content.strip())
        except Exception as e:
            print(f"[OpenAI error] {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        cli_chat()
    else:
        print("Financial app on http://127.0.0.1:5069")
        app.run(host="127.0.0.1", port=5069, debug=True, use_reloader=False)
