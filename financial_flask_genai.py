# financial_flask_genai.py
from flask import Flask, request, render_template_string, session, redirect, url_for
import fitz, tempfile, re, os, sys
from dotenv import load_dotenv
from openai import OpenAI

# --- API + Flask setup ---
load_dotenv("C:\\EUacademy\\.env")
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

# Regexes for ENBD-style statements (tweak as needed)
patterns_dual = {
    "Total Operating Income": r"Total operating income\s+([\d,]+)\s+([\d,]+)",
    "General and Administrative Expenses": r"General and administrative expenses\s+\(([\d,]+)\)\s+\(([\d,]+)\)",
    "Operating Profit Before Impairment": r"Operating profit before impairment\s+([\d,]+)\s+([\d,]+)",
    "Profit Before Tax": r"Profit for the period before taxation\s+([\d,]+)\s+([\d,]+)",
    "Taxation Charge": r"Taxation charge\s+\(([\d,]+)\)\s+\(([\d,]+)\)",
    "Profit for the Period": r"Profit for the period\s+([\d,]+)\s+([\d,]+)",
    "Earnings Per Share (AED)": r"Earnings per share\s*\(AED\)\s+([\d\.]+)\s+([\d\.]+)",
}
patterns_single = {
    "Gross Loans": r"Gross loans and receivables\s+([\d,]+)\s+[\d,]+",
    "ECL": r"Less:\s*Expected credit losses\s+\(([\d,]+)\)\s+\([\d,]+\)",
    "NPLs": r"Total of credit impaired loans and receivables\s+([\d,]+)\s+[\d,]+",
    "Total Assets": r"Segment Assets[\s\S]*?(\d{1,3}(?:,\d{3})+)\s*\n\s*Segment Liabilities",
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
    return extract_dual(txt), extract_single(txt)

def compute_ratios(dual, single):
    toi = dual["Total Operating Income"]["current"]
    ga  = dual["General and Administrative Expenses"]["current"]
    opb = dual["Operating Profit Before Impairment"]["current"]
    pat = dual["Profit for the Period"]["current"]
    pbt = dual["Profit Before Tax"]["current"]
    tax = dual["Taxation Charge"]["current"]
    eps_c = dual["Earnings Per Share (AED)"]["current"]
    eps_p = dual["Earnings Per Share (AED)"]["prior"]

    gross  = single["Gross Loans"]
    ecl    = single["ECL"]
    npl    = single["NPLs"]
    assets = single["Total Assets"]

    return [
        ("Cost-to-Income",        safe_div(ga, toi)),
        ("Net Profit Margin",     safe_div(pat, toi)),
        ("Pre-impairment Margin", safe_div(opb, toi)),
        ("NPL Ratio",             safe_div(npl, gross)),
        ("Coverage Ratio",        safe_div(ecl, npl)),
        ("ECL/Gross Loans",       safe_div(ecl, gross)),
        ("Tax Rate",              safe_div(tax, pbt)),
        ("ROA",                   safe_div(pat, assets)),
        ("EPS YoY",               safe_div((eps_c - eps_p) if (eps_c is not None and eps_p) else None, eps_p)),
    ]

def metrics_to_context(dual, single, ratios):
    lines = ["Key metrics & ratios:"]
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
      <h3 class="card-title">Extracted Metrics</h3>
      <div class="row">
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
    recs = []
    if ratios:
        d = dict(ratios)
        if d.get("Cost-to-Income") is not None and d["Cost-to-Income"] > 0.50:
            recs.append("High cost-to-income; review operating expenses.")
        if d.get("NPL Ratio") is not None and d["NPL Ratio"] > 0.06:
            recs.append("NPL ratio elevated; examine credit concentrations.")
    return render_template_string(
        TEMPLATE,
        has_context=has_context,
        ratios=ratios,
        recs=recs,
        dual=dual,
        single=single,
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
            error=None, upload_error="Please select a PDF."
        )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        f.save(tmp.name)
        dual, single = parse_pdf(tmp.name)
    finally:
        try:
            tmp.close(); os.unlink(tmp.name)
        except Exception:
            pass

    ratios = compute_ratios(dual, single)

    # store everything for re-display + chat
    session["financial_context"] = metrics_to_context(dual, single, ratios)
    session["financial_ratios"]  = ratios
    session["financial_dual"]    = dual
    session["financial_single"]  = single

    recs = []
    d = dict(ratios)
    if d.get("Cost-to-Income") is not None and d["Cost-to-Income"] > 0.50:
        recs.append("High cost-to-income; review operating expenses.")
    if d.get("NPL Ratio") is not None and d["NPL Ratio"] > 0.06:
        recs.append("NPL ratio elevated; examine credit concentrations.")

    return render_template_string(
        TEMPLATE,
        has_context=True, ratios=ratios, recs=recs,
        dual=dual, single=single,
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

    if not context or not ratios:
        return render_template_string(
            TEMPLATE,
            has_context=False, ratios=None, recs=None,
            dual={}, single={}, prompt=prompt, answer=None,
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

    # rebuild recs
    recs = []
    d = dict(ratios)
    if d.get("Cost-to-Income") is not None and d["Cost-to-Income"] > 0.50:
        recs.append("High cost-to-income; review operating expenses.")
    if d.get("NPL Ratio") is not None and d["NPL Ratio"] > 0.06:
        recs.append("NPL ratio elevated; examine credit concentrations.")

    return render_template_string(
        TEMPLATE,
        has_context=True, ratios=ratios, recs=recs,
        dual=dual, single=single,
        prompt=prompt, answer=answer,
        error=None, upload_error=None
    )

@app.route("/clear")
def clear():
    for k in ["financial_context", "financial_ratios", "financial_dual", "financial_single"]:
        session.pop(k, None)
    return redirect(url_for("home"))

@app.route("/debug")
def debug():
    return {
        "has_context": bool(session.get("financial_context")),
        "has_ratios": bool(session.get("financial_ratios")),
        "dual_keys": list((session.get("financial_dual") or {}).keys()),
        "single_keys": list((session.get("financial_single") or {}).keys()),
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
            dual, single = parse_pdf(pdf_path)
            ratios_local = compute_ratios(dual, single)
            context = metrics_to_context(dual, single, ratios_local)
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
        print("Financial app on http://127.0.0.1:5053")
        app.run(host="127.0.0.1", port=5053, debug=True, use_reloader=False)
