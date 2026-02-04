
# Financial PDF Analyzer (HDFC / ENBD) + Agentic Comparator

This repo contains small Flask apps + a CLI to:

- Extract key financial statement line-items from bank annual-report PDFs (regex + PDF text extraction; HDFC also supports OCR fallback)
- Compute useful ratios and basic recommendations
- Optionally use OpenAI to answer questions about the extracted metrics
- Compare the two extractors on the same PDF(s) and recommend the best fit (`agentic_flow.py`)

## What’s in here

- `hdfc_extraction_1.0.py` — HDFC analyzer (runs on `http://127.0.0.1:5071`) with OCR fallback via Tesseract.
- `enbd_extraction_1.0.py` — ENBD analyzer (runs on `http://127.0.0.1:5083`).
- `agentic_flow.py` — “Agentic Flow” UI + API + CLI to run both extractors and pick a winner (runs on `http://127.0.0.1:5099` by default).
- `encrypt_keys.py` — helper to encrypt API keys for use via env vars.

## Requirements

- Windows + Python `>= 3.12` (see `pyproject.toml`)
- Recommended: a virtual environment
- Optional (for HDFC OCR): Tesseract OCR installed
- Optional (for AI Q&A / LLM judge): OpenAI API key

## Setup (Windows / PowerShell)

1) Create + activate a venv

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

This repo has both `pyproject.toml` (minimal deps) and `requirements.txt` (broader deps).

Recommended (most features):

```powershell
python -m pip install -U pip
pip install -r requirements.txt
```

3) Configure environment variables

- Copy `.env.example` → `.env` and fill values (recommended)
- Or set env vars in your shell/session

## Environment variables

Common:

- `FLASK_SECRET_KEY` — Flask session secret (dev default exists, but set this for real usage)

OpenAI (optional):

- `OPENAI_API_KEY_ENCRYPTED` — encrypted OpenAI key (recommended)
- `OPENAI_PASSPHRASE` — passphrase used for encryption/decryption (default: `default_salt_2024`)
- `OPENAI_API_KEY` — plaintext OpenAI key (supported, but not recommended)

Agentic Flow (optional):

- `AGENTIC_FLOW_CHAT_MODEL` — chat model used by `agentic_flow.py` (default: `gpt-4.1`)

HDFC OCR (optional):

- `TESSERACT_CMD` — full path to `tesseract.exe` if it’s not on `PATH`
- `HDFC_TEST_PDF` — used by the `/test-pdf` debug route

## Encrypting your OpenAI key (recommended)

The apps support a simple XOR+SHA256+base64 scheme for storing an encrypted key in env vars.

1) Run:

```powershell
python encrypt_keys.py
```

2) Paste the printed encrypted value into your `.env` as `OPENAI_API_KEY_ENCRYPTED`.

3) If you used a custom passphrase, set the same value in `.env` as `OPENAI_PASSPHRASE`.

Note: don’t commit `.env` or API keys.

## Running the apps

### HDFC analyzer (Flask)

```powershell
python hdfc_extraction_1.0.py
```

- Open: `http://127.0.0.1:5071`
- Helpful debug endpoints:
	- `/debug` — shows current extraction + OpenAI/OCR status
	- `/test-pdf` — uses `HDFC_TEST_PDF` to run a fixed test PDF

### ENBD analyzer (Flask)

```powershell
python enbd_extraction_1.0.py
```

- Open: `http://127.0.0.1:5083`
- Helpful debug endpoint: `/debug`

### Agentic Flow (Flask UI + API)

Run the UI (default):

```powershell
python agentic_flow.py --serve
```

Custom host/port:

```powershell
python agentic_flow.py --serve --host 127.0.0.1 --port 5099
```

- Open: `http://127.0.0.1:5099`
- Upload one or more PDFs and it will run both extractors, score them, and recommend the best.

#### Agentic Flow CLI mode

Analyze one PDF:

```powershell
python agentic_flow.py --pdf path\to\report.pdf
```

Analyze multiple PDFs:

```powershell
python agentic_flow.py --pdf a.pdf b.pdf c.pdf
```

Print full JSON:

```powershell
python agentic_flow.py --pdf a.pdf --json
```

Enable optional “LLM judge” (requires OpenAI configured):

```powershell
python agentic_flow.py --pdf a.pdf --llm
```

## OCR notes (HDFC)

For scanned PDFs, `hdfc_extraction_1.0.py` can OCR pages using Tesseract.

- Install Tesseract and ensure `tesseract.exe` is on PATH, or set `TESSERACT_CMD`.
- Common Windows path:
	- `C:\Program Files\Tesseract-OCR\tesseract.exe`

If OCR is missing, the app will still run but extraction quality may drop on scanned documents.

## How extraction works (high level)

- PDF text is extracted via PyMuPDF (`fitz`).
- Each extractor uses regex patterns tuned for its bank’s statement format.
- Ratios are computed from extracted line items; missing values are skipped.
- If OpenAI is configured, you can ask questions in the UI about the extracted context.

## Troubleshooting

- **OpenAI not configured**: set `OPENAI_API_KEY_ENCRYPTED` + `OPENAI_PASSPHRASE` (or `OPENAI_API_KEY`).
- **OCR not available**: install Tesseract or set `TESSERACT_CMD`.
- **PDF upload too large**: Flask max upload is set to 32MB in the apps.

## Disclaimer

This is best-effort extraction from semi-structured PDFs. Results can be incomplete or incorrect depending on the report layout, scan quality, and formatting. Validate outputs before using them for decisions.

