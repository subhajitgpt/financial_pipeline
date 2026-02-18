# n8n workflow: run `agentic_flow.py` from a webhook

This folder contains an importable n8n workflow JSON plus a small Python generator.

The workflow is:

Webhook (POST JSON) → Execute Command (runs Python CLI) → Respond to Webhook

## What’s included

- `n8n/dist/agentic_flow_webhook.json`
  - Import this into n8n.
- `n8n/build_workflows.py`
  - Regenerates the JSON (useful if you want to change the webhook path, python executable, or script path).

## Prerequisites

1) Python is available where n8n runs

- n8n runs the command on the same machine/container as n8n.
- Ensure `python` works in that environment (or use an absolute python path).

2) This repo (or at least `agentic_flow.py`) is accessible where n8n runs

- If n8n runs on your Windows host: you can reference `C:\\Financial\\agentic_flow.py`.
- If n8n runs in Docker: mount the repo as a volume (see Docker notes below).

3) Execute Command node availability

Some n8n deployments disable “Execute Command” for security reasons. If the node is missing or blocked, enable it according to your deployment policy/docs.

## Generate the workflow JSON (optional)

The workflow JSON is already checked in under `n8n/dist/`, but you can regenerate it:

```powershell
python n8n\build_workflows.py
```

## Import into n8n

In n8n UI:

1. Workflows → **Import from File**
2. Select `n8n/dist/agentic_flow_webhook.json`
3. Open the “Run Agentic Flow CLI” node and confirm the command points at the correct Python + script path.

### Important: script path

The generated workflow uses a relative script name (`agentic_flow.py`). If n8n is not started from the repo root, change the command to an absolute path, for example:

```text
python C:\Financial\agentic_flow.py --pdf {{$json.pdfPath}} --json
```

You can edit it directly in the node, or update `n8n/build_workflows.py` and regenerate.

## Activate and run

1) Activate the workflow in n8n.

2) Call the webhook endpoint with JSON:

```json
{
  "pdfPath": "C:\\Financial\\data\\report.pdf",
  "llm": false
}
```

The response body is whatever `agentic_flow.py` prints to stdout (it uses `--json` so it should be JSON).

### Test call (Windows PowerShell)

Replace `<N8N_BASE_URL>` with your n8n URL (e.g. `http://localhost:5678`).

```powershell
$body = @{ pdfPath = 'C:\\Financial\\data\\report.pdf'; llm = $false } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "<N8N_BASE_URL>/webhook/agentic-flow/analyze" -ContentType "application/json" -Body $body
```

If you’re using the “test” webhook URL in n8n (while editing, not activated), n8n typically exposes a different path (often `/webhook-test/...`). Use the URL shown in the Webhook node panel.

## Docker notes (if n8n runs in a container)

- Mount the repo (or at least the scripts + PDF folder) into the container.
- Use container paths in `pdfPath` (e.g. `/data/report.pdf`).
- Update the Execute Command node to call the script by container path.

Example (conceptual):

- Host: `C:\\Financial` mounted to container `/workspace`
- Command: `python /workspace/agentic_flow.py --pdf {{$json.pdfPath}} --json`
- Request body: `{ "pdfPath": "/workspace/data/report.pdf" }`

## Troubleshooting

- If you get empty output: check the Execute Command node output for `stderr` and `exitCode`.
- If `pdfPath` contains spaces: JSON is fine, but ensure the command quoting is correct if you hard-code paths.
- If OpenAI is enabled (`llm: true`): ensure the same environment variables used locally are available to the n8n process/container.
