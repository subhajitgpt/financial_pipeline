"""Generate n8n workflow JSON files.

This repo doesn't depend on n8n libraries; workflows are plain JSON.

Usage (PowerShell):
  python n8n\build_workflows.py

It writes:
  n8n/dist/agentic_flow_webhook.json

Import that file into n8n (Workflow → Import from File).

Notes:
- n8n's built-in Code node is JavaScript; to run Python we use Execute Command.
- This workflow expects a JSON body with at least: { "pdfPath": "C:\\path\\file.pdf" }
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Node:
    id: str
    name: str
    type: str
    typeVersion: int
    position: List[int]
    parameters: Dict[str, Any]


def _workflow_base(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "nodes": [],
        "connections": {},
        "settings": {},
        "active": False,
        "pinData": {},
        "staticData": None,
        "meta": {"instanceId": ""},
        "tags": [],
    }


def _add_node(wf: Dict[str, Any], node: Node) -> None:
    wf["nodes"].append(
        {
            "id": node.id,
            "name": node.name,
            "type": node.type,
            "typeVersion": node.typeVersion,
            "position": node.position,
            "parameters": node.parameters,
        }
    )


def _connect(wf: Dict[str, Any], from_name: str, to_name: str) -> None:
    # n8n connection structure: connections[fromNode]["main"][0] = [{ node: toNode, type: "main", index: 0 }]
    conns = wf.setdefault("connections", {})
    conns.setdefault(from_name, {}).setdefault("main", []).append([
        {"node": to_name, "type": "main", "index": 0}
    ])


def build_agentic_flow_webhook_workflow(
    *,
    workflow_name: str = "Agentic Flow (Webhook → Python CLI)",
    webhook_path: str = "agentic-flow/analyze",
    python_exe: str = "python",
    script_rel_path: str = "agentic_flow.py",
) -> Dict[str, Any]:
    """Webhook accepts JSON {pdfPath, llm?} and runs agentic_flow.py CLI."""

    wf = _workflow_base(workflow_name)

    webhook = Node(
        id="b9f3d0a2-9bdf-4c7f-a1aa-2d3c2e8f4a01",
        name="Webhook",
        type="n8n-nodes-base.webhook",
        typeVersion=2,
        position=[400, 300],
        parameters={
            "httpMethod": "POST",
            "path": webhook_path,
            "responseMode": "responseNode",
            "options": {},
        },
    )

    # Build a command that supports optional LLM flag.
    # - Always uses --json for machine-readable output.
    # - Adds --llm if body.llm is truthy.
    # n8n expressions start with = and are evaluated in a string field.
    command_expr = (
        "={{ "
        "'" + python_exe + "' "
        "+ '" + script_rel_path + "' "
        "+ '--pdf ' + ($json.pdfPath || '') + ' --json' "
        "+ (($json.llm || false) ? ' --llm' : '') "
        "}}"
    )

    exec_cmd = Node(
        id="b9f3d0a2-9bdf-4c7f-a1aa-2d3c2e8f4a02",
        name="Run Agentic Flow CLI",
        type="n8n-nodes-base.executeCommand",
        typeVersion=2,
        position=[680, 300],
        parameters={
            "command": command_expr,
            "options": {
                # Keep the working directory at the n8n process cwd; if you run n8n elsewhere,
                # use an absolute script path in script_rel_path.
            },
        },
    )

    respond = Node(
        id="b9f3d0a2-9bdf-4c7f-a1aa-2d3c2e8f4a03",
        name="Respond to Webhook",
        type="n8n-nodes-base.respondToWebhook",
        typeVersion=1,
        position=[960, 300],
        parameters={
            "responseCode": 200,
            "responseData": "firstEntryJson",
            "options": {
                "responseHeaders": {
                    "entries": [
                        {"name": "Content-Type", "value": "application/json"}
                    ]
                }
            },
            # executeCommand returns { stdout, stderr, exitCode }.
            # We return stdout as the HTTP body.
            "responseBody": "={{$json.stdout}}",
        },
    )

    _add_node(wf, webhook)
    _add_node(wf, exec_cmd)
    _add_node(wf, respond)

    _connect(wf, "Webhook", "Run Agentic Flow CLI")
    _connect(wf, "Run Agentic Flow CLI", "Respond to Webhook")

    return wf


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "dist")
    os.makedirs(out_dir, exist_ok=True)

    wf = build_agentic_flow_webhook_workflow()
    out_path = os.path.join(out_dir, "agentic_flow_webhook.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
