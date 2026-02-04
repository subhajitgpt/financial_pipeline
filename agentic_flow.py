import argparse
import base64
import hashlib
import importlib.util
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import fitz
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# ===================== OpenAI Key Decryption (matches existing scripts) =====================
def decrypt_key(encrypted_data: str, passphrase: str = "default_salt_2024") -> str:
    if not encrypted_data:
        return ""
    try:
        encrypted_bytes = base64.b64decode(encrypted_data)
        key_hash = hashlib.sha256(passphrase.encode("utf-8")).digest()

        decrypted = bytearray()
        for i, byte in enumerate(encrypted_bytes):
            decrypted.append(byte ^ key_hash[i % len(key_hash)])

        return decrypted.decode("utf-8")
    except Exception:
        return ""


def _load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _workspace_file(*parts: str) -> str:
    base = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base, *parts)


HDFC_EXTRACTOR_PATH = _workspace_file("hdfc_extraction_1.0.py")
ENBD_EXTRACTOR_PATH = _workspace_file("enbd_extraction_1.0.py")

_HDFC_MOD = None
_ENBD_MOD = None


def _get_extractors():
    global _HDFC_MOD, _ENBD_MOD

    if _HDFC_MOD is None:
        _HDFC_MOD = _load_module_from_path("hdfc_extractor_v1", HDFC_EXTRACTOR_PATH)
    if _ENBD_MOD is None:
        _ENBD_MOD = _load_module_from_path("enbd_extractor_v1", ENBD_EXTRACTOR_PATH)

    return _HDFC_MOD, _ENBD_MOD


@dataclass
class ExtractorRun:
    name: str
    ok: bool
    elapsed_s: float
    error: Optional[str] = None
    units: Optional[Dict[str, Any]] = None
    dual_raw: Optional[Dict[str, Any]] = None
    single_raw: Optional[Dict[str, Any]] = None
    dual: Optional[Dict[str, Any]] = None
    single: Optional[Dict[str, Any]] = None
    ratios: Optional[List[Tuple[str, Any]]] = None
    recs: Optional[List[str]] = None
    context: Optional[str] = None


@dataclass
class ExtractorScore:
    total: float
    fill_rate: float
    filled_metrics: int
    expected_metrics: int
    ratio_count: int
    invalid_ratio_count: int
    runtime_score: float
    units_score: float
    bank_hint_bonus: float


@dataclass
class OrgPerformance:
    organisation: str
    score: float
    metrics_used: int
    metrics_available: int
    details: Dict[str, Any]


def _extract_native_text(pdf_path: str, max_chars: int = 12000) -> str:
    try:
        with fitz.open(pdf_path) as doc:
            parts = []
            total = 0
            for pg in doc:
                t = pg.get_text("text") or ""
                if t:
                    parts.append(t)
                    total += len(t)
                    if total >= max_chars:
                        break
        return "\n".join(parts)[:max_chars]
    except Exception:
        return ""


def _detect_bank_hint(text: str) -> Dict[str, Any]:
    t = (text or "").lower()

    hints = {
        "likely": None,
        "confidence": 0.0,
        "signals": [],
    }

    def add(signal: str, weight: float, likely: str):
        hints["signals"].append(signal)
        if hints["confidence"] < weight:
            hints["confidence"] = weight
            hints["likely"] = likely

    if re.search(r"\bhdfc\b", t) or re.search(r"hdfc\s+bank", t):
        add("keyword:hdfc", 0.9, "HDFC")
    if re.search(r"\b(emirates\s+nbd|enbd)\b", t):
        add("keyword:enbd", 0.9, "ENBD")

    if "₹" in text or re.search(r"\b(inr|rupees|rs\.)\b", t):
        add("currency:inr", 0.6, "HDFC")
    if re.search(r"\b(aed|uae\s*dirhams|dirhams)\b", t):
        add("currency:aed", 0.6, "ENBD")

    if re.search(r"\brbi\b", t) or re.search(r"\bgross\s+npa\b", t):
        add("domain:india-banking", 0.55, "HDFC")
    if re.search(r"\bislamic\b", t) or re.search(r"\bcentral\s+banks\b", t):
        add("domain:uae-banking", 0.55, "ENBD")

    return hints


def _count_filled_metrics(dual_raw: Optional[Dict[str, Any]], single_raw: Optional[Dict[str, Any]]) -> Tuple[int, int]:
    dual_raw = dual_raw or {}
    single_raw = single_raw or {}

    expected = len(dual_raw) + len(single_raw)

    filled_dual = 0
    for v in dual_raw.values():
        if isinstance(v, dict) and (v.get("current") is not None or v.get("prior") is not None):
            filled_dual += 1
    filled_single = sum(1 for v in single_raw.values() if v is not None)

    return filled_dual + filled_single, expected


def _ratio_invalid(name: str, val: Any) -> bool:
    try:
        x = float(val)
    except Exception:
        return True

    # broad sanity guardrails; these are generic so they don’t over-penalize edge cases
    if not (-10.0 <= x <= 10.0):
        return True

    n = (name or "").lower()
    if "ratio" in n or "margin" in n or "rate" in n or "roa" in n or "roe" in n or "ldr" in n:
        if not (-2.0 <= x <= 5.0):
            return True

    return False


def _score_run(run: ExtractorRun, bank_hint: Dict[str, Any]) -> ExtractorScore:
    filled, expected = _count_filled_metrics(run.dual_raw, run.single_raw)
    fill_rate = (filled / expected) if expected > 0 else 0.0

    ratios = run.ratios or []
    invalid_ratio_count = sum(1 for n, v in ratios if _ratio_invalid(n, v))
    ratio_count = len(ratios)

    # runtime score in [0..1]
    runtime_score = 1.0 / (1.0 + max(0.0, float(run.elapsed_s)))

    units = run.units or {}
    units_score = 0.0
    if units.get("currency"):
        units_score += 0.6
    if units.get("units_label"):
        units_score += 0.4

    likely = bank_hint.get("likely")
    conf = float(bank_hint.get("confidence") or 0.0)
    bank_hint_bonus = 0.0
    if likely and conf >= 0.55:
        bank_hint_bonus = (0.08 * conf) if likely == run.name else (-0.03 * conf)

    # weighted total in [roughly 0..1]
    total = 0.0
    total += 0.55 * fill_rate
    total += 0.20 * min(1.0, ratio_count / 10.0)
    total += 0.10 * units_score
    total += 0.10 * runtime_score
    total += bank_hint_bonus

    # penalize invalid ratios lightly
    if ratio_count > 0:
        total -= 0.08 * (invalid_ratio_count / ratio_count)

    total = max(0.0, min(1.0, total))

    return ExtractorScore(
        total=total,
        fill_rate=fill_rate,
        filled_metrics=filled,
        expected_metrics=expected,
        ratio_count=ratio_count,
        invalid_ratio_count=invalid_ratio_count,
        runtime_score=runtime_score,
        units_score=units_score,
        bank_hint_bonus=bank_hint_bonus,
    )


def _ratios_to_dict(ratios: Optional[List[Tuple[str, Any]]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, val in (ratios or []):
        try:
            out[str(name).strip().lower()] = float(val)
        except Exception:
            continue
    return out


def _find_ratio_value(ratio_map: Dict[str, float], *name_contains: str) -> Optional[float]:
    if not ratio_map:
        return None
    needles = [n.strip().lower() for n in name_contains if n]
    for k, v in ratio_map.items():
        if all(n in k for n in needles):
            return v
    return None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# ===================== Cross-bank key metrics (for UI + explainability) =====================
STANDARD_METRICS: List[Dict[str, Any]] = [
    {
        "key": "net_profit_margin",
        "label": "Net Profit Margin",
        "better": "higher",
        "fmt": "pct",
        "candidates": [("net", "profit", "margin")],
    },
    {
        "key": "roa",
        "label": "ROA",
        "better": "higher",
        "fmt": "pct",
        "candidates": [("roa",)],
    },
    {
        "key": "roe",
        "label": "ROE",
        "better": "higher",
        "fmt": "pct",
        "candidates": [("roe",)],
    },
    {
        "key": "cost_to_income",
        "label": "Cost-to-Income",
        "better": "lower",
        "fmt": "pct",
        "candidates": [("cost-to-income",), ("cost", "income")],
    },
    {
        "key": "asset_quality",
        "label": "NPL/NPA",
        "better": "lower",
        "fmt": "pct",
        "candidates": [("npl",), ("gross", "npa"), ("net", "npa")],
    },
    {
        "key": "loan_to_deposit",
        "label": "Loan-to-Deposit (LDR)",
        "better": "middle",
        "fmt": "pct",
        "candidates": [("loan-to-deposit",), ("ldr",)],
    },
    {
        "key": "liquid_assets_pct",
        "label": "Liquid Assets %",
        "better": "higher",
        "fmt": "pct",
        "candidates": [("liquid", "assets"), ("cash", "inv")],
    },
]


def _pick_ratio_value(ratio_map: Dict[str, float], candidates: List[Tuple[str, ...]]) -> Optional[float]:
    for cand in candidates:
        v = _find_ratio_value(ratio_map, *cand)
        if v is not None:
            return float(v)
    return None


def _standard_metrics_from_ratios(ratios: Optional[List[Tuple[str, Any]]]) -> Dict[str, Optional[float]]:
    ratio_map = _ratios_to_dict(ratios)
    out: Dict[str, Optional[float]] = {}
    for m in STANDARD_METRICS:
        out[m["key"]] = _pick_ratio_value(ratio_map, m.get("candidates") or [])
    return out


def _metric_advantage(a_val: Optional[float], b_val: Optional[float], better: str) -> Optional[float]:
    if a_val is None or b_val is None:
        return None
    if better == "higher":
        return a_val - b_val
    if better == "lower":
        return b_val - a_val
    if better == "middle":
        # For LDR, closer to ~85% is better (broad heuristic)
        target = 0.85
        return abs(b_val - target) - abs(a_val - target)
    return None


def _format_metric_for_reason(metric: Dict[str, Any], a_val: Optional[float], b_val: Optional[float]) -> str:
    label = metric.get("label") or metric.get("key")
    if a_val is None or b_val is None:
        return f"{label}: insufficient data"
    if metric.get("fmt") == "pct":
        return f"{label}: {a_val*100:.2f}% vs {b_val*100:.2f}%"
    return f"{label}: {a_val} vs {b_val}"


def _build_strengths_weaknesses(org_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    """Return per-org strengths/weaknesses comparing to the strongest other org."""
    org_names = list(org_metrics.keys())
    out: Dict[str, Dict[str, List[str]]] = {}
    for org in org_names:
        a = org_metrics.get(org) or {}
        # choose comparator: highest perf score among others
        others = [o for o in org_names if o != org]
        if not others:
            out[org] = {"strengths": [], "weaknesses": []}
            continue

        # Comparator = best other by perf score, fallback to first
        comp = max(
            others,
            key=lambda o: float((org_metrics.get(o) or {}).get("avg_perf_score") or 0.0),
        )
        b = org_metrics.get(comp) or {}

        scored: List[Tuple[float, str, str]] = []
        for metric in STANDARD_METRICS:
            key = metric["key"]
            adv = _metric_advantage(a.get("metrics", {}).get(key), b.get("metrics", {}).get(key), metric.get("better"))
            if adv is None:
                continue
            reason = _format_metric_for_reason(metric, a.get("metrics", {}).get(key), b.get("metrics", {}).get(key))
            scored.append((float(adv), key, reason))

        strengths = [f"{reason}" for adv, _k, reason in sorted(scored, key=lambda x: x[0], reverse=True) if adv > 0][:3]
        weaknesses = [f"{reason}" for adv, _k, reason in sorted(scored, key=lambda x: x[0]) if adv < 0][:3]
        out[org] = {"strengths": strengths, "weaknesses": weaknesses}

    return out


def _org_performance_from_ratios(organisation: str, ratios: Optional[List[Tuple[str, Any]]]) -> OrgPerformance:
    """Compute a light-weight, cross-bank comparable performance score from ratios.

    This is NOT a valuation model; it's a simple, explainable scoring heuristic.
    """
    ratio_map = _ratios_to_dict(ratios)

    npm = _find_ratio_value(ratio_map, "net", "profit", "margin")
    cost_income = _find_ratio_value(ratio_map, "cost-to-income") or _find_ratio_value(ratio_map, "cost", "income")
    roa = _find_ratio_value(ratio_map, "roa")
    roe = _find_ratio_value(ratio_map, "roe")
    npl = (
        _find_ratio_value(ratio_map, "npl")
        or _find_ratio_value(ratio_map, "gross", "npa")
        or _find_ratio_value(ratio_map, "net", "npa")
    )

    # Normalize each metric into [0..1] with broad caps; ignore missing metrics.
    metrics: List[Tuple[str, Optional[float], float]] = [
        ("Net Profit Margin", npm, 0.30),
        ("ROA", roa, 0.25),
        ("ROE", roe, 0.20),
        ("Cost-to-Income (lower is better)", cost_income, 0.15),
        ("NPL/NPA (lower is better)", npl, 0.10),
    ]

    def normalize(name: str, val: float) -> float:
        n = name.lower()
        if "net profit margin" in n:
            return _clamp01(val / 0.40)  # 40% -> 1.0
        if n == "roa":
            return _clamp01(val / 0.03)  # 3% -> 1.0
        if n == "roe":
            return _clamp01(val / 0.25)  # 25% -> 1.0
        if "cost-to-income" in n:
            return _clamp01(1.0 - (val / 0.70))  # 70% -> 0.0
        if "npl/npa" in n:
            return _clamp01(1.0 - (val / 0.08))  # 8% -> 0.0
        return 0.0

    weight_sum = 0.0
    score_sum = 0.0
    used = 0
    details: Dict[str, Any] = {}
    for metric_name, raw_val, weight in metrics:
        if raw_val is None:
            continue
        used += 1
        weight_sum += weight
        score_sum += weight * normalize(metric_name, float(raw_val))
        details[metric_name] = {"value": raw_val}

    score = (score_sum / weight_sum) if weight_sum > 0 else 0.0
    return OrgPerformance(
        organisation=organisation,
        score=_clamp01(score),
        metrics_used=used,
        metrics_available=len(metrics),
        details=details,
    )


def _safe_call_extractor(name: str, module, pdf_path: str) -> ExtractorRun:
    t0 = time.perf_counter()
    try:
        dual_raw, single_raw, dual, single, units = module.parse_pdf(pdf_path)
        ratios = []
        if hasattr(module, "compute_ratios"):
            ratios = module.compute_ratios(dual_raw, single_raw)
        recs = []
        if hasattr(module, "recommendations"):
            recs = module.recommendations(ratios)
        context = None
        if hasattr(module, "metrics_to_context"):
            context = module.metrics_to_context(dual, single, ratios, units)

        return ExtractorRun(
            name=name,
            ok=True,
            elapsed_s=time.perf_counter() - t0,
            units=units,
            dual_raw=dual_raw,
            single_raw=single_raw,
            dual=dual,
            single=single,
            ratios=ratios,
            recs=recs,
            context=context,
        )
    except Exception as e:
        return ExtractorRun(
            name=name,
            ok=False,
            elapsed_s=time.perf_counter() - t0,
            error=str(e),
        )


def _init_openai_client() -> Optional[Any]:
    load_dotenv()

    encrypted = os.getenv("OPENAI_API_KEY_ENCRYPTED") or os.getenv("OPENAI_API_KEY")
    passphrase = os.getenv("OPENAI_PASSPHRASE", "default_salt_2024")
    api_key = decrypt_key(encrypted, passphrase) if encrypted else ""

    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _init_openai_client_with_key(api_key: str) -> Optional[Any]:
    api_key = (api_key or "").strip()
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _build_analysis_context(result: Dict[str, Any], max_chars: int = 14000) -> str:
    """Build a compact, copy/paste friendly context string for LLM chat.

    Avoid embedding full extracted text; only include summary + key metrics.
    """
    if not result:
        return ""

    orgs = result.get("organisations") or {}
    schema = result.get("metric_schema") or []
    insights = result.get("insights") or {}
    sw = (insights.get("strengths_weaknesses") or {}) if isinstance(insights, dict) else {}

    def fmt_pct(v: Optional[float]) -> str:
        if v is None:
            return "—"
        try:
            return f"{float(v)*100:.2f}%"
        except Exception:
            return "—"

    lines: List[str] = []
    lines.append("Agentic Flow - Analysis Context")
    lines.append(f"PDFs analyzed: {result.get('pdf_count')}")
    lines.append(f"Recommended organisation: {result.get('recommended_organisation')}")
    reasons = (insights.get("recommendation_reasons") or []) if isinstance(insights, dict) else []
    if reasons:
        lines.append("Recommendation reasons:")
        for r in reasons[:6]:
            lines.append(f"- {r}")

    if orgs:
        lines.append("\nOrganisation summary:")
        for org, d in orgs.items():
            try:
                lines.append(
                    f"- {org}: avg_perf={float(d.get('avg_perf_score') or 0.0):.3f} "
                    f"avg_extraction={float(d.get('avg_extraction_score') or 0.0):.3f} "
                    f"pdfs={int(d.get('pdf_count') or 0)} wins(HDFC/ENBD)={((d.get('wins') or {}).get('HDFC', 0))}/{((d.get('wins') or {}).get('ENBD', 0))}"
                )
            except Exception:
                lines.append(f"- {org}: (summary unavailable)")

        lines.append("\nKey metrics (averaged per organisation):")
        for org, d in orgs.items():
            lines.append(f"{org}:")
            metrics = (d.get("metrics") or {}) if isinstance(d, dict) else {}
            for m in schema:
                k = m.get("key")
                label = m.get("label")
                if not k:
                    continue
                val = metrics.get(k)
                lines.append(f"- {label}: {fmt_pct(val)}")

        lines.append("\nStrengths & weaknesses:")
        for org in orgs.keys():
            org_sw = sw.get(org) or {}
            strengths = org_sw.get("strengths") or []
            weaknesses = org_sw.get("weaknesses") or []
            lines.append(f"{org} strengths: {('; '.join(strengths) if strengths else '—')}")
            lines.append(f"{org} weaknesses: {('; '.join(weaknesses) if weaknesses else '—')}")

    ctx = "\n".join(lines)
    if len(ctx) > max_chars:
        ctx = ctx[: max_chars - 30] + "\n…(truncated)"
    return ctx


def _final_recommendation_text(result: Dict[str, Any]) -> str:
    if not result:
        return "No analysis available."

    orgs = result.get("organisations") or {}
    insights = result.get("insights") or {}
    recommended = result.get("recommended_organisation")
    if not recommended:
        return "No recommendation could be computed (missing organisation detection and/or metrics)."

    reasons = (insights.get("recommendation_reasons") or []) if isinstance(insights, dict) else []
    sw = (insights.get("strengths_weaknesses") or {}) if isinstance(insights, dict) else {}

    other = None
    if len(orgs) > 1:
        others = [o for o in orgs.keys() if o != recommended]
        if others:
            other = max(others, key=lambda o: float((orgs.get(o) or {}).get("avg_perf_score") or 0.0))

    lines: List[str] = []
    lines.append(f"Final recommendation: {recommended}")
    if other:
        lines.append(f"Runner-up: {other}")
    lines.append(f"PDFs analyzed: {result.get('pdf_count')}")

    if reasons:
        lines.append("\nKey reasons:")
        for r in reasons[:6]:
            lines.append(f"- {r}")

    rec_sw = sw.get(recommended) or {}
    strengths = rec_sw.get("strengths") or []
    weaknesses = rec_sw.get("weaknesses") or []
    if strengths:
        lines.append("\nStrengths:")
        for s in strengths[:5]:
            lines.append(f"- {s}")
    if weaknesses:
        lines.append("\nWeaknesses / watch-outs:")
        for w in weaknesses[:5]:
            lines.append(f"- {w}")

    lines.append("\nNote: This is a heuristic comparison based on extracted ratios; it is not investment advice.")
    return "\n".join(lines)


def _llm_choose_best(
    client: Any,
    pdf_hint: Dict[str, Any],
    runs: Dict[str, ExtractorRun],
    scores: Dict[str, ExtractorScore],
) -> Dict[str, Any]:
    """Optional LLM judge; returns {"winner": "HDFC"|"ENBD"|None, "reason": str}."""

    payload = {
        "bank_hint": pdf_hint,
        "results": {
            k: {
                "ok": v.ok,
                "error": v.error,
                "units": v.units,
                "filled": scores[k].filled_metrics,
                "expected": scores[k].expected_metrics,
                "fill_rate": round(scores[k].fill_rate, 4),
                "ratio_count": scores[k].ratio_count,
                "invalid_ratio_count": scores[k].invalid_ratio_count,
                "elapsed_s": round(v.elapsed_s, 3),
            }
            for k, v in runs.items()
        },
    }

    prompt = (
        "You are judging which extraction pipeline performed better on a bank PDF. "
        "Pick the best pipeline using the provided metrics only. "
        "Prefer: higher fill_rate, more valid ratios, correct currency/units, and fewer errors. "
        "Return strict JSON with keys: winner (HDFC|ENBD|null), confidence (0-1), reasons (array of 2-4 short strings).\n\n"
        f"INPUT_JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    def _extract_first_json_object(s: str) -> str:
        s = (s or "").strip()
        if s.startswith("```"):
            # strip fenced blocks like ```json ... ```
            s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
            s = re.sub(r"```\s*$", "", s.strip())
        m = re.search(r"\{[\s\S]*\}", s)
        return m.group(0) if m else s

    try:
        resp = client.chat.completions.create(
            model=os.getenv("AGENTIC_FLOW_JUDGE_MODEL", "gpt-4.1"),
            messages=[
                {"role": "system", "content": "Return only JSON. No markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=350,
        )
        text = (resp.choices[0].message.content or "").strip()
        data = json.loads(_extract_first_json_object(text))
        return {
            "winner": data.get("winner"),
            "confidence": data.get("confidence"),
            "reasons": data.get("reasons"),
            "raw": data,
        }
    except Exception as e:
        return {"winner": None, "confidence": 0.0, "reasons": [f"LLM judge unavailable: {e}"]}


def analyze_pdf(pdf_path: str, use_llm_judge: bool = False) -> Dict[str, Any]:
    hdfc_mod, enbd_mod = _get_extractors()

    preview = _extract_native_text(pdf_path)
    bank_hint = _detect_bank_hint(preview)

    runs: Dict[str, ExtractorRun] = {
        "HDFC": _safe_call_extractor("HDFC", hdfc_mod, pdf_path),
        "ENBD": _safe_call_extractor("ENBD", enbd_mod, pdf_path),
    }

    scores: Dict[str, ExtractorScore] = {
        k: _score_run(v, bank_hint) if v.ok else ExtractorScore(
            total=0.0,
            fill_rate=0.0,
            filled_metrics=0,
            expected_metrics=0,
            ratio_count=0,
            invalid_ratio_count=0,
            runtime_score=0.0,
            units_score=0.0,
            bank_hint_bonus=0.0,
        )
        for k, v in runs.items()
    }

    # heuristic winner
    ok_names = [k for k, v in runs.items() if v.ok]
    if not ok_names:
        winner = None
    else:
        winner = max(ok_names, key=lambda n: scores[n].total)

    llm_judgement = None
    if use_llm_judge:
        client = _init_openai_client()
        if client is not None and ok_names:
            llm_judgement = _llm_choose_best(client, bank_hint, runs, scores)
            if llm_judgement.get("winner") in ("HDFC", "ENBD"):
                winner = llm_judgement["winner"]

    return {
        "pdf_path": pdf_path,
        "bank_hint": bank_hint,
        "winner": winner,
        "organisation": bank_hint.get("likely") or winner,
        "runs": {k: asdict(v) for k, v in runs.items()},
        "scores": {k: asdict(v) for k, v in scores.items()},
        "llm_judgement": llm_judgement,
    }


def analyze_pdfs(pdf_paths: List[str], use_llm_judge: bool = False) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for p in pdf_paths:
        r = analyze_pdf(p, use_llm_judge=use_llm_judge)

        org = r.get("organisation") or r.get("winner") or "Unknown"
        winner = r.get("winner")
        winner_ratios = None
        try:
            if winner in ("HDFC", "ENBD"):
                winner_ratios = (r.get("runs") or {}).get(winner, {}).get("ratios")
        except Exception:
            winner_ratios = None

        org_perf = _org_performance_from_ratios(str(org), winner_ratios)
        r["org_performance"] = asdict(org_perf)

        # Standardized key metrics for tabular comparison
        r["key_metrics"] = _standard_metrics_from_ratios(winner_ratios)
        results.append(r)

    # Aggregate per organisation
    orgs: Dict[str, Dict[str, Any]] = {}
    for r in results:
        org = str(r.get("organisation") or "Unknown")
        orgs.setdefault(
            org,
            {
                "pdf_count": 0,
                "avg_perf_score": 0.0,
                "avg_extraction_score": 0.0,
                "wins": {"HDFC": 0, "ENBD": 0, "None": 0},
                "perf_scores": [],
                "extract_scores": [],
                "coverage": [],
                "metric_values": {},
            },
        )

        orgs[org]["pdf_count"] += 1
        perf = (r.get("org_performance") or {}).get("score")
        if perf is not None:
            orgs[org]["perf_scores"].append(float(perf))
        cov = (r.get("org_performance") or {}).get("metrics_used")
        if cov is not None:
            orgs[org]["coverage"].append(int(cov))

        # Collect key metrics for the metrics tab
        try:
            km = r.get("key_metrics") or {}
            for metric in STANDARD_METRICS:
                k = metric["key"]
                v = km.get(k)
                if v is None:
                    continue
                orgs[org]["metric_values"].setdefault(k, []).append(float(v))
        except Exception:
            pass

        winner = r.get("winner") or "None"
        if winner not in ("HDFC", "ENBD"):
            winner = "None"
        orgs[org]["wins"][winner] += 1

        # Extraction score of the chosen winner (heuristic)
        try:
            if winner in ("HDFC", "ENBD"):
                sc = ((r.get("scores") or {}).get(winner) or {}).get("total")
                if sc is not None:
                    orgs[org]["extract_scores"].append(float(sc))
        except Exception:
            pass

    for org, d in orgs.items():
        ps = d.get("perf_scores") or []
        es = d.get("extract_scores") or []
        d["avg_perf_score"] = (sum(ps) / len(ps)) if ps else 0.0
        d["avg_extraction_score"] = (sum(es) / len(es)) if es else 0.0
        d["avg_metrics_used"] = (sum(d.get("coverage") or []) / len(d.get("coverage") or [1])) if d.get("coverage") else 0.0

        metric_avgs: Dict[str, Optional[float]] = {}
        for metric in STANDARD_METRICS:
            k = metric["key"]
            vals = (d.get("metric_values") or {}).get(k) or []
            metric_avgs[k] = (sum(vals) / len(vals)) if vals else None
        d["metrics"] = metric_avgs
        d["metrics_coverage"] = sum(1 for v in metric_avgs.values() if v is not None)

        # keep payload small
        d.pop("metric_values", None)

        # keep payload small
        d.pop("perf_scores", None)
        d.pop("extract_scores", None)
        d.pop("coverage", None)

    # Recommend best organisation by performance score (tie-break on extraction score)
    recommended_org = None
    if orgs:
        recommended_org = max(
            orgs.keys(),
            key=lambda o: (orgs[o].get("avg_perf_score", 0.0), orgs[o].get("avg_extraction_score", 0.0), orgs[o].get("pdf_count", 0)),
        )

    # Build strengths/weaknesses and recommendation explanation
    org_metrics_for_insights = {
        org: {
            "avg_perf_score": d.get("avg_perf_score", 0.0),
            "metrics": d.get("metrics") or {},
        }
        for org, d in orgs.items()
    }
    sw = _build_strengths_weaknesses(org_metrics_for_insights)

    recommendation_reasons: List[str] = []
    if recommended_org and len(orgs) > 1:
        # Compare against the strongest runner-up by avg_perf_score
        others = [o for o in orgs.keys() if o != recommended_org]
        runner_up = max(others, key=lambda o: float(orgs[o].get("avg_perf_score") or 0.0))
        a = (orgs.get(recommended_org) or {}).get("metrics") or {}
        b = (orgs.get(runner_up) or {}).get("metrics") or {}
        advantages: List[Tuple[float, str]] = []
        for metric in STANDARD_METRICS:
            k = metric["key"]
            adv = _metric_advantage(a.get(k), b.get(k), metric.get("better"))
            if adv is None:
                continue
            if adv <= 0:
                continue
            advantages.append((float(adv), _format_metric_for_reason(metric, a.get(k), b.get(k))))
        recommendation_reasons = [txt for _adv, txt in sorted(advantages, key=lambda x: x[0], reverse=True)[:4]]
    elif recommended_org:
        recommendation_reasons = [
            "Only one organisation detected from uploaded PDFs.",
        ]

    result = {
        "pdf_count": len(results),
        "recommended_organisation": recommended_org,
        "organisations": orgs,
        "results": results,
        "use_llm_judge": use_llm_judge,
        "metric_schema": [{k: v for k, v in m.items() if k in ("key", "label", "better", "fmt")} for m in STANDARD_METRICS],
        "insights": {
            "strengths_weaknesses": sw,
            "recommendation_reasons": recommendation_reasons,
        },
    }

    # Add small helper strings for UI panels
    result["final_recommendation"] = _final_recommendation_text(result)
    result["llm_context"] = _build_analysis_context(result)
    return result


# ===================== Flask UI =====================
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Agentic Flow - Extractor Comparator</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
  <style>
    body { background: #f7f7f8; }
    .card { border: 1px solid #e5e5e5; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
  </style>
</head>
<body>
<div class="container py-4">
  <div class="row mb-3">
    <div class="col">
      <h3 class="mb-1">Agentic Flow</h3>
            <div class="text-muted">Upload one or more PDFs, run both extractors, score extraction quality, and recommend the best performing organisation based on key ratios.</div>
    </div>
  </div>

  <div class="card p-3 mb-3">
        <form method="post" action="/upload" enctype="multipart/form-data" class="row g-2 align-items-end">
            <div class="col-md-7">
                <label class="form-label">PDFs (you can select multiple)</label>
                <input type="file" name="pdf_files" accept="application/pdf" class="form-control" multiple required />
                <div class="form-text">Tip: select multiple PDFs to get an overall recommendation.</div>
            </div>
      <div class="col-md-3">
        <label class="form-label">LLM Judge</label>
        <select name="llm" class="form-select">
          <option value="0" selected>Off (heuristic only)</option>
          <option value="1">On (uses OpenAI key if configured)</option>
        </select>
      </div>
      <div class="col-md-2">
        <button class="btn btn-primary w-100" type="submit">Analyze</button>
      </div>
      {% if error %}
        <div class="col-12"><div class="alert alert-danger mb-0">{{ error }}</div></div>
      {% endif %}
    </form>
  </div>

        {% if result %}
            {% if result.results %}
                <ul class="nav nav-tabs" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="tab-summary-btn" data-bs-toggle="tab" data-bs-target="#tab-summary" type="button" role="tab">Summary</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="tab-metrics-btn" data-bs-toggle="tab" data-bs-target="#tab-metrics" type="button" role="tab">Key Metrics</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="tab-final-btn" data-bs-toggle="tab" data-bs-target="#tab-final" type="button" role="tab">Final Recommendation</button>
                    </li>
                </ul>
                <div class="tab-content pt-3">
                    <div class="tab-pane fade show active" id="tab-summary" role="tabpanel" aria-labelledby="tab-summary-btn">
                        <div class="card p-3 mb-3">
                            <div class="d-flex justify-content-between flex-wrap gap-2">
                                <div>
                                    <div class="text-muted">Recommended organisation (performance score)</div>
                                    <div class="h4 mb-0">{{ result.recommended_organisation or 'None' }}</div>
                                    <div class="text-muted">Analyzed PDFs: {{ result.pdf_count }}</div>
                                </div>
                                <div>
                                    <div class="text-muted">Scoring note</div>
                                    <div class="mono">Uses ratios like NPM, ROA/ROE, cost-to-income, and NPL/NPA when available.</div>
                                </div>
                            </div>

                            {% if result.insights and result.insights.recommendation_reasons and result.insights.recommendation_reasons|length > 0 %}
                                <hr/>
                                <div class="text-muted mb-1">Why this recommendation</div>
                                <ul class="mb-0">
                                    {% for rr in result.insights.recommendation_reasons %}
                                        <li>{{ rr }}</li>
                                    {% endfor %}
                                </ul>
                            {% endif %}

                            {% if result.organisations and result.organisations|length > 0 %}
                                <hr/>
                                <div class="text-muted mb-2">Organisation summary</div>
                                <div class="table-responsive">
                                    <table class="table table-sm align-middle mb-0">
                                        <thead>
                                            <tr>
                                                <th>Organisation</th>
                                                <th>PDFs</th>
                                                <th>Avg performance</th>
                                                <th>Avg extraction</th>
                                                <th>Wins (HDFC/ENBD)</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {% for org, d in result.organisations.items() %}
                                                <tr>
                                                    <td>{{ org }}</td>
                                                    <td>{{ d.pdf_count }}</td>
                                                    <td><span class="mono">{{ "%.3f"|format(d.avg_perf_score) }}</span></td>
                                                    <td><span class="mono">{{ "%.3f"|format(d.avg_extraction_score) }}</span></td>
                                                    <td><span class="mono">{{ d.wins.HDFC }}/{{ d.wins.ENBD }}</span></td>
                                                </tr>
                                            {% endfor %}
                                        </tbody>
                                    </table>
                                </div>
                            {% endif %}
                        </div>

                        <div class="card p-3 mb-3">
                            <div class="text-muted mb-2">Per-PDF details</div>
                            {% for r in result.results %}
                                <details class="mb-2">
                                    <summary>
                                        <span class="mono">{{ r.pdf_path }}</span>
                                        · org=<span class="mono">{{ r.organisation }}</span>
                                        · winner=<span class="mono">{{ r.winner }}</span>
                                        · perf=<span class="mono">{{ "%.3f"|format(r.org_performance.score) }}</span>
                                    </summary>

                                    <div class="mt-2">
                                        <div class="text-muted">Bank hint</div>
                                        <div class="mono">likely={{ r.bank_hint.likely }} confidence={{ "%.2f"|format(r.bank_hint.confidence) }} signals={{ r.bank_hint.signals }}</div>
                                        <div class="text-muted mt-2">Performance metrics used</div>
                                        <div class="mono">used={{ r.org_performance.metrics_used }}/{{ r.org_performance.metrics_available }}</div>
                                    </div>

                                    {% if r.key_metrics %}
                                        <div class="mt-3">
                                            <div class="text-muted mb-2">Key metrics (from winner ratios)</div>
                                            <div class="table-responsive">
                                                <table class="table table-sm align-middle mb-0">
                                                    <thead>
                                                        <tr><th>Metric</th><th>Value</th></tr>
                                                    </thead>
                                                    <tbody>
                                                        {% for m in result.metric_schema %}
                                                            {% set v = r.key_metrics[m.key] %}
                                                            {% if v is not none %}
                                                                <tr>
                                                                    <td>{{ m.label }}</td>
                                                                    <td class="mono">{{ "%.2f"|format(v*100) }}%</td>
                                                                </tr>
                                                            {% endif %}
                                                        {% endfor %}
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    {% endif %}

                                    <div class="row g-3 mt-1">
                                        {% for name in ['HDFC', 'ENBD'] %}
                                            {% set run = r.runs[name] %}
                                            {% set sc = r.scores[name] %}
                                            <div class="col-lg-6">
                                                <div class="card p-3 h-100">
                                                    <div class="d-flex justify-content-between">
                                                        <div class="h6 mb-0">{{ name }}</div>
                                                        <div class="badge text-bg-{{ 'success' if run.ok else 'danger' }}">{{ 'OK' if run.ok else 'ERROR' }}</div>
                                                    </div>
                                                    <div class="text-muted">score={{ "%.3f"|format(sc.total) }} · fill={{ sc.filled_metrics }}/{{ sc.expected_metrics }} ({{ "%.1f"|format(sc.fill_rate*100) }}%) · ratios={{ sc.ratio_count }} · invalid_ratios={{ sc.invalid_ratio_count }} · time={{ "%.2f"|format(run.elapsed_s) }}s</div>
                                                    <hr/>
                                                    {% if run.ok %}
                                                        <div class="mb-2"><span class="text-muted">Units:</span> <span class="mono">{{ run.units }}</span></div>
                                                        {% if run.recs and run.recs|length > 0 %}
                                                            <div class="text-muted">Extractor recommendations</div>
                                                            <ul>
                                                                {% for rr in run.recs %}
                                                                    <li>{{ rr }}</li>
                                                                {% endfor %}
                                                            </ul>
                                                        {% endif %}
                                                        <details>
                                                            <summary class="text-muted">Context preview</summary>
                                                            <pre class="mono mt-2" style="white-space: pre-wrap;">{{ (run.context or '')[:1200] }}</pre>
                                                        </details>
                                                    {% else %}
                                                        <div class="alert alert-danger mb-0">{{ run.error }}</div>
                                                    {% endif %}
                                                </div>
                                            </div>
                                        {% endfor %}
                                    </div>

                                    {% if r.llm_judgement %}
                                        <div class="mt-3">
                                            <div class="text-muted mb-1">LLM judgement</div>
                                            <div class="mono">winner={{ r.llm_judgement.winner }} confidence={{ r.llm_judgement.confidence }}</div>
                                        </div>
                                    {% endif %}
                                </details>
                            {% endfor %}
                        </div>

                        <div class="mt-3">
                            <a class="btn btn-outline-secondary" href="/">Analyze another</a>
                            <a class="btn btn-outline-primary" href="/api/last">Download last result (JSON)</a>
                        </div>
                    </div>

                    <div class="tab-pane fade" id="tab-metrics" role="tabpanel" aria-labelledby="tab-metrics-btn">
                        <div class="card p-3 mb-3">
                            <div class="text-muted mb-2">Organisation key metrics (averaged across uploaded PDFs)</div>
                            <div class="table-responsive">
                                <table class="table table-sm align-middle mb-0">
                                    <thead>
                                        <tr>
                                            <th>Organisation</th>
                                            <th>PDFs</th>
                                            <th>Metrics covered</th>
                                            {% for m in result.metric_schema %}
                                                <th>{{ m.label }}</th>
                                            {% endfor %}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {% for org, d in result.organisations.items() %}
                                            <tr>
                                                <td>{{ org }}</td>
                                                <td class="mono">{{ d.pdf_count }}</td>
                                                <td class="mono">{{ d.metrics_coverage or 0 }}/{{ result.metric_schema|length }}</td>
                                                {% for m in result.metric_schema %}
                                                    {% set v = (d.metrics or {}).get(m.key) %}
                                                    <td class="mono">{% if v is not none %}{{ "%.2f"|format(v*100) }}%{% else %}—{% endif %}</td>
                                                {% endfor %}
                                            </tr>
                                        {% endfor %}
                                    </tbody>
                                </table>
                            </div>
                            <div class="form-text">Note: metrics are read from the winner extractor per PDF; missing fields are shown as —.</div>
                        </div>

                        <div class="card p-3 mb-3">
                            <div class="text-muted mb-2">Strengths & weaknesses (relative comparison)</div>
                            {% for org, d in result.organisations.items() %}
                                <div class="mb-3">
                                    <div class="h6 mb-1">{{ org }}</div>
                                    {% set sw = (result.insights.strengths_weaknesses or {}).get(org) %}
                                    <div class="row g-3">
                                        <div class="col-md-6">
                                            <div class="text-muted">Strengths</div>
                                            {% if sw and sw.strengths and sw.strengths|length > 0 %}
                                                <ul class="mb-0">
                                                    {% for s in sw.strengths %}
                                                        <li>{{ s }}</li>
                                                    {% endfor %}
                                                </ul>
                                            {% else %}
                                                <div class="mono">No strong differentiators found (or insufficient data).</div>
                                            {% endif %}
                                        </div>
                                        <div class="col-md-6">
                                            <div class="text-muted">Weaknesses</div>
                                            {% if sw and sw.weaknesses and sw.weaknesses|length > 0 %}
                                                <ul class="mb-0">
                                                    {% for w in sw.weaknesses %}
                                                        <li>{{ w }}</li>
                                                    {% endfor %}
                                                </ul>
                                            {% else %}
                                                <div class="mono">No clear weaknesses found (or insufficient data).</div>
                                            {% endif %}
                                        </div>

                                        <div class="tab-pane fade" id="tab-final" role="tabpanel" aria-labelledby="tab-final-btn">
                                            <div class="card p-3 mb-3">
                                                <div class="d-flex justify-content-between flex-wrap gap-2">
                                                    <div>
                                                        <div class="text-muted">Final recommendation</div>
                                                        <div class="h4 mb-0">{{ result.recommended_organisation or 'None' }}</div>
                                                        <div class="text-muted">Based on averaged key ratios across uploaded PDFs.</div>
                                                    </div>
                                                    <div>
                                                        <div class="text-muted">LLM chat</div>
                                                        <div class="mono">Ask questions using the context panel below.</div>
                                                    </div>
                                                </div>
                                                <hr/>
                                                <pre class="mono mb-0" style="white-space: pre-wrap;">{{ result.final_recommendation or '' }}</pre>
                                            </div>

                                            <div class="card p-3 mb-3">
                                                <div class="d-flex justify-content-between flex-wrap gap-2 align-items-end">
                                                    <div>
                                                        <div class="text-muted">OpenAI panel</div>
                                                        <div class="form-text">Paste context (or load from analysis) and chat. API key is optional if configured via env.</div>
                                                    </div>
                                                    <div class="d-flex gap-2">
                                                        <button class="btn btn-outline-secondary btn-sm" type="button" id="btn-load-context">Load analysis context</button>
                                                        <button class="btn btn-outline-secondary btn-sm" type="button" id="btn-clear-chat">Clear chat</button>
                                                    </div>
                                                </div>

                                                <div class="row g-3 mt-1">
                                                    <div class="col-lg-6">
                                                        <label class="form-label">Context</label>
                                                        <textarea id="llm-context" class="form-control mono" rows="14" placeholder="Paste or load analysis context here..."></textarea>
                                                        <div class="form-text">This context is sent to `/api/chat` with your message.</div>

                                                        <label class="form-label mt-2">Optional OpenAI API key (used for this request only)</label>
                                                        <input id="llm-api-key" class="form-control mono" type="password" placeholder="sk-..." />
                                                        <div class="form-text">Not stored server-side. Leave blank if configured in env.</div>

                                                        <textarea id="llm-context-hidden" class="d-none">{{ result.llm_context or '' }}</textarea>
                                                    </div>

                                                    <div class="col-lg-6">
                                                        <label class="form-label">Chat</label>
                                                        <div id="chat-box" class="border rounded p-2" style="height: 340px; overflow:auto; background:#fff;">
                                                            <div class="text-muted">Ask: “Summarize why the recommended organisation is better”</div>
                                                        </div>

                                                        <div class="input-group mt-2">
                                                            <input id="chat-input" class="form-control" placeholder="Type a question..." />
                                                            <button id="chat-send" class="btn btn-primary" type="button">Send</button>
                                                        </div>
                                                        <div class="form-text">If OpenAI isn’t configured, the API returns an error message.</div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                {% if not loop.last %}<hr/>{% endif %}
                            {% endfor %}
                        </div>

                        <div class="card p-3 mb-3">
                            <div class="text-muted mb-2">Per-PDF key metrics (winner-based)</div>
                            <div class="table-responsive">
                                <table class="table table-sm align-middle mb-0">
                                    <thead>
                                        <tr>
                                            <th>PDF</th>
                                            <th>Organisation</th>
                                            <th>Winner</th>
                                            {% for m in result.metric_schema %}
                                                <th>{{ m.label }}</th>
                                            {% endfor %}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {% for r in result.results %}
                                            <tr>
                                                <td class="mono">{{ r.pdf_path }}</td>
                                                <td class="mono">{{ r.organisation }}</td>
                                                <td class="mono">{{ r.winner }}</td>
                                                {% for m in result.metric_schema %}
                                                    {% set v = (r.key_metrics or {}).get(m.key) %}
                                                    <td class="mono">{% if v is not none %}{{ "%.2f"|format(v*100) }}%{% else %}—{% endif %}</td>
                                                {% endfor %}
                                            </tr>
                                        {% endfor %}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            {% else %}
                <div class="alert alert-warning">No results.</div>
            {% endif %}
        {% endif %}
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
    (function() {
        const chatBox = document.getElementById('chat-box');
        const chatInput = document.getElementById('chat-input');
        const chatSend = document.getElementById('chat-send');
        const btnLoad = document.getElementById('btn-load-context');
        const btnClear = document.getElementById('btn-clear-chat');
        const ctx = document.getElementById('llm-context');
        const hidden = document.getElementById('llm-context-hidden');
        const apiKey = document.getElementById('llm-api-key');

        if (!chatBox || !chatInput || !chatSend || !ctx) return;

        let history = [];

        function addMsg(role, content) {
            const wrap = document.createElement('div');
            wrap.className = 'mb-2';
            const badge = document.createElement('span');
            badge.className = 'badge me-2 ' + (role === 'user' ? 'text-bg-secondary' : 'text-bg-success');
            badge.textContent = role;
            const pre = document.createElement('pre');
            pre.className = 'mono mb-0';
            pre.style.whiteSpace = 'pre-wrap';
            pre.textContent = content;
            wrap.appendChild(badge);
            wrap.appendChild(pre);
            chatBox.appendChild(wrap);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        async function send() {
            const msg = (chatInput.value || '').trim();
            if (!msg) return;
            const context = (ctx.value || '').trim();
            addMsg('user', msg);
            history.push({role:'user', content: msg});
            chatInput.value = '';

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        message: msg,
                        context: context,
                        history: history,
                        api_key: (apiKey && apiKey.value ? apiKey.value : '')
                    })
                });
                const data = await res.json();
                if (!res.ok) {
                    const err = data && data.error ? data.error : ('HTTP ' + res.status);
                    addMsg('assistant', 'Error: ' + err);
                    history.push({role:'assistant', content: 'Error: ' + err});
                    return;
                }
                const answer = (data && data.answer) ? data.answer : '';
                addMsg('assistant', answer);
                history.push({role:'assistant', content: answer});
            } catch (e) {
                addMsg('assistant', 'Error: ' + e);
                history.push({role:'assistant', content: 'Error: ' + e});
            }
        }

        chatSend.addEventListener('click', send);
        chatInput.addEventListener('keydown', (ev) => {
            if (ev.key === 'Enter') {
                ev.preventDefault();
                send();
            }
        });

        if (btnLoad) {
            btnLoad.addEventListener('click', () => {
                const v = hidden ? (hidden.value || '') : '';
                if (v) ctx.value = v;
            });
        }

        if (btnClear) {
            btnClear.addEventListener('click', () => {
                history = [];
                chatBox.innerHTML = '<div class="text-muted">Chat cleared.</div>';
            });
        }
    })();
</script>
</body>
</html>
"""


_LAST_RESULT: Optional[Dict[str, Any]] = None


@app.route("/", methods=["GET"])
def home():
    return render_template_string(TEMPLATE, result=None, error=None)


@app.route("/upload", methods=["POST"])
def upload():
    global _LAST_RESULT

    files = request.files.getlist("pdf_files")
    files = [f for f in files if f and f.filename]
    if not files:
        return render_template_string(TEMPLATE, result=None, error="Please select at least one PDF file.")
    if any(not f.filename.lower().endswith(".pdf") for f in files):
        return render_template_string(TEMPLATE, result=None, error="Please upload PDF files only.")

    use_llm = (request.form.get("llm") or "0").strip() == "1"

    tmp_paths: List[str] = []
    try:
        for f in files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            f.save(tmp.name)
            tmp.close()
            tmp_paths.append(tmp.name)

        result = analyze_pdfs(tmp_paths, use_llm_judge=use_llm)
        _LAST_RESULT = result
        return render_template_string(TEMPLATE, result=result, error=None)
    except Exception as e:
        return render_template_string(TEMPLATE, result=None, error=f"Error analyzing PDFs: {e}")
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    files = request.files.getlist("pdf_files")
    if not files:
        # backwards compatibility
        single = request.files.get("pdf_file")
        files = [single] if single and single.filename else []

    files = [f for f in files if f and f.filename]
    if not files:
        return jsonify({"error": "missing pdf_files"}), 400

    use_llm = (request.form.get("llm") or "0").strip() == "1"

    tmp_paths: List[str] = []
    try:
        for f in files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            f.save(tmp.name)
            tmp.close()
            tmp_paths.append(tmp.name)

        if len(tmp_paths) == 1:
            result = analyze_pdf(tmp_paths[0], use_llm_judge=use_llm)
        else:
            result = analyze_pdfs(tmp_paths, use_llm_judge=use_llm)
        return jsonify(result)
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass


@app.route("/api/last", methods=["GET"])
def api_last():
    if _LAST_RESULT is None:
        return jsonify({"error": "no result yet"}), 404
    return jsonify(_LAST_RESULT)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Chat endpoint for the OpenAI panel.

    Expected JSON body:
      {
        "message": "...",
        "context": "...",   # optional
        "history": [{"role":"user"|"assistant","content":"..."}, ...],  # optional
        "api_key": "sk-..." # optional; used for this request only
      }
    """
    try:
        payload = request.get_json(silent=True) or {}
    except Exception:
        payload = {}

    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"error": "missing message"}), 400

    context = (payload.get("context") or "").strip()
    history = payload.get("history") or []
    api_key = (payload.get("api_key") or "").strip()

    client = _init_openai_client_with_key(api_key) if api_key else _init_openai_client()
    if client is None:
        return jsonify({"error": "OpenAI not configured. Set OPENAI_API_KEY_ENCRYPTED/OPENAI_API_KEY or provide api_key in the panel."}), 400

    # Sanitize/limit history
    safe_history: List[Dict[str, str]] = []
    if isinstance(history, list):
        for m in history[-12:]:
            if not isinstance(m, dict):
                continue
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                safe_history.append({"role": role, "content": content[:4000]})

    sys = (
        "You are a financial analyst assistant. "
        "Use only the provided context; if context is insufficient, ask a clarifying question. "
        "Be concise, use bullet points where helpful, and clearly separate strengths, weaknesses, and caveats."
    )

    # Keep context bounded
    if len(context) > 16000:
        context = context[:15950] + "\n…(truncated)"

    messages = [{"role": "system", "content": sys}]
    if context:
        messages.append({"role": "user", "content": f"CONTEXT:\n{context}"})
    messages.extend(safe_history)
    messages.append({"role": "user", "content": message})

    try:
        resp = client.chat.completions.create(
            model=os.getenv("AGENTIC_FLOW_CHAT_MODEL", "gpt-4.1"),
            messages=messages,
            temperature=0.2,
            max_tokens=700,
        )
        answer = (resp.choices[0].message.content or "").strip()
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": f"chat failed: {e}"}), 500


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Run both extractors and suggest the better one.")
    parser.add_argument("--pdf", required=False, nargs="+", help="Path(s) to PDF(s) to analyze")
    parser.add_argument("--llm", action="store_true", help="Enable optional OpenAI judge")
    parser.add_argument("--json", action="store_true", help="Print full JSON result")
    parser.add_argument("--serve", action="store_true", help="Run Flask UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5099, type=int)
    args = parser.parse_args()

    if args.serve or not args.pdf:
        print("\n🧠 Agentic Flow - Extractor Comparator")
        print(f"🌐 Running on http://{args.host}:{args.port}")
        print("Tip: set OPENAI_API_KEY_ENCRYPTED + OPENAI_PASSPHRASE to enable LLM judge.")
        app.run(host=args.host, port=args.port, debug=True, use_reloader=False)
        return 0

    if len(args.pdf) == 1:
        result: Dict[str, Any] = analyze_pdf(args.pdf[0], use_llm_judge=args.llm)
    else:
        result = analyze_pdfs(args.pdf, use_llm_judge=args.llm)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if "results" in result:
            print(f"recommended_organisation={result.get('recommended_organisation')}")
            print(f"pdf_count={result.get('pdf_count')}")
            for org, d in (result.get("organisations") or {}).items():
                print(
                    f"- {org}: avg_perf={d.get('avg_perf_score', 0.0):.3f} avg_extraction={d.get('avg_extraction_score', 0.0):.3f} "
                    f"pdfs={d.get('pdf_count', 0)} wins(HDFC/ENBD)={d.get('wins', {}).get('HDFC', 0)}/{d.get('wins', {}).get('ENBD', 0)}"
                )
        else:
            print(f"winner={result.get('winner')}")
            for name in ("HDFC", "ENBD"):
                run = result["runs"][name]
                sc = result["scores"][name]
                print(
                    f"- {name}: ok={run['ok']} score={sc['total']:.3f} fill={sc['filled_metrics']}/{sc['expected_metrics']} "
                    f"ratios={sc['ratio_count']} invalid_ratios={sc['invalid_ratio_count']} time={run['elapsed_s']:.2f}s"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
