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
from flask import Flask, Response, jsonify, render_template_string, request, stream_with_context

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


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return None


def _fx_rates() -> Dict[str, float]:
    """Return currency-per-USD FX rates used for UI normalization.

    Defaults are conservative and are meant for display/relative comparison only.
    Override via env for accuracy:
      - AGENTIC_FLOW_FX_AED_PER_USD (default 3.6725)
      - AGENTIC_FLOW_FX_INR_PER_USD (default 83.0)
    """

    def env_float(key: str, default: float) -> float:
        raw = (os.getenv(key) or "").strip()
        v = _to_float(raw)
        return v if (v is not None and v > 0) else default

    return {
        "AED": env_float("AGENTIC_FLOW_FX_AED_PER_USD", 3.6725),
        "INR": env_float("AGENTIC_FLOW_FX_INR_PER_USD", 83.0),
        "USD": 1.0,
    }


_UNITS_MULTIPLIER: Dict[str, float] = {
    "units": 1.0,
    "unit": 1.0,
    "": 1.0,
    None: 1.0,  # type: ignore[dict-item]
    "thousand": 1e3,
    "thousands": 1e3,
    "lakh": 1e5,
    "lakhs": 1e5,
    "crore": 1e7,
    "crores": 1e7,
    "million": 1e6,
    "millions": 1e6,
    "billion": 1e9,
    "billions": 1e9,
}


def _normalize_units_label(label: Any) -> Optional[str]:
    if label is None:
        return None
    s = str(label).strip().lower()
    if not s:
        return None
    if "billion" in s:
        return "billions"
    if "million" in s:
        return "millions"
    if "thousand" in s:
        return "thousands"
    if "crore" in s:
        return "crores"
    if "lakh" in s:
        return "lakhs"
    return s


def to_usd(value: Any, units: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """Convert a reported metric value into USD.

    Interprets `value` as being in `units['currency']` and scaled by `units['units_label']`.
    Example: currency=AED, units_label=millions, value=100 => 100,000,000 AED.
    """

    v = _to_float(value)
    if v is None:
        return None

    units = units or {}
    currency = str(units.get("currency") or "").upper() or None
    units_label = _normalize_units_label(units.get("units_label"))

    mult = _UNITS_MULTIPLIER.get(units_label, 1.0)
    fx = _fx_rates()
    per_usd = fx.get(currency or "", None)
    if per_usd is None:
        return None
    return (v * mult) / per_usd


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
        return f"{label}: {a_val*100:,.2f}% vs {b_val*100:,.2f}%"
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

    encrypted = (os.getenv("OPENAI_API_KEY_ENCRYPTED") or "").strip()
    plaintext = (os.getenv("OPENAI_API_KEY") or "").strip()
    passphrase = os.getenv("OPENAI_PASSPHRASE", "default_salt_2024")

    api_key = ""
    if encrypted:
        api_key = (decrypt_key(encrypted, passphrase) or "").strip()
    elif plaintext:
        # Plaintext keys should be used as-is (do not attempt decryption).
        api_key = plaintext

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
    lines.append("Financial Statement Analyzer and Recommendation Agent")
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


def _build_metrics_tab_context(result: Dict[str, Any], max_chars: int = 14000) -> str:
    """Build context aligned to the Key Metrics tab.

    This intentionally focuses on averaged per-organisation metrics (compact + comparable).
    """
    if not result:
        return ""

    orgs = result.get("organisations") or {}
    schema = result.get("metric_schema") or []

    def fmt_pct(v: Optional[float]) -> str:
        if v is None:
            return "—"
        try:
            return f"{float(v) * 100:.2f}%"
        except Exception:
            return "—"

    lines: List[str] = []
    lines.append("Financial Statement Analyzer and Recommendation Agent")
    lines.append("Context: Key Metrics tab")
    lines.append(f"PDFs analyzed: {result.get('pdf_count')}")
    lines.append(f"Recommended organisation: {result.get('recommended_organisation')}")
    if orgs:
        lines.append("\nKey metrics (averaged per organisation):")
        for org, d in orgs.items():
            lines.append(f"{org}:")
            metrics = (d.get("metrics") or {}) if isinstance(d, dict) else {}
            for m in schema:
                k = m.get("key")
                label = m.get("label")
                if not k:
                    continue
                lines.append(f"- {label}: {fmt_pct(metrics.get(k))}")

    ctx = "\n".join(lines)
    if len(ctx) > max_chars:
        ctx = ctx[: max_chars - 30] + "\n…(truncated)"
    return ctx


def _build_extractor_tab_context(result: Dict[str, Any], extractor_name: str, max_chars: int = 14000) -> str:
    """Build context aligned to the ENBD/HDFC extractor tabs."""
    if not result:
        return ""

    extractor_name = (extractor_name or "").strip().upper()
    if extractor_name not in ("ENBD", "HDFC"):
        return ""

    def fmt_pct(v: Any) -> str:
        vv = _to_float(v)
        if vv is None:
            return "—"
        try:
            return f"{vv * 100:.2f}%"
        except Exception:
            return "—"

    lines: List[str] = []
    lines.append("Financial Statement Analyzer and Recommendation Agent")
    lines.append(f"Context: {extractor_name} tab")
    lines.append(f"PDFs analyzed: {result.get('pdf_count')}")
    lines.append(f"Recommended organisation: {result.get('recommended_organisation')}")

    res_list = result.get("results") or []
    for r in res_list:
        if not isinstance(r, dict):
            continue
        pdf_name = r.get("pdf_name") or r.get("filename") or "(unknown pdf)"
        org = r.get("organisation") or "(unknown org)"
        runs = r.get("runs") or {}
        run = runs.get(extractor_name) if isinstance(runs, dict) else None

        lines.append(f"\nPDF: {pdf_name}")
        lines.append(f"Organisation: {org}")
        if not isinstance(run, dict) or not run.get("ok"):
            err = None
            try:
                err = run.get("error") if isinstance(run, dict) else None
            except Exception:
                err = None
            lines.append(f"Extractor status: FAILED{(' - ' + str(err)) if err else ''}")
            continue

        units = run.get("units") or {}
        if isinstance(units, dict) and units:
            u = units.get("display") or units.get("currency") or units.get("name")
            if u:
                lines.append(f"Units: {u}")

        ratios = run.get("ratios") or []
        if ratios:
            lines.append("Ratios:")
            try:
                for name, val in list(ratios)[:18]:
                    lines.append(f"- {name}: {fmt_pct(val)}")
            except Exception:
                pass

        recs = run.get("recs") or []
        if recs:
            lines.append("Extractor recommendations:")
            for rr in recs[:10]:
                lines.append(f"- {rr}")

        ctx = (run.get("context") or "").strip()
        if ctx:
            # Keep it compact; the raw PDF text isn't embedded here anyway.
            lines.append("Context snippet:")
            lines.append(ctx[:2600] + ("\n…(truncated)" if len(ctx) > 2600 else ""))

        # Truncate early if we already exceed max_chars.
        if len("\n".join(lines)) > max_chars:
            break

    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[: max_chars - 30] + "\n…(truncated)"
    return out


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


def _final_recommendation_struct(result: Dict[str, Any]) -> Dict[str, Any]:
    """Structured recommendation payload for nicer UI rendering."""
    if not result:
        return {
            "recommended": None,
            "runner_up": None,
            "pdf_count": 0,
            "reasons": [],
            "strengths": [],
            "weaknesses": [],
            "note": "",
        }

    orgs = result.get("organisations") or {}
    insights = result.get("insights") or {}
    recommended = result.get("recommended_organisation")

    runner_up = None
    if recommended and len(orgs) > 1:
        others = [o for o in orgs.keys() if o != recommended]
        if others:
            runner_up = max(others, key=lambda o: float((orgs.get(o) or {}).get("avg_perf_score") or 0.0))

    reasons = (insights.get("recommendation_reasons") or []) if isinstance(insights, dict) else []
    sw = (insights.get("strengths_weaknesses") or {}) if isinstance(insights, dict) else {}
    rec_sw = sw.get(recommended) or {}
    strengths = rec_sw.get("strengths") or []
    weaknesses = rec_sw.get("weaknesses") or []

    note = "This is a heuristic comparison based on extracted ratios; validate against source statements."

    return {
        "recommended": recommended,
        "runner_up": runner_up,
        "pdf_count": int(result.get("pdf_count") or 0),
        "reasons": reasons,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "note": note,
    }


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
    result["final_recommendation_struct"] = _final_recommendation_struct(result)
    # FinanceBot contexts (tab-aware)
    result["llm_context"] = _build_analysis_context(result)
    result["llm_context_summary"] = result["llm_context"]
    result["llm_context_metrics"] = _build_metrics_tab_context(result)
    result["llm_context_enbd"] = _build_extractor_tab_context(result, "ENBD")
    result["llm_context_hdfc"] = _build_extractor_tab_context(result, "HDFC")
    return result


# ===================== Flask UI =====================
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


@app.template_filter("usd")
def jinja_usd(value: Any, units: Optional[Dict[str, Any]] = None) -> str:
    v = to_usd(value, units)
    if v is None:
        return "—"
    try:
        # compact but readable for large statements
        if abs(v) >= 1e9:
            return f"${v/1e9:,.2f}B"
        if abs(v) >= 1e6:
            return f"${v/1e6:,.2f}M"
        if abs(v) >= 1e3:
            return f"${v/1e3:,.2f}K"
        return f"${v:,.2f}"
    except Exception:
        return "—"


@app.template_filter("fmt_num")
def jinja_fmt_num(value: Any) -> str:
    v = _to_float(value)
    if v is None:
        return "—"
    try:
        return f"{v:,.2f}"
    except Exception:
        return str(value)


@app.template_filter("pct")
def jinja_pct(value: Any) -> str:
    """Format a ratio value (e.g., 0.5237) as a percentage string (e.g., 52.37%)."""
    v = _to_float(value)
    if v is None:
        return "—"
    try:
        return f"{(v * 100):,.2f}%"
    except Exception:
        return "—"


def _template_context(**extra: Any) -> Dict[str, Any]:
    ctx = {
        "openai": _openai_status(),
        "sources": {"HDFC": HDFC_EXTRACTOR_PATH, "ENBD": ENBD_EXTRACTOR_PATH},
        "fx": _fx_rates(),
    }
    ctx.update(extra)
    return ctx


def _openai_status() -> Dict[str, Any]:
    """Return minimal OpenAI configuration status for UI display (never returns key material)."""
    load_dotenv()
    encrypted = (os.getenv("OPENAI_API_KEY_ENCRYPTED") or "").strip()
    plaintext = (os.getenv("OPENAI_API_KEY") or "").strip()
    passphrase = os.getenv("OPENAI_PASSPHRASE", "default_salt_2024")

    source = "none"
    raw = ""
    if encrypted:
        source = "encrypted"
        raw = encrypted
    elif plaintext:
        source = "plaintext"
        raw = plaintext

    api_key = ""
    if source == "encrypted":
        api_key = (decrypt_key(raw, passphrase) if raw else "") or ""
    elif source == "plaintext":
        api_key = raw

    ready = bool(api_key) and OpenAI is not None
    return {
        "ready": ready,
        "source": source,
        "chat_model": os.getenv("AGENTIC_FLOW_CHAT_MODEL", "gpt-4.1"),
        "judge_model": os.getenv("AGENTIC_FLOW_JUDGE_MODEL", "gpt-4.1"),
        "has_openai_sdk": OpenAI is not None,
    }


TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <meta name="theme-color" content="#10a37f"/>
    <title>financebit</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #10a37f;
            --secondary-color: #0066cc;
            --bg-light: #f7f7f8;
            --card-border: rgba(15, 23, 42, 0.10);
            --soft: rgba(15, 23, 42, 0.06);
            --shadow: 0 8px 28px rgba(15, 23, 42, 0.06);
            --shadow-lg: 0 18px 60px rgba(15, 23, 42, 0.10);
            --text: #0f172a;
            --muted: #64748b;
            --radius: 16px;
        }
        body {
            background:
                radial-gradient(1200px 520px at 12% -4%, rgba(0, 102, 204, 0.12) 0%, rgba(0, 102, 204, 0) 55%),
                radial-gradient(900px 480px at 88% 0%, rgba(16, 163, 127, 0.12) 0%, rgba(16, 163, 127, 0) 55%),
                var(--bg-light);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: var(--text);
        }
        a { text-decoration: none; }
        .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
        .card { border: 1px solid var(--card-border); box-shadow: var(--shadow); border-radius: var(--radius); }
        .card .card-title { letter-spacing: -0.01em; }
        .btn { border-radius: 12px; }
        .btn-primary {
            border: none;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            box-shadow: 0 10px 26px rgba(0,0,0,0.10);
        }
        .btn-primary:hover { filter: brightness(0.98); }
        .btn-outline-secondary, .btn-outline-primary { border-radius: 12px; }
        .form-control, .form-select { border-radius: 12px; border-color: rgba(15,23,42,0.12); }
        .form-control:focus, .form-select:focus { border-color: rgba(16,163,127,0.5); box-shadow: 0 0 0 0.25rem rgba(16,163,127,0.10); }
        .af-header {
            border: 1px solid var(--card-border);
            background:
                linear-gradient(135deg, rgba(255,255,255,0.92), rgba(248,250,252,0.86));
            box-shadow: var(--shadow-lg);
            border-radius: calc(var(--radius) + 2px);
            position: relative;
            overflow: hidden;
        }
        .af-header:before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                radial-gradient(700px 220px at 18% 18%, rgba(0,102,204,0.14) 0%, rgba(0,102,204,0) 60%),
                radial-gradient(700px 220px at 82% 0%, rgba(16,163,127,0.18) 0%, rgba(16,163,127,0) 60%);
            pointer-events: none;
        }
        .af-header > * { position: relative; }
        .af-header-title {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .af-icon {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            box-shadow: 0 8px 18px rgba(0,0,0,0.12);
        }
        .af-subtitle { color: var(--muted); max-width: 980px; }
        .af-kpi { background: rgba(255,255,255,0.78); border: 1px solid var(--soft); border-radius: 14px; }
        .af-tabs {
            border-bottom: 0;
            background: rgba(255,255,255,0.62);
            border: 1px solid var(--soft);
            border-radius: 14px;
            padding: 6px;
            gap: 6px;
            flex-wrap: wrap;
        }
        .af-tabs .nav-item {
            flex: 1 1 0;
        }
        .af-tabs .nav-link {
            border: 0;
            border-radius: 12px;
            color: #334155;
            padding: 10px 12px;
            width: 100%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 42px;
            white-space: nowrap;
        }
        .af-tabs .nav-link:hover { background: rgba(15,23,42,0.04); }
        .af-tabs .nav-link.active {
            font-weight: 700;
            color: #0b1220;
            background: rgba(16,163,127,0.10);
            box-shadow: 0 10px 28px rgba(15,23,42,0.08);
        }
        details { border: 1px solid var(--soft); background: rgba(255,255,255,0.70); border-radius: 14px; padding: 10px 12px; }
        details[open] { box-shadow: 0 14px 38px rgba(15,23,42,0.10); }
        details > summary { cursor: pointer; list-style: none; }
        details > summary::-webkit-details-marker { display: none; }
        details > summary { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
        details > summary:after {
            content: "\\f078";
            font-family: "Font Awesome 6 Free";
            font-weight: 900;
            color: var(--muted);
        }
        details[open] > summary:after { content: "\\f077"; }
        .table thead th { white-space: nowrap; }
        .table { --bs-table-bg: transparent; }
        .table thead { background: rgba(15, 23, 42, 0.03); }
        .table tbody tr:hover { background: rgba(2, 6, 23, 0.03); }
        .af-pre { white-space: pre-wrap; background: rgba(255,255,255,0.80); border: 1px solid var(--soft); border-radius: 12px; padding: 12px; }
        .af-muted { color: var(--muted); }
        .af-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 12px;
            border-radius: 14px;
            border: 1px solid var(--soft);
            background: rgba(255,255,255,0.72);
        }
        .af-section-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
        }
        .af-section-title {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .af-section-title i { color: rgba(15,23,42,0.65); }
        .af-soft-note { background: rgba(255,255,255,0.70); border: 1px dashed rgba(15,23,42,0.16); }

        /* Summary + Metrics polish */
        .af-hero {
            border: 1px solid var(--soft);
            background: rgba(255,255,255,0.78);
            border-radius: 18px;
            padding: 16px;
        }
        .af-hero-value {
            font-weight: 900;
            letter-spacing: -0.03em;
            font-size: clamp(28px, 2.6vw, 38px);
            line-height: 1.05;
        }
        .af-kpi-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
        @media (max-width: 992px) { .af-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
        @media (max-width: 576px) { .af-kpi-grid { grid-template-columns: 1fr; } }
        .af-kpi {
            border: 1px solid var(--soft);
            background: rgba(255,255,255,0.72);
            border-radius: 14px;
            padding: 12px;
        }
        .af-kpi .label { color: var(--muted); font-size: 12px; }
        .af-kpi .value { font-weight: 800; font-size: 18px; letter-spacing: -0.02em; margin-top: 4px; }
        .af-kpi .sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
        .af-list { list-style: none; padding-left: 0; margin: 0; }
        .af-list li {
            display: flex;
            gap: 10px;
            align-items: flex-start;
            padding: 8px 0;
            border-bottom: 1px dashed var(--soft);
        }
        .af-list li:last-child { border-bottom: 0; }
        .af-list i { color: rgba(15,23,42,0.45); margin-top: 3px; }
        .af-table-wrap {
            border: 1px solid var(--soft);
            background: rgba(255,255,255,0.72);
            border-radius: 14px;
            overflow: hidden;
        }
        .af-table-wrap .table { margin-bottom: 0; }
        .af-table-wrap .table thead { background: rgba(15, 23, 42, 0.04); }
        .af-table-wrap .table thead th { font-weight: 750; font-size: 13px; color: rgba(15,23,42,0.80); }
        .af-table-wrap .table tbody td { vertical-align: middle; }
        .af-sw-box {
            border: 1px solid var(--soft);
            background: rgba(255,255,255,0.70);
            border-radius: 14px;
            padding: 12px;
            height: 100%;
        }

        /* ENBD/HDFC extractor tab layout (screenshot-like) */
        .af-block-title {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 0;
            font-weight: 800;
            letter-spacing: -0.02em;
        }
        .af-block-subtitle { color: var(--muted); }
        .af-metrics-card { padding: 18px; }
        .af-metrics-head {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
            margin-bottom: 12px;
        }
        .af-units-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(15,23,42,0.12);
            background: rgba(255,255,255,0.75);
            color: rgba(15,23,42,0.72);
            font-size: 12px;
        }
        .af-table-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 700;
            margin: 0 0 10px 0;
        }
        .af-table-title i { color: rgba(15,23,42,0.65); }
        .af-table thead th {
            background: rgba(15,23,42,0.92);
            color: rgba(255,255,255,0.95);
            font-weight: 700;
            border-bottom: 0;
        }
        .af-table tbody tr:nth-child(odd) { background: rgba(2,6,23,0.02); }
        .af-value-usd { color: var(--muted); font-size: 12px; margin-top: 2px; }

        .af-ratios-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
        @media (max-width: 992px) { .af-ratios-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
        @media (max-width: 576px) { .af-ratios-grid { grid-template-columns: 1fr; } }
        .af-ratio-card {
            border: 1px solid rgba(15,23,42,0.10);
            background: rgba(255,255,255,0.78);
            border-radius: 14px;
            padding: 14px 14px;
            box-shadow: 0 10px 26px rgba(15,23,42,0.06);
            min-height: 86px;
        }
        .af-ratio-name { color: #2563eb; font-weight: 700; font-size: 13px; }
        .af-ratio-val { font-weight: 800; font-size: 22px; letter-spacing: -0.02em; margin-top: 6px; }

        /* OpenAI chat panel */
        .af-chat-wrap {
            border: 1px solid var(--soft);
            background: rgba(255,255,255,0.94);
            border-radius: 14px;
            overflow: hidden;
        }
        .af-chat-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 12px 14px;
            background: rgba(15, 23, 42, 0.06);
            border-bottom: 1px solid var(--soft);
        }
        .af-chat-head .title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 0;
        }
        .af-chat-log {
            padding: 14px;
            height: 360px;
            overflow: auto;
        }
        .af-bubble {
            max-width: 92%;
            border-radius: 14px;
            padding: 10px 12px;
            border: 1px solid rgba(15,23,42,0.10);
            background: rgba(255,255,255,0.96);
            box-shadow: 0 10px 26px rgba(15,23,42,0.06);
            white-space: pre-wrap;
        }
        .af-msg { display: flex; margin-bottom: 10px; }
        .af-msg.user { justify-content: flex-end; }
        .af-msg.user .af-bubble {
            border: none;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: rgba(255,255,255,0.96);
        }
        .af-msg.assistant { justify-content: flex-start; }
        .af-chat-foot {
            padding: 12px 14px;
            border-top: 1px solid var(--soft);
            background: rgba(255,255,255,0.92);
        }
        .af-typing-caret {
            display: inline-block;
            width: 8px;
            margin-left: 2px;
            border-radius: 2px;
            background: rgba(15,23,42,0.55);
            height: 1em;
            vertical-align: -0.15em;
            animation: afBlink 1s steps(2, start) infinite;
        }
        @keyframes afBlink {
            to { visibility: hidden; }
        }

        /* Floating OpenAI panel */
        .af-float-chat {
            position: fixed;
            right: 22px;
            bottom: 22px;
            z-index: 2000;
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 10px;
        }
        .af-chat-fab {
            border: none;
            border-radius: 999px;
            padding: 12px 14px;
            color: rgba(255,255,255,0.96);
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            box-shadow: 0 18px 60px rgba(15,23,42,0.18);
            display: inline-flex;
            align-items: center;
            gap: 10px;
            font-weight: 800;
            letter-spacing: -0.01em;
            cursor: pointer;
            touch-action: manipulation;
            -webkit-tap-highlight-color: transparent;
        }
        .af-chat-fab:focus-visible {
            outline: 3px solid rgba(15, 23, 42, 0.55);
            outline-offset: 2px;
        }
        .af-chat-fab:disabled {
            filter: grayscale(0.2);
            opacity: 0.75;
        }
        .af-float-panel {
            width: min(460px, calc(100vw - 44px));
            height: min(640px, calc(100vh - 110px));
            height: min(640px, calc(100dvh - 110px));
            display: none;
            flex-direction: column;
            box-shadow: var(--shadow-lg);
            transform: translateY(10px);
            opacity: 0;
            transition: transform 160ms ease, opacity 160ms ease;
        }
        .af-float-panel.open {
            display: flex;
            transform: translateY(0);
            opacity: 1;
        }
        .af-float-panel .af-chat-log {
            height: auto;
            flex: 1;
            min-height: 220px;
        }
        .af-chat-head-actions {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
            justify-content: flex-end;
            row-gap: 6px;
        }
        .af-chat-mini-btn {
            border-radius: 12px;
        }

        @media (max-width: 576px) {
            .af-float-chat {
                right: 12px;
                bottom: 12px;
            }
            .af-float-panel {
                width: calc(100vw - 24px);
                height: calc(100vh - 96px);
                height: calc(100dvh - 96px);
            }
        }

        /* Floating panel internal tabs */
        .af-chat-tabs {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 14px;
            background: rgba(255,255,255,0.65);
            border-bottom: 1px solid var(--soft);
        }
        .af-chat-tab-btn {
            border: 1px solid var(--soft);
            background: rgba(255,255,255,0.72);
            color: #334155;
            border-radius: 999px;
            padding: 6px 10px;
            font-weight: 800;
            font-size: 12px;
        }
        .af-chat-tab-btn.active {
            background: rgba(16,163,127,0.10);
            box-shadow: 0 10px 28px rgba(15,23,42,0.08);
            color: #0b1220;
        }
        .af-chat-tab-pane {
            display: none;
            flex-direction: column;
            flex: 1;
            min-height: 0;
        }
        .af-chat-tab-pane.active { display: flex; }
        .af-chat-context {
            padding: 14px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            overflow: auto;
        }
    </style>
</head>
<body>
<div class="container-xxl py-4">
    <div class="af-header p-3 p-md-4 mb-3">
        <div class="d-flex justify-content-between align-items-start flex-wrap gap-3">
            <div>
                <div class="af-header-title">
                    <div class="af-icon"><i class="fa-solid fa-chart-line"></i></div>
                    <div>
                        <div class="d-flex align-items-center flex-wrap gap-2">
                            <h3 class="mb-0">Financial Statement Analyzer and Recommendation Agent</h3>
                            <span class="badge text-bg-primary">Extractor Comparator</span>
                        </div>
                        <div class="af-subtitle mt-1">Upload PDFs, run both extractors, score extraction quality, and recommend the best-performing organisation using standardized ratios.</div>
                    </div>
                </div>
            </div>
            <div class="af-chip">
                <div class="text-muted small">OpenAI (optional)</div>
                <div>
                    {% if openai and openai.ready %}
                        <span class="badge text-bg-success">Ready</span>
                        <span class="text-muted ms-2 small">source={{ openai.source }} · model={{ openai.chat_model }}</span>
                    {% else %}
                        <span class="badge text-bg-secondary">Not configured</span>
                        <span class="text-muted ms-2 small">Heuristic-only analysis is available.</span>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>

    <div class="card p-3 p-md-4 mb-3">
        <div class="af-section-head mb-2">
            <div>
                <h5 class="af-section-title"><i class="fa-solid fa-file-arrow-up"></i>Upload PDFs</h5>
                <div class="text-muted small">Multi-select supported</div>
            </div>
            <div class="af-chip text-muted small">
                <i class="fa-solid fa-shield-halved"></i>
                Files are processed locally by the extractors.
            </div>
        </div>
        <form method="post" action="/upload" enctype="multipart/form-data" class="row g-3 align-items-end">
            <div class="col-md-9">
                <label class="form-label">Financial statement PDFs</label>
                <input type="file" name="pdf_files" accept="application/pdf" class="form-control" multiple required />
                <div class="form-text">Select one or more PDFs to generate an overall recommendation.</div>
            </div>
            <div class="col-md-3">
                <button class="btn btn-primary w-100" type="submit"><i class="fa-solid fa-magnifying-glass-chart me-2"></i>Analyze</button>
            </div>
            {% if error %}
                <div class="col-12"><div class="alert alert-danger mb-0"><i class="fa-solid fa-triangle-exclamation me-2"></i>{{ error }}</div></div>
            {% endif %}
        </form>
    </div>

    {% if result %}
        {% if result.results %}
            <ul class="nav af-tabs" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="tab-summary-btn" data-bs-toggle="tab" data-bs-target="#tab-summary" type="button" role="tab">Summary</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="tab-metrics-btn" data-bs-toggle="tab" data-bs-target="#tab-metrics" type="button" role="tab">Key Metrics</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="tab-enbd-btn" data-bs-toggle="tab" data-bs-target="#tab-enbd" type="button" role="tab">ENBD</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="tab-hdfc-btn" data-bs-toggle="tab" data-bs-target="#tab-hdfc" type="button" role="tab">HDFC</button>
                </li>
            </ul>

            <div class="tab-content pt-3">
                <div class="tab-pane fade show active" id="tab-summary" role="tabpanel" aria-labelledby="tab-summary-btn">
                    <div class="card p-3 p-md-4 mb-3">
                        <div class="row g-3 align-items-stretch">
                            <div class="col-lg-7">
                                <div class="af-hero h-100">
                                    <div class="d-flex justify-content-between gap-3 flex-wrap">
                                        <div>
                                            <div class="text-muted small">Recommended organisation</div>
                                            <div class="d-flex align-items-end gap-2 flex-wrap">
                                                <div class="af-hero-value">{{ result.recommended_organisation or '—' }}</div>
                                                {% if result.recommended_organisation %}
                                                    <span class="badge text-bg-primary">Top match</span>
                                                {% endif %}
                                            </div>
                                            <div class="text-muted">PDFs analyzed: <span class="mono">{{ result.pdf_count }}</span></div>
                                        </div>
                                        <div class="af-kpi-grid">
                                            <div class="af-kpi">
                                                <div class="label">Organisations compared</div>
                                                <div class="value mono">{{ (result.organisations or {})|length }}</div>
                                                <div class="sub">Detected from uploads</div>
                                            </div>
                                            <div class="af-kpi">
                                                <div class="label">Metrics tracked</div>
                                                <div class="value mono">{{ result.metric_schema|length }}</div>
                                                <div class="sub">Standard ratios</div>
                                            </div>
                                            <div class="af-kpi">
                                                <div class="label">PDFs analyzed</div>
                                                <div class="value mono">{{ result.pdf_count }}</div>
                                                <div class="sub">Winner-based scoring</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-lg-5">
                                <div class="af-soft-note p-3 rounded-4 h-100">
                                    <div class="text-muted small">Scoring note</div>
                                    <div class="mono small">Uses ratios like net profit margin, ROA/ROE, cost-to-income, and NPL/NPA when available.</div>
                                    <div class="text-muted small mt-2">Tip: For apples-to-apples comparison, review the USD equivalents in ENBD/HDFC tabs.</div>
                                </div>
                            </div>
                        </div>

                        {% set rec = result.final_recommendation_struct or {} %}
                        {% if rec and rec.recommended %}
                            <hr/>
                            <div class="text-muted mb-2">Recommendation</div>
                            <div class="af-hero">
                                <div class="d-flex justify-content-between align-items-start flex-wrap gap-3">
                                    <div>
                                        <div class="text-muted small">Final recommendation</div>
                                        <div class="d-flex align-items-end gap-2 flex-wrap">
                                            <div class="af-hero-value">{{ rec.recommended }}</div>
                                            <span class="badge text-bg-primary">Top match</span>
                                        </div>
                                        <div class="text-muted">PDFs analyzed: <span class="mono">{{ rec.pdf_count }}</span>{% if rec.runner_up %} · Runner-up: <span class="mono">{{ rec.runner_up }}</span>{% endif %}</div>
                                    </div>
                                    <div class="af-soft-note p-3 rounded-4" style="max-width: 520px;">
                                        <div class="text-muted small">Note</div>
                                        <div class="mono small">{{ rec.note }}</div>
                                    </div>
                                </div>

                                {% if rec.reasons and rec.reasons|length > 0 %}
                                    <hr/>
                                    <div class="text-muted mb-2">Key reasons</div>
                                    <ul class="af-list">
                                        {% for rr in rec.reasons %}
                                            <li><i class="fa-solid fa-circle-check"></i><div><strong>{{ rr.split(':')[0] }}</strong>{% if ':' in rr %}: {{ rr.split(':', 1)[1].strip() }}{% endif %}</div></li>
                                        {% endfor %}
                                    </ul>
                                {% endif %}

                                <div class="row g-3 mt-1">
                                    <div class="col-md-6">
                                        <div class="af-sw-box">
                                            <div class="af-table-title"><i class="fa-solid fa-thumbs-up"></i> Strengths</div>
                                            {% if rec.strengths and rec.strengths|length > 0 %}
                                                <ul class="mb-0">
                                                    {% for s in rec.strengths[:5] %}
                                                        <li>{{ s }}</li>
                                                    {% endfor %}
                                                </ul>
                                            {% else %}
                                                <div class="text-muted small">No clear strengths found (or insufficient data).</div>
                                            {% endif %}
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="af-sw-box">
                                            <div class="af-table-title"><i class="fa-solid fa-triangle-exclamation"></i> Weaknesses / watch-outs</div>
                                            {% if rec.weaknesses and rec.weaknesses|length > 0 %}
                                                <ul class="mb-0">
                                                    {% for w in rec.weaknesses[:5] %}
                                                        <li>{{ w }}</li>
                                                    {% endfor %}
                                                </ul>
                                            {% else %}
                                                <div class="text-muted small">No clear weaknesses found (or insufficient data).</div>
                                            {% endif %}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        {% endif %}

                        {% if result.organisations and result.organisations|length > 0 %}
                            <hr/>
                            <div class="text-muted mb-2">Organisation summary</div>
                            <div class="af-table-wrap">
                                <div class="table-responsive">
                                <table class="table table-sm align-middle">
                                    <thead>
                                        <tr>
                                            <th>Organisation</th>
                                            <th class="text-end">PDFs</th>
                                            <th class="text-end">Avg performance</th>
                                            <th class="text-end">Avg extraction</th>
                                            <th class="text-end">Wins (HDFC/ENBD)</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {% for org, d in result.organisations.items() %}
                                            <tr>
                                                <td>{{ org }}</td>
                                                <td class="mono text-end">{{ d.pdf_count }}</td>
                                                <td class="mono text-end">{{ "%.3f"|format(d.avg_perf_score) }}</td>
                                                <td class="mono text-end">{{ "%.3f"|format(d.avg_extraction_score) }}</td>
                                                <td class="mono text-end">{{ (d.wins or {}).get('HDFC', 0) }}/{{ (d.wins or {}).get('ENBD', 0) }}</td>
                                            </tr>
                                        {% endfor %}
                                    </tbody>
                                </table>
                                </div>
                            </div>
                        {% endif %}
                    </div>

                    <div class="card p-3 p-md-4 mb-3">
                        <div class="text-muted mb-2">Per-PDF details</div>
                        {% for r in result.results %}
                            <details class="mb-2">
                                <summary>
                                    <div class="d-flex flex-wrap gap-2 align-items-center">
                                        <span class="mono">{{ r.pdf_path }}</span>
                                        <span class="badge text-bg-light border">org: <span class="mono">{{ r.organisation }}</span></span>
                                        <span class="badge text-bg-light border">perf: <span class="mono">{{ "%.3f"|format(r.org_performance.score) }}</span></span>
                                    </div>
                                </summary>

                                <div class="mt-2">
                                    <div class="text-muted">Bank hint</div>
                                    <div class="mono">likely={{ r.bank_hint.likely }} confidence={{ "%.2f"|format(r.bank_hint.confidence) }} signals={{ r.bank_hint.signals }}</div>
                                    <div class="text-muted mt-2">Performance metrics used</div>
                                    <div class="mono">used={{ r.org_performance.metrics_used }}/{{ r.org_performance.metrics_available }}</div>
                                </div>

                                {% if r.key_metrics %}
                                    <div class="mt-3">
                                        <div class="text-muted mb-2">Key metrics (winner ratios)</div>
                                        <div class="af-table-wrap">
                                            <div class="table-responsive">
                                            <table class="table table-sm align-middle">
                                                <thead>
                                                    <tr><th>Metric</th><th>Value</th></tr>
                                                </thead>
                                                <tbody>
                                                    {% for m in result.metric_schema %}
                                                        {% set v = r.key_metrics[m.key] %}
                                                        {% if v is not none %}
                                                            <tr>
                                                                <td>{{ m.label }}</td>
                                                                <td class="mono">{{ v|pct }}</td>
                                                            </tr>
                                                        {% endif %}
                                                    {% endfor %}
                                                </tbody>
                                            </table>
                                            </div>
                                        </div>
                                    </div>
                                {% endif %}

                                <div class="row g-3 mt-1">
                                    {% for name in ['HDFC', 'ENBD'] %}
                                        {% set run = r.runs[name] %}
                                        {% set sc = r.scores[name] %}
                                        <div class="col-lg-6">
                                            <div class="card p-3 h-100">
                                                <div class="d-flex justify-content-between align-items-start gap-2">
                                                    <div>
                                                        <div class="h6 mb-0">{{ name }}</div>
                                                        <div class="text-muted small">score={{ "%.3f"|format(sc.total) }} · fill={{ sc.filled_metrics }}/{{ sc.expected_metrics }} ({{ "%.1f"|format(sc.fill_rate*100) }}%) · ratios={{ sc.ratio_count }} · invalid={{ sc.invalid_ratio_count }} · time={{ "%.2f"|format(run.elapsed_s) }}s</div>
                                                    </div>
                                                    <div class="badge text-bg-{{ 'success' if run.ok else 'danger' }}">{{ 'OK' if run.ok else 'ERROR' }}</div>
                                                </div>
                                                <hr/>
                                                {% if run.ok %}
                                                    <div class="mb-2"><span class="text-muted">Units:</span> <span class="mono">{{ run.units }}</span></div>
                                                    {% if run.recs and run.recs|length > 0 %}
                                                        <div class="text-muted">Extractor recommendations</div>
                                                        <ul class="mb-0">
                                                            {% for rr in run.recs %}
                                                                <li>{{ rr }}</li>
                                                            {% endfor %}
                                                        </ul>
                                                    {% endif %}
                                                    <details class="mt-2">
                                                        <summary class="text-muted">Context preview</summary>
                                                        <pre class="mono af-pre mt-2">{{ (run.context or '')[:1200] }}</pre>
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

                    <div class="d-flex gap-2 mt-3">
                        <a class="btn btn-outline-secondary" href="/">Analyze another</a>
                        <a class="btn btn-outline-primary" href="/api/last">Download last result (JSON)</a>
                    </div>
                </div>

                <div class="tab-pane fade" id="tab-metrics" role="tabpanel" aria-labelledby="tab-metrics-btn">
                    <div class="card p-3 p-md-4 mb-3">
                        <div class="d-flex justify-content-between flex-wrap gap-2 align-items-end mb-2">
                            <div>
                                <div class="text-muted">Organisation key metrics</div>
                                <div class="text-muted small">Averaged across uploaded PDFs (winner-based)</div>
                            </div>
                            <div class="text-muted small">Missing values shown as —</div>
                        </div>
                        <div class="af-table-wrap">
                        <div class="table-responsive">
                            <table class="table table-sm align-middle">
                                <thead>
                                    <tr>
                                        <th>Organisation</th>
                                        <th class="text-end">PDFs</th>
                                        <th class="text-end">Metrics covered</th>
                                        {% for m in result.metric_schema %}
                                            <th class="text-end">{{ m.label }}</th>
                                        {% endfor %}
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for org, d in result.organisations.items() %}
                                        <tr>
                                            <td>{{ org }}</td>
                                            <td class="mono text-end">{{ d.pdf_count }}</td>
                                            <td class="mono text-end">{{ d.metrics_coverage or 0 }}/{{ result.metric_schema|length }}</td>
                                            {% for m in result.metric_schema %}
                                                {% set v = (d.metrics or {}).get(m.key) %}
                                                <td class="mono text-end">{% if v is not none %}{{ v|pct }}{% else %}—{% endif %}</td>
                                            {% endfor %}
                                        </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                        </div>
                        <div class="form-text">Metrics are computed from the winner extractor per PDF.</div>
                    </div>

                    <div class="card p-3 p-md-4 mb-3">
                        <div class="text-muted mb-2">Strengths & weaknesses (relative comparison)</div>
                        {% for org, d in result.organisations.items() %}
                            <div class="mb-3">
                                <div class="h6 mb-1">{{ org }}</div>
                                {% set sw = (result.insights.strengths_weaknesses or {}).get(org) %}
                                <div class="row g-3">
                                    <div class="col-md-6">
                                        <div class="af-sw-box">
                                        <div class="text-muted">Strengths</div>
                                        {% if sw and sw.strengths and sw.strengths|length > 0 %}
                                            <ul class="af-list mt-2">
                                                {% for s in sw.strengths %}
                                                    <li><i class="fa-solid fa-circle-check"></i><div>{{ s }}</div></li>
                                                {% endfor %}
                                            </ul>
                                        {% else %}
                                            <div class="mono mt-2">No strong differentiators found (or insufficient data).</div>
                                        {% endif %}
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="af-sw-box">
                                        <div class="text-muted">Weaknesses</div>
                                        {% if sw and sw.weaknesses and sw.weaknesses|length > 0 %}
                                            <ul class="af-list mt-2">
                                                {% for w in sw.weaknesses %}
                                                    <li><i class="fa-solid fa-circle-xmark"></i><div>{{ w }}</div></li>
                                                {% endfor %}
                                            </ul>
                                        {% else %}
                                            <div class="mono mt-2">No clear weaknesses found (or insufficient data).</div>
                                        {% endif %}
                                        </div>
                                    </div>
                                </div>
                            </div>
                            {% if not loop.last %}<hr/>{% endif %}
                        {% endfor %}
                    </div>

                    <div class="card p-3 p-md-4 mb-3">
                        <div class="text-muted mb-2">Per-PDF key metrics (winner-based)</div>
                        <div class="af-table-wrap">
                        <div class="table-responsive">
                            <table class="table table-sm align-middle">
                                <thead>
                                    <tr>
                                        <th>PDF</th>
                                        <th>Organisation</th>
                                        {% for m in result.metric_schema %}
                                            <th class="text-end">{{ m.label }}</th>
                                        {% endfor %}
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for r in result.results %}
                                        <tr>
                                            <td class="mono">{{ r.pdf_path }}</td>
                                            <td class="mono">{{ r.organisation }}</td>
                                            {% for m in result.metric_schema %}
                                                {% set v = (r.key_metrics or {}).get(m.key) %}
                                                <td class="mono text-end">{% if v is not none %}{{ v|pct }}{% else %}—{% endif %}</td>
                                            {% endfor %}
                                        </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                        </div>
                    </div>
                </div>

                <div class="tab-pane fade" id="tab-enbd" role="tabpanel" aria-labelledby="tab-enbd-btn">
                    <div class="card p-3 mb-3">
                        <div class="d-flex justify-content-between flex-wrap gap-2">
                            <div>
                                <div class="h5 mb-0">ENBD extraction</div>
                                <div class="text-muted">Raw extracted metrics and computed ratios from the ENBD extractor.</div>
                                <div class="text-muted small mt-1">Source: <span class="mono">{{ (sources or {}).get('ENBD') }}</span></div>
                            </div>
                            <div class="af-chip small text-muted">
                                <i class="fa-solid fa-dollar-sign"></i>
                                USD normalization uses FX: AED/USD={{ (fx or {}).get('AED') }}, INR/USD={{ (fx or {}).get('INR') }}
                            </div>
                        </div>
                    </div>
                    {% for r in result.results %}
                        {% if r.organisation == 'ENBD' %}
                        {% set run = (r.runs or {}).get('ENBD') %}
                        <div class="card p-3 mb-3">
                            <div class="d-flex justify-content-between flex-wrap gap-2">
                                <div>
                                    <div class="h6 mb-0"><span class="mono">{{ r.pdf_path }}</span></div>
                                    <div class="text-muted small">org={{ r.organisation }}</div>
                                </div>
                                <div>
                                    <span class="badge text-bg-{{ 'success' if run and run.ok else 'danger' }}">{{ 'OK' if run and run.ok else 'ERROR' }}</span>
                                </div>
                            </div>
                            <hr/>
                            {% if run and run.ok %}
                                {% set cu = (run.units or {}).get('currency') %}
                                {% set ul = (run.units or {}).get('units_label') %}
                                <div class="af-metrics-card">
                                    <div class="af-metrics-head">
                                        <div>
                                            <h4 class="af-block-title mb-1"><i class="fa-solid fa-table"></i>Extracted Financial Metrics</h4>
                                            <div class="af-block-subtitle">{% if cu %}Values in {{ cu }}{% if ul %} {{ ul }}{% endif %}{% endif %}</div>
                                        </div>
                                        <div class="af-units-pill" title="{{ run.units }}"><i class="fa-solid fa-coins"></i>
                                            {% if cu %}{{ cu }}{% else %}—{% endif %}{% if ul %} {{ ul }}{% endif %}
                                        </div>
                                    </div>

                                    <div class="row g-4">
                                        <div class="col-lg-7">
                                            <div class="af-table-title"><i class="fa-solid fa-list"></i>Income Statement (Current vs Prior)</div>
                                            <div class="table-responsive">
                                                <table class="table table-sm align-middle af-table mb-0">
                                                    <thead><tr><th>Line Item</th><th class="text-end">Current</th><th class="text-end">Prior</th></tr></thead>
                                                    <tbody>
                                                        {% for k, v in (run.dual or {}).items() %}
                                                            <tr>
                                                                <td><strong>{{ k }}</strong></td>
                                                                <td class="mono text-end">
                                                                    {% if v.current is not none %}{{ v.current|fmt_num }}{% else %}—{% endif %}
                                                                    {% if v.current is not none %}<div class="af-value-usd">{{ v.current|usd(run.units) }}</div>{% endif %}
                                                                </td>
                                                                <td class="mono text-end">
                                                                    {% if v.prior is not none %}{{ v.prior|fmt_num }}{% else %}—{% endif %}
                                                                    {% if v.prior is not none %}<div class="af-value-usd">{{ v.prior|usd(run.units) }}</div>{% endif %}
                                                                </td>
                                                            </tr>
                                                        {% endfor %}
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                        <div class="col-lg-5">
                                            <div class="af-table-title"><i class="fa-solid fa-scale-balanced"></i>Balance Sheet Items</div>
                                            <div class="table-responsive">
                                                <table class="table table-sm align-middle af-table mb-0">
                                                    <thead><tr><th>Item</th><th class="text-end">Value</th></tr></thead>
                                                    <tbody>
                                                        {% for k, v in (run.single or {}).items() %}
                                                            <tr>
                                                                <td><strong>{{ k }}</strong></td>
                                                                <td class="mono text-end">
                                                                    {{ v|fmt_num }}
                                                                    {% if v is not none %}<div class="af-value-usd">{{ v|usd(run.units) }}</div>{% endif %}
                                                                </td>
                                                            </tr>
                                                        {% endfor %}
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <div class="card p-3 mt-3">
                                    <h4 class="af-block-title mb-0"><i class="fa-solid fa-calculator"></i>Financial Ratios</h4>
                                    <div class="mt-3 af-ratios-grid">
                                        {% for name, val in (run.ratios or []) %}
                                            {% if val is not none %}
                                                <div class="af-ratio-card">
                                                    <div class="af-ratio-name">{{ name }}</div>
                                                    <div class="af-ratio-val">{{ "%.2f"|format(val*100) }}%</div>
                                                </div>
                                            {% endif %}
                                        {% endfor %}
                                    </div>
                                </div>

                                {% if run.recs and run.recs|length > 0 %}
                                    <div class="mt-3">
                                        <div class="text-muted">Extractor recommendations</div>
                                        <ul class="mb-0">
                                            {% for rr in run.recs %}
                                                <li>{{ rr }}</li>
                                            {% endfor %}
                                        </ul>
                                    </div>
                                {% endif %}

                                <details class="mt-3">
                                    <summary class="text-muted">Context preview</summary>
                                    <pre class="mono af-pre mt-2">{{ (run.context or '')[:1800] }}</pre>
                                </details>
                            {% else %}
                                <div class="alert alert-danger mb-0">{{ (run.error if run else 'Extractor failed') }}</div>
                            {% endif %}
                        </div>
                        {% endif %}
                    {% endfor %}
                </div>

                <div class="tab-pane fade" id="tab-hdfc" role="tabpanel" aria-labelledby="tab-hdfc-btn">
                    <div class="card p-3 mb-3">
                        <div class="d-flex justify-content-between flex-wrap gap-2">
                            <div>
                                <div class="h5 mb-0">HDFC extraction</div>
                                <div class="text-muted">Raw extracted metrics and computed ratios from the HDFC extractor.</div>
                                <div class="text-muted small mt-1">Source: <span class="mono">{{ (sources or {}).get('HDFC') }}</span></div>
                            </div>
                            <div class="af-chip small text-muted">
                                <i class="fa-solid fa-dollar-sign"></i>
                                USD normalization uses FX: AED/USD={{ (fx or {}).get('AED') }}, INR/USD={{ (fx or {}).get('INR') }}
                            </div>
                        </div>
                    </div>
                    {% for r in result.results %}
                        {% if r.organisation == 'HDFC' %}
                        {% set run = (r.runs or {}).get('HDFC') %}
                        <div class="card p-3 mb-3">
                            <div class="d-flex justify-content-between flex-wrap gap-2">
                                <div>
                                    <div class="h6 mb-0"><span class="mono">{{ r.pdf_path }}</span></div>
                                    <div class="text-muted small">org={{ r.organisation }}</div>
                                </div>
                                <div>
                                    <span class="badge text-bg-{{ 'success' if run and run.ok else 'danger' }}">{{ 'OK' if run and run.ok else 'ERROR' }}</span>
                                </div>
                            </div>
                            <hr/>
                            {% if run and run.ok %}
                                {% set cu = (run.units or {}).get('currency') %}
                                {% set ul = (run.units or {}).get('units_label') %}
                                <div class="af-metrics-card">
                                    <div class="af-metrics-head">
                                        <div>
                                            <h4 class="af-block-title mb-1"><i class="fa-solid fa-table"></i>Extracted Financial Metrics</h4>
                                            <div class="af-block-subtitle">{% if cu %}Values in {{ cu }}{% if ul %} {{ ul }}{% endif %}{% endif %}</div>
                                        </div>
                                        <div class="af-units-pill" title="{{ run.units }}"><i class="fa-solid fa-coins"></i>
                                            {% if cu %}{{ cu }}{% else %}—{% endif %}{% if ul %} {{ ul }}{% endif %}
                                        </div>
                                    </div>

                                    <div class="row g-4">
                                        <div class="col-lg-7">
                                            <div class="af-table-title"><i class="fa-solid fa-list"></i>Income Statement (Current vs Prior)</div>
                                            <div class="table-responsive">
                                                <table class="table table-sm align-middle af-table mb-0">
                                                    <thead><tr><th>Line Item</th><th class="text-end">Current</th><th class="text-end">Prior</th></tr></thead>
                                                    <tbody>
                                                        {% for k, v in (run.dual or {}).items() %}
                                                            <tr>
                                                                <td><strong>{{ k }}</strong></td>
                                                                <td class="mono text-end">
                                                                    {% if v.current is not none %}{{ v.current|fmt_num }}{% else %}—{% endif %}
                                                                    {% if v.current is not none %}<div class="af-value-usd">{{ v.current|usd(run.units) }}</div>{% endif %}
                                                                </td>
                                                                <td class="mono text-end">
                                                                    {% if v.prior is not none %}{{ v.prior|fmt_num }}{% else %}—{% endif %}
                                                                    {% if v.prior is not none %}<div class="af-value-usd">{{ v.prior|usd(run.units) }}</div>{% endif %}
                                                                </td>
                                                            </tr>
                                                        {% endfor %}
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                        <div class="col-lg-5">
                                            <div class="af-table-title"><i class="fa-solid fa-scale-balanced"></i>Balance Sheet Items</div>
                                            <div class="table-responsive">
                                                <table class="table table-sm align-middle af-table mb-0">
                                                    <thead><tr><th>Item</th><th class="text-end">Value</th></tr></thead>
                                                    <tbody>
                                                        {% for k, v in (run.single or {}).items() %}
                                                            <tr>
                                                                <td><strong>{{ k }}</strong></td>
                                                                <td class="mono text-end">
                                                                    {{ v|fmt_num }}
                                                                    {% if v is not none %}<div class="af-value-usd">{{ v|usd(run.units) }}</div>{% endif %}
                                                                </td>
                                                            </tr>
                                                        {% endfor %}
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <div class="card p-3 mt-3">
                                    <h4 class="af-block-title mb-0"><i class="fa-solid fa-calculator"></i>Financial Ratios</h4>
                                    <div class="mt-3 af-ratios-grid">
                                        {% for name, val in (run.ratios or []) %}
                                            {% if val is not none %}
                                                <div class="af-ratio-card">
                                                    <div class="af-ratio-name">{{ name }}</div>
                                                    <div class="af-ratio-val">{{ "%.2f"|format(val*100) }}%</div>
                                                </div>
                                            {% endif %}
                                        {% endfor %}
                                    </div>
                                </div>

                                {% if run.recs and run.recs|length > 0 %}
                                    <div class="mt-3">
                                        <div class="text-muted">Extractor recommendations</div>
                                        <ul class="mb-0">
                                            {% for rr in run.recs %}
                                                <li>{{ rr }}</li>
                                            {% endfor %}
                                        </ul>
                                    </div>
                                {% endif %}

                                <details class="mt-3">
                                    <summary class="text-muted">Context preview</summary>
                                    <pre class="mono af-pre mt-2">{{ (run.context or '')[:1800] }}</pre>
                                </details>
                            {% else %}
                                <div class="alert alert-danger mb-0">{{ (run.error if run else 'Extractor failed') }}</div>
                            {% endif %}
                        </div>
                        {% endif %}
                    {% endfor %}
                </div>

            </div>
        {% else %}
            <div class="alert alert-warning">No results.</div>
        {% endif %}
    {% endif %}
</div>

<!-- Floating FinanceBot chat widget (bottom-right) -->
<textarea id="afLlmContext" class="d-none">{{ (result.llm_context if result else '') or '' }}</textarea>
<textarea id="afLlmContextSummary" class="d-none">{{ (result.llm_context_summary if result else '') or '' }}</textarea>
<textarea id="afLlmContextMetrics" class="d-none">{{ (result.llm_context_metrics if result else '') or '' }}</textarea>
<textarea id="afLlmContextENBD" class="d-none">{{ (result.llm_context_enbd if result else '') or '' }}</textarea>
<textarea id="afLlmContextHDFC" class="d-none">{{ (result.llm_context_hdfc if result else '') or '' }}</textarea>
<div class="af-float-chat" id="afFloatChat">
    <button class="af-chat-fab" id="afChatFab" type="button" aria-controls="afChatPanel" aria-expanded="false" onclick="if(window.__FinanceBotMainBound){return;}(function(){try{var p=document.getElementById('afChatPanel');var b=document.getElementById('afChatFab');if(!p||!b)return;var open=p.classList.contains('open');if(open){p.classList.remove('open');p.style.display='';p.setAttribute('aria-hidden','true');b.setAttribute('aria-expanded','false');}else{p.classList.add('open');p.style.display='flex';p.setAttribute('aria-hidden','false');b.setAttribute('aria-expanded','true');}}catch(e){try{console.error('afChatFab onclick error',e);}catch(_){}}})();">
        <i class="fa-solid fa-wand-magic-sparkles"></i>
        <span>FinanceBot</span>
    </button>
    <div class="af-chat-wrap af-float-panel" id="afChatPanel" aria-hidden="true">
        <div class="af-chat-head">
            <div class="title"><i class="fa-solid fa-comments"></i> FinanceBot</div>
            <div class="af-chat-head-actions">
                {% if openai and openai.ready %}
                    <span class="badge text-bg-success">Ready</span>
                    <span class="text-muted small">model=<span class="mono">{{ openai.chat_model }}</span></span>
                {% else %}
                    <span class="badge text-bg-secondary">Not configured</span>
                {% endif %}
                <button class="btn btn-outline-secondary btn-sm af-chat-mini-btn" id="afChatStop" type="button" title="Stop generating" disabled>
                    <i class="fa-solid fa-stop"></i>
                </button>
                <button class="btn btn-outline-secondary btn-sm af-chat-mini-btn" id="afChatReset" type="button" title="Reset chat">
                    <i class="fa-solid fa-rotate-left"></i>
                </button>
                <button class="btn btn-outline-secondary btn-sm af-chat-mini-btn" id="afChatClose" type="button" title="Minimize" aria-label="Minimize FinanceBot panel" onclick="if(window.__FinanceBotMainBound){return;}(function(){try{var p=document.getElementById('afChatPanel');var b=document.getElementById('afChatFab');if(!p||!b)return;p.classList.remove('open');p.style.display='';p.setAttribute('aria-hidden','true');b.setAttribute('aria-expanded','false');}catch(e){try{console.error('afChatClose onclick error',e);}catch(_){}}})();">
                    <i class="fa-solid fa-minus"></i>
                </button>
            </div>
        </div>
        <div class="af-chat-tabs" role="tablist" aria-label="FinanceBot panel tabs">
            <button class="af-chat-tab-btn active" id="afChatTabChatBtn" type="button" aria-controls="afChatTabChatPane" aria-selected="true">Chat</button>
            <button class="af-chat-tab-btn" id="afChatTabContextBtn" type="button" aria-controls="afChatTabContextPane" aria-selected="false">Insert context</button>
        </div>

        <div class="af-chat-tab-pane active" id="afChatTabChatPane" role="tabpanel" aria-labelledby="afChatTabChatBtn">
            <div class="af-chat-log" id="afChatLog" aria-live="polite"></div>
            <div class="af-chat-foot">
                <div class="row g-2 align-items-end">
                    <div class="col-9">
                        <label class="form-label small text-muted mb-1" for="afChatInput">Your question</label>
                        <textarea class="form-control" id="afChatInput" rows="2" placeholder="Ask about the current analysis…"></textarea>
                    </div>
                    <div class="col-3 d-grid">
                        <button class="btn btn-primary" id="afChatSend" type="button" onclick="if(window.__FinanceBotMainBound){return;}(function(){try{var logEl=document.getElementById('afChatLog');var inputEl=document.getElementById('afChatInput');var activeBtn=document.querySelector('.af-tabs .nav-link.active');var target=(activeBtn&&activeBtn.getAttribute?activeBtn.getAttribute('data-bs-target'):null)||'#tab-summary';var map={'#tab-summary':'afLlmContextSummary','#tab-metrics':'afLlmContextMetrics','#tab-enbd':'afLlmContextENBD','#tab-hdfc':'afLlmContextHDFC'};var ctxEl=document.getElementById(map[target]||'afLlmContext')||document.getElementById('afLlmContext');var msg=(inputEl&&inputEl.value?inputEl.value:'').trim();if(!msg||!logEl||!inputEl)return;var ctx=(ctxEl&&ctxEl.value?ctxEl.value:'');var append=function(role,text){var row=document.createElement('div');row.className='af-msg '+role;var bubble=document.createElement('div');bubble.className='af-bubble';bubble.textContent=text||'';row.appendChild(bubble);logEl.appendChild(row);try{logEl.scrollTop=logEl.scrollHeight;}catch(e){}};append('user',msg);inputEl.value='';inputEl.disabled=true;var btn=document.getElementById('afChatSend');if(btn)btn.disabled=true;append('assistant','…');var lastBubble=logEl.lastChild&&logEl.lastChild.querySelector?logEl.lastChild.querySelector('.af-bubble'):null;fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg,context:ctx,history:[]})}).then(function(r){return r.json().catch(function(){return {error:r.statusText||'bad response'};});}).then(function(data){var out=(data&&data.answer)?data.answer:('Error: '+((data&&data.error)?data.error:'Unknown error'));if(lastBubble)lastBubble.textContent=out;}).catch(function(e){if(lastBubble)lastBubble.textContent='Error: '+e;}).finally(function(){inputEl.disabled=false;try{inputEl.focus();}catch(e){}if(btn)btn.disabled=false;});}catch(e){try{console.error('afChatSend fallback error',e);}catch(_){} }})();"><i class="fa-solid fa-paper-plane me-2"></i>Send</button>
                    </div>
                </div>
                <div class="form-text">Tip: Press <span class="mono">Enter</span> to send · <span class="mono">Shift+Enter</span> for a new line.</div>
            </div>
        </div>

        <div class="af-chat-tab-pane" id="afChatTabContextPane" role="tabpanel" aria-labelledby="afChatTabContextBtn">
            <div class="af-chat-context">
                <div class="text-muted small">Uses the compact analysis context for the currently active main tab (no PDFs are uploaded to OpenAI). Click “Insert into message” to paste it into the chat box.</div>
                <textarea class="form-control mono" id="afContextPreview" rows="10" readonly></textarea>
                <div class="d-flex gap-2">
                    <button class="btn btn-outline-secondary btn-sm" id="afInsertCtxFromTab" type="button" title="Insert analysis context into the message box" onclick="if(window.__FinanceBotMainBound){return;}(function(){try{var activeBtn=document.querySelector('.af-tabs .nav-link.active');var target=(activeBtn&&activeBtn.getAttribute?activeBtn.getAttribute('data-bs-target'):null)||'#tab-summary';var map={'#tab-summary':'afLlmContextSummary','#tab-metrics':'afLlmContextMetrics','#tab-enbd':'afLlmContextENBD','#tab-hdfc':'afLlmContextHDFC'};var ctxEl=document.getElementById(map[target]||'afLlmContext')||document.getElementById('afLlmContext');var inputEl=document.getElementById('afChatInput');var tabBtn=document.getElementById('afChatTabChatBtn');var ctx=(ctxEl&&ctxEl.value?ctxEl.value:'').trim();if(!ctx||!inputEl)return;var existing=(inputEl.value||'').trim();inputEl.value=existing?(existing+'\n\n'+ctx):ctx;try{tabBtn&&tabBtn.click();}catch(e){}try{inputEl.focus();}catch(e){}}catch(e){try{console.error('afInsertCtxFromTab onclick error',e);}catch(_){}}})();">
                        <i class="fa-solid fa-file-import me-1"></i>Insert into message
                    </button>
                    <button class="btn btn-outline-secondary btn-sm" id="afGoToChat" type="button" title="Back to chat">
                        <i class="fa-solid fa-comments me-1"></i>Go to chat
                    </button>
                </div>
                <div class="form-text">If the context is empty, run an analysis first.</div>
            </div>
        </div>
    </div>
</div>

<div class="container-xxl pb-4">
    <div class="text-center small text-muted">
        Designed for quick extractor comparison. Always validate results against source statements.
    </div>
</div>

<script>
(() => {
    try { if (typeof window.__FinanceBotMainBound === 'undefined') window.__FinanceBotMainBound = false; } catch (e) {}
    try {
    const openaiReady = {{ 'true' if (openai and openai.ready) else 'false' }};

    const fabEl = document.getElementById('afChatFab');
    const panelEl = document.getElementById('afChatPanel');
    const closeEl = document.getElementById('afChatClose');
    const stopEl = document.getElementById('afChatStop');
    const resetEl = document.getElementById('afChatReset');

    const logEl = document.getElementById('afChatLog');
    const inputEl = document.getElementById('afChatInput');
    const sendBtn = document.getElementById('afChatSend');
    const getAnalysisContext = () => {
        try {
            const activeBtn = document.querySelector('.af-tabs .nav-link.active');
            const target = (activeBtn && activeBtn.getAttribute) ? (activeBtn.getAttribute('data-bs-target') || '') : '';
            const map = {
                '#tab-summary': 'afLlmContextSummary',
                '#tab-metrics': 'afLlmContextMetrics',
                '#tab-enbd': 'afLlmContextENBD',
                '#tab-hdfc': 'afLlmContextHDFC',
            };
            const id = map[target] || 'afLlmContext';
            const el = document.getElementById(id) || document.getElementById('afLlmContext');
            return el ? (el.value || '') : '';
        } catch (e) {
            return '';
        }
    };

    const tabChatBtn = document.getElementById('afChatTabChatBtn');
    const tabContextBtn = document.getElementById('afChatTabContextBtn');
    const chatPaneEl = document.getElementById('afChatTabChatPane');
    const ctxPaneEl = document.getElementById('afChatTabContextPane');
    const ctxPreviewEl = document.getElementById('afContextPreview');
    const insertCtxFromTabBtn = document.getElementById('afInsertCtxFromTab');
    const goToChatBtn = document.getElementById('afGoToChat');

    const setInnerTab = (tabName) => {
        const isChat = tabName === 'chat';
        if (tabChatBtn) {
            tabChatBtn.classList.toggle('active', isChat);
            tabChatBtn.setAttribute('aria-selected', isChat ? 'true' : 'false');
        }
        if (tabContextBtn) {
            tabContextBtn.classList.toggle('active', !isChat);
            tabContextBtn.setAttribute('aria-selected', isChat ? 'false' : 'true');
        }
        if (chatPaneEl) chatPaneEl.classList.toggle('active', isChat);
        if (ctxPaneEl) ctxPaneEl.classList.toggle('active', !isChat);
        if (!isChat) {
            const ctxNow = getAnalysisContext();
            if (ctxPreviewEl) ctxPreviewEl.value = ctxNow || '';
            if (insertCtxFromTabBtn) insertCtxFromTabBtn.disabled = !(ctxNow || '').trim();
        }
        if (isChat) {
            try { inputEl && inputEl.focus(); } catch (e) {}
        }
    };

    if (ctxPreviewEl) ctxPreviewEl.value = getAnalysisContext() || '';
    if (insertCtxFromTabBtn) insertCtxFromTabBtn.disabled = !(getAnalysisContext() || '').trim();

    const setPanelOpen = (open) => {
        if (!panelEl || !fabEl) {
            try { console.warn('[FinanceBot] Chat panel elements missing', { fabEl: !!fabEl, panelEl: !!panelEl }); } catch (e) {}
            return;
        }
        if (open) {
            panelEl.classList.add('open');
            panelEl.style.display = 'flex';
            panelEl.setAttribute('aria-hidden', 'false');
            fabEl.setAttribute('aria-expanded', 'true');
            try {
                const ctxNow = getAnalysisContext();
                if (ctxPreviewEl) ctxPreviewEl.value = ctxNow || '';
                if (insertCtxFromTabBtn) insertCtxFromTabBtn.disabled = !(ctxNow || '').trim();
            } catch (e) {}
            setInnerTab('chat');
            try { inputEl && inputEl.focus(); } catch (e) {}
        } else {
            panelEl.classList.remove('open');
            panelEl.style.display = '';
            panelEl.setAttribute('aria-hidden', 'true');
            fabEl.setAttribute('aria-expanded', 'false');
        }
    };

    if (fabEl) {
        try { window.__FinanceBotMainBound = true; } catch (e) {}
        fabEl.addEventListener('click', () => {
            if (!panelEl) return;
            const open = panelEl.classList.contains('open');
            setPanelOpen(!open);
        });
    }
    if (closeEl) closeEl.addEventListener('click', () => setPanelOpen(false));

    if (tabChatBtn) tabChatBtn.addEventListener('click', () => setInnerTab('chat'));
    if (tabContextBtn) tabContextBtn.addEventListener('click', () => setInnerTab('context'));
    if (goToChatBtn) goToChatBtn.addEventListener('click', () => setInnerTab('chat'));

    // If the user changes the main page tab while on the context pane, refresh the preview.
    try {
        const mainTabBtns = document.querySelectorAll('.af-tabs .nav-link');
        mainTabBtns.forEach((btn) => {
            btn.addEventListener('click', () => {
                setTimeout(() => {
                    if (!ctxPaneEl || !ctxPaneEl.classList.contains('active')) return;
                    const ctxNow = getAnalysisContext();
                    if (ctxPreviewEl) ctxPreviewEl.value = ctxNow || '';
                    if (insertCtxFromTabBtn) insertCtxFromTabBtn.disabled = !(ctxNow || '').trim();
                }, 0);
            });
        });
    } catch (e) {}

    // Insert-context should work even if chat widgets are unavailable.
    if (insertCtxFromTabBtn) {
        insertCtxFromTabBtn.addEventListener('click', () => {
            const ctx = (getAnalysisContext() || '').trim();
            if (!ctx) return;
            if (!inputEl) return;
            const existing = (inputEl.value || '').trim();
            inputEl.value = existing ? (existing + "\n\n" + ctx) : ctx;
            setInnerTab('chat');
            try { inputEl.focus(); } catch (e) {}
        });
    }

    if (!logEl || !inputEl || !sendBtn) return;

    const history = []; // {role, content}
    let currentAbort = null;
    let currentTyping = null; // { row, bubble }

    const scrollToBottom = () => {
        try { logEl.scrollTop = logEl.scrollHeight; } catch (e) {}
    };

    const addMsg = (role, content, { typing = false } = {}) => {
        const row = document.createElement('div');
        row.className = `af-msg ${role}`;
        const bubble = document.createElement('div');
        bubble.className = 'af-bubble';

        if (typing) {
            bubble.textContent = content || '';
            const caret = document.createElement('span');
            caret.className = 'af-typing-caret';
            caret.setAttribute('aria-hidden', 'true');
            bubble.appendChild(caret);
            row.dataset.typing = '1';
        } else {
            bubble.textContent = content || '';
        }

        row.appendChild(bubble);
        logEl.appendChild(row);
        scrollToBottom();
        return { row, bubble };
    };

    const setSending = (sending) => {
        sendBtn.disabled = !!sending;
        inputEl.disabled = !!sending;
        if (stopEl) stopEl.disabled = !sending;
        if (resetEl) resetEl.disabled = false;
    };

    const stopCurrent = () => {
        try {
            if (currentAbort) currentAbort.abort();
        } catch (e) {}
        currentAbort = null;
        if (currentTyping) {
            try {
                stopTyping(currentTyping.row, currentTyping.bubble);
                const prev = (currentTyping.bubble.textContent || '').trim();
                if (!prev) currentTyping.bubble.textContent = '(stopped)';
            } catch (e) {}
        }
        currentTyping = null;
        setSending(false);
    };

    const resetChat = () => {
        stopCurrent();
        history.length = 0;
        try { logEl.innerHTML = ''; } catch (e) {}
        try { inputEl.value = ''; } catch (e) {}
        addMsg('assistant', "Ask a question about the current analysis and I’ll answer using the computed metrics.");
        try { inputEl.focus(); } catch (e) {}
    };

    if (stopEl) stopEl.addEventListener('click', stopCurrent);
    if (resetEl) resetEl.addEventListener('click', resetChat);

    const trimHistory = () => {
        while (history.length > 12) history.shift();
    };

    const updateTypingBubble = (bubble, text) => {
        const caret = bubble.querySelector('.af-typing-caret');
        bubble.textContent = text || '';
        if (caret) bubble.appendChild(caret);
        scrollToBottom();
    };

    const stopTyping = (row, bubble) => {
        if (!row) return;
        row.dataset.typing = '0';
        const caret = bubble.querySelector('.af-typing-caret');
        if (caret) caret.remove();
    };

    const parseSseLines = (buffer) => {
        const events = [];
        const parts = buffer.split(/\n\n/);
        const rest = parts.pop();
        for (const chunk of parts) {
            let eventName = 'message';
            let dataLines = [];
            for (const line of chunk.split(/\n/)) {
                if (line.startsWith('event:')) eventName = line.slice(6).trim();
                if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
            }
            events.push({ event: eventName, dataText: dataLines.join('\n') });
        }
        return { events, rest: rest || '' };
    };

    const send = async () => {
        const msg = (inputEl.value || '').trim();
        if (!msg) return;

        // If something is already running, stop it first.
        if (sendBtn.disabled) {
            stopCurrent();
        }

        if (!openaiReady) {
            addMsg('assistant', 'OpenAI is not configured. Set OPENAI_API_KEY_ENCRYPTED/OPENAI_API_KEY (and optionally OPENAI_PASSPHRASE) and reload.');
            return;
        }

        inputEl.value = '';
        addMsg('user', msg);
        history.push({ role: 'user', content: msg });
        trimHistory();

        const assistant = addMsg('assistant', '', { typing: true });
        currentTyping = assistant;
        let assistantText = '';
        setSending(true);

        try {
            const ctxNow = getAnalysisContext();
            currentAbort = (typeof AbortController !== 'undefined') ? new AbortController() : null;
            const resp = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: msg, context: ctxNow, history }),
                signal: currentAbort ? currentAbort.signal : undefined,
            });

            if (!resp.ok || !resp.body) {
                let errText = '';
                try { errText = await resp.text(); } catch (e) {}
                stopTyping(assistant.row, assistant.bubble);
                updateTypingBubble(assistant.bubble, `Error: ${errText || resp.statusText}`);
                history.push({ role: 'assistant', content: `Error: ${errText || resp.statusText}` });
                trimHistory();
                return;
            }

            const reader = resp.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const parsed = parseSseLines(buffer);
                buffer = parsed.rest;

                for (const ev of parsed.events) {
                    if (ev.event === 'error') {
                        let payload = null;
                        try { payload = JSON.parse(ev.dataText || '{}'); } catch (e) {}
                        const em = (payload && payload.error) ? payload.error : 'Unknown error';
                        assistantText += `\n[Error] ${em}`;
                        updateTypingBubble(assistant.bubble, assistantText);
                        stopTyping(assistant.row, assistant.bubble);
                    } else if (ev.event === 'done') {
                        stopTyping(assistant.row, assistant.bubble);
                    } else {
                        let payload = null;
                        try { payload = JSON.parse(ev.dataText || '{}'); } catch (e) {}
                        const delta = (payload && payload.delta) ? payload.delta : '';
                        if (delta) {
                            assistantText += delta;
                            updateTypingBubble(assistant.bubble, assistantText);
                        }
                    }
                }
            }

            stopTyping(assistant.row, assistant.bubble);
            history.push({ role: 'assistant', content: (assistantText || '').trim() || '(no output)' });
            trimHistory();
        } catch (e) {
            // Aborted requests should be treated as a stop, not an error.
            const aborted = (e && (e.name === 'AbortError' || String(e).includes('AbortError')));
            if (aborted) {
                stopTyping(assistant.row, assistant.bubble);
                if (!(assistant.bubble.textContent || '').trim()) updateTypingBubble(assistant.bubble, '(stopped)');
            } else {
                stopTyping(assistant.row, assistant.bubble);
                updateTypingBubble(assistant.bubble, `Error: ${e}`);
                history.push({ role: 'assistant', content: `Error: ${e}` });
                trimHistory();
            }
        } finally {
            currentAbort = null;
            currentTyping = null;
            setSending(false);
            try { inputEl.focus(); } catch (e) {}
        }
    };

    sendBtn.addEventListener('click', send);
    inputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            send();
            return;
        }
    });

    // (insertCtxFromTabBtn handler registered above)

    addMsg('assistant', "Ask a question about the current analysis and I’ll answer using the computed metrics.");
    } catch (e) {
        try { console.error('[FinanceBot] init failed', e); } catch (_) {}
    }
})();
</script>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""


_LAST_RESULT: Optional[Dict[str, Any]] = None


@app.route("/", methods=["GET"])
def home():
    return render_template_string(TEMPLATE, **_template_context(result=None, error=None))


@app.route("/upload", methods=["POST"])
def upload():
    global _LAST_RESULT

    files = request.files.getlist("pdf_files")
    files = [f for f in files if f and f.filename]
    if not files:
        return render_template_string(TEMPLATE, **_template_context(result=None, error="Please select at least one PDF file."))
    if any(not f.filename.lower().endswith(".pdf") for f in files):
        return render_template_string(TEMPLATE, **_template_context(result=None, error="Please upload PDF files only."))

    # Web UI uses heuristic-only comparison to avoid confusing LLM toggles.
    use_llm = False

    tmp_paths: List[str] = []
    display_names: List[str] = []
    try:
        for f in files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            f.save(tmp.name)
            tmp.close()
            tmp_paths.append(tmp.name)
            display_names.append(f.filename)

        result = analyze_pdfs(tmp_paths, use_llm_judge=use_llm)
        # Replace temp paths with original filenames for UI display.
        try:
            for i, r in enumerate(result.get("results") or []):
                if i < len(display_names):
                    r["pdf_path"] = display_names[i]
        except Exception:
            pass
        _LAST_RESULT = result
        return render_template_string(TEMPLATE, **_template_context(result=result, error=None))
    except Exception as e:
        return render_template_string(TEMPLATE, **_template_context(result=None, error=f"Error analyzing PDFs: {e}"))
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
    """Chat endpoint for optional OpenAI-powered Q&A.

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
        return jsonify({"error": "OpenAI not configured. Set OPENAI_API_KEY_ENCRYPTED/OPENAI_API_KEY or provide api_key in the request."}), 400

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


@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    """Streaming chat endpoint for the UI (SSE over fetch).

    Response is text/event-stream with payloads:
      data: {"delta": "..."}\n\n
      event: done\ndata: {}\n\n
      event: error\ndata: {"error": "..."}\n\n
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

    client = _init_openai_client()
    if client is None:
        return jsonify({"error": "OpenAI not configured. Set OPENAI_API_KEY_ENCRYPTED (with OPENAI_PASSPHRASE) or OPENAI_API_KEY."}), 400

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

    if len(context) > 16000:
        context = context[:15950] + "\n…(truncated)"

    messages = [{"role": "system", "content": sys}]
    if context:
        messages.append({"role": "user", "content": f"CONTEXT:\n{context}"})
    messages.extend(safe_history)
    messages.append({"role": "user", "content": message})

    def _extract_delta(chunk: Any) -> str:
        try:
            delta = chunk.choices[0].delta
            return (getattr(delta, "content", None) or "")
        except Exception:
            pass
        try:
            return (((chunk or {}).get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
        except Exception:
            return ""

    @stream_with_context
    def generate():
        yield "event: open\ndata: {}\n\n"
        try:
            stream = client.chat.completions.create(
                model=os.getenv("AGENTIC_FLOW_CHAT_MODEL", "gpt-4.1"),
                messages=messages,
                temperature=0.2,
                max_tokens=700,
                stream=True,
            )
            for chunk in stream:
                delta = _extract_delta(chunk)
                if delta:
                    yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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
        print("\nfinancebit")
        print(f"Running on http://{args.host}:{args.port}")
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
