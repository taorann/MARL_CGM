from __future__ import annotations

import re
from typing import Any


_API_PATTERNS = [
    re.compile(
        r"(?P<symbol>[A-Za-z_][\w.]*)\(\) takes (?P<expected>\d+) positional arguments? but (?P<actual>\d+) were given",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<symbol>[A-Za-z_][\w.]*)\(\) got an unexpected keyword argument ['\"](?P<argument>[^'\"]+)['\"]",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<symbol>[A-Za-z_][\w.]*)\(\) missing (?P<count>\d+) required positional arguments?: (?P<arguments>[^\n]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"['\"](?P<object>[^'\"]+)['\"] object has no attribute ['\"](?P<symbol>[A-Za-z_]\w*)['\"]",
        re.IGNORECASE,
    ),
]


def api_signature_failure_hint(*results: dict[str, Any] | None) -> dict[str, str] | None:
    """Extract a compact API/signature lesson from failed patch feedback."""

    for result in results:
        feedback = _failure_feedback(result)
        if not feedback:
            continue
        error_summary = str(feedback.get("error_summary") or "")
        if not error_summary.strip():
            continue
        for pattern in _API_PATTERNS:
            match = pattern.search(error_summary)
            if not match:
                continue
            symbol = str(match.groupdict().get("symbol") or "").strip()
            if not symbol:
                continue
            return {
                "api_symbol": symbol,
                "error_excerpt": _one_line(match.group(0)),
                "required_evidence": (
                    f"Read implementation code proving the signature/usage of {symbol} "
                    "or an existing sibling call site before another repair."
                ),
            }
    return None


def evidence_mentions_api(params: dict[str, Any], symbol: str) -> bool:
    needle = str(symbol or "").strip().lower()
    if not needle:
        return False
    evidence_chain = params.get("evidence_chain")
    if not isinstance(evidence_chain, list):
        return False
    for item in evidence_chain:
        if not isinstance(item, dict):
            continue
        haystack = " ".join(
            str(item.get(key) or "")
            for key in ("node_id", "role", "evidence")
        ).lower()
        if needle in haystack:
            return True
    return False


def _failure_feedback(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    feedback = result.get("failure_feedback")
    if isinstance(feedback, dict):
        return feedback
    return None


def _one_line(text: str, *, limit: int = 220) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 14] + "...<truncated>"
