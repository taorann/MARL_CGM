from __future__ import annotations

import json
from typing import Any


def info(message: str) -> None:
    print(message, flush=True)


def compact_json(value: Any, limit: int = 500) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...<truncated {len(text) - limit} chars>"


def summarize_action_result(result: dict[str, Any] | None) -> str:
    if not isinstance(result, dict):
        return ""
    if result.get("blocked"):
        return f"blocked: {result.get('reason', '')}"
    if result.get("error"):
        return f"error: {result.get('error')} {result.get('reason', '')}".strip()
    tool = str(result.get("tool") or "")
    if tool == "run_failed_test":
        test = result.get("test") if isinstance(result.get("test"), dict) else {}
        return f"test={test.get('status')} rc={test.get('returncode')} frames={len(test.get('implementation_frames') or [])}"
    if tool == "explore_find":
        return f"results={len(result.get('results') or [])}" + (f" warning={result.get('warning')}" if result.get("warning") else "")
    if tool == "grep_code":
        return f"hits={len(result.get('hits') or [])} path_glob={result.get('path_glob')}"
    if tool == "explore_expand":
        mode = result.get("expand_mode")
        symbol = result.get("symbol")
        suffix = f" mode={mode}" if mode else ""
        suffix += f" symbol={symbol}" if symbol else ""
        return f"results={len(result.get('results') or [])}{suffix}"
    if tool == "read":
        node = result.get("node") if isinstance(result.get("node"), dict) else {}
        return f"{node.get('kind')} {node.get('path')}:{(node.get('lines') or ['?','?'])[0]}-{(node.get('lines') or ['?','?'])[1]}"
    if tool == "memory_commit":
        changed = result.get("memory_changed")
        changed_text = "" if changed is None else f" changed={changed}"
        return f"committed={len(result.get('committed') or [])} memory={len(result.get('memory') or [])}{changed_text}"
    if tool == "memory_delete":
        changed = result.get("memory_changed")
        changed_text = "" if changed is None else f" changed={changed}"
        return f"deleted={len(result.get('deleted_ids') or [])} memory={len(result.get('memory') or [])}{changed_text}"
    if tool == "repair":
        touched = ",".join(result.get("touched_paths") or [])
        return f"status={result.get('status')} rolled_back={result.get('rolled_back')} paths={touched}"
    return compact_json(result, limit=300)


def summarize_action_status(result: dict[str, Any] | None) -> str:
    if not isinstance(result, dict):
        return "unknown"
    if result.get("blocked"):
        return "blocked"
    if result.get("error"):
        return "error"
    status = result.get("status")
    if isinstance(status, str) and status:
        return status
    test = result.get("test") if isinstance(result.get("test"), dict) else None
    if isinstance(test, dict) and test.get("status"):
        return str(test["status"])
    return "ok"


def summarize_cgm_response(response: Any) -> str:
    if not isinstance(response, dict):
        return compact_json(response, limit=400)
    patch = response.get("patch")
    if isinstance(patch, dict):
        edits = patch.get("edits")
        edit_count = len(edits) if isinstance(edits, list) else 0
        summary = str(patch.get("summary") or "").strip()
        return f"edits={edit_count}" + (f" summary={summary}" if summary else "")
    return compact_json(response, limit=400)
