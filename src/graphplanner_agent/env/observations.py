from __future__ import annotations

import json
import re

from graphplanner_agent.datasets import TaskSpec
from graphplanner_agent.env.repair_feedback import api_signature_failure_hint
from graphplanner_agent.memory.cgm_memory import CgmMemory
from graphplanner_agent.memory.text_notes import TextNotes
from graphplanner_agent.memory.working import WorkingMemory, code_status_for


def _runtime_facts(
    failure_summary: dict[str, object] | None,
    memory: CgmMemory,
    repair_feedback: str | None,
    latest_result: dict[str, object] | None,
    last_repair_attempt: dict[str, object] | None,
    last_repair_review: dict[str, object] | None,
    verified: bool,
) -> dict[str, object]:
    missing = [node.id for node in memory.nodes.values() if not node.has_code]
    status = latest_result.get("status") if latest_result else None
    api_hint = api_signature_failure_hint(latest_result, last_repair_attempt)
    return {
        "verified": verified,
        "done_after_verified_repair": verified,
        "last_action_blocked": bool(latest_result and latest_result.get("blocked")),
        "last_block_reason": latest_result.get("reason") if latest_result and latest_result.get("blocked") else None,
        "latest_patch_status": status if status in {"patch_rejected", "syntax_failed", "test_failed", "passed"} else None,
        "source_after_failed_patch": "rolled_back_or_unchanged" if status in {"patch_rejected", "syntax_failed", "test_failed"} else None,
        "last_repair_error_origin": last_repair_attempt.get("error_origin") if last_repair_attempt else None,
        "last_repair_review_verdict": (
            last_repair_review.get("review", {}).get("verdict")
            if isinstance(last_repair_review, dict) and isinstance(last_repair_review.get("review"), dict)
            else None
        ),
        "fail_to_pass_behavior_present": failure_summary is not None,
        "repair_memory_node_count": len(memory.nodes),
        "unhydrated_memory_node_ids": missing,
        "repair_feedback_present": bool(repair_feedback),
        "last_patch_api_signature_failure": api_hint,
    }


def _collect_unread_symbol_references(latest_result: dict[str, object] | None, *, limit: int = 10) -> list[dict[str, object]]:
    if not latest_result:
        return []
    references: list[dict[str, object]] = []
    seen: set[str] = set()

    def add_many(items) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("id") or "")
            if not node_id or node_id in seen:
                continue
            seen.add(node_id)
            references.append(item)
            if len(references) >= limit:
                return

    add_many(latest_result.get("unread_local_symbol_references"))
    results = latest_result.get("results")
    if isinstance(results, list):
        for result in results:
            if not isinstance(result, dict):
                continue
            add_many(result.get("unread_local_symbol_references"))
            if len(references) >= limit:
                break
    return references[:limit]


def _collect_dispatch_tables(latest_result: dict[str, object] | None, *, limit: int = 6) -> list[dict[str, object]]:
    if not latest_result:
        return []
    tables: list[dict[str, object]] = []

    def add_many(items) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            tables.append(item)
            if len(tables) >= limit:
                return

    add_many(latest_result.get("dispatch_tables"))
    results = latest_result.get("results")
    if isinstance(results, list):
        for result in results:
            if not isinstance(result, dict):
                continue
            add_many(result.get("dispatch_tables"))
            if len(tables) >= limit:
                break
    return tables[:limit]


def _truncate_middle(text: str, limit: int) -> tuple[str, dict[str, object] | None]:
    if len(text) <= limit:
        return text, None
    if limit < 40:
        return text[:limit], {"original_chars": len(text), "emitted_chars": limit, "omitted_chars": len(text) - limit}
    keep_head = max(1, limit // 2)
    keep_tail = max(1, limit - keep_head)
    marker = f"\n...<truncated {len(text) - limit} chars>...\n"
    truncated = text[:keep_head] + marker + text[-keep_tail:]
    return truncated, {"original_chars": len(text), "emitted_chars": len(truncated), "omitted_chars": len(text) - limit}


def _compact_behavior_summary(
    summary: dict[str, object] | None,
    *,
    field: str,
    command_limit: int = 900,
    excerpt_limit: int = 2200,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    report = {"field": field, "truncated": False, "truncated_items": []}
    if summary is None:
        return None, report
    compact: dict[str, object] = {
        "status": summary.get("status"),
        "returncode": summary.get("returncode"),
        "resolved": summary.get("resolved"),
        "tests_status": summary.get("tests_status"),
        "implementation_frames": summary.get("implementation_frames"),
        "runtime_observations": summary.get("runtime_observations"),
        "command_omitted_for_benchmark_hygiene": summary.get("command_omitted_for_benchmark_hygiene"),
        "parser_error": summary.get("parser_error"),
    }
    command = summary.get("command")
    if command is not None:
        compact_command, item = _truncate_middle(str(command), command_limit)
        compact["command"] = compact_command
        if item:
            item.update({"path": f"{field}.command", "reason": "command exceeded observation budget"})
            report["truncated_items"].append(item)
    excerpt = str(summary.get("excerpt") or "").strip()
    if excerpt:
        compact_excerpt, item = _truncate_middle(excerpt, excerpt_limit)
        compact["excerpt"] = compact_excerpt
        if item:
            item.update({"path": f"{field}.excerpt", "reason": "test output excerpt exceeded observation budget"})
            report["truncated_items"].append(item)
    report["truncated"] = bool(report["truncated_items"])
    return compact, report


def _compact_latest_result(result: dict[str, object] | None) -> tuple[dict[str, object] | None, dict[str, object]]:
    report = {"field": "latest_action_result", "truncated": False, "truncated_items": []}
    if result is None:
        return None, report
    tool = str(result.get("tool") or "")
    if tool == "repair":
        status = str(result.get("status") or "")
        compact = {
            "tool": tool,
            "status": status or result.get("status"),
            "blocked": result.get("blocked"),
            "reason": result.get("reason"),
            "rolled_back": result.get("rolled_back"),
            "done": result.get("done"),
            "touched_paths": result.get("touched_paths"),
            "summary": result.get("summary"),
            "error_origin": result.get("error_origin"),
            "source_tree_state": result.get("source_tree_state"),
        }
        if status in {"patch_rejected", "syntax_failed", "test_failed", "infra_bug"}:
            compact["failure_feedback"] = result.get("failure_feedback") or _repair_failure_feedback_from_result(result)
        else:
            if result.get("patch_preview"):
                compact["patch_preview"] = result.get("patch_preview")
            test_summary, test_report = _compact_behavior_summary(
                result.get("test_summary") if isinstance(result.get("test_summary"), dict) else None,
                field="latest_action_result.test_summary",
            )
            if test_summary:
                compact["test_summary"] = test_summary
            if test_report["truncated"]:
                report["truncated_items"].extend(test_report["truncated_items"])
    elif tool == "repair_review":
        compact = {
            "tool": tool,
            "status": result.get("status"),
            "blocked": result.get("blocked"),
            "reason": result.get("reason"),
            "review": result.get("review"),
            "note_to_planner": result.get("note_to_planner"),
            "cgm_payload": result.get("cgm_payload"),
            "cgm_response": result.get("cgm_response"),
            "error_origin": result.get("error_origin"),
        }
    elif tool == "run_failed_test":
        test, test_report = _compact_behavior_summary(
            result.get("test") if isinstance(result.get("test"), dict) else None,
            field="latest_action_result.test",
        )
        compact = {"tool": tool, "test": test}
        if test_report["truncated"]:
            report["truncated_items"].extend(test_report["truncated_items"])
    else:
        compact = result
    report["truncated"] = bool(report["truncated_items"])
    return compact, report


def _repair_failure_feedback_from_result(result: dict[str, object]) -> dict[str, object]:
    test_summary = result.get("test_summary") if isinstance(result.get("test_summary"), dict) else None
    return {
        "failed_patch": _compact_observation_value(result.get("patch_preview"), 2500),
        "failed_tests": _failed_test_selectors(test_summary),
        "error_summary": _repair_error_summary(result, test_summary),
    }


def _failed_test_selectors(test_summary: dict[str, object] | None) -> list[str]:
    if not isinstance(test_summary, dict):
        return []
    observations = test_summary.get("runtime_observations")
    if isinstance(observations, dict):
        selectors = observations.get("failed_selectors")
        if isinstance(selectors, list):
            return [str(item) for item in selectors if str(item).strip()]
    tests_status = test_summary.get("tests_status")
    out: list[str] = []
    if isinstance(tests_status, dict):
        for bucket in tests_status.values():
            if not isinstance(bucket, dict):
                continue
            failures = bucket.get("failure")
            if not isinstance(failures, list):
                continue
            for selector in failures:
                text = str(selector).strip()
                if text and text not in out:
                    out.append(text)
    return out


def _repair_error_summary(result: dict[str, object], test_summary: dict[str, object] | None) -> str:
    parts: list[str] = []
    for key in ["generated_patch_error_excerpt", "reason", "summary"]:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    if isinstance(test_summary, dict):
        observations = test_summary.get("runtime_observations")
        if isinstance(observations, dict):
            for key in ["exception_types", "actual_messages", "actual_assertion_values"]:
                values = observations.get(key)
                if isinstance(values, list) and values:
                    parts.append(f"{key}: " + "; ".join(str(item) for item in values[:4]))
        excerpt = str(test_summary.get("excerpt") or "").strip()
        if excerpt:
            parts.append(excerpt)
    if not parts:
        parts.append(str(result.get("error_origin") or "repair failed"))
    text, _ = _truncate_middle("\n".join(parts), 1800)
    return text


def _compact_observation_value(value: object, limit: int):
    if value is None:
        return None
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(value)
    if len(text) <= limit:
        return value
    compact, _ = _truncate_middle(text, limit)
    return compact


def _merge_truncation_reports(*reports: dict[str, object]) -> dict[str, object]:
    return {
        "truncated": any(bool(report.get("truncated")) for report in reports),
        "fields": list(reports),
    }


def _read_not_committed(working: WorkingMemory, memory: CgmMemory, *, limit: int = 12) -> list[dict[str, object]]:
    committed = set(memory.nodes)
    items = [
        entry
        for node_id, entry in working.entries.items()
        if node_id not in committed and entry.node.has_code and _is_explicit_read_source(entry.source)
    ]
    items.sort(key=lambda entry: entry.last_step, reverse=True)
    return [
        {
            "id": entry.node.id,
            "kind": _public_node_kind(entry.node.kind),
            "name": entry.node.name,
            "path": entry.node.path,
            "lines": [entry.node.start_line, entry.node.end_line],
            "source": entry.source,
            "code_status": code_status_for(entry.source, entry.node),
        }
        for entry in items[:limit]
    ]


def _working_code(
    working: WorkingMemory,
    memory: CgmMemory,
    *,
    affordance_symbols: list[str] | None = None,
    total_limit: int = 36000,
    per_node_limit: int = 9000,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    items = [entry for entry in working.entries.values() if entry.node.has_code]
    items.sort(key=lambda entry: entry.last_step)
    committed = set(memory.nodes)
    remaining = total_limit
    out: list[dict[str, object]] = []
    truncated_nodes: list[dict[str, object]] = []
    omitted_nodes: list[dict[str, object]] = []
    for idx, entry in enumerate(items):
        if remaining <= 0:
            for omitted in items[idx:]:
                omitted_nodes.append(
                    {
                        "id": omitted.node.id,
                        "name": omitted.node.name,
                        "path": omitted.node.path,
                        "lines": [omitted.node.start_line, omitted.node.end_line],
                        "reason": "working_code_W total character budget exhausted",
                    }
                )
            break
        node = entry.node
        full_code = _line_numbered_text(node.text or "", node.start_line)
        code = full_code
        truncated = False
        node_limit = min(per_node_limit, remaining)
        if len(code) > node_limit:
            omitted_chars = len(code) - node_limit
            code = code[:node_limit] + f"\n...<truncated {omitted_chars} chars>"
            truncated = True
            truncated_nodes.append(
                {
                    "id": node.id,
                    "name": node.name,
                    "path": node.path,
                    "lines": [node.start_line, node.end_line],
                    "emitted_chars": node_limit,
                    "omitted_chars": omitted_chars,
                    "reason": "per-node or remaining total character budget",
                }
            )
        remaining -= len(code)
        out.append(
            {
                "id": node.id,
                "kind": _public_node_kind(node.kind),
                "name": node.name,
                "path": node.path,
                "lines": [node.start_line, node.end_line],
                "source": entry.source,
                "code_status": code_status_for(entry.source, node),
                "evidence_status": _node_evidence_status(entry.source, node, node.id in committed),
                "last_step": entry.last_step,
                "in_repair_memory_M": node.id in committed,
                "available_expansions": _available_expansions(node, affordance_symbols or []),
                "code": code,
                "truncated": truncated,
                "full_code_chars": len(full_code),
                "emitted_code_chars": len(code),
            }
        )
    report = {
        "field": "working_code_W",
        "truncated": bool(truncated_nodes or omitted_nodes),
        "total_limit_chars": total_limit,
        "per_node_limit_chars": per_node_limit,
        "source_node_count": len(items),
        "emitted_node_count": len(out),
        "omitted_node_count": len(omitted_nodes),
        "remaining_chars": max(remaining, 0),
        "truncated_nodes": truncated_nodes,
        "omitted_nodes": omitted_nodes,
    }
    return out, report


def _node_evidence_status(source: str, node, committed: bool) -> str:
    if committed:
        return "committed_to_repair_memory_M"
    status = code_status_for(source, node)
    if status in {"read", "hydrated"}:
        return "read_not_committed"
    if status == "preview":
        return "orientation_preview_read_before_commit"
    return "working_context"


def _available_expansions(node, symbols: list[str]) -> list[dict[str, object]]:
    if _public_node_kind(node.kind) not in {"class", "function", "method"}:
        return []
    expansions: list[dict[str, object]] = [
        {
            "label": "mechanism",
            "action": {"tool": "explore_expand", "params": {"anchor": node.id, "expand_mode": "mechanism"}},
            "relations_expected": ["parent/base", "override", "composition", "pipeline"],
        }
    ]
    for symbol in symbols[:3]:
        expansions.append(
            {
                "label": f"owner_flow:{symbol}",
                "action": {
                    "tool": "explore_expand",
                    "params": {"anchor": node.id, "expand_mode": "owner_flow", "symbol": symbol},
                },
                "relations_expected": ["attribute_owner", "symbol_consumer"],
            }
        )
    return expansions


_AFFORDANCE_STOPWORDS = {
    "assert",
    "body",
    "class",
    "code",
    "column",
    "columns",
    "data",
    "else",
    "error",
    "false",
    "format",
    "from",
    "function",
    "html",
    "import",
    "issue",
    "lambda",
    "line",
    "none",
    "object",
    "output",
    "pass",
    "path",
    "python",
    "return",
    "self",
    "status",
    "table",
    "test",
    "text",
    "true",
    "value",
    "values",
    "with",
}


def _affordance_symbols(
    task: TaskSpec,
    failure_summary: dict[str, object] | None,
    latest_result: dict[str, object] | None,
    last_repair_attempt: dict[str, object] | None,
    last_repair_review: dict[str, object] | None,
    *,
    limit: int = 5,
) -> list[str]:
    chunks = [
        task.issue_title or "",
        task.issue_body or "",
        _json_for_symbol_scan(failure_summary),
        _json_for_symbol_scan(latest_result),
        _json_for_symbol_scan(last_repair_attempt),
        _json_for_symbol_scan(last_repair_review),
    ]
    text = "\n".join(chunk for chunk in chunks if chunk)
    quoted = re.findall(r"[`'\"]([A-Za-z_][A-Za-z0-9_]*)[`'\"]", text)
    identifiers = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", text)
    ordered: list[str] = []
    for name in [*quoted, *identifiers]:
        key = name.strip()
        if not key or key.lower() in _AFFORDANCE_STOPWORDS:
            continue
        if len(key) < 3:
            continue
        if key.startswith("_") and len(key) < 4:
            continue
        if key not in ordered:
            ordered.append(key)
        if len(ordered) >= limit:
            break
    return ordered


def _json_for_symbol_scan(value: object) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _line_numbered_text(text: str, start_line: int) -> str:
    return "\n".join(f"{idx:4d}: {line}" for idx, line in enumerate(text.splitlines(), start_line))


def _is_explicit_read_source(source: str) -> bool:
    return str(source or "").startswith(("read:", "hydrated_for_memory"))


def _evidence_status(
    failure_summary: dict[str, object] | None,
    memory: CgmMemory,
    latest_result: dict[str, object] | None,
    unread_references: list[dict[str, object]],
    read_not_committed: list[dict[str, object]],
    repair_feedback: str | None,
    last_repair_attempt: dict[str, object] | None,
    last_repair_review: dict[str, object] | None,
    working_code: list[dict[str, object]],
    input_truncation_report: dict[str, object],
) -> dict[str, object]:
    hydrated = [node.id for node in memory.nodes.values() if node.has_code]
    missing = [node.id for node in memory.nodes.values() if not node.has_code]
    return {
        "fail_to_pass_behavior_present": failure_summary is not None,
        "memory_node_count": len(memory.nodes),
        "hydrated_memory_node_ids": hydrated,
        "unhydrated_memory_node_ids": missing,
        "unread_symbol_reference_count": len(unread_references),
        "read_not_committed_count": len(read_not_committed),
        "working_code_node_count": len(working_code),
        "input_truncated": bool(input_truncation_report.get("truncated")),
        "input_truncation_report": input_truncation_report if input_truncation_report.get("truncated") else None,
        "repair_feedback_present": bool(repair_feedback),
        "latest_patch_status": latest_result.get("status") if latest_result else None,
        "last_repair_attempt_status": last_repair_attempt.get("status") if last_repair_attempt else None,
        "last_repair_error_origin": last_repair_attempt.get("error_origin") if last_repair_attempt else None,
        "last_repair_review_verdict": (
            last_repair_review.get("review", {}).get("verdict")
            if isinstance(last_repair_review, dict) and isinstance(last_repair_review.get("review"), dict)
            else None
        ),
        "repair_requires_evidence_package": True,
        "repair_package_required_fields": [
            "failure_seen",
            "evidence_chain",
            "target_nodes",
            "intent_analysis",
            "confidence",
        ],
    }


def _repair_failure_followup(
    latest_result: dict[str, object] | None,
    last_repair_attempt: dict[str, object] | None,
) -> dict[str, object] | None:
    if not latest_result or latest_result.get("tool") != "repair":
        return None
    status = str(latest_result.get("status") or "")
    if status not in {"test_failed", "syntax_failed", "patch_rejected"}:
        return None

    last_origin = last_repair_attempt.get("error_origin") if last_repair_attempt else None
    error_origin = str(latest_result.get("error_origin") or last_origin or "")
    generic = [
        "Do not repeat the same repair intent after a failed patch.",
        "First use last_repair_attempt.failure_feedback.failed_patch to explain why the previous patch did not resolve the runtime behavior.",
        "Then read/commit new implementation evidence that can distinguish the old intent_analysis from the remaining failure.",
        "The next repair must change evidence_chain, target_nodes, intent_analysis, or confidence.",
    ]
    if status == "test_failed":
        focus = (
            "The patch applied and rolled back after tests failed, so the syntax/range was usable but the behavior was still wrong. "
            "Inspect last_repair_attempt.failure_feedback, then look for an unhandled consumer, output builder, "
            "state propagation step, parent/base behavior, or sibling implementation."
        )
    elif error_origin == "duplicate_patch":
        focus = (
            "The generated edits duplicate a patch already tried or rejected. This is a hard signal that the current "
            "intent_analysis or target evidence is stale; change the evidence_chain or target before repair."
        )
    elif status == "syntax_failed":
        focus = (
            "The generated patch was syntactically invalid and rolled back. The original source is not broken. "
            "A format-only retry is acceptable only if the evidence_chain is still sound; otherwise read more code first."
        )
    else:
        focus = (
            "The patch was rejected before behavior testing. If the rejection was not purely patch formatting, collect new "
            "code evidence before trying another repair."
        )

    return {
        "status": status,
        "error_origin": error_origin or None,
        "instruction": "deepen_before_next_repair",
        "focus": focus,
        "required_before_next_repair": generic,
    }


def _current_turn_protocol(
    failure_summary: dict[str, object] | None,
    memory: CgmMemory,
    read_not_committed: list[dict[str, object]],
    input_truncation_report: dict[str, object],
    latest_result: dict[str, object] | None,
    last_repair_attempt: dict[str, object] | None,
    last_repair_review: dict[str, object] | None,
    repair_disabled_reason: str | None,
) -> dict[str, object]:
    hydrated = [node.id for node in memory.nodes.values() if node.has_code]
    unhydrated = [node.id for node in memory.nodes.values() if not node.has_code]
    candidate_ids = [str(item["id"]) for item in read_not_committed if item.get("id")]
    blockers: list[str] = []
    valid_next_actions: list[str] = []
    invalid_next_actions: list[str] = []

    if failure_summary is None:
        blockers.append("repair is blocked until fail-to-pass runtime behavior is collected")
        valid_next_actions.append("run_failed_test to collect behavior evidence")
    if not hydrated:
        if candidate_ids:
            blockers.append("repair_memory_M has no hydrated code; CGM repair sees M, not W")
            valid_next_actions.append("memory_commit one or more already-read code nodes if they belong in the evidence_chain")
        else:
            blockers.append("repair_memory_M has no hydrated code and no read code is available to commit")
            valid_next_actions.append("explore_find, grep_code with path_glob, or explore_expand to locate implementation nodes, then read them")
    if unhydrated:
        blockers.append("repair_memory_M contains unhydrated nodes without code bodies")
        valid_next_actions.append("read or re-commit unhydrated memory nodes so CGM receives real code")
    if input_truncation_report.get("truncated"):
        blockers.append("some observation fields were truncated; omitted code is uncertainty, not evidence")
    if latest_result and latest_result.get("blocked"):
        blockers.append(f"latest action was blocked: {latest_result.get('reason')}")
        suggested = latest_result.get("suggested_next_actions")
        if isinstance(suggested, list):
            valid_next_actions.extend(str(item) for item in suggested if str(item).strip())
        reason = str(latest_result.get("reason") or "").lower()
        if "zero results" in reason or (latest_result.get("tool") == "explore_find" and not latest_result.get("results")):
            valid_next_actions.append(
                "if a search returned zero results, change route: use grep_code with a known path_glob, search parameter/helper names, expand from a related node, or read a file/sibling symbol instead of repeating the same query"
            )
    repair_failure_followup = _repair_failure_followup(latest_result, last_repair_attempt)
    if repair_failure_followup:
        blockers.append(str(repair_failure_followup["focus"]))
        valid_next_actions.extend(str(item) for item in repair_failure_followup["required_before_next_repair"])
        valid_next_actions.append(
            "repair_review can ask CGM to critique the current intent/evidence without applying a patch; use it when M is plausible but the failed patch suggests target or mechanism uncertainty"
        )
        status = str(repair_failure_followup.get("status") or "")
        error_origin = str(repair_failure_followup.get("error_origin") or "")
        if status == "test_failed":
            valid_next_actions.append(
                "search/read the downstream code that consumes the changed value or renders the remaining output; use grep_code with a scoped path_glob for broad mechanism terms; do not assume the first TypeError/raise site is the whole bug"
            )
            valid_next_actions.append(
                "if the failed patch hit a missing attribute or wrong object owner, use explore_expand with expand_mode=owner_flow and symbol=<attribute>, or expand_mode=mechanism from the patch-site class/method, before another local search"
            )
        if error_origin == "duplicate_patch":
            valid_next_actions.append(
                "because the patch was a duplicate, do not call repair until M or evidence_chain/intent_analysis changes to cover a new mechanism"
            )
    api_hint = api_signature_failure_hint(latest_result, last_repair_attempt)
    if api_hint:
        symbol = str(api_hint.get("api_symbol") or "").strip()
        blockers.append(
            "previous patch failed due to an unverified API/signature: "
            f"{api_hint.get('error_excerpt')}. Read code proving {symbol} before another repair."
        )
        valid_next_actions.append(
            f"explore_find for symbol:{symbol}, grep_code for {symbol} with a scoped path_glob, or expand from the caller, then read the exact definition/signature or an existing implementation call site"
        )
        valid_next_actions.append(
            f"if {symbol} is an attribute/parameter owner problem, use explore_expand owner_flow with symbol={symbol} from the failed patch-site class/method"
        )
        valid_next_actions.append(
            f"include the read {symbol} evidence in evidence_chain, and commit it to M if CGM must call or modify that API"
        )
        invalid_next_actions.append("repair before reading the API/signature named in last_repair_attempt")
    if latest_result and latest_result.get("tool") == "memory_commit" and latest_result.get("memory_changed") is False:
        blockers.append("latest memory_commit did not change repair_memory_M")
        valid_next_actions.append(
            "do not repeat the same memory_commit; repair with the current curated M, commit a different causal node, delete stale memory, or explore/read new evidence"
        )
    if repair_disabled_reason:
        blockers.append(repair_disabled_reason)
        invalid_next_actions.append("repair")
        if failure_summary is None:
            valid_next_actions.append("run_failed_test; repair is not an available tool yet")
        elif candidate_ids:
            valid_next_actions.append("if repair still lacks evidence, explicitly memory_commit only nodes needed by target_nodes/evidence_chain")
        elif latest_result and latest_result.get("tool") == "explore_find" and latest_result.get("results"):
            valid_next_actions.append("read one node id from latest_action_result.results before searching again")
        else:
            valid_next_actions.append("explore_find/grep_code/explore_expand/read to collect new implementation evidence before the next repair")

    if candidate_ids and hydrated:
        valid_next_actions.append("W may contain extra context; commit only nodes that prove target_nodes/evidence_chain or use memory_delete to remove stale M nodes")
    if hydrated and failure_summary is not None and not unhydrated:
        valid_next_actions.append(
            "repair is allowed only after target_nodes are committed in M, evidence_chain uses read code ids, target_nodes appear in evidence_chain, "
            "intent_analysis explains the local mechanism, and confidence is numeric"
        )
        valid_next_actions.append(
            "repair_review is allowed with the same structured evidence package when you want CGM to critique intent/target before generating a patch"
        )
    if not hydrated or failure_summary is None or unhydrated:
        invalid_next_actions.append("repair")
        invalid_next_actions.append("repair_review")

    if latest_result and latest_result.get("tool") == "repair_review":
        review = latest_result.get("review") if isinstance(latest_result.get("review"), dict) else {}
        verdict = str(review.get("verdict") or "")
        evidence_gaps = review.get("evidence_gaps") or review.get("missing_evidence")
        has_evidence_gaps = isinstance(evidence_gaps, list) and any(str(item).strip() for item in evidence_gaps)
        if verdict in {"needs_more_evidence", "change_target", "avoid_patch"}:
            blockers.append(
                "latest repair_review did not endorse immediate patching; decide whether to adopt, revise, or reject its evidence_gaps/target_assessment/suggested_next_action before repair"
            )
            valid_next_actions.append("validate or falsify latest_action_result.review using implementation code; then update M/evidence_chain, repair, or run another repair_review")
        elif verdict == "ready" and has_evidence_gaps:
            valid_next_actions.append(
                "latest repair_review is ready but lists evidence_gaps; decide whether those gaps are essential. "
                "If target/mechanism are already supported by visible code, repair may adopt the review; otherwise validate/falsify the gaps first."
            )
        elif verdict == "ready":
            valid_next_actions.append(
                "CGM repair_review says ready; decide whether to accept its critique. If accepted, call repair with the same target_nodes/evidence_chain. "
                "If you disagree, revise evidence/intent or call repair_review again with review_focus."
            )

    if last_repair_review and isinstance(last_repair_review.get("review"), dict):
        review = last_repair_review["review"]
        verdict = str(review.get("verdict") or "")
        evidence_gaps = review.get("evidence_gaps") or review.get("missing_evidence")
        has_evidence_gaps = isinstance(evidence_gaps, list) and any(str(item).strip() for item in evidence_gaps)
        if verdict in {"needs_more_evidence", "change_target", "avoid_patch"}:
            blockers.append(
                f"last repair_review verdict={verdict}; repair with the same M/target/evidence package is blocked until the critique is adopted with changed evidence, falsified by code evidence, or another repair_review returns ready"
            )
            valid_next_actions.append(
                "validate or falsify last_repair_review.evidence_gaps using implementation code/runtime output only, then update M/evidence_chain or request another repair_review"
            )
            invalid_next_actions.append("repair with unchanged M/target/evidence after non-ready repair_review")
        elif verdict == "ready" and has_evidence_gaps:
            valid_next_actions.append(
                "repair using the reviewed evidence package if the gaps are non-essential, or validate/falsify evidence_gaps before repair if they affect target/mechanism"
            )
        elif verdict == "ready":
            valid_next_actions.append(
                "repair using the same evidence package if you adopt the critique, or repair_review with review_focus to ask CGM for deeper analysis"
            )

    if not blockers:
        blockers.append("no hard blocker detected, but repair still requires a structured evidence package grounded in committed M nodes")

    return {
        "observation_priority": "read this first; it states which planner actions are currently valid",
        "w_m_rule": "working_code_W includes read code and explore_find previews; repair_memory_M is model-curated CGM evidence; memory_commit never auto-adds related nodes and requires explicit read evidence",
        "repair_blocked_now": bool(invalid_next_actions),
        "repair_temporarily_disabled": bool(repair_disabled_reason),
        "repair_disabled_reason": repair_disabled_reason,
        "blockers": blockers,
        "valid_next_actions": valid_next_actions,
        "invalid_next_actions": invalid_next_actions,
        "candidate_memory_commit_ids": candidate_ids,
        "candidate_memory_commit_nodes": read_not_committed,
        "committed_hydrated_memory_ids": hydrated,
        "committed_unhydrated_memory_ids": unhydrated,
        "repair_failure_followup": repair_failure_followup,
        "repair_mechanism_requirements": {
            "evidence_chain": "observed runtime behavior -> implementation entry/state/decision/output -> patch target; unsupported links mean explore/read/commit, not repair",
            "failure_seen": "actual issue/runtime failure only",
            "target_nodes": "committed M nodes CGM should treat as the patch locus; each target must appear in evidence_chain",
            "intent_analysis": "short advisory mechanism analysis, not exact patch text",
            "confidence": "planner self-score from 0 to 1; lower it when localization or behavior details are uncertain",
            "repair_review": "same evidence package as repair, but CGM returns critique only; it does not apply or test a patch",
            "review_adoption": (
                "repair_review returns critique/adoption_advice, not a binding contract. Decide whether to adopt, revise, or reject it using visible code evidence."
            ),
            "memory_curation_rule": "commit only nodes you intentionally want CGM to use; use memory_delete for stale or noisy M nodes",
            "fallback_after_blocked_repair": (
                "do not repeat the same repair; close the blocker by searching alternate terms, reading caller/consumer/parent/sibling code, "
                "using explore_expand mechanism/owner_flow for base-class, composed-object, or owner-flow context, or committing the already code-bearing W node that proves the claim"
            ),
            "fallback_after_failed_patch": (
                "a test_failed, syntax_failed, patch_rejected, or duplicate_patch result must change the next intent_analysis or evidence_chain. "
                "Use last_repair_attempt.failure_feedback.failed_patch, failed_tests, and error_summary to identify what the patch failed to cover; "
                "then read or commit new mechanism evidence before another repair. Missing-attribute failures should trigger owner_flow/mechanism expand, not another local grep for the same missing name."
            ),
        },
    }


def _json_block(value: object) -> str:
    if value is None:
        return "null"
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _format_node_line(item: dict[str, object]) -> str:
    lines = item.get("lines")
    if isinstance(lines, list) and len(lines) == 2:
        line_text = f":{lines[0]}-{lines[1]}"
    else:
        line_text = ""
    return f"- {item.get('id')} ({item.get('kind')}) {item.get('name')} at {item.get('path')}{line_text}"


def _render_text_observation(state: dict[str, object]) -> str:
    protocol = state["current_turn_protocol"] if isinstance(state.get("current_turn_protocol"), dict) else {}
    issue = state["issue"] if isinstance(state.get("issue"), dict) else {}
    lines: list[str] = []

    lines.append("CURRENT TURN PROTOCOL")
    lines.append(str(protocol.get("observation_priority") or "read this first"))
    lines.append(f"W/M rule: {protocol.get('w_m_rule')}")
    lines.append(f"Repair blocked now: {protocol.get('repair_blocked_now')}")
    lines.append("")
    lines.append("Current blockers:")
    for item in protocol.get("blockers") or []:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Valid next actions:")
    for item in protocol.get("valid_next_actions") or []:
        lines.append(f"- {item}")
    invalid = protocol.get("invalid_next_actions") or []
    if invalid:
        lines.append("")
        lines.append("Invalid next actions right now:")
        for item in invalid:
            lines.append(f"- {item}")
    candidates = protocol.get("candidate_memory_commit_nodes") or []
    if candidates:
        lines.append("")
        lines.append("Read but not committed to repair_memory_M:")
        for item in candidates:
            if isinstance(item, dict):
                lines.append(_format_node_line(item))
    requirements = protocol.get("repair_mechanism_requirements")
    if isinstance(requirements, dict):
        lines.append("")
        lines.append("Repair mechanism requirements:")
        for key, value in requirements.items():
            lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("ISSUE")
    lines.append(f"Title: {issue.get('title')}")
    body = str(issue.get("body") or "").strip()
    lines.append("Body:")
    lines.append(body or "<empty>")

    lines.append("")
    lines.append("FAILED TEST RUNTIME SUMMARY")
    lines.append(_json_block(state.get("test_behavior")))

    lines.append("")
    lines.append("LATEST ACTION RESULT")
    lines.append(_json_block(state.get("latest_action_result")))

    lines.append("")
    lines.append("REPAIR MEMORY M")
    lines.append(_json_block(state.get("repair_memory_M")))

    lines.append("")
    lines.append("UNREAD IMPLEMENTATION REFERENCES")
    lines.append(_json_block(state.get("unread_symbol_references")))

    lines.append("")
    lines.append("WORKING CODE W")
    working_code = state.get("working_code_W")
    if isinstance(working_code, list) and working_code:
        for item in working_code:
            if not isinstance(item, dict):
                continue
            lines.append("")
            lines.append(f"Node: {item.get('id')}")
            lines.append(f"Kind/name: {item.get('kind')} {item.get('name')}")
            lines.append(f"Path/lines: {item.get('path')} {item.get('lines')}")
            lines.append(f"Source/status: {item.get('source')} / {item.get('code_status')} / {item.get('evidence_status')}")
            lines.append(f"In repair_memory_M: {item.get('in_repair_memory_M')}")
            expansions = item.get("available_expansions")
            if isinstance(expansions, list) and expansions:
                lines.append("Available expansions:")
                lines.append(_json_block(expansions))
            lines.append("Code:")
            lines.append("```text")
            lines.append(str(item.get("code") or ""))
            lines.append("```")
    else:
        lines.append("<no code has been read yet>")

    lines.append("")
    lines.append("TRAJECTORY SUMMARY")
    lines.append(_json_block(state.get("trajectory_summary")))

    lines.append("")
    lines.append("COMPACT STRUCTURED STATE")
    compact = {
        "task": state.get("task"),
        "graph_summary": state.get("graph_summary"),
        "retrieval_scope": state.get("retrieval_scope"),
        "working_subgraph_W": state.get("working_subgraph_W"),
        "working_vs_memory": state.get("working_vs_memory"),
        "evidence_status": state.get("evidence_status"),
        "input_truncation_report": state.get("input_truncation_report"),
        "last_repair_attempt": state.get("last_repair_attempt"),
        "last_repair_review": state.get("last_repair_review"),
        "planner_diagnostics": state.get("planner_diagnostics"),
        "recent_actions": state.get("recent_actions"),
        "recent_action_signatures": state.get("recent_action_signatures"),
        "runtime_facts": state.get("runtime_facts"),
        "verified": state.get("verified"),
    }
    lines.append(_json_block(compact))
    return "\n".join(lines)


def build_observation(
    task: TaskSpec,
    working: WorkingMemory,
    memory: CgmMemory,
    notes: TextNotes,
    latest_result: dict[str, object] | None,
    failure_summary: dict[str, object] | None,
    repair_feedback: str | None,
    last_repair_attempt: dict[str, object] | None,
    last_repair_review: dict[str, object] | None,
    trajectory: list[dict[str, object]],
    planner_diagnostics: list[dict[str, object]],
    recent_actions: list[str],
    recent_action_signatures: list[str],
    graph_node_count: int,
    graph_edge_count: int,
    sandbox_backend: str,
    verified: bool,
    repair_disabled_reason: str | None = None,
    observation_mode: str = "json",
) -> str:
    unread_references = _collect_unread_symbol_references(latest_result)
    dispatch_tables = _collect_dispatch_tables(latest_result)
    read_not_committed = _read_not_committed(working, memory)
    affordance_symbols = _affordance_symbols(task, failure_summary, latest_result, last_repair_attempt, last_repair_review)
    working_code, working_code_report = _working_code(working, memory, affordance_symbols=affordance_symbols)
    compact_failure_summary, failure_report = _compact_behavior_summary(failure_summary, field="test_behavior")
    compact_latest_result, latest_report = _compact_latest_result(latest_result)
    input_truncation_report = _merge_truncation_reports(working_code_report, failure_report, latest_report)
    current_turn_protocol = _current_turn_protocol(
        failure_summary,
        memory,
        read_not_committed,
        input_truncation_report,
        compact_latest_result,
        last_repair_attempt,
        last_repair_review,
        repair_disabled_reason,
    )
    state = {
        "current_turn_protocol": current_turn_protocol,
        "task": {
            "task_id": task.task_id,
            "base_commit": task.base_commit,
            "sandbox_backend": sandbox_backend,
            "docker_or_sif_image": task.docker_image,
            "repo_path": str(task.repo_path),
        },
        "issue": {"title": task.issue_title, "body": task.issue_body},
        "graph_summary": {"nodes": graph_node_count, "edges": graph_edge_count},
        "retrieval_scope": {
            "indexed_code": "implementation code only; benchmark test paths are excluded from search, expand, read, and memory_commit",
            "test_source_visibility": "benchmark test source is not exposed as repair evidence",
            "test_output_visibility": "test output, traceback symptoms, selectors, and pass/fail status may appear as behavior evidence",
        },
        "test_behavior": compact_failure_summary,
        "working_subgraph_W": working.summary(),
        "repair_memory_M": memory.summary(),
        "working_vs_memory": {
            "repair_uses_only_M": True,
            "read_not_committed_to_M": read_not_committed,
        },
        "working_code_W": working_code,
        "code_node_affordance_symbols": affordance_symbols,
        "input_truncation_report": input_truncation_report,
        "text_notes_T": notes.summary(),
        "latest_action_result": compact_latest_result,
        "dispatch_tables": dispatch_tables,
        "unread_symbol_references": unread_references,
        "evidence_status": _evidence_status(
            failure_summary,
            memory,
            latest_result,
            unread_references,
            read_not_committed,
            repair_feedback,
            last_repair_attempt,
            last_repair_review,
            working_code,
            input_truncation_report,
        ),
        "last_repair_attempt": last_repair_attempt,
        "last_repair_review": last_repair_review,
        "trajectory_summary": trajectory,
        "planner_diagnostics": planner_diagnostics,
        "verified": verified,
        "recent_actions": recent_actions[-8:],
        "recent_action_signatures": recent_action_signatures[-5:],
        "runtime_facts": _runtime_facts(failure_summary, memory, repair_feedback, latest_result, last_repair_attempt, last_repair_review, verified),
    }
    if str(observation_mode or "json").strip().lower() == "text":
        return _render_text_observation(state)
    return json.dumps(state, indent=2, sort_keys=True)


def _public_node_kind(kind: str) -> str:
    return "assignment" if kind in {"assignment", "module_assignment"} else kind
