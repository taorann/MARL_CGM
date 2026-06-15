from __future__ import annotations

from dataclasses import asdict, replace
import json

from graphplanner_agent.graph.expand import MECHANISM_MODES, expand, expand_with_context
from graphplanner_agent.graph.file_discovery import discover_implementation_files, synthetic_file_node
from graphplanner_agent.graph.grep import grep_code
from graphplanner_agent.graph.guards import is_test_path
from graphplanner_agent.graph.read import line_numbered
from graphplanner_agent.graph.schema import RepoGraph
from graphplanner_agent.graph.search import SearchResult, search_graph
from graphplanner_agent.env.evidence import (
    dispatch_relationship_context,
    dispatch_tables,
    is_small_grain_node,
    local_symbol_references,
    node_brief,
    normalize_find_type,
    preview_node_code,
    public_node_kind,
    read_node_for_evidence,
    top_symbols_for_file,
    value_flow_context,
)
from graphplanner_agent.env.repair_feedback import api_signature_failure_hint, evidence_mentions_api
from graphplanner_agent.memory.hydration import hydrate_node_from_runtime
from graphplanner_agent.planner.protocol import PlannerAction
from graphplanner_agent.repair.cgm_context import build_cgm_payload, new_file_target_path, summarize_cgm_payload, validate_cgm_payload
from graphplanner_agent.repair.cgm_client import CgmUnavailableError
from graphplanner_agent.repair.patch_apply import apply_patch, normalize_patch_with_runtime, validate_patch_with_runtime
from graphplanner_agent.repair.patch_quality import syntax_check_python
from graphplanner_agent.repair.patch_schema import parse_cgm_output
from graphplanner_agent.repair.patch_schema import Patch
from graphplanner_agent.runtime.test_runner import behavior_summary


def handle_action(env, action: PlannerAction) -> dict[str, object]:
    if action.tool == "run_failed_test":
        result = env.runtime.run_fail_to_pass(env.task)
        env.failure_summary = behavior_summary(result)
        return {"tool": action.tool, "test": env.failure_summary}

    if action.tool == "explore_find":
        query = str(action.params.get("query", ""))
        find_type = normalize_find_type(str(action.params.get("find_type", "any")))
        class_name = action.params.get("class_name")
        path_glob = action.params.get("path_glob")
        results, warning = search_graph(
            env.graph,
            query,
            find_type,
            class_name,
            root=env.runtime.root,
            path_glob=str(path_glob) if path_glob else None,
        )
        if not results:
            discovered = discover_implementation_files(
                env.runtime,
                query=query,
                path_glob=str(path_glob) if path_glob else "",
                limit=12,
            )
            if discovered:
                results = [
                    SearchResult(
                        node=synthetic_file_node(item.path, item.line_count),
                        score=item.score,
                        source=item.source,
                    )
                    for item in discovered
                ]
                discovery_warning = "Graph search had no hit; runtime implementation-file discovery returned scoped file candidates."
                warning = f"{warning}; {discovery_warning}" if warning else discovery_warning
        result_payloads = []
        for result in results:
            payload, working_node = _find_result_payload(env, result)
            result_payloads.append(payload)
            source = f"find_preview:{result.source}" if working_node.has_code else result.source
            env.working.add(working_node, source, result.score, env.step_count)
        return {
            "tool": action.tool,
            "warning": warning,
            "path_glob": str(path_glob) if path_glob else None,
            "result_policy": (
                "function/class/method/assignment results include a small implementation preview in W for orientation only; "
                "read the node before memory_commit or repair evidence. file-level results list top symbols instead of full file text"
            ),
            "results": result_payloads,
        }

    if action.tool == "grep_code":
        pattern = str(action.params.get("pattern", ""))
        path_glob = str(action.params.get("path_glob", ""))
        context_lines = int(action.params.get("context_lines", 2))
        limit = int(action.params.get("limit", 20))
        regex = bool(action.params.get("regex", False))
        hits = grep_code(env.graph, env.runtime, pattern, path_glob, context_lines=context_lines, limit=limit, regex=regex)
        payloads = []
        for hit in hits:
            covering = node_brief(hit.covering_node) if hit.covering_node else None
            if hit.covering_node:
                env.working.add(_candidate_only(hit.covering_node), "grep_hit:covering_node", 2.0, env.step_count)
            payloads.append(
                {
                    "path": hit.path,
                    "line": hit.line,
                    "text": hit.text,
                    "context": hit.context,
                    "covering_node": covering,
                    "suggested_read": (
                        {"node_id": hit.covering_node.id, "view": f"around_line:{hit.line}"} if hit.covering_node else None
                    ),
                }
            )
        return {
            "tool": action.tool,
            "pattern": pattern,
            "path_glob": path_glob,
            "result_policy": (
                "grep_code returns line-level navigation context only; read the covering_node before memory_commit or repair evidence"
            ),
            "hits": payloads,
        }

    if action.tool == "explore_expand":
        anchor = str(action.params.get("anchor", ""))
        mode = str(action.params.get("expand_mode", "related"))
        symbol = str(action.params.get("symbol", "")).strip()
        resolved_anchor, candidates = _resolve_node_ref(env, anchor)
        if candidates:
            return {"tool": action.tool, "blocked": True, "reason": f"ambiguous anchor: {anchor}", "candidates": candidates}
        if resolved_anchor:
            anchor = resolved_anchor.id
        if mode in MECHANISM_MODES:
            expanded = expand_with_context(_graph_with_working_code(env), anchor, mode, symbol=symbol)
            result_payloads = []
            for item in expanded:
                payload = {
                    "id": item.node.id,
                    "kind": public_node_kind(item.node.kind),
                    "name": item.node.name,
                    "path": item.node.path,
                    "lines": [item.node.start_line, item.node.end_line],
                    "relation": item.relation,
                    "reason": item.reason,
                    "suggested_read": {"node_id": item.node.id, "view": "body"},
                }
                preview = preview_node_code(env.runtime, item.node)
                working_node = _candidate_only(item.node)
                source = f"expand:{mode}:{item.relation}"
                if preview.error:
                    payload["code_preview_error"] = preview.error
                elif preview.text.strip():
                    payload["code"] = preview.line_numbered
                    payload["code_preview_lines"] = [preview.start_line, preview.end_line]
                    payload["code_preview_truncated"] = preview.truncated
                    working_node = replace(item.node, text=preview.text)
                    source = f"expand_preview:{mode}:{item.relation}"
                env.working.add(working_node, source, 1.5, env.step_count)
                result_payloads.append(payload)
            return {
                "tool": action.tool,
                "anchor": anchor,
                "expand_mode": mode,
                "symbol": symbol or None,
                "result_policy": (
                    "mechanism/owner_flow expand returns lazy AST relation candidates with code previews for orientation. "
                    "Read exact nodes before memory_commit or repair evidence."
                ),
                "results": result_payloads,
            }
        nodes = expand(env.graph, anchor, mode)
        for node in nodes:
            env.working.add(_candidate_only(node), f"expand:{mode}", 1.0, env.step_count)
        return {
            "tool": action.tool,
            "anchor": anchor,
            "expand_mode": mode,
            "results": [
                {"id": n.id, "kind": public_node_kind(n.kind), "name": n.name, "path": n.path, "lines": [n.start_line, n.end_line]}
                for n in nodes
            ],
        }

    if action.tool == "read":
        node_id = str(action.params.get("node_id", ""))
        view = str(action.params.get("view", "body"))
        node, candidates = _resolve_node_ref(env, node_id)
        if candidates:
            return {"tool": action.tool, "blocked": True, "reason": f"ambiguous node_id: {node_id}", "candidates": candidates}
        if not node:
            return {"tool": action.tool, "blocked": True, "reason": f"unknown node_id: {node_id}"}
        if is_test_path(node.path):
            return {"tool": action.tool, "blocked": True, "reason": "benchmark test nodes are not readable repair evidence"}
        read = read_node_for_evidence(env.runtime, node, view)
        env.working.add(read, f"read:{view}", 10.0, env.step_count)
        references = local_symbol_references(env.graph, read, read.text or "", read_node_ids=_read_node_ids(env))
        unread_references = [item for item in references if item.get("read_status") == "unread"]
        tables = dispatch_tables(env.graph, read, read.text or "", read_node_ids=_read_node_ids(env))
        dispatch_context, related_nodes = dispatch_relationship_context(
            env.graph,
            env.runtime,
            read,
            read.text or "",
            issue_text=_issue_context_text(env),
            read_node_ids=_read_node_ids(env),
        )
        flow_context, flow_nodes = value_flow_context(
            env.graph,
            env.runtime,
            read,
            read.text or "",
            read_node_ids=_read_node_ids(env),
        )
        for related in related_nodes:
            env.working.add(related, "relation_context:consumer_candidate_preview", 3.0, env.step_count)
        for related in flow_nodes:
            env.working.add(_candidate_only(related), "value_flow_context:candidate", 2.5, env.step_count)
        return {
            "tool": action.tool,
            "node": node_brief(read),
            "code": line_numbered(read),
            "local_symbol_references": references,
            "unread_local_symbol_references": unread_references,
            "dispatch_tables": tables,
            "dispatch_relationship_context": dispatch_context,
            "value_flow_context": flow_context,
            "value_flow_context_policy": (
                "value_flow_context is best-effort implementation evidence from call expressions and signatures; "
                "it shows upstream/downstream argument-to-parameter flow but is not a proof of runtime values. "
                "Related caller/callee candidates are added to W; read exact nodes before memory_commit/repair."
            ),
            "relationship_context_policy": (
                "consumer candidates are auto-added to W as orientation code only; read the exact node before memory_commit/repair evidence"
            ),
        }

    if action.tool == "memory_commit":
        raw_select_ids = action.params.get("select_ids")
        if not raw_select_ids:
            return {
                "tool": action.tool,
                "blocked": True,
                "reason": "memory_commit requires explicit select_ids; M is model-curated and nodes are never auto-selected",
            }
        select_ids = []
        seen_select_ids: set[str] = set()
        for raw_id in raw_select_ids:
            node_id = str(raw_id)
            if node_id not in seen_select_ids:
                select_ids.append(node_id)
                seen_select_ids.add(node_id)
        before_ids = set(env.memory.nodes)
        keep_ids = set(str(node_id) for node_id in action.params["keep_ids"]) if "keep_ids" in action.params else set(env.memory.nodes)
        dropped_by_keep_ids = sorted(before_ids - keep_ids)
        note = action.params.get("note")
        selected = []
        for node_id in select_ids:
            if node_id not in env.memory.nodes:
                entry = env.working.entries.get(node_id)
                if not entry or not _is_explicit_read_source(entry.source):
                    return {
                        "tool": action.tool,
                        "blocked": True,
                        "reason": (
                            f"memory_commit requires an explicit read before commit: {node_id}. "
                            "explore_find previews and expand candidates are orientation context, not repair evidence."
                        ),
                        "suggested_next_actions": [
                            f"read node_id={node_id} with view=body or a focused around_line/file_window view",
                            "then memory_commit the read node only if it belongs in the evidence_chain",
                        ],
                    }
            node = hydrate_node_from_runtime(env.runtime, env.graph, env.working, str(node_id))
            if is_test_path(node.path):
                return {"tool": action.tool, "blocked": True, "reason": f"benchmark test node cannot be committed: {node.id}"}
            env.working.add(node, "hydrated_for_memory", 10.0, env.step_count)
            selected.append(node)
        kept = [env.memory.nodes[nid] for nid in keep_ids if nid in env.memory.nodes]
        env.memory.nodes = {node.id: node for node in kept}
        env.memory.commit(selected, note=note)
        if note:
            env.notes.add(str(note), action.params.get("tag"))
        after_ids = set(env.memory.nodes)
        selected_ids = [n.id for n in selected]
        newly_added_ids = [node_id for node_id in selected_ids if node_id not in before_ids]
        already_present_ids = [node_id for node_id in selected_ids if node_id in before_ids]
        memory_changed = before_ids != after_ids
        result = {
            "tool": action.tool,
            "committed": selected_ids,
            "explicitly_selected_ids": select_ids,
            "newly_added_ids": newly_added_ids,
            "already_present_ids": already_present_ids,
            "dropped_by_keep_ids": dropped_by_keep_ids,
            "memory_changed": memory_changed,
            "memory": env.memory.summary(),
        }
        if not memory_changed:
            result["note_to_planner"] = (
                "No new repair evidence was added to M. Do not repeat this commit; either repair with the curated M, "
                "commit a different causal node, delete stale memory, or explore/read new evidence."
            )
        return result

    if action.tool == "memory_delete":
        before_ids = set(env.memory.nodes)
        delete_ids = [str(node_id) for node_id in action.params.get("delete_ids") or []]
        keep_ids = [str(node_id) for node_id in action.params.get("keep_ids")] if "keep_ids" in action.params else None
        env.memory.delete(delete_ids, keep_ids, action.params.get("note"))
        if action.params.get("note"):
            env.notes.add(str(action.params["note"]), action.params.get("tag"))
        after_ids = set(env.memory.nodes)
        return {
            "tool": action.tool,
            "deleted_ids": sorted(before_ids - after_ids),
            "requested_delete_ids": delete_ids,
            "requested_keep_ids": keep_ids,
            "memory_changed": before_ids != after_ids,
            "memory": env.memory.summary(),
        }

    if action.tool == "memory_commit_note":
        env.notes.add(str(action.params.get("note", "")), action.params.get("tag"))
        return {"tool": action.tool, "notes": env.notes.summary()}

    if action.tool == "repair_review":
        return _repair_review(env, action)

    if action.tool == "repair":
        return _repair(env, action)

    if action.tool == "repair_propose":
        return _repair(env, action, propose=True)

    if action.tool == "repair_revise":
        return _repair(env, action, propose=True, revise=True)

    if action.tool == "repair_submit":
        return _repair_submit(env, action)

    if action.tool == "discard_pending_patch":
        return _discard_pending_patch(env, action)

    if action.tool == "repair_chunk":
        return _repair(env, action, chunk=True)

    return {"tool": action.tool, "blocked": True, "reason": "unhandled tool"}


def _find_result_payload(env, result) -> tuple[dict[str, object], object]:
    node = result.node
    payload = node_brief(node)
    payload["score"] = result.score
    payload["source"] = result.source
    if node.kind == "file":
        payload["code_preview_policy"] = "file-level result omits full text; read a focused symbol or file_window if needed"
        payload["top_symbols"] = top_symbols_for_file(env.graph, node.path)
        return payload, _candidate_only(node)
    if not is_small_grain_node(node):
        return payload, _candidate_only(node)
    working_node = _candidate_only(node)
    preview = preview_node_code(env.runtime, node)
    if preview.error:
        payload["code_preview_error"] = preview.error
        return payload, working_node
    payload["code"] = preview.line_numbered
    payload["code_preview_truncated"] = preview.truncated
    payload["code_preview_lines"] = [preview.start_line, preview.end_line]
    payload["code_preview_policy"] = "orientation preview only; call read on this node before memory_commit or repair"
    if preview.text.strip():
        working_node = replace(node, text=preview.text)
    references = local_symbol_references(env.graph, node, preview.text, read_node_ids=_read_node_ids(env))
    payload["local_symbol_references"] = references
    payload["unread_local_symbol_references"] = [item for item in references if item.get("read_status") == "unread"]
    payload["dispatch_tables"] = dispatch_tables(env.graph, node, preview.text, read_node_ids=_read_node_ids(env))
    dispatch_context, related_nodes = dispatch_relationship_context(
        env.graph,
        env.runtime,
        node,
        preview.text,
        issue_text=_issue_context_text(env),
        read_node_ids=_read_node_ids(env),
    )
    payload["dispatch_relationship_context"] = dispatch_context
    if related_nodes:
        payload["relationship_context_policy"] = (
            "consumer candidates are auto-added to W as orientation code only; read the exact node before memory_commit/repair evidence"
        )
        for related in related_nodes:
            env.working.add(related, "relation_context:consumer_candidate_preview", 3.0, env.step_count)
    return payload, working_node


def _read_node_ids(env) -> set[str]:
    ids = {
        node_id
        for node_id, entry in env.working.entries.items()
        if entry.node.has_code and _is_explicit_read_source(entry.source)
    }
    ids.update(node_id for node_id, node in env.memory.nodes.items() if node.has_code)
    return ids


def _is_explicit_read_source(source: str) -> bool:
    return str(source or "").startswith(("read:", "hydrated_for_memory"))


def _issue_context_text(env) -> str:
    parts = [env.task.issue_title or "", env.task.issue_body or ""]
    if isinstance(env.failure_summary, dict):
        for key in ("command", "excerpt"):
            value = env.failure_summary.get(key)
            if isinstance(value, str):
                parts.append(value)
        observations = env.failure_summary.get("runtime_observations")
        if isinstance(observations, dict):
            for value in observations.values():
                if isinstance(value, str):
                    parts.append(value)
                elif isinstance(value, list):
                    parts.extend(str(item) for item in value if isinstance(item, (str, int, float)))
    return "\n".join(part for part in parts if part)


def _candidate_only(node):
    return replace(node, text=None)


def _graph_with_working_code(env) -> RepoGraph:
    """Return a transient graph view where hydrated W code overrides graph stubs."""
    graph = RepoGraph(root=env.graph.root)
    graph.nodes.update(env.graph.nodes)
    graph.edges.extend(env.graph.edges)
    graph._edge_set.update(env.graph._edge_set)
    for node_id, entry in env.working.entries.items():
        if entry.node.has_code:
            graph.nodes[node_id] = entry.node
    return graph


def _resolve_node_ref(env, raw: str):
    key = str(raw or "").strip()
    if not key:
        return None, []
    direct = env.working.get(key) or env.graph.nodes.get(key)
    if direct:
        return direct, []
    working_matches = _matching_nodes(
        (entry.node for entry in env.working.entries.values() if not is_test_path(entry.node.path)),
        key,
    )
    if working_matches:
        if len(working_matches) == 1:
            return working_matches[0], []
        return None, [node_brief(node) for node in working_matches[:8]]
    graph_matches = _matching_nodes((node for node in env.graph.nodes.values() if not is_test_path(node.path)), key)
    if not graph_matches:
        return None, []
    if len(graph_matches) == 1:
        return graph_matches[0], []
    return None, [node_brief(node) for node in graph_matches[:8]]


def _matching_nodes(nodes, key: str):
    matches = [node for node in nodes if _node_alias_matches(node, key)]
    concrete = [node for node in matches if node.kind not in {"usage", "import"}]
    if concrete:
        matches = concrete
    matches.sort(key=lambda node: (node.name.lower() != key.lower(), node.path.lower() != key.lower(), node.kind not in {"function", "method"}, node.path, node.start_line))
    return matches


def _node_alias_matches(node, key: str) -> bool:
    key_lower = key.lower()
    name = str(node.name or "")
    path = str(node.path or "")
    if key == name or key == path or key == node.id:
        return True
    if key_lower == name.lower() or key_lower == path.lower():
        return True
    return key_lower in {part.lower() for part in (node.id, name, path)}


def _repair_submit(env, action: PlannerAction) -> dict[str, object]:
    patch = env.pending_patch
    if not isinstance(patch, Patch):
        return {"tool": "repair_submit", "blocked": True, "reason": "repair_submit requires a pending patch"}
    origin = env.pending_patch_origin if isinstance(env.pending_patch_origin, dict) else {}
    memory_ids = [str(node_id) for node_id in origin.get("memory_node_ids", [])] or list(env.memory.nodes)
    target_nodes = [str(node_id) for node_id in origin.get("target_nodes", [])]
    patch_preview = _patch_preview(patch)
    snapshot = env.runtime.snapshot(patch.touched_paths)
    apply_patch(env.runtime, patch)
    syntax = syntax_check_python(env.runtime, patch)
    if syntax and not syntax.passed:
        env.runtime.rollback(snapshot)
        env.repair_feedback = (
            "syntax_failed: submitted pending patch was syntactically invalid and rolled back; "
            f"compiler excerpt: {syntax.summary()}"
        )
        env.pending_patch = None
        env.pending_patch_origin = None
        return _finish_repair(
            env,
            {
                "tool": "repair_submit",
                "status": "syntax_failed",
                "rolled_back": True,
                "reason": "submitted pending patch was syntactically invalid",
                "summary": syntax.summary(),
                "error_origin": "pending_patch",
                "source_tree_state": "rolled_back_to_original",
                "submit_decision": str(action.params.get("decision") or "").strip(),
                "patch_preview": patch_preview,
                "cgm_response": origin.get("cgm_response"),
            },
            memory_ids,
            target_nodes,
        )
    test = env.runtime.run_fail_to_pass(env.task)
    if test.passed:
        env.verified = True
        env.done = True
        env.status = "pass"
        env.pending_patch = None
        env.pending_patch_origin = None
        env.repair_feedback = f"verified pending patch applied: {patch.summary}"
        return _finish_repair(
            env,
            {
                "tool": "repair_submit",
                "status": "passed",
                "rolled_back": False,
                "done": True,
                "touched_paths": patch.touched_paths,
                "summary": patch.summary,
                "submit_decision": str(action.params.get("decision") or "").strip(),
                "test_summary": behavior_summary(test),
                "patch_preview": patch_preview,
                "cgm_response": origin.get("cgm_response"),
            },
            memory_ids,
            target_nodes,
        )
    env.runtime.rollback(snapshot)
    env.pending_patch = None
    env.pending_patch_origin = None
    if test.status == "infra_bug":
        env.done = True
        env.status = "bug"
    env.repair_feedback = (
        "test_failed and rolled back: submitted pending patch applied but fail-to-pass behavior is still wrong. "
        "Use recent_repair_attempts, recent_cgm_insights, and failure_feedback before generating another candidate."
    )
    return _finish_repair(
        env,
        {
            "tool": "repair_submit",
            "status": "test_failed" if test.status != "infra_bug" else "infra_bug",
            "rolled_back": True,
            "done": test.status == "infra_bug",
            "touched_paths": patch.touched_paths,
            "summary": patch.summary,
            "submit_decision": str(action.params.get("decision") or "").strip(),
            "test_summary": behavior_summary(test),
            "error_origin": "generated_patch_behavior" if test.status != "infra_bug" else "test_infra",
            "source_tree_state": "rolled_back_to_original",
            "patch_preview": patch_preview,
            "cgm_response": origin.get("cgm_response"),
        },
        memory_ids,
        target_nodes,
    )


def _discard_pending_patch(env, action: PlannerAction) -> dict[str, object]:
    summary = _pending_patch_summary(env)
    reason = str(action.params.get("reason") or "").strip()
    env.pending_patch = None
    env.pending_patch_origin = None
    env.repair_feedback = f"pending patch discarded: {reason or 'planner discarded candidate patch'}"
    return {
        "tool": "discard_pending_patch",
        "status": "discarded",
        "reason": reason,
        "discarded_patch": summary,
        "source_tree_state": "unchanged",
    }


def _repair(env, action: PlannerAction, *, chunk: bool = False, propose: bool = False, revise: bool = False) -> dict[str, object]:
    tool_name = "repair_revise" if revise else "repair_propose" if propose else "repair_chunk" if chunk else "repair"
    if env.config.require_failed_test_before_repair and env.failure_summary is None:
        return {"tool": tool_name, "blocked": True, "reason": f"run_failed_test is required before {tool_name}"}
    if not env.memory.nodes:
        return {"tool": tool_name, "blocked": True, "reason": f"{tool_name} requires at least one committed memory node with code"}
    missing = [node.id for node in env.memory.nodes.values() if not node.has_code]
    if missing:
        return {"tool": tool_name, "blocked": True, "reason": f"memory nodes lack code bodies: {missing}"}
    contract_errors = _repair_contract_errors(env, action.params)
    if contract_errors:
        return {
            "tool": tool_name,
            "blocked": True,
            "reason": "; ".join(contract_errors),
            "contract_errors": contract_errors,
            "suggested_next_actions": _repair_blocked_guidance(action.params, contract_errors),
        }
    plan = _repair_plan_text(action.params)
    if chunk:
        plan = _chunk_plan_text(plan, action.params)
    if propose:
        plan = _propose_plan_text(plan, action.params, revise=revise)
    confidence = _confidence(action.params.get("confidence"))
    memory_ids = list(env.memory.nodes)
    target_nodes = [str(node_id) for node_id in action.params.get("target_nodes", [])]
    pending_patch_summary = _pending_patch_summary(env) if revise else None
    planner_decision_context = _planner_decision_context(action.params, tool_name, pending_patch_summary)
    payload = build_cgm_payload(
        env.task,
        env.graph,
        env.memory,
        plan,
        confidence,
        env.failure_summary,
        env.repair_feedback,
        env.config.max_patch_edits,
        target_node_ids=target_nodes,
        repair_history=env.repair_attempts,
        pending_patch=pending_patch_summary,
        cgm_insights=env.cgm_insights,
        planner_decision_context=planner_decision_context,
    )
    if revise and env.pending_patch is None:
        return {
            "tool": tool_name,
            "blocked": True,
            "reason": "repair_revise requires an existing pending_patch from repair_propose or a prior repair_revise",
        }
    payload_errors = validate_cgm_payload(payload)
    payload_summary = summarize_cgm_payload(payload)
    if payload_errors:
        env.repair_feedback = "patch_rejected: invalid CGM payload: " + "; ".join(payload_errors)
        return _finish_repair(
            env,
            {
                "tool": tool_name,
                "status": "patch_rejected",
                "reason": env.repair_feedback,
                "error_origin": "cgm_payload_validation",
                "cgm_payload": payload_summary,
            },
            memory_ids,
            target_nodes,
        )
    try:
        raw = env.cgm.generate_patch(payload)
    except CgmUnavailableError as exc:
        env.repair_feedback = (
            f"infra_retryable: CGM unavailable during repair generation: {exc}. "
            "No patch was generated or applied; source tree is unchanged. "
            "You may retry repair with the same evidence if the evidence package is still sound, "
            "or continue reading if confidence is low."
        )
        return _finish_repair(
            env,
            {
                "tool": tool_name,
                "status": "infra_retryable",
                "retryable": True,
                "done": False,
                "reason": env.repair_feedback,
                "error_origin": "cgm_unavailable",
                "source_tree_state": "unchanged",
                "cgm_payload": payload_summary,
            },
            memory_ids,
            target_nodes,
        )
    except Exception as exc:
        env.repair_feedback = f"patch_rejected: CGM generation failed: {exc}"
        return _finish_repair(
            env,
            {
                "tool": tool_name,
                "status": "patch_rejected",
                "reason": env.repair_feedback,
                "error_origin": "cgm_generation",
                "cgm_payload": payload_summary,
            },
            memory_ids,
            target_nodes,
        )
    cgm_response = _compact_cgm_response(raw)
    protocol_error = _cgm_output_protocol_error(raw)
    if protocol_error:
        env.repair_feedback = f"patch_rejected: {protocol_error}"
        return _finish_repair(
            env,
            {
                "tool": tool_name,
                "status": "patch_rejected",
                "reason": protocol_error,
                "error_origin": "cgm_output_protocol",
                "cgm_payload": payload_summary,
                "cgm_output": _compact_cgm_output_protocol(raw),
                "cgm_response": cgm_response,
            },
            memory_ids,
            target_nodes,
        )
    try:
        patch = parse_cgm_output(raw)
    except Exception as exc:
        env.repair_feedback = f"patch_rejected: {exc}"
        return _finish_repair(
            env,
            {
                "tool": tool_name,
                "status": "patch_rejected",
                "reason": str(exc),
                "error_origin": "cgm_patch_schema",
                "cgm_payload": payload_summary,
                "cgm_response": cgm_response,
            },
            memory_ids,
            target_nodes,
        )
    patch, normalization_notes = normalize_patch_with_runtime(env.runtime, patch)
    patch_preview = _patch_preview(patch)
    if normalization_notes:
        patch_preview["normalization_notes"] = normalization_notes
    decision = validate_patch_with_runtime(env.runtime, patch, env.config.max_patch_edits, env.config.allow_test_changes)
    if not decision.ok:
        retry = _retry_patch_generation_after_validation_failure(env, payload, decision.reason, patch_preview)
        if retry:
            patch, patch_preview, decision = retry
        if not decision.ok:
            env.repair_feedback = f"patch_rejected: {decision.reason}"
            error_origin = "patch_format_validation" if _is_patch_format_validation(decision.reason) else "patch_validation"
            return _finish_repair(
                env,
                {
                    "tool": tool_name,
                    "status": "patch_rejected",
                    "reason": decision.reason,
                    "error_origin": error_origin,
                    "patch_preview": patch_preview,
                    "cgm_payload": payload_summary,
                    "cgm_response": cgm_response,
                },
                memory_ids,
                target_nodes,
            )
    signature = json.dumps({"edits": [asdict(edit) for edit in patch.edits]}, sort_keys=True)
    if env.repair_history.duplicate(signature):
        env.repair_feedback = (
            "patch_rejected: duplicate patch attempt. The exact generated edits have already been tried or rejected. "
            "Do not repeat the same repair intent. Treat this as evidence that the current intent_analysis is stale or incomplete; "
            "revise the evidence_chain or intent_analysis and read or commit a new caller, consumer, output-shaping, "
            "parent, sibling, or local invariant node before another repair."
        )
        return _finish_repair(
            env,
            {
                "tool": tool_name,
                "status": "patch_rejected",
                "reason": "duplicate patch attempt",
                "error_origin": "duplicate_patch",
                "patch_preview": patch_preview,
                "cgm_payload": payload_summary,
                "cgm_response": cgm_response,
            },
            memory_ids,
            target_nodes,
        )
    env.repair_history.record(signature)
    snapshot = env.runtime.snapshot(patch.touched_paths)
    apply_patch(env.runtime, patch)
    syntax = syntax_check_python(env.runtime, patch)
    if syntax and not syntax.passed:
        env.runtime.rollback(snapshot)
        retry = _retry_patch_generation_after_generated_syntax_failure(env, payload, syntax.summary(), patch_preview)
        if retry:
            retry_patch, retry_preview, retry_decision, retry_cgm_response = retry
            retry_signature = json.dumps({"edits": [asdict(edit) for edit in retry_patch.edits]}, sort_keys=True)
            if not retry_decision.ok:
                patch_preview["internal_retry_rejected_reason"] = retry_decision.reason
                patch_preview["internal_retry_patch_preview"] = retry_preview
            elif env.repair_history.duplicate(retry_signature):
                patch_preview["internal_retry_rejected_reason"] = "duplicate patch attempt"
                patch_preview["internal_retry_patch_preview"] = retry_preview
            else:
                patch = retry_patch
                patch_preview = retry_preview
                cgm_response = retry_cgm_response
                env.repair_history.record(retry_signature)
                snapshot = env.runtime.snapshot(patch.touched_paths)
                apply_patch(env.runtime, patch)
                syntax = syntax_check_python(env.runtime, patch)
                if syntax and not syntax.passed:
                    env.runtime.rollback(snapshot)
                else:
                    syntax = None
        if syntax and not syntax.passed:
            env.repair_feedback = (
                "syntax_failed: generated patch was syntactically invalid and was rolled back; "
                "original source remains unchanged. Do not infer an original source syntax error from this. "
                f"Generated patch compiler excerpt: {syntax.summary()}"
            )
            return _finish_repair(
                env,
                {
                    "tool": tool_name,
                    "status": "syntax_failed",
                    "rolled_back": True,
                    "reason": "generated patch was syntactically invalid and rolled back; original source remains unchanged",
                    "summary": syntax.summary(),
                    "error_origin": "generated_patch",
                    "source_tree_state": "rolled_back_to_original",
                    "generated_patch_error_excerpt": syntax.summary(),
                    "patch_preview": patch_preview,
                    "cgm_payload": payload_summary,
                    "cgm_response": cgm_response,
                },
                memory_ids,
                target_nodes,
            )
    if chunk:
        env.repair_feedback = (
            f"repair_chunk applied and kept: {patch.summary}. "
            "This chunk has not been judged by fail-to-pass/PASS_TO_PASS tests; continue with the next chunk or call repair for final verification."
        )
        return _finish_repair(
            env,
            {
                "tool": tool_name,
                "status": "chunk_applied",
                "rolled_back": False,
                "done": False,
                "touched_paths": patch.touched_paths,
                "summary": patch.summary,
                "source_tree_state": "patched_unverified_chunk",
                "remaining_work": str(action.params.get("remaining_work") or "").strip(),
                "cgm_payload": payload_summary,
                "patch_preview": patch_preview,
                "cgm_response": cgm_response,
            },
            memory_ids,
            target_nodes,
        )
    if propose:
        env.runtime.rollback(snapshot)
        env.pending_patch = patch
        env.pending_patch_origin = {
            "tool": tool_name,
            "status": "patch_proposed",
            "summary": patch.summary,
            "memory_node_ids": sorted(str(node_id) for node_id in memory_ids),
            "target_nodes": sorted(str(node_id) for node_id in target_nodes),
            "planner_decision_context": planner_decision_context,
            "patch_preview": patch_preview,
            "cgm_response": cgm_response,
        }
        env.repair_feedback = (
            f"{tool_name} produced a pending patch: {patch.summary}. "
            "Planner must inspect pending_patch_summary and choose repair_submit, repair_revise, discard_pending_patch, or read more code before testing."
        )
        return _finish_repair(
            env,
            {
                "tool": tool_name,
                "status": "patch_proposed",
                "rolled_back": True,
                "done": False,
                "touched_paths": patch.touched_paths,
                "summary": patch.summary,
                "source_tree_state": "unchanged_pending_patch_saved",
                "cgm_payload": payload_summary,
                "patch_preview": patch_preview,
                "pending_patch_summary": _pending_patch_summary(env),
                "cgm_response": cgm_response,
            },
            memory_ids,
            target_nodes,
        )

    test = env.runtime.run_fail_to_pass(env.task)
    if test.passed:
        env.verified = True
        env.done = True
        env.status = "pass"
        env.repair_feedback = f"verified patch applied: {patch.summary}"
        return _finish_repair(
            env,
            {
                "tool": "repair",
                "status": "passed",
                "rolled_back": False,
                "done": True,
                "touched_paths": patch.touched_paths,
                "summary": patch.summary,
                "test_summary": behavior_summary(test),
                "cgm_payload": payload_summary,
                "patch_preview": patch_preview,
                "cgm_response": cgm_response,
            },
            memory_ids,
            target_nodes,
        )
    if test.status == "infra_bug":
        env.runtime.rollback(snapshot)
        env.repair_feedback = (
            "infra_bug and rolled back: generated patch was not judged because the test runner used an invalid "
            f"environment or could not execute the issue test reliably. Runtime summary: {test.summary()}. "
            "Do not treat this as patch behavior evidence. Fix the test environment or use the official SWE-bench "
            "eval command before drawing repair conclusions."
        )
        return _finish_repair(
            env,
            {
                "tool": "repair",
                "status": "infra_bug",
                "rolled_back": True,
                "touched_paths": patch.touched_paths,
                "summary": patch.summary,
                "test_summary": behavior_summary(test),
                "error_origin": "test_infra",
                "source_tree_state": "rolled_back_to_original",
                "cgm_payload": payload_summary,
                "patch_preview": patch_preview,
                "cgm_response": cgm_response,
            },
            memory_ids,
            target_nodes,
        )
    env.runtime.rollback(snapshot)
    env.repair_feedback = (
        "test_failed and rolled back: generated patch applied but fail-to-pass behavior is still wrong. "
        "Treat this as evidence that the prior intent_analysis or patch site is incomplete, not as permission to retry "
        f"the same patch. Runtime summary: {test.summary()}. "
        "Before the next repair, use last_repair_attempt.failure_feedback and collect new evidence for "
        "the remaining runtime behavior, especially downstream consumers, output formatting, state propagation, parent/base "
        "logic, or sibling implementations."
    )
    return _finish_repair(
        env,
        {
            "tool": "repair",
            "status": "test_failed",
            "rolled_back": True,
            "touched_paths": patch.touched_paths,
            "summary": patch.summary,
            "test_summary": behavior_summary(test),
            "error_origin": "generated_patch_behavior",
            "source_tree_state": "rolled_back_to_original",
            "cgm_payload": payload_summary,
            "patch_preview": patch_preview,
            "cgm_response": cgm_response,
        },
        memory_ids,
        target_nodes,
    )


def _repair_review(env, action: PlannerAction) -> dict[str, object]:
    if env.config.require_failed_test_before_repair and env.failure_summary is None:
        return {"tool": "repair_review", "blocked": True, "reason": "run_failed_test is required before repair_review"}
    if not env.memory.nodes:
        return {"tool": "repair_review", "blocked": True, "reason": "repair_review requires committed memory nodes with code"}
    missing = [node.id for node in env.memory.nodes.values() if not node.has_code]
    if missing:
        return {"tool": "repair_review", "blocked": True, "reason": f"memory nodes lack code bodies: {missing}"}
    contract_errors = _repair_contract_errors(env, action.params, allow_failed_same_memory=True)
    if contract_errors:
        return {
            "tool": "repair_review",
            "blocked": True,
            "reason": "; ".join(contract_errors),
            "contract_errors": contract_errors,
            "suggested_next_actions": _repair_blocked_guidance(action.params, contract_errors),
        }

    plan = _repair_plan_text(action.params)
    confidence = _confidence(action.params.get("confidence"))
    target_nodes = [str(node_id) for node_id in action.params.get("target_nodes", [])]
    payload = build_cgm_payload(
        env.task,
        env.graph,
        env.memory,
        plan,
        confidence,
        env.failure_summary,
        env.repair_feedback,
        env.config.max_patch_edits,
        target_node_ids=target_nodes,
        repair_history=env.repair_attempts,
        pending_patch=_pending_patch_summary(env),
        cgm_insights=env.cgm_insights,
        planner_decision_context=_planner_decision_context(action.params, "repair_review", None),
    )
    payload["mode"] = "intent_review"
    payload["review_request"] = {
        "purpose": "critique repair intent and evidence only; do not generate or apply a patch",
        "planner_confidence": confidence,
        "last_repair_attempt": env.last_repair_attempt,
        "previous_review": env.last_repair_review,
        "planner_review_focus": str(action.params.get("review_focus") or "").strip(),
    }
    if env.repair_feedback:
        payload["prior_feedback"] = env.repair_feedback
    payload_errors = validate_cgm_payload(payload)
    payload_summary = summarize_cgm_payload(payload)
    if payload_errors:
        return {
            "tool": "repair_review",
            "status": "patch_rejected",
            "blocked": True,
            "reason": "invalid CGM review payload: " + "; ".join(payload_errors),
            "error_origin": "cgm_payload_validation",
            "cgm_payload": payload_summary,
        }
    try:
        raw = env.cgm.review_intent(payload)
    except CgmUnavailableError as exc:
        return {
            "tool": "repair_review",
            "status": "infra_retryable",
            "retryable": True,
            "done": False,
            "reason": f"CGM unavailable during repair_review: {exc}",
            "error_origin": "cgm_unavailable",
            "cgm_payload": payload_summary,
        }
    except Exception as exc:
        return {
            "tool": "repair_review",
            "status": "patch_rejected",
            "reason": f"CGM repair_review failed: {exc}",
            "error_origin": "cgm_generation",
            "cgm_payload": payload_summary,
        }
    review = _compact_cgm_review(raw)
    env.last_repair_review = {
        "status": "reviewed",
        "review": review,
        "package_signature": _repair_package_signature(env, action.params),
    }
    result = {
        "tool": "repair_review",
        "status": "reviewed",
        "review": review,
        "cgm_payload": payload_summary,
        "cgm_response": _compact_cgm_response(raw),
        "note_to_planner": (
            "Use this CGM critique to revise intent_analysis, target_nodes, evidence_chain, memory_delete stale nodes, "
            "or proceed to repair if verdict/confidence support the current mechanism."
        ),
    }
    note = _review_note(review)
    if note:
        env.notes.add(note, tag="repair_review")
    return result


def _repair_contract_errors(env, params: dict[str, object], *, allow_failed_same_memory: bool = False) -> list[str]:
    errors: list[str] = []
    required_text = ["failure_seen", "intent_analysis"]
    for key in required_text:
        if not str(params.get(key, "")).strip():
            errors.append(f"repair requires non-empty {key}")
    confidence = _confidence(params.get("confidence"))
    if confidence is None:
        errors.append("repair requires numeric confidence between 0 and 1")

    target_nodes = _string_list(params.get("target_nodes"))
    if not target_nodes:
        errors.append("repair requires non-empty target_nodes")

    evidence_chain = params.get("evidence_chain")
    chain_node_ids: list[str] = []
    if not isinstance(evidence_chain, list) or not evidence_chain:
        errors.append("repair requires non-empty evidence_chain")
    else:
        for idx, item in enumerate(evidence_chain):
            if not isinstance(item, dict):
                errors.append(f"evidence_chain[{idx}] must be an object")
                continue
            node_id = str(item.get("node_id", "")).strip()
            role = str(item.get("role", "")).strip()
            evidence = str(item.get("evidence", "")).strip()
            if not node_id:
                errors.append(f"evidence_chain[{idx}] requires node_id")
                continue
            if not role:
                errors.append(f"evidence_chain[{idx}] requires role")
            if not evidence:
                errors.append(f"evidence_chain[{idx}] requires evidence")
            chain_node_ids.append(node_id)

    memory_ids = set(env.memory.nodes)
    read_ids = _read_node_ids(env)
    for node_id in target_nodes:
        if new_file_target_path(node_id):
            continue
        if node_id.startswith("new_file:"):
            errors.append(f"invalid new_file target path: {node_id}")
        elif node_id not in memory_ids:
            errors.append(f"target node is not in repair memory M: {node_id}")
    for node_id in chain_node_ids:
        if new_file_target_path(node_id):
            continue
        if node_id not in read_ids:
            errors.append(f"evidence_chain node is not a read/committed code node: {node_id}")
    missing_targets_from_chain = [
        node_id
        for node_id in target_nodes
        if node_id not in chain_node_ids and not new_file_target_path(node_id)
    ]
    if missing_targets_from_chain:
        errors.append(f"target_nodes must appear in evidence_chain: {missing_targets_from_chain}")

    if (
        not allow_failed_same_memory
        and env.repair_history.failed_with_same_memory(list(env.memory.nodes))
        and not _last_review_ready_for_package(env, params)
    ):
        errors.append("previous repair failed and repair memory M has not changed; collect or change evidence before retrying repair")
    if not allow_failed_same_memory:
        review_error = _repair_review_gate_error(env, params)
        if review_error:
            errors.append(review_error)
        api_hint = api_signature_failure_hint(getattr(env, "last_repair_attempt", None))
        if api_hint and not evidence_mentions_api(params, str(api_hint.get("api_symbol") or "")):
            errors.append(
                "previous patch failed because it used an unverified API/signature "
                f"({api_hint.get('error_excerpt')}); evidence_chain must include read implementation code "
                f"proving {api_hint.get('api_symbol')} usage/signature before repair"
            )
    return errors


def _repair_package_signature(env, params: dict[str, object]) -> dict[str, object]:
    return {
        "memory_node_ids": sorted(str(node_id) for node_id in env.memory.nodes),
        "target_nodes": sorted(_string_list(params.get("target_nodes"))),
        "evidence_chain_node_ids": sorted(_evidence_chain_node_ids(params.get("evidence_chain"))),
    }


def _evidence_chain_node_ids(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    ids: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id") or "").strip()
        if node_id:
            ids.append(node_id)
    return ids


def _repair_review_gate_error(env, params: dict[str, object]) -> str | None:
    state = getattr(env, "last_repair_review", None)
    if not isinstance(state, dict):
        return None
    review = state.get("review") if isinstance(state.get("review"), dict) else {}
    verdict = str(review.get("verdict") or "").strip()
    if verdict in {"", "ready"}:
        return None
    previous_signature = state.get("package_signature")
    if not isinstance(previous_signature, dict):
        return None
    current_signature = _repair_package_signature(env, params)
    if current_signature != previous_signature:
        return None
    return (
        f"latest repair_review verdict={verdict}; repair with the same memory/target/evidence package is blocked. "
        "Read/commit/delete implementation evidence, change target_nodes or evidence_chain, or run another repair_review that returns ready."
    )


def _last_review_ready_for_package(env, params: dict[str, object]) -> bool:
    state = getattr(env, "last_repair_review", None)
    if not isinstance(state, dict):
        return False
    review = state.get("review") if isinstance(state.get("review"), dict) else {}
    if str(review.get("verdict") or "").strip() != "ready":
        return False
    previous_signature = state.get("package_signature")
    if not isinstance(previous_signature, dict):
        return False
    return _repair_package_signature(env, params) == previous_signature


def _confidence(value: object) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence < 0 or confidence > 1:
        return None
    return confidence


def _repair_blocked_guidance(params: dict[str, object], errors: list[str]) -> list[str]:
    joined = " ; ".join(errors).lower()
    guidance = [
        "Do not repeat repair with the same evidence package; close the named evidence gap with explore_find, explore_expand, read, or memory_commit.",
    ]
    if "target node is not in repair memory" in joined:
        guidance.append("Commit the patch target node into repair_memory_M before repair.")
    if "evidence_chain node is not a read/committed code node" in joined:
        guidance.append("Read or locate every evidence_chain node_id that is not already code-bearing in W/M, then retry with those exact ids.")
    if "test_behavior" in joined:
        guidance.append("Do not use test_behavior as an evidence_chain node_id; put runtime/test symptoms only in failure_seen, then use read code node ids and any explicit new_file target in evidence_chain.")
    if "target_nodes must appear in evidence_chain" in joined:
        guidance.append("Add each target node to evidence_chain with a short evidence sentence explaining why it is the patch locus.")
    if "confidence" in joined:
        guidance.append("Set confidence to a number from 0 to 1; use a lower value if localization is plausible but behavior details are still uncertain.")
    return guidance


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _repair_plan_text(params: dict[str, object]) -> str:
    intent = str(params.get("intent_analysis", "")).strip()
    evidence = [
        str(params.get("failure_seen", "")).strip(),
        _format_evidence_chain(params.get("evidence_chain")),
    ]
    confidence = _confidence(params.get("confidence"))
    intent_parts = []
    if confidence is not None:
        intent_parts.append(f"confidence={confidence:.2f}")
    if intent:
        intent_parts.append(intent)
    sections = [
        (
            "Patch-site focus",
            "\n".join(f"- {node_id}" for node_id in _string_list(params.get("target_nodes"))),
        ),
        ("Evidence chain", "\n".join(part for part in evidence if part)),
        ("Planner intent analysis", "\n".join(part for part in intent_parts if part)),
        (
            "Planner constraints",
            "\n".join(
                part
                for part in [
                    "Do not let this brief override the issue text or source snippets.",
                    "Treat any proposed replacement API/attribute/keyword as tentative unless it is visible in the provided code.",
                ]
                if part
            ),
        ),
    ]
    return "\n\n".join(f"{title}:\n{body}" for title, body in sections if body)


def _chunk_plan_text(plan: str, params: dict[str, object]) -> str:
    remaining = str(params.get("remaining_work") or "").strip()
    chunk_rules = [
        "Generate only this chunk. Do not solve unrelated parts of the issue in this response.",
        "The chunk must be internally coherent and should leave the repository in a syntactically valid state.",
        "Final fail-to-pass/PASS_TO_PASS verification will happen in a later ordinary repair action.",
    ]
    if remaining:
        chunk_rules.append(f"Known remaining work after this chunk: {remaining}")
    return plan.rstrip() + "\n\nChunk mode:\n" + "\n".join(f"- {item}" for item in chunk_rules)


def _propose_plan_text(plan: str, params: dict[str, object], *, revise: bool = False) -> str:
    rules = [
        "Generate a candidate patch only; the planner will inspect it before tests run.",
        "Return a concise insight_summary if the service schema supports it.",
        "Do not repeat recent failed patch strategies from repair_history or cgm_insights.",
    ]
    if revise:
        focus = str(params.get("revision_focus") or "").strip()
        review = params.get("pending_patch_review") or {}
        rules.extend(
            [
                "Revise the current pending patch instead of producing an unrelated patch.",
                f"Revision focus: {focus}",
                "Planner pending patch review: " + json.dumps(review, ensure_ascii=False, sort_keys=True),
            ]
        )
    return plan.rstrip() + "\n\nPatch deliberation mode:\n" + "\n".join(f"- {item}" for item in rules if item)


def _planner_decision_context(params: dict[str, object], tool_name: str, pending_patch: dict[str, object] | None) -> dict[str, object]:
    context: dict[str, object] = {"tool": tool_name}
    for key in ("revision_focus", "pending_patch_review", "remaining_work"):
        value = params.get(key)
        if value:
            context[key] = value
    if pending_patch:
        context["pending_patch"] = pending_patch
    return context


def _format_evidence_chain(value: object) -> str:
    if not isinstance(value, list):
        return ""
    lines: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id", "")).strip()
        role = str(item.get("role", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        if node_id or role or evidence:
            line = f"- {role or 'role'}: {node_id}"
            if evidence:
                line += f" -- {evidence}"
            lines.append(line)
    return "\n".join(lines)


def _patch_preview(patch) -> dict[str, object]:
    return {
        "summary": patch.summary,
        "edits": [
            {
                "path": edit.path,
                "start": edit.start,
                "end": edit.end,
                "new_text": _truncate_feedback(edit.new_text, 1400),
            }
            for edit in patch.edits[:8]
        ],
        "edit_count": len(patch.edits),
    }


def _pending_patch_summary(env) -> dict[str, object] | None:
    patch = getattr(env, "pending_patch", None)
    if not isinstance(patch, Patch):
        return None
    origin = getattr(env, "pending_patch_origin", None)
    return {
        "summary": patch.summary,
        "touched_paths": patch.touched_paths,
        "patch_preview": _patch_preview(patch),
        "origin": _compact_result_value(origin, 1800) if isinstance(origin, dict) else None,
    }


def _is_patch_format_validation(reason: str) -> bool:
    lowered = reason.lower()
    return any(
        marker in lowered
        for marker in [
            "schema json embedded",
            "tuple/schema artifact",
            "standalone diff plus marker",
            "diff/conflict marker embedded",
            "collapses multi-line python span",
        ]
    )


def _cgm_output_protocol_error(raw: object) -> str | None:
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("diff --git") or stripped.startswith("--- ") or stripped.startswith("@@ "):
            return "CGM returned unified diff text, but this agent requires JSON patch object output"
        try:
            raw = json.loads(stripped)
        except Exception:
            return None
    if not isinstance(raw, dict):
        return None
    summaries: list[str] = []
    for value in (raw.get("summary"), raw.get("format"), raw.get("output_format")):
        if value is not None:
            summaries.append(str(value).strip().lower())
    patch = raw.get("patch")
    if isinstance(patch, dict):
        for value in (patch.get("summary"), patch.get("format"), patch.get("output_format")):
            if value is not None:
                summaries.append(str(value).strip().lower())
    if any(value == "codefuse-cgm-partial" for value in summaries):
        return "CGM used codefuse-cgm-partial fallback; require a complete JSON patch object instead"
    if isinstance(raw.get("diff"), str):
        return "CGM returned an unparsed diff field; require parsed patch edits or a complete JSON patch object"
    return None


def _compact_cgm_output_protocol(raw: object) -> dict[str, object]:
    if isinstance(raw, str):
        return {"type": "str", "preview": _truncate_feedback(raw, 1200)}
    if not isinstance(raw, dict):
        return {"type": type(raw).__name__}
    patch = raw.get("patch")
    edit_count = None
    patch_summary = None
    if isinstance(patch, dict):
        edits = patch.get("edits")
        edit_count = len(edits) if isinstance(edits, list) else None
        patch_summary = patch.get("summary")
    return {
        "type": "dict",
        "top_keys": sorted(str(key) for key in raw.keys()),
        "summary": raw.get("summary"),
        "patch_summary": patch_summary,
        "edit_count": edit_count,
        "runtime_mode": raw.get("runtime_mode"),
    }


def _compact_cgm_response(raw: object) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        return None
    reasoning = raw.get("reasoning_content")
    reasoning_text = reasoning if isinstance(reasoning, str) else ""
    compact = {
        "model": raw.get("model"),
        "output_format": raw.get("output_format"),
        "thinking_enabled": raw.get("thinking_enabled"),
        "reasoning_chars": raw.get("reasoning_chars", len(reasoning_text)),
        "reasoning_preview": _truncate_feedback(str(raw.get("reasoning_preview") or reasoning_text), 1200) if reasoning_text or raw.get("reasoning_preview") else None,
        "raw_preview": _truncate_feedback(str(raw.get("raw_preview") or ""), 1200) if raw.get("raw_preview") else None,
    }
    if isinstance(raw.get("insight_summary"), dict):
        compact["insight_summary"] = _compact_result_value(raw.get("insight_summary"), 1400)
    return compact


def _cgm_insight(raw_response: object, compact_response: dict[str, object] | None, patch_preview: dict[str, object] | None) -> dict[str, object] | None:
    if not isinstance(raw_response, dict) and not compact_response:
        return None
    explicit = raw_response.get("insight_summary") if isinstance(raw_response, dict) else None
    if explicit is None and isinstance(compact_response, dict):
        explicit = compact_response.get("insight_summary")
    if isinstance(explicit, dict):
        insight = dict(explicit)
    else:
        insight = {}
    if patch_preview and "patch_summary" not in insight:
        insight["patch_summary"] = patch_preview.get("summary")
        insight["touched_paths"] = sorted({str(edit.get("path")) for edit in patch_preview.get("edits", []) if isinstance(edit, dict) and edit.get("path")})
        insight["edit_count"] = patch_preview.get("edit_count")
    if compact_response:
        insight.setdefault("model", compact_response.get("model"))
        insight.setdefault("output_format", compact_response.get("output_format"))
        if compact_response.get("reasoning_preview") and "reasoning_summary" not in insight:
            insight["reasoning_summary"] = compact_response.get("reasoning_preview")
    return _compact_result_value(insight, 1800) if insight else None


def _compact_cgm_review(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        return {"verdict": "needs_more_evidence", "raw_type": type(raw).__name__}
    review = raw.get("review")
    if not isinstance(review, dict):
        review = raw
    confidence = _confidence(review.get("confidence"))

    def short_text(key: str, limit: int = 900) -> str:
        return _truncate_feedback(str(review.get(key) or ""), limit)

    def short_list(key: str, fallback_key: str | None = None) -> list[str]:
        value = review.get(key)
        if not isinstance(value, list) and fallback_key:
            value = review.get(fallback_key)
        if not isinstance(value, list):
            return []
        return [_truncate_feedback(str(item), 320) for item in value[:8] if str(item).strip()]

    verdict = str(review.get("verdict") or "needs_more_evidence").strip()
    if verdict not in {"ready", "needs_more_evidence", "change_target", "avoid_patch"}:
        verdict = "needs_more_evidence"
    evidence_gaps, removed_missing = _sanitize_review_evidence(short_list("evidence_gaps", fallback_key="missing_evidence"))
    suggested_next_action, removed_suggested = _sanitize_review_action(short_text("suggested_next_action", 500))
    removed_test_source_requests = removed_missing
    if removed_suggested:
        removed_test_source_requests.append(removed_suggested)
    adoption_caveat = ""
    if verdict == "ready" and evidence_gaps:
        adoption_caveat = (
            "CGM marked this intent ready but also listed evidence_gaps; planner may adopt the ready review "
            "if target/mechanism confidence is supported by visible code, or validate the gaps first if they look essential."
        )
    elif verdict == "ready" and _review_action_requests_more_evidence(suggested_next_action):
        adoption_caveat = (
            "CGM marked this intent ready while suggesting more inspection; planner may still adopt it when the suggested "
            "inspection is optional rather than a missing target/mechanism link."
        )
    return {
        "verdict": verdict,
        "confidence": confidence,
        "mechanism_assessment": short_text("mechanism_assessment") or short_text("issue_mechanism"),
        "target_assessment": short_text("target_assessment"),
        "evidence_gaps": evidence_gaps,
        "suggested_next_action": suggested_next_action,
        "adoption_advice": short_text("adoption_advice") or short_text("feedback_for_planner"),
        "removed_benchmark_test_source_requests": removed_test_source_requests,
        "adoption_caveat": adoption_caveat,
        "summary": raw.get("summary"),
    }


def _review_action_requests_more_evidence(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False
    evidence_verbs = ["read", "inspect", "search", "explore", "open", "look at", "check", "verify", "confirm"]
    evidence_objects = [
        "implementation",
        "code",
        "method",
        "function",
        "class",
        "api",
        "signature",
        "call site",
        "caller",
        "consumer",
        "writer",
        "line",
        "lines",
        "after",
    ]
    return any(verb in lowered for verb in evidence_verbs) and any(obj in lowered for obj in evidence_objects)


def _sanitize_review_evidence(items: list[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    removed: list[str] = []
    for item in items:
        if _requests_benchmark_test_source(item):
            removed.append(item)
        else:
            kept.append(item)
    if removed:
        replacement = (
            "Do not read benchmark test source; collect the equivalent evidence from runtime output, issue text, "
            "or implementation code paths instead."
        )
        if replacement not in kept:
            kept.append(replacement)
    return kept, removed


def _sanitize_review_action(text: str) -> tuple[str, str | None]:
    if not _requests_benchmark_test_source(text):
        return text, None
    return (
        "Do not read benchmark test source; inspect runtime output summaries and implementation code that produces the observed behavior.",
        text,
    )


def _requests_benchmark_test_source(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False
    read_like = any(word in lowered for word in ["read", "inspect", "open", "look at", "check"])
    test_source_marker = any(
        marker in lowered
        for marker in [
            "test source",
            "test file",
            "test function",
            "hidden test",
            "hidden assertion",
            "expected values from tests",
            "/tests/",
            "tests/",
            "test_",
            "_test.py",
        ]
    )
    return read_like and test_source_marker


def _review_note(review: dict[str, object]) -> str:
    verdict = str(review.get("verdict") or "").strip()
    confidence = review.get("confidence")
    parts = []
    if verdict:
        parts.append(f"verdict={verdict}")
    if confidence is not None:
        parts.append(f"confidence={confidence}")
    for key in ["mechanism_assessment", "target_assessment", "adoption_advice", "suggested_next_action"]:
        value = str(review.get(key) or "").strip()
        if value:
            parts.append(f"{key}: {value}")
    gaps = review.get("evidence_gaps")
    if isinstance(gaps, list) and gaps:
        parts.append("evidence_gaps: " + "; ".join(str(item) for item in gaps[:4]))
    if not parts:
        return ""
    return _truncate_feedback("CGM repair_review: " + " | ".join(parts), 1200)


def _retry_patch_generation_after_validation_failure(env, payload: dict[str, object], reason: str, patch_preview: dict[str, object]):
    if not _should_retry_patch_generation(reason):
        return None
    retry_payload = _payload_with_patch_retry_feedback(payload, reason, patch_preview)
    try:
        raw = env.cgm.generate_patch(retry_payload)
        patch = parse_cgm_output(raw)
        patch, normalization_notes = normalize_patch_with_runtime(env.runtime, patch)
    except Exception as exc:
        retry_preview = dict(patch_preview)
        retry_preview["internal_retry_error"] = str(exc)
        return None
    retry_preview = _patch_preview(patch)
    retry_preview["internal_retry_from"] = reason
    if normalization_notes:
        retry_preview["normalization_notes"] = normalization_notes
    decision = validate_patch_with_runtime(env.runtime, patch, env.config.max_patch_edits, env.config.allow_test_changes)
    if not decision.ok:
        retry_preview["internal_retry_rejected_reason"] = decision.reason
    return patch, retry_preview, decision


def _retry_patch_generation_after_generated_syntax_failure(
    env,
    payload: dict[str, object],
    syntax_summary: str,
    patch_preview: dict[str, object],
):
    reason = f"generated patch failed Python syntax check after application and rollback: {syntax_summary}"
    retry_payload = _payload_with_patch_retry_feedback(payload, reason, patch_preview)
    try:
        raw = env.cgm.generate_patch(retry_payload)
        retry_cgm_response = _compact_cgm_response(raw)
        protocol_error = _cgm_output_protocol_error(raw)
        if protocol_error:
            retry_preview = dict(patch_preview)
            retry_preview["internal_retry_error"] = protocol_error
            return None
        patch = parse_cgm_output(raw)
        patch, normalization_notes = normalize_patch_with_runtime(env.runtime, patch)
    except Exception:
        return None
    retry_preview = _patch_preview(patch)
    retry_preview["internal_retry_from"] = reason
    if normalization_notes:
        retry_preview["normalization_notes"] = normalization_notes
    decision = validate_patch_with_runtime(env.runtime, patch, env.config.max_patch_edits, env.config.allow_test_changes)
    if not decision.ok:
        retry_preview["internal_retry_rejected_reason"] = decision.reason
    return patch, retry_preview, decision, retry_cgm_response


def _should_retry_patch_generation(reason: str) -> bool:
    lowered = reason.lower()
    return any(
        marker in lowered
        for marker in [
            "schema json embedded",
            "tuple/schema artifact",
            "standalone diff plus marker",
            "diff/conflict marker embedded",
            "collapses multi-line python span",
            "control-flow header",
        ]
    )


def _payload_with_patch_retry_feedback(payload: dict[str, object], reason: str, patch_preview: dict[str, object]) -> dict[str, object]:
    retry_payload = dict(payload)
    preview = json.dumps(_compact_result_value(patch_preview, 1800), ensure_ascii=False, sort_keys=True)
    extra = (
        "Previous generated patch was rejected during format validation or Python syntax checking.\n"
        f"Rejection reason: {reason}\n"
        f"Rejected patch preview: {preview}\n"
        "Regenerate a clean minimal patch object only. "
        "Do not include nested JSON/tool metadata inside new_text. "
        "If the intended change is one existing line, use start=end for that exact line and preserve indentation. "
        "If multiple lines are needed, rewrite a complete coherent block."
    )
    retry_payload["plan_text"] = (str(retry_payload.get("plan_text") or "").rstrip() + "\n\n" + extra).strip()
    return retry_payload


def _finish_repair(env, result: dict[str, object], memory_ids: list[str], target_nodes: list[str]) -> dict[str, object]:
    status = str(result.get("status") or "")
    failure_feedback = _repair_failure_feedback(result)
    if failure_feedback:
        result["failure_feedback"] = failure_feedback
    patch_preview = result.get("patch_preview") if isinstance(result.get("patch_preview"), dict) else None
    cgm_response = result.get("cgm_response") if isinstance(result.get("cgm_response"), dict) else None
    if patch_preview or status in {"patch_rejected", "syntax_failed", "test_failed", "infra_bug", "passed", "patch_proposed"}:
        attempt = {
            "status": status,
            "tool": result.get("tool"),
            "summary": result.get("summary"),
            "touched_paths": result.get("touched_paths"),
            "error_origin": result.get("error_origin"),
            "rolled_back": result.get("rolled_back"),
            "source_tree_state": result.get("source_tree_state"),
            "memory_node_ids": sorted(str(node_id) for node_id in memory_ids),
            "target_nodes": sorted(str(node_id) for node_id in target_nodes),
            "patch_preview": _compact_result_value(patch_preview, 2500) if patch_preview else None,
            "failure_feedback": failure_feedback,
        }
        env.repair_attempts.append(_compact_result_value(attempt, 3600))
        del env.repair_attempts[:-5]
    insight = _cgm_insight(result.get("cgm_response_raw"), cgm_response, patch_preview)
    if insight is None:
        insight = _cgm_insight(result.get("cgm_response"), cgm_response, patch_preview)
    if insight:
        env.cgm_insights.append(insight)
        del env.cgm_insights[:-5]
    env.repair_history.record_outcome(status, memory_ids, str(result.get("error_origin") or "") or None)
    env.last_repair_attempt = {
        "status": status,
        "error_origin": result.get("error_origin"),
        "source_tree_state": result.get("source_tree_state"),
        "rolled_back": result.get("rolled_back"),
        "memory_node_ids": sorted(str(node_id) for node_id in memory_ids),
        "target_nodes": sorted(str(node_id) for node_id in target_nodes),
        "selected_fix_contract_present": bool(
            isinstance(result.get("cgm_payload"), dict)
            and result["cgm_payload"].get("selected_fix_contract_present")
        ),
    }
    if failure_feedback:
        env.last_repair_attempt["failure_feedback"] = failure_feedback
    return result


def _repair_failure_feedback(result: dict[str, object]) -> dict[str, object] | None:
    status = str(result.get("status") or "")
    if status not in {"patch_rejected", "syntax_failed", "test_failed", "infra_bug"}:
        return None
    test_summary = result.get("test_summary") if isinstance(result.get("test_summary"), dict) else None
    return {
        "failed_patch": _compact_result_value(result.get("patch_preview"), 2500),
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
    return _truncate_feedback("\n".join(parts), 1800)


def _compact_test_summary_for_repair(value) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    compact: dict[str, object] = {
        "status": value.get("status"),
        "returncode": value.get("returncode"),
        "resolved": value.get("resolved"),
        "tests_status": value.get("tests_status"),
        "implementation_frames": value.get("implementation_frames"),
        "runtime_observations": value.get("runtime_observations"),
        "command_omitted_for_benchmark_hygiene": value.get("command_omitted_for_benchmark_hygiene"),
        "parser_error": value.get("parser_error"),
    }
    command = value.get("command")
    if command is not None:
        compact["command"] = _truncate_middle(str(command), 700)
    excerpt = str(value.get("excerpt") or "").strip()
    if excerpt:
        compact["excerpt"] = _truncate_middle(excerpt, 1200)
    return compact


def _compact_result_value(value, limit: int):
    if value is None:
        return None
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(value)
    if len(text) <= limit:
        return value
    return text[:limit] + f"...<truncated {len(text) - limit} chars>"


def _truncate_feedback(text: str, limit: int = 900) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit < 40:
        return text[:limit]
    head = limit // 2
    tail = limit - head
    return text[:head] + f"\n...<truncated {len(text) - limit} chars>...\n" + text[-tail:]
