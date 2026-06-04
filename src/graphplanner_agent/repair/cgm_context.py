from __future__ import annotations

import json
import re

from graphplanner_agent.datasets import TaskSpec
from graphplanner_agent.graph.guards import is_test_path
from graphplanner_agent.graph.schema import GraphNode, RepoGraph
from graphplanner_agent.memory.cgm_memory import CgmMemory


def build_cgm_payload(
    task: TaskSpec,
    graph: RepoGraph,
    memory: CgmMemory,
    intent_analysis: str,
    confidence: float | None,
    failure_summary: dict[str, object] | None,
    prior_feedback: str | None,
    max_edits: int,
    target_node_ids: list[str] | None = None,
) -> dict[str, object]:
    nodes = list(memory.nodes.values())
    memory_node_ids = {node.id for node in nodes}
    target_ids = [str(node_id).strip() for node_id in (target_node_ids or []) if str(node_id).strip()]
    target_id_set = set(target_ids)
    target_nodes = [node for node in nodes if node.id in target_id_set] if target_id_set else nodes
    dispatch_facts = _memory_dispatch_tables(graph, memory)
    hydrated_nodes = [(node, _raw_source_text(node.text or "", node.start_line)) for node in nodes if node.text]
    snippet_nodes = _snippet_nodes(hydrated_nodes, target_id_set)
    snippets = [
        {
            "id": node.id,
            "kind": node.kind,
            "name": node.name,
            "path": node.path,
            "start": node.start_line,
            "end": node.end_line,
            "role": "target" if (not target_id_set or node.id in target_id_set) else "context",
            "is_target": not target_id_set or node.id in target_id_set,
            "text": text,
            "lines": text.splitlines(),
            "snippet_lines": text.splitlines(),
            "numbered_text": _numbered_text(text, node.start_line),
        }
        for node, text in snippet_nodes
    ]
    serialized_code = [
        {
            "id": node.id,
            "path": node.path,
            "start": node.start_line,
            "end": node.end_line,
            "kind": node.kind,
            "name": node.name,
            "numbered_text": _numbered_text(text, node.start_line),
        }
        for node, text in hydrated_nodes
    ]
    graph_nodes = _graph_context_nodes(graph, memory)
    context_node_ids = {str(node["id"]) for node in graph_nodes}
    graph_edges = [
        {"source": edge.source, "target": edge.target, "type": edge.type, "edgeType": edge.type}
        for edge in graph.edges
        if edge.source in context_node_ids and edge.target in context_node_ids
    ]
    constraints = {
        "max_edits": max_edits,
        "implementation_only": True,
        "no_test_changes": True,
        "output_format": "unified_diff",
    }
    plan_text = _plan_text(
        intent_analysis,
        confidence,
        memory,
        prior_feedback,
        failure_summary,
        task.fail_to_pass,
        dispatch_facts,
        target_nodes,
    )
    return {
        "issue": {
            "id": task.task_id,
            "title": task.issue_title,
            "body": _issue_body(task),
            "repo": _repo_name(task),
            "language": "python",
        },
        "plan": {
            "targets": [
                {
                    "id": node.id,
                    "path": node.path,
                    "start": node.start_line,
                    "end": node.end_line,
                    "why": "planner selected repair target",
                }
                for node in target_nodes
            ],
            "planner_confidence": confidence,
        },
        "plan_text": plan_text,
        # Compatibility with the old graph_planner CodeFuse-CGM service.  The
        # service renders this as [Instruction].  Keep it as a short output
        # contract; repair evidence should come from issue, graph, and snippets.
        "prompt": _cgm_prompt(max_edits, target_nodes),
        "answer": "",
        "task": "issue_fix",
        "repo": _repo_name(task),
        "language": "python",
        "subgraph": graph_nodes,
        "graph": {
            "reponame": _repo_name(task),
            "language": "python",
            "nodes": graph_nodes,
            "edges": graph_edges,
            "adjacency_edges": graph_edges,
            "adjacency_list": _adjacency_list(graph_edges),
        },
        "snippets": snippets,
        "serialized_code": serialized_code,
        "code_facts": {"dispatch_tables": dispatch_facts},
        "metadata": {
            "constraints": constraints,
            "output_format": "unified_diff",
            "planner_confidence": confidence,
            "target_node_count": len(target_nodes),
            "serialized_code_count": len(serialized_code),
            "snippet_target_count": sum(1 for snippet in snippets if snippet.get("is_target")),
            "snippet_context_count": sum(1 for snippet in snippets if not snippet.get("is_target")),
            "graph_profile": {"node_count": len(graph_nodes), "edge_count": len(graph_edges)},
        },
    }


def _snippet_nodes(
    hydrated_nodes: list[tuple[GraphNode, str]],
    target_id_set: set[str],
) -> list[tuple[GraphNode, str]]:
    if not target_id_set:
        return hydrated_nodes
    targets = [item for item in hydrated_nodes if item[0].id in target_id_set]
    contexts = [item for item in hydrated_nodes if item[0].id not in target_id_set]
    ordered = targets + contexts
    return ordered if ordered else hydrated_nodes


def summarize_cgm_payload(payload: dict[str, object]) -> dict[str, object]:
    graph = payload.get("graph", {})
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    edges = graph.get("edges", []) if isinstance(graph, dict) else []
    snippets = payload.get("snippets", [])
    serialized_code = payload.get("serialized_code", [])
    snippet_target_count = 0
    snippet_context_count = 0
    if isinstance(snippets, list):
        snippet_target_count = sum(1 for snippet in snippets if isinstance(snippet, dict) and snippet.get("is_target"))
        snippet_context_count = sum(1 for snippet in snippets if isinstance(snippet, dict) and not snippet.get("is_target"))
    issue = payload.get("issue", {})
    plan = payload.get("plan", {})
    targets = plan.get("targets", []) if isinstance(plan, dict) else []
    target_summaries: list[str] = []
    if isinstance(targets, list):
        for target in targets[:8]:
            if not isinstance(target, dict):
                continue
            path = str(target.get("path") or "").strip()
            start = target.get("start")
            end = target.get("end")
            if path:
                target_summaries.append(f"{path}:{start}-{end}")
    return {
        "issue_id": issue.get("id") if isinstance(issue, dict) else None,
        "issue_title_present": bool(issue.get("title")) if isinstance(issue, dict) else False,
        "issue_body_chars": len(str(issue.get("body") or "")) if isinstance(issue, dict) else 0,
        "plan_target_count": len(targets) if isinstance(targets, list) else 0,
        "plan_targets": target_summaries,
        "planner_confidence": plan.get("planner_confidence") if isinstance(plan, dict) else None,
        "node_count": len(nodes) if isinstance(nodes, list) else 0,
        "edge_count": len(edges) if isinstance(edges, list) else 0,
        "snippet_count": len(snippets) if isinstance(snippets, list) else 0,
        "snippet_target_count": snippet_target_count,
        "snippet_context_count": snippet_context_count,
        "serialized_code_count": len(serialized_code) if isinstance(serialized_code, list) else 0,
        "has_prompt": bool(str(payload.get("prompt") or "").strip()),
        "prompt_chars": len(str(payload.get("prompt") or "")),
        "plan_text_chars": len(str(payload.get("plan_text") or "")),
        "selected_fix_contract_present": isinstance(payload.get("selected_fix_contract"), dict)
        and any(payload["selected_fix_contract"].get(key) for key in ("must", "must_not", "acceptance_criteria")),
        "has_subgraph": isinstance(payload.get("subgraph"), list) and bool(payload.get("subgraph")),
        "has_adjacency_list": bool(graph.get("adjacency_list")) if isinstance(graph, dict) else False,
        "dispatch_table_count": len(payload.get("code_facts", {}).get("dispatch_tables", []))
        if isinstance(payload.get("code_facts"), dict)
        else 0,
        "node_paths": sorted({str(node.get("path", "")) for node in nodes if isinstance(node, dict)}),
    }


def validate_cgm_payload(payload: dict[str, object]) -> list[str]:
    errors: list[str] = []
    graph = payload.get("graph")
    if not isinstance(graph, dict):
        return ["payload.graph must be an object"]
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        errors.append("payload.graph.nodes must contain at least one memory node")
        return errors
    if not str(payload.get("prompt") or "").strip():
        errors.append("payload.prompt must contain the CGM output contract")
    if not isinstance(payload.get("subgraph"), list) or not payload.get("subgraph"):
        errors.append("payload.subgraph must contain CodeFuse-CGM compatibility nodes")
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"node {idx} must be an object")
            continue
        if not node.get("id") or not node.get("path"):
            errors.append(f"node {idx} is missing id/path")
        if node.get("is_memory_target") and not str(node.get("text") or "").strip():
            errors.append(f"node {node.get('id', idx)} is missing hydrated code text")
    snippets = payload.get("snippets")
    if not isinstance(snippets, list) or not snippets:
        errors.append("payload.snippets must contain hydrated memory code")
    for idx, snippet in enumerate(snippets if isinstance(snippets, list) else []):
        if not isinstance(snippet, dict):
            errors.append(f"snippet {idx} must be an object")
            continue
        if not isinstance(snippet.get("lines"), list) or not snippet.get("lines"):
            errors.append(f"snippet {idx} is missing line-numberable lines")
        if not str(snippet.get("numbered_text") or "").strip():
            errors.append(f"snippet {idx} is missing numbered_text")
    serialized = payload.get("serialized_code")
    if not isinstance(serialized, list) or not serialized:
        errors.append("payload.serialized_code must contain numbered memory code")
    return errors


def _graph_context_nodes(graph: RepoGraph, memory: CgmMemory, max_nodes: int = 48) -> list[dict[str, object]]:
    memory_ids = set(memory.nodes)
    ordered_ids: list[str] = list(memory.nodes)
    for node_id in list(memory_ids):
        for neighbor in graph.neighbors(node_id):
            if neighbor.id in memory_ids or neighbor.id in ordered_ids:
                continue
            if not neighbor.path or is_test_path(neighbor.path):
                continue
            ordered_ids.append(neighbor.id)
            if len(ordered_ids) >= max_nodes:
                break
        if len(ordered_ids) >= max_nodes:
            break

    out: list[dict[str, object]] = []
    for node_id in ordered_ids:
        node = memory.nodes.get(node_id) or graph.nodes.get(node_id)
        if node is None or not node.path or is_test_path(node.path):
            continue
        is_memory = node.id in memory_ids
        out.append(
            {
                "id": node.id,
                "type": node.kind,
                "kind": node.kind,
                "nodeType": node.kind.title(),
                "name": node.name,
                "path": node.path,
                "start_line": node.start_line,
                "end_line": node.end_line,
                "text": _raw_source_text(node.text or "", node.start_line),
                "is_memory_target": is_memory,
            }
        )
    return out


def _memory_dispatch_tables(graph: RepoGraph, memory: CgmMemory) -> list[dict[str, object]]:
    from graphplanner_agent.env.evidence import dispatch_tables

    read_node_ids = {node_id for node_id, node in memory.nodes.items() if node.has_code}
    tables: list[dict[str, object]] = []
    for node in memory.nodes.values():
        if not node.text:
            continue
        tables.extend(dispatch_tables(graph, node, node.text, read_node_ids=read_node_ids))
    return tables


def _adjacency_list(edges: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = {}
    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if not source or not target:
            continue
        out.setdefault(source, []).append({"target": target, "type": str(edge.get("type") or edge.get("edgeType") or "RELATED")})
    return out


def _numbered_text(text: str, start_line: int) -> str:
    lines = (text or "").splitlines()
    return "\n".join(f"{start_line + idx:>5}: {line}" for idx, line in enumerate(lines))


def _raw_source_text(text: str, start_line: int) -> str:
    """Return raw source, stripping display-only line-number prefixes if present."""

    value = text or ""
    lines = value.splitlines()
    if not lines:
        return value
    stripped: list[str] = []
    numbered = 0
    expected = max(1, int(start_line or 1))
    for idx, line in enumerate(lines):
        match = re.match(r"^\s*(\d+): ?(.*)$", line)
        if match and int(match.group(1)) == expected + idx:
            numbered += 1
            stripped.append(match.group(2))
        else:
            stripped.append(line)
    nonblank = sum(1 for line in lines if line.strip())
    if numbered and numbered >= max(1, nonblank // 2):
        suffix = "\n" if value.endswith("\n") else ""
        return "\n".join(stripped) + suffix
    return value


def _cgm_prompt(max_edits: int, target_nodes: list[object]) -> str:
    paths = []
    for node in target_nodes:
        path = str(getattr(node, "path", "") or "").strip()
        if path and path not in paths:
            paths.append(path)
    guide_lines = [
        "Generate a minimal implementation patch for the issue using the provided target code and graph context.",
        "Return exactly one complete unified diff and nothing else.",
        "Use only editable target files and exact paths from snippets.",
        "Keep the patch minimal and syntactically valid.",
        "Do not output markdown, prose, JSON, shell commands, logs, tests, reproduction scripts, new files, deletes, or renames.",
        "Source snippets and graph node text are authoritative.",
        f"Maximum edits: {max_edits}.",
    ]
    if paths:
        guide_lines.append("Editable target files: " + ", ".join(paths[:6]) + ".")
    return "\n".join(guide_lines).strip()


def _repo_name(task: TaskSpec) -> str:
    repo = task.metadata.get("repo") if isinstance(task.metadata, dict) else None
    if isinstance(repo, str) and repo.strip():
        return repo.strip().split("/")[-1]
    return task.repo_path.name or "repo"


def _issue_body(task: TaskSpec) -> str:
    sections = [f"Title: {task.issue_title}".strip(), _clean_issue_text(task.issue_body)]
    return "\n\n".join(section for section in sections if section)


def _clean_issue_text(text: str, *, max_chars: int = 5000) -> str:
    """Keep user-visible bug evidence and remove benchmark/noisy boilerplate."""

    value = str(text or "")
    value = re.sub(r"<!--[\s\S]*?-->", "", value)
    value = re.sub(
        r"(?ims)^###\s*(System Details|Versions?)\b[\s\S]*?(?=^###\s+|\Z)",
        "",
        value,
    )
    value = re.sub(r"(?im)^Base commit:\s*[0-9a-f]{7,40}\s*$", "", value)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    if len(value) > max_chars:
        head = max_chars // 2
        tail = max_chars - head
        value = value[:head].rstrip() + f"\n...<issue truncated {len(value) - max_chars} chars>...\n" + value[-tail:].lstrip()
    return value


def _plan_text(
    planner_brief: str,
    confidence: float | None,
    memory: CgmMemory,
    prior_feedback: str | None,
    failure_summary: dict[str, object] | None,
    fail_to_pass: list[str],
    dispatch_facts: list[dict[str, object]],
    target_nodes: list[object],
) -> str:
    sections: list[str] = []
    if target_nodes:
        sections.append("Suggested starting point:\n" + _format_target_nodes(target_nodes))
    if planner_brief.strip():
        confidence_line = ""
        if confidence is not None:
            confidence_line = f"Planner confidence in this intent analysis: {confidence:.2f}\n"
        sections.append(
            "Planner intent analysis (advisory; issue text and source snippets are authoritative):\n"
            + confidence_line
            + _clip_text(planner_brief.strip(), 2200)
        )
    if dispatch_facts:
        sections.append("Relevant structured source facts:\n" + _format_dispatch_facts(dispatch_facts))
    if memory.notes:
        sections.append("Useful memory notes:\n" + "\n".join(f"- {note}" for note in memory.notes[-2:]))
    if prior_feedback:
        sections.append("Avoid repeating prior failed patch pattern:\n" + prior_feedback.strip())
    return "\n\n".join(section for section in sections if section.strip())


def _clip_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    return text[:head].rstrip() + f"\n...<planner brief truncated {len(text) - limit} chars>...\n" + text[-tail:].lstrip()


def _format_target_nodes(target_nodes: list[object]) -> str:
    lines: list[str] = []
    for node in target_nodes[:8]:
        path = str(getattr(node, "path", "") or "").strip()
        name = str(getattr(node, "name", "") or "").strip()
        kind = str(getattr(node, "kind", "") or "").strip()
        start = getattr(node, "start_line", None)
        end = getattr(node, "end_line", None)
        if not path:
            continue
        label = f"{path}:{start}-{end}"
        details = " ".join(part for part in [kind, name] if part)
        if details:
            label += f" ({details})"
        lines.append(f"- {label}")
    return "\n".join(lines)


def _format_dispatch_facts(dispatch_facts: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for table in dispatch_facts[:6]:
        name = str(table.get("name") or "<table>")
        for entry in table.get("entries", []) if isinstance(table.get("entries"), list) else []:
            if not isinstance(entry, dict):
                continue
            lines.append(f"- {name}[{entry.get('key')!r}] -> {entry.get('target')}")
            if len(lines) >= 24:
                return "\n".join(lines)
    return "\n".join(lines)


def _format_failure_summary(summary: dict[str, object], excerpt_limit: int = 1200) -> str:
    compact = {
        "status": summary.get("status"),
        "command": summary.get("command"),
        "returncode": summary.get("returncode"),
        "resolved": summary.get("resolved"),
        "tests_status": summary.get("tests_status"),
        "implementation_frames": summary.get("implementation_frames"),
        "runtime_observations": summary.get("runtime_observations"),
        "command_omitted_for_benchmark_hygiene": summary.get("command_omitted_for_benchmark_hygiene"),
        "parser_error": summary.get("parser_error"),
    }
    lines = [json.dumps(compact, ensure_ascii=False, sort_keys=True)]
    excerpt = str(summary.get("excerpt") or "").strip()
    if excerpt:
        if len(excerpt) > excerpt_limit:
            excerpt = excerpt[-excerpt_limit:]
        lines.append("Output excerpt:\n" + excerpt)
    return "\n".join(lines)


def _format_failure_brief(summary: dict[str, object]) -> str:
    compact = {
        "status": summary.get("status"),
        "returncode": summary.get("returncode"),
        "resolved": summary.get("resolved"),
        "tests_status": summary.get("tests_status"),
        "implementation_frames": summary.get("implementation_frames"),
        "runtime_observations": summary.get("runtime_observations"),
        "command_omitted_for_benchmark_hygiene": summary.get("command_omitted_for_benchmark_hygiene"),
        "parser_error": summary.get("parser_error"),
    }
    return json.dumps(compact, ensure_ascii=False, sort_keys=True)
