from __future__ import annotations

import ast
import keyword
import re
import textwrap
from dataclasses import dataclass

from graphplanner_agent.graph.guards import is_test_path
from graphplanner_agent.graph.read import line_numbered, read_node_from_runtime
from graphplanner_agent.graph.schema import GraphNode, RepoGraph


ASSIGNMENT_KINDS = {"assignment", "module_assignment"}
SMALL_GRAIN_KINDS = {"class", "function", "method"} | ASSIGNMENT_KINDS
PUBLIC_FIND_TYPES = {"file", "class", "function", "method", "assignment", "any"}
FIND_TYPE_ALIASES = {
    "module_assignment": "assignment",
}
LEAD_KINDS = SMALL_GRAIN_KINDS
MAX_FIND_PREVIEW_LINES = 30
MAX_FIND_PREVIEW_CHARS = 1200
MAX_LEADS = 8
MAX_TOP_SYMBOLS = 12
MAX_DISPATCH_TABLES = 4
MAX_DISPATCH_ENTRIES = 24
MAX_DISPATCH_CONTEXT_FACTS = 4
MAX_CONSUMER_CANDIDATES = 6
MAX_RELATED_CODE_LINES = 80
MAX_RELATED_CODE_CHARS = 5000

_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_STOP_IDENTIFIERS = set(keyword.kwlist) | {
    "False",
    "None",
    "True",
    "self",
    "cls",
    "args",
    "kwargs",
    "np",
    "numpy",
    "pd",
    "typing",
}


@dataclass(slots=True)
class CodePreview:
    text: str
    start_line: int
    end_line: int
    line_numbered: str
    truncated: bool = False
    error: str | None = None
    source_node: GraphNode | None = None


def node_brief(node: GraphNode) -> dict[str, object]:
    return {
        "id": node.id,
        "kind": public_node_kind(node.kind),
        "name": node.name,
        "path": node.path,
        "lines": [node.start_line, node.end_line],
    }


def public_node_kind(kind: str) -> str:
    if kind in ASSIGNMENT_KINDS:
        return "assignment"
    return kind


def normalize_find_type(find_type: str) -> str:
    value = str(find_type or "any").strip() or "any"
    return FIND_TYPE_ALIASES.get(value, value)


def is_small_grain_node(node: GraphNode) -> bool:
    return node.kind in SMALL_GRAIN_KINDS and not is_test_path(node.path)


def read_node_for_evidence(runtime, node: GraphNode, view: str = "body") -> GraphNode:
    if view == "body" and node.kind in ASSIGNMENT_KINDS and node.end_line - node.start_line <= 2:
        return read_node_from_runtime(runtime, node, f"file_window:{node.start_line}-{node.start_line + 30}")
    return read_node_from_runtime(runtime, node, view)


def preview_node_code(runtime, node: GraphNode, *, max_lines: int = MAX_FIND_PREVIEW_LINES, max_chars: int = MAX_FIND_PREVIEW_CHARS) -> CodePreview:
    try:
        if node.text and node.text.strip():
            source = node
        else:
            source = read_node_for_evidence(runtime, node, "body")
    except Exception as exc:
        return CodePreview(text="", start_line=node.start_line, end_line=node.end_line, line_numbered="", error=str(exc))

    text, end_line, truncated = _clip_text(source.text or "", source.start_line, max_lines=max_lines, max_chars=max_chars)
    preview_node = GraphNode(
        id=source.id,
        kind=source.kind,
        name=source.name,
        path=source.path,
        start_line=source.start_line,
        end_line=end_line,
        text=text,
        preview=source.preview,
        parent_id=source.parent_id,
    )
    return CodePreview(
        text=text,
        start_line=source.start_line,
        end_line=end_line,
        line_numbered=line_numbered(preview_node),
        truncated=truncated,
        source_node=source,
    )


def top_symbols_for_file(graph: RepoGraph, path: str, *, limit: int = MAX_TOP_SYMBOLS) -> list[dict[str, object]]:
    symbols = [
        node
        for node in graph.nodes.values()
        if node.path == path and node.kind in SMALL_GRAIN_KINDS and not is_test_path(node.path)
    ]
    symbols.sort(key=lambda node: (node.start_line, node.end_line - node.start_line, node.name))
    return [node_brief(node) for node in symbols[:limit]]


def local_symbol_references(
    graph: RepoGraph,
    source_node: GraphNode,
    text: str,
    *,
    read_node_ids: set[str] | None = None,
    limit: int = MAX_LEADS,
) -> list[dict[str, object]]:
    if not text.strip():
        return []
    identifier_positions: dict[str, int] = {}
    for match in _IDENTIFIER_RE.finditer(text):
        item = match.group(0)
        if item in _STOP_IDENTIFIERS or item.startswith("__"):
            continue
        identifier_positions.setdefault(item, match.start())
    if not identifier_positions:
        return []
    read_node_ids = read_node_ids or set()
    own_names = {source_node.name, source_node.name.rsplit(".", 1)[-1]}
    candidates: list[tuple[int, int, GraphNode, str]] = []
    for node in graph.nodes.values():
        if node.id == source_node.id or node.path != source_node.path:
            continue
        if node.kind not in LEAD_KINDS or is_test_path(node.path):
            continue
        names = {node.name, node.name.rsplit(".", 1)[-1]}
        names = {name for name in names if name and name not in own_names}
        matched = sorted(names & set(identifier_positions), key=lambda item: identifier_positions[item])
        if not matched:
            continue
        symbol = matched[0]
        candidates.append((identifier_positions[symbol], node.start_line, node, symbol))

    candidates.sort(key=lambda item: (item[0], item[1], item[2].name))
    references = []
    seen: set[str] = set()
    for appearance_index, _, node, symbol in candidates:
        if node.id in seen:
            continue
        seen.add(node.id)
        brief = node_brief(node)
        brief.update(
            {
                "relation": "symbol_reference",
                "symbol": symbol,
                "appearance_index": appearance_index,
                "read_status": "read" if node.id in read_node_ids else "unread",
                "source": "referenced in the current implementation snippet",
            }
        )
        references.append(brief)
        if len(references) >= limit:
            break
    return references


def dispatch_tables(
    graph: RepoGraph,
    source_node: GraphNode,
    text: str,
    *,
    read_node_ids: set[str] | None = None,
    table_limit: int = MAX_DISPATCH_TABLES,
    entry_limit: int = MAX_DISPATCH_ENTRIES,
) -> list[dict[str, object]]:
    """Extract table-driven dispatch facts from dict assignments in a snippet."""
    if not text.strip():
        return []
    try:
        tree = ast.parse(textwrap.dedent(text).strip() + "\n")
    except SyntaxError:
        return []

    read_node_ids = read_node_ids or set()
    tables: list[dict[str, object]] = []
    for stmt in tree.body:
        targets, value = _assignment_targets_and_value(stmt)
        if not targets or not isinstance(value, ast.Dict):
            continue
        entries = _dispatch_entries(graph, source_node, value, read_node_ids=read_node_ids, limit=entry_limit)
        if not entries:
            continue
        for target in targets:
            tables.append(
                {
                    "relation": "dispatch_table",
                    "name": target,
                    "source": "dict assignment in the current implementation snippet",
                    "source_node": node_brief(source_node),
                    "entries": entries,
                }
            )
            if len(tables) >= table_limit:
                return tables
    return tables


def dispatch_relationship_context(
    graph: RepoGraph,
    runtime,
    source_node: GraphNode,
    text: str,
    *,
    issue_text: str = "",
    read_node_ids: set[str] | None = None,
    fact_limit: int = MAX_DISPATCH_CONTEXT_FACTS,
) -> tuple[list[dict[str, object]], list[GraphNode]]:
    """Return issue-bound dispatcher/consumer facts plus preview nodes.

    The facts are intentionally evidence-shaped rather than target-shaped:
    they say which keys/values appear to drive dispatch and which implementation
    nodes should be read next, but they do not mark any candidate as the repair
    target.
    """
    if not text.strip():
        return [], []
    issue_literals = _issue_keyword_literals(issue_text)
    issue_param_names = _issue_param_names(issue_text)
    try:
        tree = ast.parse(textwrap.dedent(text).strip() + "\n")
    except SyntaxError:
        return [], []

    read_node_ids = read_node_ids or set()
    facts: list[dict[str, object]] = []
    related_nodes: list[GraphNode] = []
    for call in [node for node in ast.walk(tree) if isinstance(node, ast.Call)]:
        call_name = _call_name(call.func) or ""
        if not _looks_like_dispatch_call(call_name, call, issue_literals, issue_param_names):
            continue
        key_candidates = _call_dispatch_key_candidates(call, issue_literals, issue_param_names)
        if not key_candidates and issue_literals and any(keyword.arg is None for keyword in call.keywords):
            key_candidates = [
                {
                    "key": key,
                    "values": values,
                    "status": "candidate_from_issue_forwarded_via_kwargs",
                    "evidence": f"issue/repro names {key!r}; current call forwards **kwargs",
                }
                for key, values in sorted(issue_literals.items())
            ]
        if not key_candidates:
            key_candidates = [
                {
                    "key": "unknown",
                    "values": [],
                    "status": "unverified_dispatch_key",
                    "evidence": "current call looks like a dispatcher, but no issue literal was bound to a key",
                }
            ]

        consumer_candidates: list[dict[str, object]] = []
        downranked_noise: list[dict[str, object]] = []
        for candidate in key_candidates:
            for value in candidate.get("values") or []:
                ranked, noise = _consumer_candidates_for_value(graph, source_node, str(value), issue_text, read_node_ids)
                consumer_candidates.extend(ranked)
                downranked_noise.extend(noise)
        consumer_candidates = _dedupe_dict_nodes(consumer_candidates, limit=MAX_CONSUMER_CANDIDATES)
        downranked_noise = _dedupe_dict_nodes(downranked_noise, limit=4)

        for candidate in consumer_candidates[:2]:
            node_id = str(candidate.get("id") or "")
            node = graph.nodes.get(node_id)
            if node is None or node_id == source_node.id:
                continue
            preview = preview_node_code(
                runtime,
                node,
                max_lines=MAX_RELATED_CODE_LINES,
                max_chars=MAX_RELATED_CODE_CHARS,
            )
            if preview.error or not preview.text.strip():
                candidate["code_preview_error"] = preview.error or "empty preview"
                continue
            candidate["code"] = preview.line_numbered
            candidate["code_preview_lines"] = [preview.start_line, preview.end_line]
            candidate["code_preview_truncated"] = preview.truncated
            related_nodes.append(
                GraphNode(
                    id=node.id,
                    kind=node.kind,
                    name=node.name,
                    path=node.path,
                    start_line=node.start_line,
                    end_line=node.end_line,
                    text=preview.text,
                    preview=node.preview,
                    parent_id=node.parent_id,
                )
            )

        facts.append(
            {
                "relation": "dispatcher_context",
                "source_node": node_brief(source_node),
                "call": _unparse(call) or call_name,
                "call_name": call_name,
                "dispatcher_status": _dispatcher_status(call_name, call),
                "dispatch_key_candidates": key_candidates,
                "consumer_candidates": consumer_candidates,
                "downranked_noise": downranked_noise,
                "planner_guidance": (
                    "Treat these as relation facts and next-read candidates, not final targets. "
                    "If a wrapper/dispatcher is read, close the chain by reading the actual consumer before repair."
                ),
            }
        )
        if len(facts) >= fact_limit:
            break
    return facts, related_nodes


def _clip_text(text: str, start_line: int, *, max_lines: int, max_chars: int) -> tuple[str, int, bool]:
    lines = text.splitlines()
    truncated = len(lines) > max_lines or len(text) > max_chars
    kept: list[str] = []
    char_count = 0
    for line in lines[:max_lines]:
        projected = char_count + len(line) + 1
        if projected > max_chars and kept:
            truncated = True
            break
        if projected > max_chars:
            kept.append(line[: max(0, max_chars - char_count)])
            truncated = True
            break
        kept.append(line)
        char_count = projected
    clipped = "\n".join(kept)
    if clipped:
        clipped += "\n"
    end_line = start_line + max(0, len(kept) - 1)
    return clipped, end_line, truncated


def _assignment_targets_and_value(stmt: ast.stmt) -> tuple[list[str], ast.AST | None]:
    if isinstance(stmt, ast.Assign):
        targets = [_target_name(target) for target in stmt.targets]
        return [target for target in targets if target], stmt.value
    if isinstance(stmt, ast.AnnAssign):
        target = _target_name(stmt.target)
        return ([target] if target else []), stmt.value
    return [], None


def _dispatch_entries(
    graph: RepoGraph,
    source_node: GraphNode,
    dict_node: ast.Dict,
    *,
    read_node_ids: set[str],
    limit: int,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for key_node, value_node in zip(dict_node.keys, dict_node.values):
        if key_node is None:
            continue
        key = _key_text(key_node)
        target = _target_name(value_node)
        if key is None or not target:
            continue
        target_node = _resolve_same_file_symbol(graph, source_node, target)
        entry: dict[str, object] = {
            "key": key,
            "target": target,
            "target_symbol": target.rsplit(".", 1)[-1],
        }
        if target_node is not None:
            entry.update(node_brief(target_node))
            entry["read_status"] = "read" if target_node.id in read_node_ids else "unread"
        else:
            entry["read_status"] = "unknown"
        entries.append(entry)
        if len(entries) >= limit:
            break
    return entries


def _issue_keyword_literals(issue_text: str) -> dict[str, list[str]]:
    text = issue_text or ""
    literals: dict[str, list[str]] = {}
    for key, value in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*['\"]([^'\"]+)['\"]", text):
        if _is_low_signal_issue_key(key):
            continue
        literals.setdefault(key, [])
        if value not in literals[key]:
            literals[key].append(value)
    return literals


def _issue_param_names(issue_text: str) -> set[str]:
    text = issue_text or ""
    names = set()
    for key in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=", text):
        if not _is_low_signal_issue_key(key):
            names.add(key)
    return names


def _is_low_signal_issue_key(key: str) -> bool:
    return key in {"self", "cls", "x", "y", "i", "j"} or key.startswith("_")


def _looks_like_dispatch_call(
    call_name: str,
    call: ast.Call,
    issue_literals: dict[str, list[str]],
    issue_param_names: set[str],
) -> bool:
    lowered = call_name.lower()
    if any(marker in lowered for marker in ("registry", "dispatch", "handler", "lookup", "operator")):
        return True
    if any(keyword.arg is None for keyword in call.keywords) and issue_param_names:
        if any(marker in lowered for marker in ("write", "read", "open", "load", "dump", "parse", "format")):
            return True
    if any(keyword.arg in issue_literals for keyword in call.keywords if keyword.arg):
        return any(marker in lowered for marker in ("write", "read", "open", "load", "dump", "parse"))
    return False


def _call_dispatch_key_candidates(
    call: ast.Call,
    issue_literals: dict[str, list[str]],
    issue_param_names: set[str],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for keyword_arg in call.keywords:
        if not keyword_arg.arg:
            continue
        if keyword_arg.arg in issue_literals:
            status = "verified_keyword_from_issue"
            values = issue_literals[keyword_arg.arg]
            evidence = f"issue/repro has {keyword_arg.arg}=... and current call passes keyword {keyword_arg.arg}"
        elif isinstance(keyword_arg.value, ast.Name) and keyword_arg.value.id in issue_literals:
            status = "verified_forwarded_issue_value"
            values = issue_literals[keyword_arg.value.id]
            evidence = f"current call passes {keyword_arg.arg}={keyword_arg.value.id}; issue/repro binds {keyword_arg.value.id}"
        else:
            continue
        out.append({"key": keyword_arg.arg, "values": values, "status": status, "evidence": evidence})
    if out:
        return out
    if any(keyword.arg is None for keyword in call.keywords):
        for key in sorted(issue_param_names & set(issue_literals)):
            out.append(
                {
                    "key": key,
                    "values": issue_literals[key],
                    "status": "candidate_from_issue_forwarded_via_kwargs",
                    "evidence": f"issue/repro has {key}=... and current call forwards **kwargs",
                }
            )
    return out


def _consumer_candidates_for_value(
    graph: RepoGraph,
    source_node: GraphNode,
    value: str,
    issue_text: str,
    read_node_ids: set[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    key = value.strip().lower()
    if not key or len(key) < 2:
        return [], []
    scored: list[tuple[float, GraphNode, list[str], bool]] = []
    issue_lower = (issue_text or "").lower()
    for node in graph.nodes.values():
        if node.id == source_node.id or node.kind not in SMALL_GRAIN_KINDS or is_test_path(node.path):
            continue
        hay_name = node.name.lower()
        hay_path = node.path.lower()
        hay = " ".join([hay_name, hay_path, node.preview.lower(), (node.text or "").lower()])
        if key not in hay:
            continue
        score = 0.0
        reasons: list[str] = []
        path_base = hay_path.rsplit("/", 1)[-1]
        short_name = hay_name.rsplit(".", 1)[-1]
        if path_base == f"{key}.py":
            score += 8.0
            reasons.append(f"path basename matches dispatch value {value!r}")
        if short_name == key or hay_name == key:
            score += 7.0
            reasons.append(f"symbol name matches dispatch value {value!r}")
        if hay_name.endswith(".write") or short_name == "write":
            score += 4.0
            reasons.append("writer-like method")
        if "write" in issue_lower and "write" in hay_name:
            score += 2.0
            reasons.append("issue mentions write and candidate is a write symbol")
        if "format" in issue_lower and ("format" in hay or path_base == f"{key}.py"):
            score += 1.5
            reasons.append("format-related issue and candidate has format/path evidence")
        noise = _looks_like_lexical_noise(node, key, issue_lower)
        if noise:
            score -= 8.0
            reasons.append(noise)
        if score <= 0:
            continue
        scored.append((score, node, reasons, bool(noise)))
    scored.sort(key=lambda item: (item[0], item[1].kind in {"method", "function", "class"}, -len(item[1].path)), reverse=True)
    candidates: list[dict[str, object]] = []
    noise: list[dict[str, object]] = []
    for score, node, reasons, is_noise in scored:
        item = node_brief(node)
        item.update(
            {
                "relation": "candidate_consumer" if not is_noise else "downranked_noise",
                "dispatch_value": value,
                "score": round(score, 3),
                "reasons": reasons,
                "read_status": "read" if node.id in read_node_ids else "unread",
            }
        )
        if is_noise:
            noise.append(item)
        else:
            candidates.append(item)
    return candidates[:MAX_CONSUMER_CANDIDATES], noise[:4]


def _looks_like_lexical_noise(node: GraphNode, key: str, issue_lower: str) -> str:
    name = node.name.lower()
    path = node.path.lower()
    if f"_repr_{key}" in name or f"repr_{key}" in name:
        return "display representation helper; lexical match but not a write/dispatch consumer"
    if "validator" in path and "validator" not in issue_lower:
        return "validator path; lexical match but issue does not mention validation"
    if "pandas" in path and "pandas" not in issue_lower:
        return "pandas/display adapter; lexical match but issue does not mention pandas"
    return ""


def _dedupe_dict_nodes(items: list[dict[str, object]], *, limit: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in items:
        node_id = str(item.get("id") or "")
        key = node_id or jsonish_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def jsonish_key(item: dict[str, object]) -> str:
    return "|".join(f"{key}={item.get(key)}" for key in sorted(item))


def _dispatcher_status(call_name: str, call: ast.Call) -> str:
    lowered = call_name.lower()
    if "registry" in lowered:
        return "dispatcher_wrapper_registry_call"
    if any(keyword.arg is None for keyword in call.keywords):
        return "wrapper_forwards_kwargs"
    return "dispatcher_or_consumer_call"


def _key_text(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, str):
            return value
        if value is None or isinstance(value, (bool, int, float)):
            return repr(value)
    if isinstance(node, ast.Name):
        return node.id
    return _unparse(node)


def _target_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _target_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _call_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return _unparse(node)


def _resolve_same_file_symbol(graph: RepoGraph, source_node: GraphNode, target: str) -> GraphNode | None:
    target_names = {target, target.rsplit(".", 1)[-1]}
    matches = [
        node
        for node in graph.nodes.values()
        if node.path == source_node.path
        and node.id != source_node.id
        and node.kind in LEAD_KINDS
        and not is_test_path(node.path)
        and (node.name in target_names or node.name.rsplit(".", 1)[-1] in target_names)
    ]
    if not matches:
        return None
    matches.sort(key=lambda node: (node.name.rsplit(".", 1)[-1] not in target_names, abs(node.start_line - source_node.start_line), node.name))
    return matches[0]


def _unparse(node: ast.AST) -> str | None:
    try:
        return ast.unparse(node)
    except Exception:
        return None
