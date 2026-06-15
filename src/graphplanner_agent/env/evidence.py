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
MAX_VALUE_FLOW_FACTS = 8
MAX_RELATED_CODE_LINES = 80
MAX_RELATED_CODE_CHARS = 5000

_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_PY_DEF_RE = re.compile(r"\b(?:async\s+def|def)\s+[A-Za-z_][A-Za-z0-9_.]*\s*\((?P<params>[^)]*)\)")
_GO_FUNC_RE = re.compile(r"^\s*func\s+(?:\((?P<recv>[^)]*)\)\s*)?(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^)]*)\)")
_GO_KEYWORDS = {
    "break",
    "case",
    "chan",
    "const",
    "continue",
    "default",
    "defer",
    "else",
    "fallthrough",
    "for",
    "func",
    "go",
    "goto",
    "if",
    "import",
    "interface",
    "map",
    "package",
    "range",
    "return",
    "select",
    "struct",
    "switch",
    "type",
    "var",
}
_JS_TS_EXTS = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
_JS_KEYWORDS = {
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "default",
    "delete",
    "do",
    "else",
    "export",
    "extends",
    "finally",
    "for",
    "function",
    "if",
    "import",
    "in",
    "instanceof",
    "let",
    "new",
    "return",
    "switch",
    "throw",
    "try",
    "typeof",
    "var",
    "void",
    "while",
    "with",
    "yield",
}
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


def value_flow_context(
    graph: RepoGraph,
    runtime,
    source_node: GraphNode,
    text: str,
    *,
    read_node_ids: set[str] | None = None,
    limit: int = MAX_VALUE_FLOW_FACTS,
) -> tuple[list[dict[str, object]], list[GraphNode]]:
    """Expose caller/callee argument-to-parameter flow around a read node.

    This is intentionally best-effort. It does not try to prove runtime values;
    it only reports implementation facts visible in call expressions, such as
    ``walkDirTree(ctx, root, results)`` passing ``root`` into the callee's
    ``rootFolder`` parameter. These facts help the planner follow upstream and
    downstream data movement before committing repair evidence.
    """
    if not source_node.id or not text.strip():
        return [], []
    read_node_ids = read_node_ids or set()
    facts: list[dict[str, object]] = []
    related: list[GraphNode] = []
    seen_nodes: set[str] = set()
    outgoing = [
        edge
        for edge in graph.edges
        if edge.type == "CALLS" and edge.source == source_node.id and edge.target in graph.nodes
    ]
    incoming = [
        edge
        for edge in graph.edges
        if edge.type == "CALLS" and edge.target == source_node.id and edge.source in graph.nodes
    ]

    downstream_budget = max(1, limit // 2)
    upstream_budget = max(1, limit - downstream_budget)

    source_calls = _extract_calls_for_node(source_node, text)
    for edge in outgoing[: downstream_budget * 3]:
        callee = graph.nodes.get(edge.target)
        if callee is None or is_test_path(callee.path):
            continue
        params = _node_params(callee)
        calls = _matching_calls(source_calls, callee)
        if not calls:
            continue
        call = calls[0]
        fact = _value_flow_fact(
            relation="value_flow_downstream",
            source=source_node,
            target=callee,
            call=call,
            params=params,
            read_node_ids=read_node_ids,
        )
        facts.append(fact)
        if callee.id not in seen_nodes:
            seen_nodes.add(callee.id)
            related.append(callee)
        if len([f for f in facts if f.get("relation") == "value_flow_downstream"]) >= downstream_budget:
            break

    for edge in incoming[: upstream_budget * 3]:
        caller = graph.nodes.get(edge.source)
        if caller is None or is_test_path(caller.path):
            continue
        caller_text = _node_text_for_flow(runtime, caller)
        if not caller_text:
            continue
        caller_calls = _extract_calls_for_node(caller, caller_text)
        calls = _matching_calls(caller_calls, source_node)
        if not calls:
            continue
        call = calls[0]
        fact = _value_flow_fact(
            relation="value_flow_upstream",
            source=caller,
            target=source_node,
            call=call,
            params=_node_params(source_node),
            read_node_ids=read_node_ids,
        )
        facts.append(fact)
        if caller.id not in seen_nodes:
            seen_nodes.add(caller.id)
            related.append(caller)
        if len([f for f in facts if f.get("relation") == "value_flow_upstream"]) >= upstream_budget:
            break

    return facts[:limit], related


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


def _node_text_for_flow(runtime, node: GraphNode) -> str:
    if node.text and node.text.strip():
        return node.text
    try:
        hydrated = read_node_for_evidence(runtime, node, "body")
    except Exception:
        return ""
    return hydrated.text or ""


def _extract_calls_for_node(node: GraphNode, text: str) -> list[dict[str, object]]:
    if node.path.endswith(".go"):
        return _extract_go_calls(text)
    if node.path.endswith(_JS_TS_EXTS):
        return _extract_js_calls(text)
    return _extract_python_calls(text)


def _extract_python_calls(text: str) -> list[dict[str, object]]:
    if not text.strip():
        return []
    try:
        tree = ast.parse(textwrap.dedent(text).strip() + "\n")
    except SyntaxError:
        return []
    calls: list[dict[str, object]] = []
    for call in [item for item in ast.walk(tree) if isinstance(item, ast.Call)]:
        name = _call_name(call.func) or ""
        if not name:
            continue
        args = [_unparse(arg) or "" for arg in call.args]
        for keyword_arg in call.keywords:
            key = str(keyword_arg.arg or "").strip()
            value = _unparse(keyword_arg.value) or ""
            if key:
                args.append(f"{key}={value}")
            else:
                args.append("**" + value)
        calls.append(
            {
                "name": name,
                "base_name": name.rsplit(".", 1)[-1],
                "args": [arg for arg in args if arg],
                "line": int(getattr(call, "lineno", 0) or 0),
                "call_text": _call_expr_text(name, args),
            }
        )
    return calls


def _extract_go_calls(text: str) -> list[dict[str, object]]:
    clean = "\n".join(_strip_go_line_comment(line) for line in text.splitlines())
    calls: list[dict[str, object]] = []
    i = 0
    while i < len(clean):
        ch = clean[i]
        if not (ch == "_" or ch.isalpha()):
            i += 1
            continue
        if i > 0 and (clean[i - 1] == "_" or clean[i - 1].isalnum()):
            i += 1
            continue
        j = i + 1
        while j < len(clean) and (clean[j] == "_" or clean[j].isalnum()):
            j += 1
        first = clean[i:j]
        k = _skip_space(clean, j)
        owner = ""
        name = first
        if k < len(clean) and clean[k] == ".":
            k2 = _skip_space(clean, k + 1)
            if k2 < len(clean) and (clean[k2] == "_" or clean[k2].isalpha()):
                j2 = k2 + 1
                while j2 < len(clean) and (clean[j2] == "_" or clean[j2].isalnum()):
                    j2 += 1
                owner = first
                name = clean[k2:j2]
                k = _skip_space(clean, j2)
        if name in _GO_KEYWORDS:
            i = j
            continue
        if k >= len(clean) or clean[k] != "(":
            i = j
            continue
        if _go_identifier_is_declaration(clean, i):
            i = k + 1
            continue
        end = _matching_paren(clean, k)
        if end <= k:
            i = k + 1
            continue
        arg_text = clean[k + 1 : end]
        args = [arg.strip() for arg in _split_top_level_commas(arg_text) if arg.strip()]
        line = clean.count("\n", 0, i) + 1
        call_name = f"{owner}.{name}" if owner else name
        calls.append(
            {
                "name": call_name,
                "base_name": name,
                "owner": owner,
                "args": args,
                "line": line,
                "call_text": _call_expr_text(call_name, args),
            }
        )
        i = end + 1
    return calls


def _extract_js_calls(text: str) -> list[dict[str, object]]:
    clean = "\n".join(_strip_go_line_comment(line) for line in text.splitlines())
    calls: list[dict[str, object]] = []
    ident_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_$")
    i = 0
    while i < len(clean):
        ch = clean[i]
        if not (ch == "_" or ch == "$" or ch.isalpha()):
            i += 1
            continue
        if i > 0 and clean[i - 1] in ident_chars:
            i += 1
            continue
        j = i + 1
        while j < len(clean) and clean[j] in ident_chars:
            j += 1
        first = clean[i:j]
        k = _skip_space(clean, j)
        owner = ""
        name = first
        if k < len(clean) and clean[k] == ".":
            k2 = _skip_space(clean, k + 1)
            if k2 < len(clean) and (clean[k2] == "_" or clean[k2] == "$" or clean[k2].isalpha()):
                j2 = k2 + 1
                while j2 < len(clean) and clean[j2] in ident_chars:
                    j2 += 1
                owner = first
                name = clean[k2:j2]
                k = _skip_space(clean, j2)
        if name in _JS_KEYWORDS:
            i = j
            continue
        if k >= len(clean) or clean[k] != "(":
            i = j
            continue
        if _js_identifier_is_declaration(clean, i):
            i = k + 1
            continue
        if i > 0 and clean[i - 1] == "." and not owner:
            i = j
            continue
        end = _matching_paren(clean, k)
        if end <= k:
            i = k + 1
            continue
        if _js_call_is_probably_declaration(clean, i, k, end):
            i = end + 1
            continue
        arg_text = clean[k + 1 : end]
        args = [arg.strip() for arg in _split_top_level_commas(arg_text) if arg.strip()]
        line = clean.count("\n", 0, i) + 1
        call_name = f"{owner}.{name}" if owner else name
        calls.append(
            {
                "name": call_name,
                "base_name": name,
                "owner": owner,
                "args": args,
                "line": line,
                "call_text": _call_expr_text(call_name, args),
            }
        )
        i = end + 1
    return calls


def _matching_calls(calls: list[dict[str, object]], target: GraphNode) -> list[dict[str, object]]:
    target_names = _node_callable_names(target)
    if not target_names:
        return []
    out: list[dict[str, object]] = []
    for call in calls:
        names = {
            str(call.get("name") or ""),
            str(call.get("base_name") or ""),
        }
        if names & target_names:
            out.append(call)
    out.sort(key=lambda item: int(item.get("line") or 0))
    return out


def _node_callable_names(node: GraphNode) -> set[str]:
    name = str(node.name or "").strip()
    if not name:
        return set()
    tail = name.rsplit(".", 1)[-1]
    return {item for item in {name, tail} if item}


def _node_params(node: GraphNode) -> list[str]:
    signature = _node_signature_text(node)
    text = node.text or ""
    if node.path.endswith(".go") or signature.lstrip().startswith("func "):
        return _go_params(signature or _first_code_line(text))
    if node.path.endswith(_JS_TS_EXTS):
        return _js_params(text or signature)
    return _python_params(text or signature)


def _node_signature_text(node: GraphNode) -> str:
    if node.preview and node.preview.strip():
        return node.preview.strip()
    return _first_code_line(node.text or "")


def _first_code_line(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _python_params(text_or_sig: str) -> list[str]:
    text = textwrap.dedent(text_or_sig or "").strip()
    if text:
        try:
            tree = ast.parse(text + "\n")
            func = next((item for item in ast.walk(tree) if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
            if func is not None:
                params: list[str] = []
                params.extend(arg.arg for arg in getattr(func.args, "posonlyargs", []) or [])
                params.extend(arg.arg for arg in getattr(func.args, "args", []) or [])
                params.extend(arg.arg for arg in getattr(func.args, "kwonlyargs", []) or [])
                if func.args.vararg is not None:
                    params.append("*" + func.args.vararg.arg)
                if func.args.kwarg is not None:
                    params.append("**" + func.args.kwarg.arg)
                return [param for param in params if param not in {"self", "cls"}]
        except SyntaxError:
            pass
    match = _PY_DEF_RE.search(text_or_sig or "")
    if not match:
        return []
    return _simple_python_param_names(match.group("params") or "")


def _simple_python_param_names(params_text: str) -> list[str]:
    params: list[str] = []
    for raw in _split_top_level_commas(params_text):
        part = raw.strip()
        if not part or part in {"/", "*"}:
            continue
        part = part.lstrip("*")
        part = part.split("=", 1)[0].split(":", 1)[0].strip()
        if part and part not in {"self", "cls"}:
            params.append(part)
    return params


def _go_params(signature: str) -> list[str]:
    match = _GO_FUNC_RE.search(signature or "")
    if not match:
        return []
    raw_params = match.group("params") or ""
    params: list[str] = []
    pending_names: list[str] = []
    for raw in _split_top_level_commas(raw_params):
        part = raw.strip()
        if not part:
            continue
        tokens = part.replace("*", " ").split()
        if len(tokens) == 1:
            token = tokens[0].strip()
            if _looks_like_identifier(token) and not _looks_like_go_type(token):
                pending_names.append(token)
            continue
        first = tokens[0].strip()
        names = list(pending_names)
        pending_names.clear()
        if _looks_like_identifier(first) and not _looks_like_go_type(first):
            names.append(first)
        for name in names:
            if name and name != "_" and name not in params:
                params.append(name)
    return params


def _js_params(text_or_sig: str) -> list[str]:
    line = _first_code_line(text_or_sig or "")
    if not line:
        return []
    match = re.search(r"\bfunction\s+[A-Za-z_$][A-Za-z0-9_$]*\s*\((?P<params>[^)]*)\)", line)
    if not match:
        match = re.search(r"[A-Za-z_$][A-Za-z0-9_$]*\s*\((?P<params>[^)]*)\)\s*(?::[^{=]+)?\{?", line)
    if not match:
        match = re.search(r"=\s*(?:async\s*)?\((?P<params>[^)]*)\)\s*=>", line)
    if not match:
        one = re.search(r"=\s*(?:async\s*)?(?P<param>[A-Za-z_$][A-Za-z0-9_$]*)\s*=>", line)
        return [one.group("param")] if one else []
    params: list[str] = []
    for raw in _split_top_level_commas(match.group("params") or ""):
        part = raw.strip()
        if not part:
            continue
        part = part.split("=", 1)[0].strip()
        part = part.lstrip(".").lstrip("*").strip()
        part = part.split(":", 1)[0].strip()
        part = part.rstrip("?").strip()
        if part.startswith("{") or part.startswith("["):
            params.append(part)
        elif _looks_like_identifier(part):
            params.append(part)
    return params


def _value_flow_fact(
    *,
    relation: str,
    source: GraphNode,
    target: GraphNode,
    call: dict[str, object],
    params: list[str],
    read_node_ids: set[str],
) -> dict[str, object]:
    args = [str(arg) for arg in call.get("args", []) if str(arg)]
    mappings: list[dict[str, object]] = []
    used_params: set[str] = set()
    for idx, arg in enumerate(args):
        param = params[idx] if idx < len(params) else ""
        if "=" in arg and not arg.lstrip().startswith(("==", ">=", "<=", "!=")):
            key, value = arg.split("=", 1)
            key = key.strip().lstrip("*")
            if key in params:
                param = key
                arg = value.strip()
        item: dict[str, object] = {"position": idx, "argument": arg}
        if param:
            item["parameter"] = param
            used_params.add(param)
        mappings.append(item)
    fact = {
        "relation": relation,
        "source": node_brief(source),
        "target": node_brief(target),
        "call": str(call.get("call_text") or ""),
        "call_line": int(call.get("line") or 0) or None,
        "argument_to_parameter": mappings,
        "unmapped_parameters": [param for param in params if param not in used_params],
        "target_read_status": "read" if target.id in read_node_ids else "unread",
        "source_read_status": "read" if source.id in read_node_ids else "unread",
    }
    return fact


def _call_expr_text(name: str, args: list[str]) -> str:
    text = f"{name}({', '.join(str(arg) for arg in args)})"
    if len(text) > 240:
        return text[:237] + "..."
    return text


def _strip_go_line_comment(line: str) -> str:
    out: list[str] = []
    in_string = ""
    escaped = False
    i = 0
    while i < len(line):
        ch = line[i]
        nxt = line[i + 1] if i + 1 < len(line) else ""
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\" and in_string != "`":
                escaped = True
            elif ch == in_string:
                in_string = ""
            i += 1
            continue
        if ch in {'"', "'", "`"}:
            in_string = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            break
        out.append(ch)
        i += 1
    return "".join(out)


def _skip_space(text: str, idx: int) -> int:
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx


def _matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    in_string = ""
    escaped = False
    for idx in range(open_idx, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\" and in_string != "`":
                escaped = True
            elif ch == in_string:
                in_string = ""
            continue
        if ch in {'"', "'", "`"}:
            in_string = ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    in_string = ""
    escaped = False
    for idx, ch in enumerate(text or ""):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\" and in_string != "`":
                escaped = True
            elif ch == in_string:
                in_string = ""
            continue
        if ch in {'"', "'", "`"}:
            in_string = ch
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}" and depth > 0:
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:idx])
            start = idx + 1
    parts.append(text[start:])
    return parts


def _go_identifier_is_declaration(text: str, idx: int) -> bool:
    line_start = text.rfind("\n", 0, idx) + 1
    prefix = text[line_start:idx].strip()
    return prefix == "func" or prefix.startswith("func ") or prefix.startswith("func(") or prefix.startswith("func (")


def _js_identifier_is_declaration(text: str, idx: int) -> bool:
    line_start = text.rfind("\n", 0, idx) + 1
    prefix = text[line_start:idx].strip()
    return bool(
        re.search(r"\b(function|class|interface|type|enum)\s*$", prefix)
        or re.search(r"\b(const|let|var)\s+[A-Za-z_$][A-Za-z0-9_$]*\s*=\s*$", prefix)
        or prefix.endswith("=>")
    )


def _js_call_is_probably_declaration(text: str, name_start: int, open_idx: int, close_idx: int) -> bool:
    line_start = text.rfind("\n", 0, name_start) + 1
    prefix = text[line_start:name_start]
    if _js_identifier_is_declaration(text, name_start):
        return True
    suffix_start = _skip_space(text, close_idx + 1)
    suffix = text[suffix_start : min(len(text), suffix_start + 4)]
    args = text[open_idx + 1 : close_idx]
    if not prefix.strip() and (suffix.startswith("{") or suffix.startswith(":")) and ":" in args:
        return True
    return False


def _looks_like_identifier(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value or ""))


def _looks_like_go_type(value: str) -> bool:
    if not value:
        return False
    if value in {"string", "int", "int64", "uint32", "bool", "error", "byte", "rune", "float64"}:
        return True
    if value[0].isupper() or "." in value or value.startswith(("[]", "map[", "chan", "<-")):
        return True
    return False


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
