from __future__ import annotations

import re
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from .guards import is_test_path
from .schema import GraphNode, RepoGraph


@dataclass(slots=True)
class SearchResult:
    node: GraphNode
    score: float
    source: str


def _parse_terms(query: str) -> tuple[list[str], list[str], list[str]]:
    strong = re.findall(r"\+([A-Za-z0-9_./:-]+)", query)
    negative = re.findall(r"-([A-Za-z0-9_./:-]+)", query)
    phrases = re.findall(r'"([^"]+)"', query)
    cleaned = re.sub(r"[+-][A-Za-z0-9_./:-]+", " ", query)
    cleaned = re.sub(r'"[^"]+"', " ", cleaned)
    ordinary = [t for t in re.split(r"\W+", cleaned) if len(t) > 1]
    return strong + phrases, negative, ordinary


def search_graph(
    graph: RepoGraph,
    query: str,
    find_type: str = "any",
    class_name: str | None = None,
    limit: int = 12,
    root: Path | None = None,
    path_glob: str | None = None,
) -> tuple[list[SearchResult], str | None]:
    if class_name:
        filtered, warning = _search_graph_once(graph, query, find_type, class_name, limit, root=None, path_glob=path_glob)
        if filtered:
            return filtered, warning
        unfiltered, fallback_warning = _search_graph_once(graph, query, find_type, None, limit, root=root, path_glob=path_glob)
        if unfiltered:
            warning = f"class_name filter {class_name!r} produced no hits; retried without it"
            if fallback_warning:
                warning += f"; {fallback_warning}"
            return unfiltered, warning
        return [], None
    return _search_graph_once(graph, query, find_type, None, limit, root=root, path_glob=path_glob)


def _search_graph_once(
    graph: RepoGraph,
    query: str,
    find_type: str,
    class_name: str | None,
    limit: int,
    root: Path | None,
    path_glob: str | None,
) -> tuple[list[SearchResult], str | None]:
    strong, negative, ordinary = _parse_terms(query)
    wanted = None if find_type in {"", "any"} else find_type
    results: list[SearchResult] = []
    for node in graph.nodes.values():
        if node.kind == "repository" or is_test_path(node.path):
            continue
        if path_glob and not _path_matches(node.path, path_glob):
            continue
        if wanted and not _kind_matches(node.kind, wanted, node.name):
            continue
        if class_name and class_name not in (node.name + " " + (node.parent_id or "")):
            continue
        hay = " ".join([node.name, node.path, node.preview, node.text or ""]).lower()
        if any(term.lower() in hay for term in negative):
            continue
        if strong and not all(term.lower().replace("symbol:", "") in hay for term in strong):
            continue
        score = 0.0
        for term in strong:
            score += 5.0 if term.lower().replace("symbol:", "") in hay else 0.0
        for term in ordinary:
            score += 1.0 if term.lower() in hay else 0.0
        if query.lower() in hay:
            score += 3.0
        if score > 0 or not (strong or ordinary):
            results.append(SearchResult(node=node, score=score, source="graph"))
    results.sort(key=lambda r: (_exact_name_match(r.node, query), r.score, r.node.kind in {"function", "method", "assignment", "module_assignment"}, -len(r.node.path)), reverse=True)
    if results or root is None:
        return results[:limit], None
    fallback = _filesystem_fallback(graph, root, strong + ordinary, wanted, limit, path_glob=path_glob)
    warning = "Graph search had no hit; filesystem fallback mapped text spans back to graph nodes." if fallback else None
    return fallback, warning


def _filesystem_fallback(
    graph: RepoGraph,
    root: Path,
    terms: list[str],
    wanted: str | None,
    limit: int,
    *,
    path_glob: str | None,
) -> list[SearchResult]:
    clean_terms = [term.lower().replace("symbol:", "").replace("path:", "") for term in terms if len(term) > 1]
    if not clean_terms:
        return []
    hits: dict[str, SearchResult] = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if is_test_path(rel):
            continue
        if path_glob and not _path_matches(rel, path_glob):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(lines, start=1):
            hay = line.lower()
            score = sum(1.0 for term in clean_terms if term in hay)
            if not score:
                continue
            node = _covering_node(graph, rel, line_no, wanted)
            if not node:
                continue
            current = hits.get(node.id)
            if not current or score > current.score:
                hits[node.id] = SearchResult(node=node, score=score, source="filesystem_fallback")
    ranked = list(hits.values())
    ranked.sort(key=lambda r: (_exact_name_match(r.node, " ".join(terms)), r.score, r.node.kind in {"function", "method", "assignment", "module_assignment"}, -len(r.node.path)), reverse=True)
    return ranked[:limit]


def _path_matches(path: str, pattern: str) -> bool:
    normalized = path.replace("\\", "/")
    pattern = pattern.strip().replace("\\", "/")
    if not pattern:
        return True
    patterns = _path_pattern_variants(pattern)
    if any(ch in pattern for ch in "*?[]"):
        return any(fnmatch(normalized, candidate) for candidate in patterns)
    return normalized == pattern or normalized.startswith(pattern.rstrip("/") + "/")


def _path_pattern_variants(pattern: str) -> set[str]:
    patterns = {pattern}
    if "/**/" in pattern:
        patterns.add(pattern.replace("/**/", "/"))
    if pattern.startswith("**/"):
        patterns.add(pattern[3:])
    return patterns


def _covering_node(graph: RepoGraph, path: str, line_no: int, wanted: str | None) -> GraphNode | None:
    candidates = [
        node
        for node in graph.nodes.values()
        if node.path == path
        and node.kind != "repository"
        and node.start_line <= line_no <= node.end_line
        and (wanted is None or _kind_matches(node.kind, wanted, node.name) or node.kind == "file")
    ]
    if not candidates:
        return None
    non_file = [node for node in candidates if node.kind != "file"]
    pool = non_file or candidates
    return min(pool, key=lambda node: (node.end_line - node.start_line, node.kind == "file"))


def _kind_matches(node_kind: str, wanted: str, node_name: str = "") -> bool:
    if node_kind == wanted:
        return True
    if wanted == "assignment" and node_kind in {"assignment", "module_assignment"}:
        return True
    if wanted == "method" and node_kind == "function" and "." in node_name:
        return True
    return False


def _exact_name_match(node: GraphNode, query: str) -> bool:
    key = query.strip().lower()
    if not key or any(ch.isspace() for ch in key):
        return False
    return node.name.lower() == key or node.id.lower() == key
