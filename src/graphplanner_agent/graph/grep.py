from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
import re

from .file_discovery import discover_implementation_files, is_code_file_path, synthetic_file_node
from .guards import is_test_path
from .schema import GraphNode, RepoGraph


@dataclass(slots=True)
class GrepHit:
    path: str
    line: int
    text: str
    context: str
    covering_node: GraphNode | None


def grep_code(
    graph: RepoGraph,
    runtime,
    pattern: str,
    path_glob: str,
    *,
    context_lines: int = 2,
    limit: int = 20,
    regex: bool = False,
) -> list[GrepHit]:
    synthetic_nodes: dict[str, GraphNode] = {}
    paths = _candidate_paths(graph, path_glob)
    if not paths:
        discovered = discover_implementation_files(runtime, path_glob=path_glob, limit=max(limit, 20))
        paths = [item.path for item in discovered]
        synthetic_nodes = {item.path: synthetic_file_node(item.path, item.line_count) for item in discovered}
    if not paths:
        return []
    hits: list[GrepHit] = []
    compiled = re.compile(pattern) if regex else None
    for path in paths:
        text = runtime.read_file(path)
        lines = text.splitlines()
        for line_no, line in enumerate(lines, start=1):
            matched = bool(compiled.search(line)) if compiled else pattern in line
            if not matched:
                continue
            start = max(1, line_no - context_lines)
            end = min(len(lines), line_no + context_lines)
            context = "\n".join(f"{idx:>4}: {lines[idx - 1]}" for idx in range(start, end + 1))
            hits.append(
                GrepHit(
                    path=path,
                    line=line_no,
                    text=line,
                    context=context,
                    covering_node=_covering_node(graph, path, line_no) or synthetic_nodes.get(path),
                )
            )
            if len(hits) >= limit:
                return hits
    return hits


def _candidate_paths(graph: RepoGraph, path_glob: str) -> list[str]:
    pattern = path_glob.strip().replace("\\", "/")
    paths = {
        node.path
        for node in graph.nodes.values()
        if is_code_file_path(node.path) and not is_test_path(node.path) and _path_matches(node.path, pattern)
    }
    return sorted(paths)


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


def _covering_node(graph: RepoGraph, path: str, line_no: int) -> GraphNode | None:
    candidates = [
        node
        for node in graph.nodes.values()
        if node.path == path
        and node.kind != "repository"
        and node.start_line <= line_no <= node.end_line
        and not is_test_path(node.path)
    ]
    if not candidates:
        return None
    non_file = [node for node in candidates if node.kind != "file"]
    pool = non_file or candidates
    return min(pool, key=lambda node: (node.end_line - node.start_line, node.kind == "file"))
