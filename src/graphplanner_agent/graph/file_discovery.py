from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
import json
import re
import shlex

from graphplanner_agent.graph.guards import is_test_path
from graphplanner_agent.graph.schema import GraphNode


CODE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".mjs",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".scala",
    ".scss",
    ".swift",
    ".ts",
    ".tsx",
    ".vue",
}


@dataclass(slots=True)
class DiscoveredFile:
    path: str
    line_count: int
    score: float
    source: str


def is_code_file_path(path: str) -> bool:
    lowered = path.lower().replace("\\", "/")
    return any(lowered.endswith(ext) for ext in CODE_EXTENSIONS)


def synthetic_file_node(path: str, line_count: int = 1) -> GraphNode:
    normalized = path.replace("\\", "/").strip()
    return GraphNode(
        id=f"file:{normalized}",
        kind="file",
        name=normalized.rsplit("/", 1)[-1],
        path=normalized,
        start_line=1,
        end_line=max(1, int(line_count or 1)),
        preview="filesystem implementation file",
    )


def discover_implementation_files(
    runtime,
    *,
    query: str = "",
    path_glob: str = "",
    limit: int = 50,
) -> list[DiscoveredFile]:
    records = _runtime_file_records(runtime)
    pattern = path_glob.strip().replace("\\", "/")
    terms = _query_terms(query)
    scoped: list[DiscoveredFile] = []
    matched: list[DiscoveredFile] = []
    for record in records:
        path = str(record.get("path") or "").strip().replace("\\", "/")
        if not path or is_test_path(path) or not is_code_file_path(path):
            continue
        if pattern and not _path_matches(path, pattern):
            continue
        line_count = int(record.get("line_count") or 1)
        score = _score_path(path, terms)
        source = "filesystem_file_search"
        if score > 0 or not terms:
            matched.append(DiscoveredFile(path=path, line_count=line_count, score=score, source=source))
        elif pattern:
            scoped.append(DiscoveredFile(path=path, line_count=line_count, score=0.1, source="filesystem_scope_listing"))
    pool = matched or scoped
    pool.sort(key=lambda item: (item.score, _basename_exact(item.path, query), -len(item.path)), reverse=True)
    return pool[:limit]


def _runtime_file_records(runtime) -> list[dict[str, object]]:
    script = r'''
import json
from pathlib import Path

EXTS = set(__EXTENSIONS__)
SKIP_DIRS = {
    ".git", ".hg", ".svn", ".tox", ".venv", "venv", "env",
    "node_modules", "dist", "build", "coverage", ".cache", "__pycache__",
}

records = []
for path in Path(".").rglob("*"):
    try:
        if not path.is_file():
            continue
    except OSError:
        continue
    rel = path.as_posix()
    parts = rel.split("/")
    if any(part in SKIP_DIRS for part in parts):
        continue
    if path.suffix.lower() not in EXTS:
        continue
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            line_count = sum(1 for _ in fh)
    except OSError:
        line_count = 1
    records.append({"path": rel, "line_count": max(1, line_count)})
print(json.dumps(records, ensure_ascii=False))
'''.replace("__EXTENSIONS__", json.dumps(sorted(CODE_EXTENSIONS)))
    result = runtime.run("python -c " + shlex.quote(script), timeout=180)
    if result.returncode != 0:
        return []
    text = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "[]"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _query_terms(query: str) -> list[str]:
    value = str(query or "").strip().lower().replace("\\", "/")
    if not value:
        return []
    terms = [value]
    terms.extend(part for part in re.split(r"[^a-z0-9_./-]+", value) if len(part) > 1)
    out: list[str] = []
    for term in terms:
        cleaned = term.strip().replace("symbol:", "").replace("path:", "")
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def _score_path(path: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    hay = path.lower()
    basename = hay.rsplit("/", 1)[-1]
    stem = basename.rsplit(".", 1)[0]
    score = 0.0
    for term in terms:
        if term == hay or term == basename or term == stem:
            score += 8.0
        elif term in basename:
            score += 4.0
        elif term in hay:
            score += 2.0
    return score


def _basename_exact(path: str, query: str) -> bool:
    key = str(query or "").strip().lower().replace("\\", "/")
    if not key:
        return False
    basename = path.rsplit("/", 1)[-1].lower()
    return key in {path.lower(), basename, basename.rsplit(".", 1)[0]}


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
