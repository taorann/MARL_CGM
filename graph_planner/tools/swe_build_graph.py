from __future__ import annotations

"""
Repo-level graph builder for GraphPlanner.

This tool scans a repository and extracts a lightweight code graph from Python sources.

Outputs:
- JSON (default, backward compatible):
    {"nodes": [...], "edges": [...]}
- JSONL:
    One JSON object per line with {"type":"node", ...} or {"type":"edge", ...}
- base64(gzip(JSONL)):
    Printed to stdout when --emit-base64-gzip is set (for SSH/stdout transport)

Compatibility notes:
- --issue-id is accepted but ignored (legacy callers may still pass it).
"""

import argparse
import ast
import base64
import gzip
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


SKIP_DIRS: Set[str] = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    "node_modules",
    "site-packages",
}


def _rel_posix(path: Path, repo: Path) -> str:
    try:
        return path.relative_to(repo).as_posix()
    except Exception:
        return path.as_posix()


def iter_python_files(repo: Path) -> Iterable[Path]:
    repo = repo.resolve()
    for root, dirs, files in os.walk(repo):
        # prune dirs in-place
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in files:
            if fn.endswith(".py"):
                yield Path(root) / fn


def _node_span(n: ast.AST) -> Tuple[int, int]:
    start = int(getattr(n, "lineno", 1) or 1)
    end = int(getattr(n, "end_lineno", start) or start)
    return start, end


def _safe_parse(src: str, filename: str) -> Optional[ast.AST]:
    try:
        return ast.parse(src, filename=filename, type_comments=True)
    except Exception:
        return None


class GraphBuilder(ast.NodeVisitor):
    def __init__(self, file_rel: str, file_node_id: str) -> None:
        self.file_rel = file_rel
        self.file_node_id = file_node_id
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        # stack of container node ids (file/class/function)
        self.container_stack: List[str] = [file_node_id]
        # stack of names for qualname
        self.scope_names: List[str] = []

    def _add_node(self, node_id: str, payload: Dict[str, Any]) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = payload

    def _add_edge(self, src: str, dst: str, kind: str) -> None:
        self.edges.append({"src": src, "dst": dst, "kind": kind})

    def _current_container(self) -> str:
        return self.container_stack[-1] if self.container_stack else self.file_node_id

    def _qualname(self, name: str) -> str:
        if not self.scope_names:
            return name
        return ".".join(self.scope_names + [name])

    # ---- imports (edge: imports) ----
    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            mod = alias.name
            mid = f"module:{mod}"
            self._add_node(
                mid,
                {
                    "id": mid,
                    "kind": "module",
                    "name": mod,
                    "path": None,
                    "span": None,
                },
            )
            self._add_edge(self.file_node_id, mid, "imports")
        return self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        base = node.module or ""
        prefix = "." * int(getattr(node, "level", 0) or 0)
        mod = prefix + base
        if mod:
            mid = f"module:{mod}"
            self._add_node(
                mid,
                {
                    "id": mid,
                    "kind": "module",
                    "name": mod,
                    "path": None,
                    "span": None,
                },
            )
            self._add_edge(self.file_node_id, mid, "imports")
        return self.generic_visit(node)

    # ---- class/function nodes (edge: contains) ----
    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        qn = self._qualname(node.name)
        start, end = _node_span(node)
        nid = f"class:{self.file_rel}:{qn}:{start}"
        self._add_node(
            nid,
            {
                "id": nid,
                "kind": "class",
                "name": qn,
                "path": self.file_rel,
                "span": {"start": start, "end": end},
            },
        )
        self._add_edge(self._current_container(), nid, "contains")

        self.container_stack.append(nid)
        self.scope_names.append(node.name)
        for stmt in node.body:
            self.visit(stmt)
        self.scope_names.pop()
        self.container_stack.pop()
        # do not generic_visit to avoid double-visiting body
        return None

    def _visit_function(self, node: ast.AST, name: str, is_async: bool) -> Any:
        qn = self._qualname(name)
        start, end = _node_span(node)
        nid = f"func:{self.file_rel}:{qn}:{start}"
        self._add_node(
            nid,
            {
                "id": nid,
                "kind": "function",
                "name": qn,
                "path": self.file_rel,
                "span": {"start": start, "end": end},
                "async": bool(is_async),
            },
        )
        self._add_edge(self._current_container(), nid, "contains")

        self.container_stack.append(nid)
        self.scope_names.append(name)
        body = getattr(node, "body", [])
        for stmt in body:
            self.visit(stmt)
        self.scope_names.pop()
        self.container_stack.pop()
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        return self._visit_function(node, node.name, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        return self._visit_function(node, node.name, is_async=True)


def build_repo_graph(repo: Path) -> Dict[str, Any]:
    repo = repo.resolve()
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    for fp in iter_python_files(repo):
        rel = _rel_posix(fp, repo)
        file_id = f"file:{rel}"
        # file node
        if file_id not in nodes:
            nodes[file_id] = {
                "id": file_id,
                "kind": "file",
                "name": rel,
                "path": rel,
                "span": None,
            }

        try:
            src = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        tree = _safe_parse(src, filename=rel)
        if tree is None:
            continue

        gb = GraphBuilder(file_rel=rel, file_node_id=file_id)
        gb.visit(tree)

        # merge
        for nid, n in gb.nodes.items():
            if nid not in nodes:
                nodes[nid] = n
        edges.extend(gb.edges)

    return {"nodes": list(nodes.values()), "edges": edges}


def to_jsonl_lines(graph: Dict[str, Any]) -> List[str]:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    # stable ordering
    nodes_sorted = sorted(nodes, key=lambda x: str(x.get("id", "")))
    edges_sorted = sorted(edges, key=lambda x: (str(x.get("src", "")), str(x.get("dst", "")), str(x.get("kind", ""))))

    lines: List[str] = []
    for n in nodes_sorted:
        obj = dict(n)
        obj["type"] = "node"
        lines.append(json.dumps(obj, ensure_ascii=False))
    for e in edges_sorted:
        obj = dict(e)
        obj["type"] = "edge"
        lines.append(json.dumps(obj, ensure_ascii=False))
    return lines


def emit_base64_gzip(text: str) -> str:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
        f.write(text.encode("utf-8"))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build repo-level code graph (JSON/JSONL/base64+gzip(JSONL)).")
    p.add_argument("--repo", required=True, help="Path to repository root (inside container usually /repo).")
    # legacy arg (accepted but ignored)
    p.add_argument("--issue-id", default=None, help="(legacy) accepted but ignored; repo-level graph is built.")
    p.add_argument("--output", default="-", help="Output path. Use '-' for stdout. Default: stdout.")
    p.add_argument("--format", choices=["json", "jsonl"], default="json", help="Output format. Default: json.")
    p.add_argument(
        "--emit-base64-gzip",
        action="store_true",
        help="Print base64(gzip(JSONL)) to stdout (ignores --output, forces JSONL).",
    )

    args = p.parse_args(argv)

    repo = Path(args.repo)
    graph = build_repo_graph(repo)

    if args.emit_base64_gzip:
        lines = to_jsonl_lines(graph)
        payload = "\n".join(lines) + ("\n" if lines else "")
        b64 = emit_base64_gzip(payload)
        sys.stdout.write(b64)
        if not b64.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    if args.format == "jsonl":
        lines = to_jsonl_lines(graph)
        payload = "\n".join(lines) + ("\n" if lines else "")
    else:
        payload = json.dumps(graph, ensure_ascii=False)

    if args.output == "-" or args.output is None:
        sys.stdout.write(payload)
        if not payload.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
