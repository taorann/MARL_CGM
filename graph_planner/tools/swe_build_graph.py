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
import textwrap
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


def _clip_snippet_lines(
    file_lines: Sequence[str],
    start_line: int,
    end_line: int,
    *,
    max_lines: int,
) -> List[str]:
    """Extract [start_line, end_line] (1-indexed, inclusive) and clip length."""
    if not file_lines:
        return []
    s = max(1, int(start_line or 1))
    e = max(s, int(end_line or s))
    seg = [str(x).rstrip("\n") for x in list(file_lines[s - 1 : e])]

    # If the snippet is long, keep both head and tail to preserve endings.
    if max_lines >= 0 and len(seg) > max_lines:
        if max_lines <= 3:
            return seg[:max_lines]
        head = max_lines // 2
        tail = max_lines - head - 1
        seg = seg[:head] + ["... <clipped>"] + seg[-tail:]
    return seg


def _extract_sig_from_snippet(snippet_lines: Sequence[str]) -> str:
    for ln in snippet_lines[:12]:
        s = str(ln).strip()
        if s.startswith("async def ") or s.startswith("def ") or s.startswith("class "):
            return s
    return ""


def _truncate_doc(doc: Optional[str], max_chars: int = 240) -> str:
    if not doc:
        return ""
    d = " ".join(str(doc).strip().split())
    if len(d) > max_chars:
        return d[: max_chars - 3] + "..."
    return d


def _file_summary_snippet(file_lines: Sequence[str], tree: Optional[ast.AST], *, max_lines: int) -> List[str]:
    """Make a high-signal file-level snippet without embedding child bodies.

    We include:
      - a tiny module docstring (truncated),
      - a compact index of top-level def/class symbols (sig @ line).

    NOTE: We intentionally do **NOT** include raw file header code. The full
    code is pushed down to child nodes (func/class) to avoid a single file node
    exploding the prompt/context.
    """
    if not file_lines:
        return []
    max_lines = int(max_lines or 0)
    if max_lines <= 0:
        return []

    out: List[str] = []

    # 1) Minimal docstring (if any)
    try:
        doc = (ast.get_docstring(tree) or "").strip() if tree is not None else ""
    except Exception:
        doc = ""
    if doc:
        out.append("# Module docstring (truncated):")
        wrapped = textwrap.wrap(" ".join(doc.split()), width=110)
        # keep it small; docstring is not the code context we care about
        for ln in wrapped[: min(6, max_lines - 1)]:
            out.append(f"# {ln}")

    # 2) Symbol index (sig @ line)
    symbols: List[str] = []
    try:
        body = getattr(tree, "body", []) if tree is not None else []
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lineno = int(getattr(n, "lineno", 0) or 0)
                sig = ""
                if 1 <= lineno <= len(file_lines):
                    sig = str(file_lines[lineno - 1]).strip()
                if not sig:
                    sig = f"def {n.name}(...)"
                if len(sig) > 160:
                    sig = sig[:157] + "..."
                symbols.append(f"{sig}  # L{lineno if lineno else '?'}")
            elif isinstance(n, ast.ClassDef):
                lineno = int(getattr(n, "lineno", 0) or 0)
                sig = ""
                if 1 <= lineno <= len(file_lines):
                    sig = str(file_lines[lineno - 1]).strip()
                if not sig:
                    sig = f"class {n.name}(...)"
                if len(sig) > 160:
                    sig = sig[:157] + "..."
                symbols.append(f"{sig}  # L{lineno if lineno else '?'}")
    except Exception:
        symbols = []

    if symbols and len(out) < max_lines:
        if out and out[-1] != "":
            out.append("")
        out.append("# Top-level symbols (sig @ line):")
        for sym in symbols:
            if len(out) >= max_lines:
                break
            out.append(sym)

    # Clip final
    return out[:max_lines]


def _class_summary_snippet(file_lines: Sequence[str], cls: ast.ClassDef, *, max_lines: int) -> List[str]:
    """Compact class-level snippet (avoid embedding the whole class body).

    The goal is to keep the class node lightweight, and push real code into
    method/function child nodes.
    """
    if max_lines <= 0 or not file_lines:
        return []

    out: List[str] = []
    # Signature line
    try:
        sig = file_lines[int(cls.lineno) - 1].rstrip("\n")
        out.append(sig)
    except Exception:
        out.append(f"class {cls.name}:")

    # Docstring (as comments)
    doc = ast.get_docstring(cls) or ""
    if doc:
        out.append("# Docstring (truncated):")
        for line in textwrap.wrap(doc, width=100)[: min(6, max_lines - len(out))]:
            out.append(f"# {line}")

    # Method index
    methods: List[Tuple[int, str]] = []
    for stmt in cls.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append((int(getattr(stmt, "lineno", 0) or 0), getattr(stmt, "name", "")))
    if methods and len(out) < max_lines:
        out.append("# Methods:")
        for ln, name in methods[: min(20, max_lines - len(out))]:
            if ln <= 0:
                out.append(f"- {name}()")
            else:
                try:
                    line = file_lines[ln - 1].strip()
                except Exception:
                    line = f"def {name}(...):"
                if len(line) > 200:
                    line = line[:200] + "…"
                out.append(f"- {line}  # L{ln}")

    return out[:max_lines]


class GraphBuilder(ast.NodeVisitor):
    def __init__(
        self,
        file_rel: str,
        file_node_id: str,
        *,
        file_lines: Optional[Sequence[str]] = None,
        embed_snippets: bool = True,
        max_snippet_lines: int = 120,
        max_class_snippet_lines: int = 32,
    ) -> None:
        self.file_rel = file_rel
        self.file_node_id = file_node_id
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self._file_lines = list(file_lines or [])
        self._embed_snippets = bool(embed_snippets)
        self._max_snippet_lines = int(max_snippet_lines)
        self._max_class_snippet_lines = int(max_class_snippet_lines)
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
        payload: Dict[str, Any] = {
            "id": nid,
            "kind": "class",
            "name": qn,
            "path": self.file_rel,
            "span": {"start": start, "end": end},
        }
        snip: List[str] = []
        if self._embed_snippets and self._file_lines:
            # IMPORTANT: keep class nodes compact; method/function bodies are separate nodes.
            snip = _class_summary_snippet(self._file_lines, node, max_lines=self._max_class_snippet_lines)
            if snip:
                payload["snippet_lines"] = snip
        payload["sig"] = _extract_sig_from_snippet(snip) or f"class {qn}"
        payload["doc"] = _truncate_doc(ast.get_docstring(node) or "", max_chars=240)
        self._add_node(nid, payload)
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
        payload2: Dict[str, Any] = {
            "id": nid,
            "kind": "function",
            "name": qn,
            "path": self.file_rel,
            "span": {"start": start, "end": end},
            "async": bool(is_async),
        }
        snip: List[str] = []
        if self._embed_snippets and self._file_lines:
            snip = _clip_snippet_lines(self._file_lines, start, end, max_lines=self._max_snippet_lines)
            if snip:
                payload2["snippet_lines"] = snip
        kind_kw = "async def" if is_async else "def"
        payload2["sig"] = _extract_sig_from_snippet(snip) or f"{kind_kw} {qn}(...)"
        payload2["doc"] = _truncate_doc(ast.get_docstring(node) or "", max_chars=240)
        self._add_node(nid, payload2)
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

    embed_snippets = str(os.environ.get("GP_EMBED_REPO_SNIPPETS", "1")).strip().lower() in {"1", "true", "yes", "y"}
    # Snippet budgets:
    #   - File nodes: keep a compact module header + symbol index (avoid noisy code).
    #   - Def nodes (func/class): allow a larger embedded snippet to support CGM.
    # Backward-compat: GP_MAX_SNIPPET_LINES still works as a global fallback.
    # Keep defaults conservative to avoid prompt explosion.
    # You can override per-node budgets with:
    #   - GP_MAX_FILE_SNIPPET_LINES (file node summary)
    #   - GP_MAX_DEF_SNIPPET_LINES  (func/class node code)
    max_snippet_lines = int(os.environ.get("GP_MAX_SNIPPET_LINES", "80") or 80)
    max_file_snippet_lines = int(os.environ.get("GP_MAX_FILE_SNIPPET_LINES", "40") or 40)
    max_def_snippet_lines = int(os.environ.get("GP_MAX_DEF_SNIPPET_LINES", "80") or 80)
    max_class_snippet_lines = int(os.environ.get("GP_MAX_CLASS_SNIPPET_LINES", "32") or 32)
    # Back-compat: if a budget is set to 0, fall back to the global.
    if max_file_snippet_lines <= 0:
        max_file_snippet_lines = max_snippet_lines
    if max_def_snippet_lines <= 0:
        max_def_snippet_lines = max_snippet_lines

    for fp in iter_python_files(repo):
        rel = _rel_posix(fp, repo)
        file_id = f"file:{rel}"
        try:
            src = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        file_lines = src.splitlines()

        tree = _safe_parse(src, filename=rel)
        if tree is None:
            continue

        # file node: keep it compact (module header + symbol index), *not* raw code.
        # This prevents weak-term explosion during lexical graph search.
        if file_id not in nodes:
            file_payload: Dict[str, Any] = {
                "id": file_id,
                "kind": "file",
                "name": rel,
                "path": rel,
                "span": None,
                "sig": f"module {rel}",
                "doc": _truncate_doc(ast.get_docstring(tree) or "", max_chars=320),
            }
            if embed_snippets and file_lines:
                file_payload["snippet_lines"] = _file_summary_snippet(
                    file_lines,
                    tree,
                    max_lines=max_file_snippet_lines,
                )
            nodes[file_id] = file_payload

        gb = GraphBuilder(
            file_rel=rel,
            file_node_id=file_id,
            file_lines=file_lines,
            embed_snippets=embed_snippets,
            max_snippet_lines=max_def_snippet_lines,
            max_class_snippet_lines=max_class_snippet_lines,
        )
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
    p.add_argument("--repo", required=True, help="Path to repository root (inside container usually /testbed).")
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
