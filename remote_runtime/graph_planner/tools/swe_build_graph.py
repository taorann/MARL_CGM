from __future__ import annotations

"""
Repo-level graph builder for GraphPlanner.

This tool scans a repository and extracts a lightweight code graph from Python
and Go sources. Python uses the existing AST/tree-sitter frontend; Go uses a
parser-free structural scanner so graph building still works inside lean
benchmark containers.

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
import re
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


def iter_go_files(repo: Path) -> Iterable[Path]:
    repo = repo.resolve()
    include_tests = str(os.environ.get("GP_GRAPH_INCLUDE_TESTS", "0")).strip().lower() in {"1", "true", "yes", "y"}
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in files:
            if not fn.endswith(".go"):
                continue
            if (not include_tests) and fn.endswith("_test.go"):
                continue
            yield Path(root) / fn


JS_TS_EXTS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
GENERIC_CODE_EXTS = {
    ".java",
    ".kt",
    ".kts",
    ".rs",
    ".rb",
    ".php",
    ".cs",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".swift",
    ".scala",
}


def _is_js_ts_test_file(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    if "__tests__" in parts or "__mocks__" in parts:
        return True
    name = path.name.lower()
    return any(marker in name for marker in (".test.", ".spec.", "-test.", "_test."))


def iter_js_ts_files(repo: Path) -> Iterable[Path]:
    repo = repo.resolve()
    include_tests = str(os.environ.get("GP_GRAPH_INCLUDE_TESTS", "0")).strip().lower() in {"1", "true", "yes", "y"}
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in files:
            path = Path(root) / fn
            if path.suffix.lower() not in JS_TS_EXTS:
                continue
            if path.name.endswith(".d.ts"):
                continue
            if (not include_tests) and _is_js_ts_test_file(path):
                continue
            yield path


def iter_generic_code_files(repo: Path) -> Iterable[Path]:
    repo = repo.resolve()
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in files:
            path = Path(root) / fn
            if path.suffix.lower() in GENERIC_CODE_EXTS:
                yield path


def _node_span(n: ast.AST) -> Tuple[int, int]:
    start = int(getattr(n, "lineno", 1) or 1)
    end = int(getattr(n, "end_lineno", start) or start)
    return start, end


def _safe_parse(src: str, filename: str) -> Optional[ast.AST]:
    try:
        return ast.parse(src, filename=filename, type_comments=True)
    except Exception:
        return None


_QUALNAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _dedupe_preserve(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in items or []:
        s = str(item or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _normalize_call_owner_name(owner: str) -> str:
    """Normalize an attribute-call owner to a class/module-ish qualname.

    Tree-sitter gives the object side of ``ClassName(...).method()`` as raw
    source text (``ClassName(...)``). For call graph purposes, the useful owner
    is the constructor/class expression before the opening parenthesis.
    """
    raw = str(owner or "").strip()
    if not raw:
        return ""
    if _QUALNAME_RE.match(raw):
        return raw
    m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\(", raw)
    if m:
        return m.group(1).strip()
    return raw


def _ast_expr_qualname(expr: ast.AST) -> str:
    """Best-effort qualname for a call owner expression."""
    try:
        if isinstance(expr, ast.Name):
            return str(expr.id or "").strip()
        if isinstance(expr, ast.Attribute):
            owner = _ast_expr_qualname(expr.value)
            attr = str(getattr(expr, "attr", "") or "").strip()
            if owner and attr:
                return f"{owner}.{attr}"
            return attr
        if isinstance(expr, ast.Call):
            return _ast_expr_qualname(expr.func)
    except Exception:
        return ""
    return ""


_TS_PARSER: Any = None
_TS_PARSER_INIT = False


def _get_tree_sitter_python_parser() -> Any:
    global _TS_PARSER, _TS_PARSER_INIT
    if _TS_PARSER_INIT:
        return _TS_PARSER
    _TS_PARSER_INIT = True
    try:
        from tree_sitter_languages import get_parser  # type: ignore

        _TS_PARSER = get_parser("python")
        return _TS_PARSER
    except Exception:
        pass

    # Fallback for environments where `tree_sitter_languages` wheels are unavailable
    # (e.g. newer Python versions): use `tree_sitter` + `tree_sitter_python`.
    try:
        from tree_sitter import Language, Parser  # type: ignore
        import tree_sitter_python  # type: ignore

        lang_obj: Any = None
        capsule = tree_sitter_python.language()
        try:
            lang_obj = Language(capsule)
        except Exception:
            lang_obj = capsule

        parser = Parser()
        if hasattr(parser, "set_language"):
            parser.set_language(lang_obj)
        else:
            parser.language = lang_obj
        _TS_PARSER = parser
        return _TS_PARSER
    except Exception:
        _TS_PARSER = None
        return None


def _safe_parse_treesitter(src: str) -> Any:
    parser = _get_tree_sitter_python_parser()
    if parser is None:
        return None
    try:
        return parser.parse(src.encode("utf-8", errors="replace"))
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


class _BodyCallCollector(ast.NodeVisitor):
    """Collect call targets from a function body, excluding nested defs/classes."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # pragma: no cover - trivial
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:  # pragma: no cover - trivial
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:  # pragma: no cover - trivial
        return None

    def visit_Call(self, node: ast.Call) -> Any:
        try:
            fn = node.func
            if isinstance(fn, ast.Name):
                name = str(fn.id or "").strip()
                if name:
                    self.calls.append({"kind": "name", "target": name})
            elif isinstance(fn, ast.Attribute):
                attr = str(getattr(fn, "attr", "") or "").strip()
                owner = _ast_expr_qualname(fn.value)
                if attr:
                    if owner in {"self", "cls"}:
                        self.calls.append({"kind": "self_attr", "target": attr})
                    elif owner:
                        self.calls.append({"kind": "obj_attr", "target": attr, "owner": owner})
        except Exception:
            pass
        return self.generic_visit(node)


class _BodyUsageCollector(ast.NodeVisitor):
    """Collect name/attribute reads from a function body, excluding nested defs/classes."""

    def __init__(self) -> None:
        self.names: Set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # pragma: no cover - trivial
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:  # pragma: no cover - trivial
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:  # pragma: no cover - trivial
        return None

    def visit_Name(self, node: ast.Name) -> Any:
        try:
            if isinstance(getattr(node, "ctx", None), ast.Load):
                name = str(getattr(node, "id", "") or "").strip()
                if name:
                    self.names.add(name)
        except Exception:
            pass
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        try:
            attr = str(getattr(node, "attr", "") or "").strip()
            if attr:
                self.names.add(attr)
        except Exception:
            pass
        return self.generic_visit(node)


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
        # deferred call sites to be resolved after all defs are known
        self._pending_calls: List[Dict[str, str]] = []
        self._pending_usages: List[Dict[str, str]] = []
        self._pending_import_refs: List[Dict[str, str]] = []
        self._imported_symbols: Dict[str, Tuple[str, str]] = {}
        self._assignments_by_name: Dict[str, List[str]] = {}

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

    def _current_class_qualname(self) -> str:
        for cid in reversed(self.container_stack):
            if not str(cid).startswith("class:"):
                continue
            node = self.nodes.get(cid) or {}
            qn = str(node.get("name") or "").strip()
            if qn:
                return qn
        return ""

    def _collect_calls_for_function(self, node: ast.AST, caller_id: str) -> None:
        body = list(getattr(node, "body", []) or [])
        if not body:
            return
        collector = _BodyCallCollector()
        for stmt in body:
            collector.visit(stmt)
        if not collector.calls:
            return
        class_ctx = self._current_class_qualname()
        for c in collector.calls:
            rec: Dict[str, str] = {
                "caller": caller_id,
                "kind": str(c.get("kind") or ""),
                "target": str(c.get("target") or ""),
            }
            if class_ctx:
                rec["class_ctx"] = class_ctx
            owner = str(c.get("owner") or "").strip()
            if owner:
                rec["owner"] = owner
            self._pending_calls.append(rec)

    def _collect_usages_for_function(self, node: ast.AST, caller_id: str) -> None:
        body = list(getattr(node, "body", []) or [])
        if not body:
            return
        collector = _BodyUsageCollector()
        for stmt in body:
            collector.visit(stmt)
        for name in sorted(collector.names):
            if name:
                self._pending_usages.append({"src": caller_id, "name": name})

    def _target_names_from_ast(self, node: ast.AST) -> List[str]:
        names: List[str] = []

        def visit_target(t: ast.AST) -> None:
            if isinstance(t, ast.Name):
                if t.id and t.id not in names:
                    names.append(str(t.id))
            elif isinstance(t, ast.Attribute):
                attr = str(getattr(t, "attr", "") or "").strip()
                if attr and attr not in names:
                    names.append(attr)
            elif isinstance(t, (ast.Tuple, ast.List)):
                for elt in list(getattr(t, "elts", []) or []):
                    visit_target(elt)

        try:
            if isinstance(node, ast.Assign):
                for t in list(getattr(node, "targets", []) or []):
                    visit_target(t)
            elif isinstance(node, ast.AnnAssign):
                visit_target(node.target)
            elif isinstance(node, ast.AugAssign):
                visit_target(node.target)
        except Exception:
            return names
        return names

    def _current_assignment_scope(self) -> str:
        cur = self._current_container()
        if cur == self.file_node_id:
            return "module"
        if str(cur).startswith("class:"):
            return "class"
        return "local"

    def _visit_assignment(self, node: ast.AST) -> Any:
        scope = self._current_assignment_scope()
        if scope not in {"module", "class"}:
            # Local assignments are too numerous for now; function bodies already
            # provide sufficient repair context as function nodes.
            return self.generic_visit(node)
        names = self._target_names_from_ast(node)
        if not names:
            return self.generic_visit(node)
        start, end = _node_span(node)
        snippet = _clip_snippet_lines(
            self._file_lines,
            start,
            min(end, start + 20),
            max_lines=min(24, self._max_snippet_lines),
        ) if self._embed_snippets and self._file_lines else []
        kind = "module_assignment" if scope == "module" else "class_assignment"
        for name in names:
            qn = self._qualname(name) if scope == "class" else name
            nid = f"{kind}:{self.file_rel}:{qn}:{start}"
            payload: Dict[str, Any] = {
                "id": nid,
                "kind": kind,
                "name": qn,
                "symbol": name,
                "path": self.file_rel,
                "span": {"start": start, "end": end},
                "scope": scope,
                "sig": (snippet[0].strip() if snippet else f"{name} = ..."),
            }
            if snippet:
                payload["snippet_lines"] = snippet
            self._add_node(nid, payload)
            self._add_edge(self._current_container(), nid, "contains")
            self._assignments_by_name.setdefault(name, []).append(nid)
        return self.generic_visit(node)

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
            for alias in node.names:
                imported = str(getattr(alias, "name", "") or "").strip()
                if not imported or imported == "*":
                    continue
                local = str(getattr(alias, "asname", "") or "").strip() or imported
                self._imported_symbols[local] = (mod, imported)
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
        self._collect_calls_for_function(node, nid)
        self._collect_usages_for_function(node, nid)

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

    def visit_Assign(self, node: ast.Assign) -> Any:
        return self._visit_assignment(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        return self._visit_assignment(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        return self._visit_assignment(node)

    def finalize_call_edges(self) -> None:
        """Resolve collected call targets to in-file function nodes."""
        funcs_by_base: Dict[str, List[str]] = {}
        classes_by_base: Dict[str, List[str]] = {}
        methods_by_class: Dict[Tuple[str, str], List[str]] = {}
        caller_qual: Dict[str, str] = {}

        def add_index(index: Dict[Any, List[str]], key: Any, nid: str) -> None:
            bucket = index.setdefault(key, [])
            if nid not in bucket:
                bucket.append(nid)

        for nid, node in self.nodes.items():
            kind = str(node.get("kind") or "").lower()
            if kind == "class":
                qn = str(node.get("name") or "").strip()
                if qn:
                    add_index(classes_by_base, qn.split(".")[-1], nid)

        for nid, node in self.nodes.items():
            kind = str(node.get("kind") or "").lower()
            if kind not in {"function", "func", "method"}:
                continue
            qn = str(node.get("name") or "").strip()
            if not qn:
                continue
            caller_qual[nid] = qn
            base = qn.split(".")[-1]
            add_index(funcs_by_base, base, nid)
            if "." in qn:
                cls_qn = qn.rsplit(".", 1)[0]
                cls_short = cls_qn.split(".")[-1]
                add_index(methods_by_class, (cls_qn, base), nid)
                add_index(methods_by_class, (cls_short, base), nid)

        if self._pending_calls:
            seen_edges: Set[Tuple[str, str]] = set()
            for rec in self._pending_calls:
                caller = str(rec.get("caller") or "")
                kind = str(rec.get("kind") or "")
                target = str(rec.get("target") or "")
                class_ctx = str(rec.get("class_ctx") or "")
                owner = _normalize_call_owner_name(str(rec.get("owner") or ""))
                if not caller or not target or caller not in caller_qual:
                    continue

                candidates: List[str] = []
                if kind == "self_attr":
                    if class_ctx:
                        candidates = list(methods_by_class.get((class_ctx, target), []))
                    if not candidates:
                        candidates = list(funcs_by_base.get(target, []))
                elif kind == "obj_attr":
                    if owner in self._imported_symbols:
                        mod, imported = self._imported_symbols[owner]
                        self._pending_import_refs.append(
                            {
                                "src": caller,
                                "module": mod,
                                "symbol": imported,
                                "attr": target,
                                "kind": "calls",
                            }
                        )
                    if owner:
                        candidates = list(methods_by_class.get((owner, target), []))
                        owner_short = owner.split(".")[-1]
                        if owner_short and owner_short != owner:
                            candidates.extend(methods_by_class.get((owner_short, target), []))
                    if not candidates and class_ctx:
                        candidates = list(methods_by_class.get((class_ctx, target), []))
                else:  # kind == "name"
                    if target in self._imported_symbols:
                        mod, imported = self._imported_symbols[target]
                        self._pending_import_refs.append(
                            {
                                "src": caller,
                                "module": mod,
                                "symbol": imported,
                                "kind": "calls",
                            }
                        )
                    caller_qn = caller_qual.get(caller, "")
                    caller_base = caller_qn.split(".")[-1] if caller_qn else ""
                    if caller_base and caller_base == target:
                        candidates = [caller]
                    elif class_ctx:
                        candidates = list(methods_by_class.get((class_ctx, target), []))
                    if not candidates:
                        candidates = list(funcs_by_base.get(target, []))
                    if not candidates:
                        # Constructor calls are a real implementation dependency.
                        # A factory/dispatcher often returns SomeClass(...), and the
                        # concrete class is what the repair model needs to inspect.
                        candidates = list(classes_by_base.get(target, []))

                if not candidates:
                    continue
                candidates = _dedupe_preserve(candidates)
                # Keep links conservative: only add when target is unambiguous.
                if len(candidates) != 1:
                    continue
                dst = candidates[0]
                if dst == caller:
                    continue
                key = (caller, dst)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                self._add_edge(caller, dst, "calls")

        seen_uses: Set[Tuple[str, str]] = set()
        for rec in self._pending_usages:
            src = str(rec.get("src") or "")
            name = str(rec.get("name") or "")
            if not src or not name:
                continue
            if name in self._imported_symbols:
                mod, imported = self._imported_symbols[name]
                self._pending_import_refs.append(
                    {
                        "src": src,
                        "module": mod,
                        "symbol": imported,
                        "kind": "uses",
                    }
                )
            targets = list(self._assignments_by_name.get(name, []))
            if not targets:
                continue
            # Keep intra-file symbol linkage conservative. Multiple assignments
            # to the same name still provide useful candidates, but cap fan-out.
            for dst in targets[:4]:
                key = (src, dst)
                if key in seen_uses:
                    continue
                seen_uses.add(key)
                self._add_edge(src, dst, "uses")


class TreeSitterGraphBuilder(GraphBuilder):
    """Best-effort tree-sitter frontend compatible with GraphBuilder output schema."""

    def __init__(
        self,
        file_rel: str,
        file_node_id: str,
        *,
        file_lines: Optional[Sequence[str]] = None,
        src_bytes: bytes,
        embed_snippets: bool = True,
        max_snippet_lines: int = 120,
        max_class_snippet_lines: int = 32,
    ) -> None:
        super().__init__(
            file_rel=file_rel,
            file_node_id=file_node_id,
            file_lines=file_lines,
            embed_snippets=embed_snippets,
            max_snippet_lines=max_snippet_lines,
            max_class_snippet_lines=max_class_snippet_lines,
        )
        self._src_bytes = src_bytes

    def _node_text(self, node: Any) -> str:
        try:
            b0 = int(getattr(node, "start_byte", 0) or 0)
            b1 = int(getattr(node, "end_byte", b0) or b0)
            if b1 <= b0:
                return ""
            return self._src_bytes[b0:b1].decode("utf-8", "replace")
        except Exception:
            return ""

    def _node_span(self, node: Any) -> Tuple[int, int]:
        try:
            s = int(getattr(node, "start_point", (0, 0))[0]) + 1
            e = int(getattr(node, "end_point", (s - 1, 0))[0]) + 1
            if e < s:
                e = s
            return s, e
        except Exception:
            return 1, 1

    def _child_by_field_name(self, node: Any, name: str) -> Any:
        try:
            return node.child_by_field_name(name)
        except Exception:
            return None

    def _named_children(self, node: Any) -> List[Any]:
        try:
            return list(getattr(node, "named_children", []) or [])
        except Exception:
            return []

    def _iter_body_calls(self, body_node: Any) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if body_node is None:
            return out

        def walk(n: Any, *, root: bool = False) -> None:
            ntype = str(getattr(n, "type", "") or "")
            if (not root) and ntype in {"function_definition", "async_function_definition", "class_definition"}:
                return
            if ntype == "decorated_definition":
                # decorators wrap nested defs; skip to avoid leaking nested-body calls.
                return
            if ntype == "call":
                fn = self._child_by_field_name(n, "function")
                ftype = str(getattr(fn, "type", "") or "") if fn is not None else ""
                if ftype == "identifier":
                    target = self._node_text(fn).strip()
                    if target:
                        out.append({"kind": "name", "target": target})
                elif ftype == "attribute":
                    owner_n = self._child_by_field_name(fn, "object")
                    attr_n = self._child_by_field_name(fn, "attribute")
                    owner = self._node_text(owner_n).strip() if owner_n is not None else ""
                    target = self._node_text(attr_n).strip() if attr_n is not None else ""
                    if target:
                        if owner in {"self", "cls"}:
                            out.append({"kind": "self_attr", "target": target})
                        elif owner:
                            out.append({"kind": "obj_attr", "target": target, "owner": owner})
            for ch in self._named_children(n):
                walk(ch, root=False)

        walk(body_node, root=True)
        return out

    def _iter_body_usages(self, body_node: Any) -> Set[str]:
        names: Set[str] = set()
        if body_node is None:
            return names

        def walk(n: Any, *, root: bool = False) -> None:
            ntype = str(getattr(n, "type", "") or "")
            if (not root) and ntype in {"function_definition", "async_function_definition", "class_definition"}:
                return
            if ntype == "decorated_definition":
                return
            if ntype == "identifier":
                text = self._node_text(n).strip()
                if text:
                    names.add(text)
            elif ntype == "attribute":
                attr_n = self._child_by_field_name(n, "attribute")
                text = self._node_text(attr_n).strip() if attr_n is not None else ""
                if text:
                    names.add(text)
            for ch in self._named_children(n):
                walk(ch, root=False)

        walk(body_node, root=True)
        return names

    def _assignment_target_names_ts(self, node: Any) -> List[str]:
        names: List[str] = []

        def add_name(n: Any) -> None:
            if n is None:
                return
            ntype = str(getattr(n, "type", "") or "")
            if ntype == "identifier":
                text = self._node_text(n).strip()
                if text and text not in names:
                    names.append(text)
            elif ntype == "attribute":
                attr_n = self._child_by_field_name(n, "attribute")
                text = self._node_text(attr_n).strip() if attr_n is not None else ""
                if text and text not in names:
                    names.append(text)
            elif ntype in {"pattern_list", "tuple", "list"}:
                for ch in self._named_children(n):
                    add_name(ch)

        left = self._child_by_field_name(node, "left")
        target = self._child_by_field_name(node, "target")
        add_name(left or target)
        if not names:
            for ch in self._named_children(node)[:2]:
                add_name(ch)
        return names

    def _visit_assignment_ts(self, node: Any) -> None:
        scope = self._current_assignment_scope()
        if scope not in {"module", "class"}:
            self._visit_block(node)
            return
        names = self._assignment_target_names_ts(node)
        if not names:
            self._visit_block(node)
            return
        start, end = self._node_span(node)
        snippet = _clip_snippet_lines(
            self._file_lines,
            start,
            min(end, start + 20),
            max_lines=min(24, self._max_snippet_lines),
        ) if self._embed_snippets and self._file_lines else []
        kind = "module_assignment" if scope == "module" else "class_assignment"
        for name in names:
            qn = self._qualname(name) if scope == "class" else name
            nid = f"{kind}:{self.file_rel}:{qn}:{start}"
            payload: Dict[str, Any] = {
                "id": nid,
                "kind": kind,
                "name": qn,
                "symbol": name,
                "path": self.file_rel,
                "span": {"start": start, "end": end},
                "scope": scope,
                "sig": (snippet[0].strip() if snippet else f"{name} = ..."),
            }
            if snippet:
                payload["snippet_lines"] = snippet
            self._add_node(nid, payload)
            self._add_edge(self._current_container(), nid, "contains")
            self._assignments_by_name.setdefault(name, []).append(nid)
        self._visit_block(node)

    def _add_import_edge(self, mod: str) -> None:
        mod = (mod or "").strip()
        if not mod:
            return
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

    def _record_imported_symbol(self, local: str, mod: str, imported: str) -> None:
        local = (local or "").strip()
        mod = (mod or "").strip()
        imported = (imported or "").strip()
        if local and mod and imported and imported != "*":
            self._imported_symbols[local] = (mod, imported)

    def _visit_import_statement(self, node: Any) -> None:
        text = self._node_text(node).strip()
        if not text.startswith("import "):
            return
        rhs = text[len("import ") :]
        for part in rhs.split(","):
            p = part.strip()
            if not p:
                continue
            name = p.split(" as ", 1)[0].strip()
            if name:
                self._add_import_edge(name)

    def _visit_import_from_statement(self, node: Any) -> None:
        text = self._node_text(node).strip()
        m = re.match(r"from\s+([^\s]+)\s+import\s+", text)
        if not m:
            return
        mod = (m.group(1) or "").strip()
        self._add_import_edge(mod)
        rhs = text[m.end():].strip()
        # Best-effort import-list parsing; enough for `from x import a, b as c`.
        rhs = rhs.strip("()")
        for part in rhs.split(","):
            raw = part.strip()
            if not raw:
                continue
            if " as " in raw:
                imported, local = [x.strip() for x in raw.split(" as ", 1)]
            else:
                imported = raw.strip()
                local = imported
            self._record_imported_symbol(local, mod, imported)

    def _visit_class_definition(self, node: Any) -> None:
        name_node = self._child_by_field_name(node, "name")
        name = self._node_text(name_node).strip() if name_node is not None else ""
        if not name:
            return
        qn = self._qualname(name)
        start, end = self._node_span(node)
        nid = f"class:{self.file_rel}:{qn}:{start}"
        payload: Dict[str, Any] = {
            "id": nid,
            "kind": "class",
            "name": qn,
            "path": self.file_rel,
            "span": {"start": start, "end": end},
            "sig": f"class {qn}",
            "doc": "",
        }
        if self._embed_snippets and self._file_lines:
            snip = _clip_snippet_lines(self._file_lines, start, end, max_lines=self._max_class_snippet_lines)
            if snip:
                payload["snippet_lines"] = snip
                payload["sig"] = _extract_sig_from_snippet(snip) or payload["sig"]
        self._add_node(nid, payload)
        self._add_edge(self._current_container(), nid, "contains")

        self.container_stack.append(nid)
        self.scope_names.append(name)
        body = self._child_by_field_name(node, "body")
        self._visit_block(body)
        self.scope_names.pop()
        self.container_stack.pop()

    def _visit_function_definition(self, node: Any, *, is_async: bool) -> None:
        name_node = self._child_by_field_name(node, "name")
        name = self._node_text(name_node).strip() if name_node is not None else ""
        if not name:
            return
        qn = self._qualname(name)
        start, end = self._node_span(node)
        nid = f"func:{self.file_rel}:{qn}:{start}"
        kind_kw = "async def" if is_async else "def"
        payload: Dict[str, Any] = {
            "id": nid,
            "kind": "function",
            "name": qn,
            "path": self.file_rel,
            "span": {"start": start, "end": end},
            "async": bool(is_async),
            "sig": f"{kind_kw} {qn}(...)",
            "doc": "",
        }
        if self._embed_snippets and self._file_lines:
            snip = _clip_snippet_lines(self._file_lines, start, end, max_lines=self._max_snippet_lines)
            if snip:
                payload["snippet_lines"] = snip
                payload["sig"] = _extract_sig_from_snippet(snip) or payload["sig"]
        self._add_node(nid, payload)
        self._add_edge(self._current_container(), nid, "contains")

        class_ctx = self._current_class_qualname()
        body_node = self._child_by_field_name(node, "body")
        for c in self._iter_body_calls(body_node):
            rec: Dict[str, str] = {
                "caller": nid,
                "kind": str(c.get("kind") or ""),
                "target": str(c.get("target") or ""),
            }
            if class_ctx:
                rec["class_ctx"] = class_ctx
            owner = str(c.get("owner") or "").strip()
            if owner:
                rec["owner"] = owner
            self._pending_calls.append(rec)
        for name_used in sorted(self._iter_body_usages(body_node)):
            if name_used:
                self._pending_usages.append({"src": nid, "name": name_used})

        self.container_stack.append(nid)
        self.scope_names.append(name)
        self._visit_block(body_node)
        self.scope_names.pop()
        self.container_stack.pop()

    def _visit_block(self, block_node: Any) -> None:
        if block_node is None:
            return
        for ch in self._named_children(block_node):
            self._visit_node(ch)

    def _visit_node(self, node: Any) -> None:
        ntype = str(getattr(node, "type", "") or "")
        if ntype == "decorated_definition":
            for ch in self._named_children(node):
                ctype = str(getattr(ch, "type", "") or "")
                if ctype in {"function_definition", "async_function_definition", "class_definition"}:
                    self._visit_node(ch)
                    return
            return
        if ntype == "import_statement":
            self._visit_import_statement(node)
            return
        if ntype == "import_from_statement":
            self._visit_import_from_statement(node)
            return
        if ntype == "class_definition":
            self._visit_class_definition(node)
            return
        if ntype == "function_definition":
            self._visit_function_definition(node, is_async=False)
            return
        if ntype == "async_function_definition":
            self._visit_function_definition(node, is_async=True)
            return
        if ntype in {"assignment", "augmented_assignment", "typed_parameter", "type_alias_statement"}:
            if ntype in {"assignment", "augmented_assignment"}:
                self._visit_assignment_ts(node)
                return
        for ch in self._named_children(node):
            self._visit_node(ch)

    def build(self, tree: Any) -> None:
        root = getattr(tree, "root_node", None)
        if root is None:
            return
        self._visit_block(root)
        self.finalize_call_edges()


_GO_FUNC_RE = re.compile(r"^\s*func\s+(?:\((?P<recv>[^)]*)\)\s*)?(?P<name>[A-Za-z_]\w*)\s*\(")
_GO_TYPE_RE = re.compile(r"^\s*type\s+(?P<name>[A-Za-z_]\w*)\s+(?P<form>struct|interface)\b")
_GO_TYPE_BLOCK_ITEM_RE = re.compile(r"^\s*(?P<name>[A-Za-z_]\w*)\s+(?P<form>struct|interface)\b")
_GO_TYPE_ALIAS_RE = re.compile(r"^\s*(?:type\s+)?(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<rhs>.+)$")
_GO_VAR_CONST_RE = re.compile(r"^\s*(?P<kw>var|const)\s+(?P<name>[A-Za-z_]\w*)\b")
_GO_CALL_NAME_RE = re.compile(r"(?<![.])\b([A-Za-z_]\w*)\s*\(")
_GO_CALL_SELECTOR_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)\s*\(")
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


def _go_without_line_comment(line: str) -> str:
    """Strip Go // comments well enough for brace/call scanning."""
    out: List[str] = []
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


def _go_find_brace_span(lines: Sequence[str], start_idx: int) -> Tuple[int, int]:
    """Return inclusive 1-based span for a Go declaration starting at start_idx."""
    balance = 0
    seen_open = False
    for idx in range(start_idx, len(lines)):
        text = _go_without_line_comment(str(lines[idx]))
        for ch in text:
            if ch == "{":
                balance += 1
                seen_open = True
            elif ch == "}" and seen_open:
                balance -= 1
        if seen_open and balance <= 0:
            return start_idx + 1, idx + 1
    return start_idx + 1, start_idx + 1


def _go_receiver_parts(receiver: str) -> Tuple[str, str]:
    raw = str(receiver or "").strip()
    if not raw:
        return "", ""
    raw = raw.replace("*", " ")
    parts = raw.split()
    if not parts:
        return "", ""
    var_name = parts[0].strip() if len(parts) >= 2 else ""
    typ = parts[-1].strip()
    typ = typ.split("[", 1)[0]
    typ = typ.split(".")[-1]
    return var_name, re.sub(r"[^A-Za-z0-9_]", "", typ)


def _go_receiver_type(receiver: str) -> str:
    return _go_receiver_parts(receiver)[1]


def _go_file_summary_snippet(file_lines: Sequence[str], symbols: Sequence[Tuple[int, str]], *, max_lines: int) -> List[str]:
    if max_lines <= 0:
        return []
    out: List[str] = []
    if symbols:
        out.append("// Top-level Go symbols (sig @ line):")
        for lineno, sig in symbols:
            if len(out) >= max_lines:
                break
            sig = " ".join(str(sig or "").strip().split())
            if len(sig) > 160:
                sig = sig[:157] + "..."
            out.append(f"{sig}  // L{lineno}")
    return out[:max_lines]


def _go_collect_call_refs(
    lines: Sequence[str],
    start: int,
    end: int,
    *,
    receiver_name: str = "",
    receiver_type: str = "",
) -> List[Dict[str, str]]:
    text = "\n".join(str(x) for x in lines[max(0, start - 1) : max(start - 1, end)])
    text = "\n".join(_go_without_line_comment(x) for x in text.splitlines())
    refs: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str, str]] = set()
    for m in _GO_CALL_SELECTOR_RE.finditer(text):
        owner = str(m.group(1) or "").strip()
        name = str(m.group(2) or "").strip()
        if (
            receiver_name
            and receiver_type
            and owner == receiver_name
            and name
            and name not in _GO_KEYWORDS
        ):
            key = ("method", receiver_type, name)
            if key not in seen:
                seen.add(key)
                refs.append({"kind": "method", "receiver": receiver_type, "target": name})
    for m in _GO_CALL_NAME_RE.finditer(text):
        name = str(m.group(1) or "").strip()
        if name and name not in _GO_KEYWORDS:
            key = ("bare", "", name)
            if key not in seen:
                seen.add(key)
                refs.append({"kind": "bare", "target": name})
    return refs


def _build_go_file_graph(
    rel: str,
    src: str,
    *,
    file_id: str,
    embed_snippets: bool,
    max_file_snippet_lines: int,
    max_def_snippet_lines: int,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]]]:
    """Best-effort Go structural graph.

    This intentionally stays parser-free so the graph builder works inside lean
    benchmark containers. It extracts enough structure for planner search,
    read hydration, and expand: files, top-level types, functions/methods, and
    conservative intra-repo call references resolved after all Go files are read.
    """
    lines = src.splitlines()
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    pending_calls: List[Dict[str, str]] = []
    type_nodes: Dict[str, str] = {}
    symbols: List[Tuple[int, str]] = []

    def add_node(nid: str, payload: Dict[str, Any]) -> None:
        if nid not in nodes:
            nodes[nid] = payload

    def add_edge(src_id: str, dst_id: str, kind: str) -> None:
        edges.append({"src": src_id, "dst": dst_id, "kind": kind})

    in_type_block = False
    for idx, line in enumerate(lines):
        stripped = _go_without_line_comment(line).strip()
        if not stripped:
            continue
        if re.match(r"^type\s*\(\s*$", stripped):
            in_type_block = True
            continue
        if in_type_block and stripped == ")":
            in_type_block = False
            continue

        m = _GO_TYPE_RE.match(line)
        if not m and in_type_block:
            m = _GO_TYPE_BLOCK_ITEM_RE.match(line)
        if not m:
            continue
        name = str(m.group("name") or "").strip()
        form = str(m.group("form") or "").strip()
        if not name:
            continue
        start, end = _go_find_brace_span(lines, idx)
        nid = f"class:{rel}:{name}:{start}"
        snippet = _clip_snippet_lines(lines, start, end, max_lines=max_def_snippet_lines) if embed_snippets else []
        payload: Dict[str, Any] = {
            "id": nid,
            "kind": "class",
            "name": name,
            "symbol": name,
            "path": rel,
            "span": {"start": start, "end": end},
            "sig": stripped,
            "language": "go",
            "go_kind": form,
        }
        if snippet:
            payload["snippet_lines"] = snippet
        add_node(nid, payload)
        add_edge(file_id, nid, "contains")
        type_nodes[name] = nid
        symbols.append((start, stripped))

    for idx, line in enumerate(lines):
        stripped = _go_without_line_comment(line).strip()
        m = _GO_FUNC_RE.match(line)
        if not m:
            continue
        name = str(m.group("name") or "").strip()
        recv_name, recv_type = _go_receiver_parts(str(m.group("recv") or ""))
        if not name:
            continue
        start, end = _go_find_brace_span(lines, idx)
        qn = f"{recv_type}.{name}" if recv_type else name
        nid = f"func:{rel}:{qn}:{start}"
        snippet = _clip_snippet_lines(lines, start, end, max_lines=max_def_snippet_lines) if embed_snippets else []
        payload2: Dict[str, Any] = {
            "id": nid,
            "kind": "function",
            "name": qn,
            "symbol": name,
            "path": rel,
            "span": {"start": start, "end": end},
            "sig": stripped,
            "language": "go",
        }
        if recv_type:
            payload2["receiver"] = recv_type
        if snippet:
            payload2["snippet_lines"] = snippet
        add_node(nid, payload2)
        parent = type_nodes.get(recv_type) if recv_type else ""
        add_edge(parent or file_id, nid, "contains")
        symbols.append((start, stripped))
        for call_ref in _go_collect_call_refs(
            lines,
            start,
            end,
            receiver_name=recv_name,
            receiver_type=recv_type,
        ):
            call_name = str(call_ref.get("target") or "").strip()
            if call_name == name:
                continue
            rec = {"src": nid, "target": call_name, "kind": str(call_ref.get("kind") or "bare")}
            receiver = str(call_ref.get("receiver") or "").strip()
            if receiver:
                rec["receiver"] = receiver
            pending_calls.append(rec)

    for idx, line in enumerate(lines):
        stripped = _go_without_line_comment(line).strip()
        m = _GO_VAR_CONST_RE.match(line)
        if not m:
            m_alias = _GO_TYPE_ALIAS_RE.match(line)
            if not (m_alias and stripped.startswith("type ")):
                continue
            name = str(m_alias.group("name") or "").strip()
            kw = "type"
        else:
            name = str(m.group("name") or "").strip()
            kw = str(m.group("kw") or "").strip()
        if not name:
            continue
        start = idx + 1
        nid = f"module_assignment:{rel}:{name}:{start}"
        payload3: Dict[str, Any] = {
            "id": nid,
            "kind": "module_assignment",
            "name": name,
            "symbol": name,
            "path": rel,
            "span": {"start": start, "end": start},
            "sig": stripped,
            "language": "go",
            "go_kind": kw,
        }
        if embed_snippets:
            payload3["snippet_lines"] = [line.rstrip("\n")]
        add_node(nid, payload3)
        add_edge(file_id, nid, "contains")
        symbols.append((start, stripped))

    if file_id not in nodes:
        payload_file: Dict[str, Any] = {
            "id": file_id,
            "kind": "file",
            "name": rel,
            "path": rel,
            "span": None,
            "sig": f"go file {rel}",
            "language": "go",
        }
        if embed_snippets:
            summary = _go_file_summary_snippet(lines, sorted(symbols), max_lines=max_file_snippet_lines)
            if summary:
                payload_file["snippet_lines"] = summary
        add_node(file_id, payload_file)

    return nodes, edges, pending_calls


def _resolve_go_call_refs(nodes: Dict[str, Dict[str, Any]], edges: List[Dict[str, Any]], refs: Sequence[Dict[str, str]]) -> None:
    if not refs:
        return
    by_base: Dict[str, List[str]] = {}
    by_method: Dict[Tuple[str, str], List[str]] = {}
    for nid, node in nodes.items():
        if str(node.get("kind") or "").lower() not in {"function", "func", "method"}:
            continue
        symbol = str(node.get("symbol") or "").strip()
        name = str(node.get("name") or "").strip()
        base = symbol or (name.split(".")[-1] if name else "")
        if base:
            by_base.setdefault(base, []).append(nid)
        receiver = str(node.get("receiver") or "").strip()
        if receiver and base:
            by_method.setdefault((receiver, base), []).append(nid)
    seen: Set[Tuple[str, str, str]] = {
        (str(e.get("src") or ""), str(e.get("dst") or ""), str(e.get("kind") or ""))
        for e in edges
        if isinstance(e, dict)
    }
    for rec in refs:
        src = str(rec.get("src") or "").strip()
        target = str(rec.get("target") or "").strip()
        if not src or not target:
            continue
        kind = str(rec.get("kind") or "bare").strip()
        receiver = str(rec.get("receiver") or "").strip()
        if kind == "method" and receiver:
            candidates = [x for x in by_method.get((receiver, target), []) if x != src]
        else:
            candidates = [x for x in by_base.get(target, []) if x != src]
        if len(candidates) != 1:
            continue
        dst = candidates[0]
        key = (src, dst, "calls")
        if key in seen:
            continue
        seen.add(key)
        edges.append({"src": src, "dst": dst, "kind": "calls"})


_JS_CLASS_RE = re.compile(r"^\s*(?:export\s+default\s+|export\s+)?class\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\b")
_JS_FUNCTION_RE = re.compile(
    r"^\s*(?:export\s+default\s+|export\s+)?(?:async\s+)?function\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*\((?P<params>[^)]*)\)"
)
_JS_ARROW_RE = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\((?P<params1>[^)]*)\)|(?P<param2>[A-Za-z_$][A-Za-z0-9_$]*))\s*=>"
)
_JS_FUNCTION_ASSIGN_RE = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*(?::[^=]+)?=\s*(?:async\s+)?function\b\s*(?:[A-Za-z_$][A-Za-z0-9_$]*)?\s*\((?P<params>[^)]*)\)"
)
_JS_METHOD_RE = re.compile(
    r"^\s*(?:public\s+|private\s+|protected\s+|static\s+|async\s+|override\s+|readonly\s+|get\s+|set\s+|\*)*(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*\((?P<params>[^)]*)\)\s*(?::[^{]+)?\{?"
)
_JS_CALL_IDENT_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_$")
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


def _js_without_line_comment(line: str) -> str:
    out: List[str] = []
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


def _brace_span_generic(lines: Sequence[str], start_idx: int, *, comment_stripper) -> Tuple[int, int]:
    balance = 0
    seen_open = False
    for idx in range(start_idx, len(lines)):
        text = comment_stripper(str(lines[idx]))
        for ch in text:
            if ch == "{":
                balance += 1
                seen_open = True
            elif ch == "}" and seen_open:
                balance -= 1
        if seen_open and balance <= 0:
            return start_idx + 1, idx + 1
    return start_idx + 1, start_idx + 1


def _js_file_summary_snippet(file_lines: Sequence[str], symbols: Sequence[Tuple[int, str]], *, max_lines: int) -> List[str]:
    if max_lines <= 0:
        return []
    out: List[str] = []
    if symbols:
        out.append("// Top-level JS/TS symbols (sig @ line):")
        for lineno, sig in symbols:
            if len(out) >= max_lines:
                break
            sig = " ".join(str(sig or "").strip().split())
            if len(sig) > 160:
                sig = sig[:157] + "..."
            out.append(f"{sig}  // L{lineno}")
    return out[:max_lines]


def _js_is_decl_line(prefix: str) -> bool:
    stripped = prefix.strip()
    return bool(
        re.search(r"\b(function|class|interface|type|enum)\s*$", stripped)
        or re.search(r"\b(const|let|var)\s+[A-Za-z_$][A-Za-z0-9_$]*\s*=\s*$", stripped)
        or stripped.endswith("=>")
    )


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


def _js_call_is_probably_declaration(text: str, name_start: int, open_idx: int, close_idx: int) -> bool:
    line_start = text.rfind("\n", 0, name_start) + 1
    prefix = text[line_start:name_start]
    if _js_is_decl_line(prefix):
        return True
    suffix_start = _skip_space(text, close_idx + 1)
    suffix = text[suffix_start : min(len(text), suffix_start + 4)]
    args = text[open_idx + 1 : close_idx]
    # Class/object methods often look like `name(arg: Type) { ... }` inside a
    # snippet. Calls almost never have typed parameters immediately before `{`.
    if not prefix.strip() and (suffix.startswith("{") or suffix.startswith(":")) and ":" in args:
        return True
    return False


def _js_collect_call_refs(lines: Sequence[str], start: int, end: int, *, class_name: str = "") -> List[Dict[str, str]]:
    text = "\n".join(str(x) for x in lines[max(0, start - 1) : max(start - 1, end)])
    text = "\n".join(_js_without_line_comment(x) for x in text.splitlines())
    refs: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str, str]] = set()
    i = 0
    while i < len(text):
        ch = text[i]
        if not (ch == "_" or ch == "$" or ch.isalpha()):
            i += 1
            continue
        if i > 0 and text[i - 1] in _JS_CALL_IDENT_CHARS:
            i += 1
            continue
        j = i + 1
        while j < len(text) and text[j] in _JS_CALL_IDENT_CHARS:
            j += 1
        first = text[i:j]
        k = _skip_space(text, j)
        owner = ""
        name = first
        if k < len(text) and text[k] == ".":
            k2 = _skip_space(text, k + 1)
            if k2 < len(text) and (text[k2] == "_" or text[k2] == "$" or text[k2].isalpha()):
                j2 = k2 + 1
                while j2 < len(text) and text[j2] in _JS_CALL_IDENT_CHARS:
                    j2 += 1
                owner = first
                name = text[k2:j2]
                k = _skip_space(text, j2)
        if name in _JS_KEYWORDS:
            i = j
            continue
        if k >= len(text) or text[k] != "(":
            i = j
            continue
        line_start = text.rfind("\n", 0, i) + 1
        if _js_is_decl_line(text[line_start:i]):
            i = k + 1
            continue
        if i > 0 and text[i - 1] == "." and not owner:
            i = j
            continue
        end_paren = _matching_paren(text, k)
        if end_paren <= k:
            i = k + 1
            continue
        if _js_call_is_probably_declaration(text, i, k, end_paren):
            i = end_paren + 1
            continue
        if owner == "this" and class_name:
            key = ("method", class_name, name)
            if key not in seen:
                seen.add(key)
                refs.append({"kind": "method", "class": class_name, "target": name})
        elif not owner:
            key = ("bare", "", name)
            if key not in seen:
                seen.add(key)
                refs.append({"kind": "bare", "target": name})
        i = end_paren + 1
    return refs


def _build_js_ts_file_graph(
    rel: str,
    src: str,
    *,
    file_id: str,
    language: str,
    embed_snippets: bool,
    max_file_snippet_lines: int,
    max_def_snippet_lines: int,
    max_class_snippet_lines: int,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]]]:
    lines = src.splitlines()
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    pending_calls: List[Dict[str, str]] = []
    class_nodes: Dict[str, Tuple[str, int, int]] = {}
    symbols: List[Tuple[int, str]] = []
    covered_function_lines: Set[int] = set()

    def add_node(nid: str, payload: Dict[str, Any]) -> None:
        if nid not in nodes:
            nodes[nid] = payload

    def add_edge(src_id: str, dst_id: str, kind: str) -> None:
        edges.append({"src": src_id, "dst": dst_id, "kind": kind})

    for idx, line in enumerate(lines):
        stripped = _js_without_line_comment(line).strip()
        m = _JS_CLASS_RE.match(line)
        if not m:
            continue
        name = str(m.group("name") or "").strip()
        if not name:
            continue
        start, end = _brace_span_generic(lines, idx, comment_stripper=_js_without_line_comment)
        nid = f"class:{rel}:{name}:{start}"
        snippet = _clip_snippet_lines(lines, start, end, max_lines=max_class_snippet_lines) if embed_snippets else []
        payload: Dict[str, Any] = {
            "id": nid,
            "kind": "class",
            "name": name,
            "symbol": name,
            "path": rel,
            "span": {"start": start, "end": end},
            "sig": stripped,
            "language": language,
        }
        if snippet:
            payload["snippet_lines"] = snippet
        add_node(nid, payload)
        add_edge(file_id, nid, "contains")
        class_nodes[name] = (nid, start, end)
        symbols.append((start, stripped))

    for class_name, (class_id, class_start, class_end) in sorted(class_nodes.items(), key=lambda item: item[1][1]):
        brace_depth = 0
        for idx in range(class_start - 1, min(class_end, len(lines))):
            line = lines[idx]
            clean = _js_without_line_comment(line)
            stripped = clean.strip()
            depth_before = brace_depth
            for ch in clean:
                if ch == "{":
                    brace_depth += 1
                elif ch == "}" and brace_depth > 0:
                    brace_depth -= 1
            if idx == class_start - 1:
                continue
            if depth_before != 1 or not stripped:
                continue
            if stripped.startswith(("if ", "if(", "for ", "for(", "while ", "while(", "switch ", "switch(", "catch ", "catch(")):
                continue
            m = _JS_METHOD_RE.match(line)
            if not m:
                continue
            name = str(m.group("name") or "").strip()
            if not name or name in _JS_KEYWORDS:
                continue
            start, end = _brace_span_generic(lines, idx, comment_stripper=_js_without_line_comment)
            qn = f"{class_name}.{name}"
            nid = f"func:{rel}:{qn}:{start}"
            snippet = _clip_snippet_lines(lines, start, end, max_lines=max_def_snippet_lines) if embed_snippets else []
            payload2: Dict[str, Any] = {
                "id": nid,
                "kind": "function",
                "name": qn,
                "symbol": name,
                "path": rel,
                "span": {"start": start, "end": end},
                "sig": stripped,
                "language": language,
            }
            if snippet:
                payload2["snippet_lines"] = snippet
            add_node(nid, payload2)
            add_edge(class_id, nid, "contains")
            covered_function_lines.add(start)
            symbols.append((start, stripped))
            for call_ref in _js_collect_call_refs(lines, start, end, class_name=class_name):
                target = str(call_ref.get("target") or "").strip()
                if not target or target == name:
                    continue
                rec = {"src": nid, "target": target, "kind": str(call_ref.get("kind") or "bare")}
                cls = str(call_ref.get("class") or "").strip()
                if cls:
                    rec["class"] = cls
                pending_calls.append(rec)

    for idx, line in enumerate(lines):
        stripped = _js_without_line_comment(line).strip()
        if not stripped:
            continue
        m = _JS_FUNCTION_RE.match(line) or _JS_ARROW_RE.match(line) or _JS_FUNCTION_ASSIGN_RE.match(line)
        if not m:
            continue
        name = str(m.group("name") or "").strip()
        if not name:
            continue
        start, end = _brace_span_generic(lines, idx, comment_stripper=_js_without_line_comment)
        if start in covered_function_lines:
            continue
        nid = f"func:{rel}:{name}:{start}"
        snippet = _clip_snippet_lines(lines, start, end, max_lines=max_def_snippet_lines) if embed_snippets else []
        payload3: Dict[str, Any] = {
            "id": nid,
            "kind": "function",
            "name": name,
            "symbol": name,
            "path": rel,
            "span": {"start": start, "end": end},
            "sig": stripped,
            "language": language,
        }
        if snippet:
            payload3["snippet_lines"] = snippet
        add_node(nid, payload3)
        add_edge(file_id, nid, "contains")
        symbols.append((start, stripped))
        for call_ref in _js_collect_call_refs(lines, start, end):
            target = str(call_ref.get("target") or "").strip()
            if target and target != name:
                pending_calls.append({"src": nid, "target": target, "kind": str(call_ref.get("kind") or "bare")})

    if file_id not in nodes:
        payload_file: Dict[str, Any] = {
            "id": file_id,
            "kind": "file",
            "name": rel,
            "path": rel,
            "span": None,
            "sig": f"{language} file {rel}",
            "language": language,
        }
        if embed_snippets:
            summary = _js_file_summary_snippet(lines, sorted(symbols), max_lines=max_file_snippet_lines)
            if summary:
                payload_file["snippet_lines"] = summary
        add_node(file_id, payload_file)

    return nodes, edges, pending_calls


def _resolve_js_call_refs(nodes: Dict[str, Dict[str, Any]], edges: List[Dict[str, Any]], refs: Sequence[Dict[str, str]]) -> None:
    if not refs:
        return
    by_base: Dict[str, List[str]] = {}
    by_method: Dict[Tuple[str, str], List[str]] = {}
    for nid, node in nodes.items():
        if str(node.get("kind") or "").lower() not in {"function", "func", "method"}:
            continue
        language = str(node.get("language") or "").lower()
        if language not in {"javascript", "typescript"}:
            continue
        symbol = str(node.get("symbol") or "").strip()
        name = str(node.get("name") or "").strip()
        base = symbol or (name.rsplit(".", 1)[-1] if name else "")
        if base:
            by_base.setdefault(base, []).append(nid)
        if "." in name and base:
            cls = name.rsplit(".", 1)[0]
            by_method.setdefault((cls, base), []).append(nid)
    seen: Set[Tuple[str, str, str]] = {
        (str(e.get("src") or ""), str(e.get("dst") or ""), str(e.get("kind") or ""))
        for e in edges
        if isinstance(e, dict)
    }
    for rec in refs:
        src = str(rec.get("src") or "").strip()
        target = str(rec.get("target") or "").strip()
        if not src or not target:
            continue
        cls = str(rec.get("class") or "").strip()
        if str(rec.get("kind") or "") == "method" and cls:
            candidates = [x for x in by_method.get((cls, target), []) if x != src]
        else:
            candidates = [x for x in by_base.get(target, []) if x != src]
        if len(candidates) != 1:
            continue
        dst = candidates[0]
        key = (src, dst, "calls")
        if key in seen:
            continue
        seen.add(key)
        edges.append({"src": src, "dst": dst, "kind": "calls"})


_GENERIC_SYMBOL_RE = re.compile(
    r"^\s*(?:export\s+|public\s+|private\s+|protected\s+|static\s+|final\s+|override\s+|async\s+)*"
    r"(?:(?P<class_kw>class|interface|struct|enum)\s+(?P<class_name>[A-Za-z_][A-Za-z0-9_]*)|"
    r"(?P<func_kw>def|fn|func|function)\s+(?P<func_name>[A-Za-z_][A-Za-z0-9_]*)\s*\()"
)


def _build_generic_file_graph(
    rel: str,
    src: str,
    *,
    file_id: str,
    language: str,
    embed_snippets: bool,
    max_file_snippet_lines: int,
    max_def_snippet_lines: int,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    lines = src.splitlines()
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    symbols: List[Tuple[int, str]] = []

    def add_node(nid: str, payload: Dict[str, Any]) -> None:
        if nid not in nodes:
            nodes[nid] = payload

    def add_edge(src_id: str, dst_id: str, kind: str) -> None:
        edges.append({"src": src_id, "dst": dst_id, "kind": kind})

    for idx, line in enumerate(lines):
        stripped = line.strip()
        m = _GENERIC_SYMBOL_RE.match(line)
        if not m:
            continue
        name = str(m.group("class_name") or m.group("func_name") or "").strip()
        if not name:
            continue
        kind = "class" if m.group("class_kw") else "function"
        start, end = _brace_span_generic(lines, idx, comment_stripper=lambda x: str(x).split("//", 1)[0])
        nid = f"{kind}:{rel}:{name}:{start}"
        snippet = _clip_snippet_lines(lines, start, end, max_lines=max_def_snippet_lines) if embed_snippets else []
        payload: Dict[str, Any] = {
            "id": nid,
            "kind": kind,
            "name": name,
            "symbol": name,
            "path": rel,
            "span": {"start": start, "end": end},
            "sig": stripped,
            "language": language,
        }
        if snippet:
            payload["snippet_lines"] = snippet
        add_node(nid, payload)
        add_edge(file_id, nid, "contains")
        symbols.append((start, stripped))

    payload_file: Dict[str, Any] = {
        "id": file_id,
        "kind": "file",
        "name": rel,
        "path": rel,
        "span": None,
        "sig": f"{language} file {rel}",
        "language": language,
    }
    if embed_snippets and symbols:
        payload_file["snippet_lines"] = _js_file_summary_snippet(lines, sorted(symbols), max_lines=max_file_snippet_lines)
    add_node(file_id, payload_file)
    return nodes, edges


def _module_to_candidate_paths(module: str) -> List[str]:
    mod = (module or "").strip()
    if not mod:
        return []
    # Relative imports are common inside packages. For repo-level postprocessing,
    # strip leading dots and match by suffix so `.serializer` can resolve to
    # `django/db/migrations/serializer.py`.
    mod = mod.lstrip(".")
    path = mod.replace(".", "/")
    out = []
    if path:
        out.append(f"{path}.py")
        out.append(f"{path}/__init__.py")
    return out


def _resolve_import_refs(
    nodes: Dict[str, Dict[str, Any]],
    edges: List[Dict[str, Any]],
    refs: Sequence[Dict[str, str]],
) -> None:
    """Resolve cross-file imported symbol references into graph edges.

    Per-file call collection can see `serializer_factory(value)` and
    `Serializer.register(...)`, but it cannot resolve those imported names until
    all repo nodes are known. This postpass links those references back to the
    concrete function/class/method nodes in the imported module.
    """
    if not refs:
        return

    by_path: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for nid, node in nodes.items():
        if not isinstance(node, dict):
            continue
        path = str(node.get("path") or "").strip()
        if not path:
            continue
        by_path.setdefault(path, []).append((nid, node))

    seen: Set[Tuple[str, str, str]] = {
        (str(e.get("src") or ""), str(e.get("dst") or ""), str(e.get("kind") or ""))
        for e in edges
        if isinstance(e, dict)
    }

    def candidates_for(module: str, symbol: str, attr: str = "") -> List[str]:
        symbol = (symbol or "").strip()
        attr = (attr or "").strip()
        if not symbol:
            return []
        path_suffixes = _module_to_candidate_paths(module)
        matched_nodes: List[Tuple[str, Dict[str, Any]]] = []
        for path, items in by_path.items():
            if any(path == ps or path.endswith("/" + ps) for ps in path_suffixes):
                matched_nodes.extend(items)
        if not matched_nodes:
            return []

        exact: List[str] = []
        class_method: List[str] = []
        class_node: List[str] = []
        for nid, node in matched_nodes:
            kind = str(node.get("kind") or "").lower()
            name = str(node.get("name") or node.get("symbol") or "").strip()
            base = name.split(".")[-1] if name else ""
            if attr:
                if name == f"{symbol}.{attr}" or name.endswith(f".{symbol}.{attr}"):
                    class_method.append(nid)
                continue
            if base == symbol or name == symbol:
                if kind in {"function", "func", "method", "class", "module_assignment", "class_assignment"}:
                    exact.append(nid)
                    if kind == "class":
                        class_node.append(nid)

        if attr:
            return class_method[:4]
        # Prefer callable/value serializer nodes over a file/module-level node.
        if exact:
            return exact[:4]
        if class_node:
            return class_node[:4]
        return []

    for rec in refs:
        src = str(rec.get("src") or "").strip()
        mod = str(rec.get("module") or "").strip()
        sym = str(rec.get("symbol") or "").strip()
        attr = str(rec.get("attr") or "").strip()
        kind = str(rec.get("kind") or "uses").strip() or "uses"
        if not src or src not in nodes or not mod or not sym:
            continue
        for dst in candidates_for(mod, sym, attr):
            if not dst or dst == src:
                continue
            key = (src, dst, kind)
            if key in seen:
                continue
            seen.add(key)
            edges.append({"src": src, "dst": dst, "kind": kind, "resolved_from_import": True})


def build_repo_graph(repo: Path) -> Dict[str, Any]:
    repo = repo.resolve()
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    file_count = 0
    py_files = 0
    go_files = 0
    ast_files = 0
    ts_files = 0
    js_files = 0
    generic_files = 0
    skipped_files = 0
    import_refs: List[Dict[str, str]] = []
    go_call_refs: List[Dict[str, str]] = []
    js_call_refs: List[Dict[str, str]] = []

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
    graph_frontend = str(os.environ.get("GP_GRAPH_FRONTEND", "treesitter") or "treesitter").strip().lower()
    if graph_frontend in {"tree-sitter", "tree_sitter", "ts"}:
        graph_frontend = "treesitter"
    if graph_frontend not in {"ast", "treesitter", "auto"}:
        graph_frontend = "ast"
    # Back-compat: if a budget is set to 0, fall back to the global.
    if max_file_snippet_lines <= 0:
        max_file_snippet_lines = max_snippet_lines
    if max_def_snippet_lines <= 0:
        max_def_snippet_lines = max_snippet_lines

    for fp in iter_python_files(repo):
        file_count += 1
        py_files += 1
        rel = _rel_posix(fp, repo)
        file_id = f"file:{rel}"
        try:
            src = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        file_lines = src.splitlines()

        ast_tree = _safe_parse(src, filename=rel)
        ts_tree = None
        use_treesitter = False

        if graph_frontend == "treesitter":
            ts_tree = _safe_parse_treesitter(src)
            use_treesitter = ts_tree is not None
        elif graph_frontend == "auto":
            # Keep AST as default for stability; only fall back when AST fails.
            if ast_tree is None:
                ts_tree = _safe_parse_treesitter(src)
                use_treesitter = ts_tree is not None

        if (not use_treesitter) and ast_tree is None:
            skipped_files += 1
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
                "doc": _truncate_doc(ast.get_docstring(ast_tree) or "", max_chars=320) if ast_tree is not None else "",
            }
            if embed_snippets and file_lines:
                file_payload["snippet_lines"] = _file_summary_snippet(
                    file_lines,
                    ast_tree,
                    max_lines=max_file_snippet_lines,
                )
            nodes[file_id] = file_payload

        if use_treesitter and ts_tree is not None:
            ts_files += 1
            gb_ts = TreeSitterGraphBuilder(
                file_rel=rel,
                file_node_id=file_id,
                file_lines=file_lines,
                src_bytes=src.encode("utf-8", errors="replace"),
                embed_snippets=embed_snippets,
                max_snippet_lines=max_def_snippet_lines,
                max_class_snippet_lines=max_class_snippet_lines,
            )
            gb_ts.build(ts_tree)
            gb_nodes = gb_ts.nodes
            gb_edges = gb_ts.edges
            import_refs.extend(getattr(gb_ts, "_pending_import_refs", []) or [])
        else:
            ast_files += 1
            gb = GraphBuilder(
                file_rel=rel,
                file_node_id=file_id,
                file_lines=file_lines,
                embed_snippets=embed_snippets,
                max_snippet_lines=max_def_snippet_lines,
                max_class_snippet_lines=max_class_snippet_lines,
            )
            gb.visit(ast_tree)
            gb.finalize_call_edges()
            gb_nodes = gb.nodes
            gb_edges = gb.edges
            import_refs.extend(getattr(gb, "_pending_import_refs", []) or [])

        # merge
        for nid, n in gb_nodes.items():
            if nid not in nodes:
                nodes[nid] = n
        edges.extend(gb_edges)

    _resolve_import_refs(nodes, edges, import_refs)

    for fp in iter_go_files(repo):
        file_count += 1
        go_files += 1
        rel = _rel_posix(fp, repo)
        file_id = f"file:{rel}"
        try:
            src = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            skipped_files += 1
            continue
        gb_nodes, gb_edges, gb_calls = _build_go_file_graph(
            rel,
            src,
            file_id=file_id,
            embed_snippets=embed_snippets,
            max_file_snippet_lines=max_file_snippet_lines,
            max_def_snippet_lines=max_def_snippet_lines,
        )
        for nid, n in gb_nodes.items():
            if nid not in nodes:
                nodes[nid] = n
        edges.extend(gb_edges)
        go_call_refs.extend(gb_calls)

    _resolve_go_call_refs(nodes, edges, go_call_refs)

    for fp in iter_js_ts_files(repo):
        file_count += 1
        rel = _rel_posix(fp, repo)
        file_id = f"file:{rel}"
        language = "typescript" if fp.suffix.lower() in {".ts", ".tsx"} else "javascript"
        if language == "typescript":
            ts_files += 1
        else:
            js_files += 1
        try:
            src = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            skipped_files += 1
            continue
        gb_nodes, gb_edges, gb_calls = _build_js_ts_file_graph(
            rel,
            src,
            file_id=file_id,
            language=language,
            embed_snippets=embed_snippets,
            max_file_snippet_lines=max_file_snippet_lines,
            max_def_snippet_lines=max_def_snippet_lines,
            max_class_snippet_lines=max_class_snippet_lines,
        )
        for nid, n in gb_nodes.items():
            if nid not in nodes:
                nodes[nid] = n
        edges.extend(gb_edges)
        js_call_refs.extend(gb_calls)

    _resolve_js_call_refs(nodes, edges, js_call_refs)

    for fp in iter_generic_code_files(repo):
        file_count += 1
        generic_files += 1
        rel = _rel_posix(fp, repo)
        file_id = f"file:{rel}"
        language = fp.suffix.lower().lstrip(".") or "code"
        try:
            src = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            skipped_files += 1
            continue
        gb_nodes, gb_edges = _build_generic_file_graph(
            rel,
            src,
            file_id=file_id,
            language=language,
            embed_snippets=embed_snippets,
            max_file_snippet_lines=max_file_snippet_lines,
            max_def_snippet_lines=max_def_snippet_lines,
        )
        for nid, n in gb_nodes.items():
            if nid not in nodes:
                nodes[nid] = n
        edges.extend(gb_edges)

    try:
        print(
            (
                f"[swe_build_graph] repo={str(repo)} files={int(file_count)} "
                f"py_files={py_files} go_files={go_files} ast_files={ast_files} "
                f"js_files={js_files} ts_files={ts_files} generic_files={generic_files} skipped={skipped_files} "
                f"frontend={graph_frontend} nodes={len(nodes)} edges={len(edges)}"
            ),
            file=sys.stderr,
            flush=True,
        )
    except Exception:
        pass
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
