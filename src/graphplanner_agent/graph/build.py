from __future__ import annotations

import ast
from pathlib import Path

from .guards import is_test_path
from .schema import GraphNode, RepoGraph


def _lines_for(source: str, start: int, end: int) -> str:
    lines = source.splitlines()
    return "\n".join(lines[start - 1 : end]) + ("\n" if end >= start else "")


def _node_id(kind: str, path: str, name: str, start: int) -> str:
    safe = path.replace("/", "::")
    return f"{kind}::{safe}::{name}::{start}"


class _Builder(ast.NodeVisitor):
    def __init__(self, graph: RepoGraph, path: str, source: str, file_id: str):
        self.graph = graph
        self.path = path
        self.source = source
        self.stack = [file_id]
        self.functions: dict[str, str] = {}

    @property
    def parent(self) -> str:
        return self.stack[-1]

    def _add_symbol(self, node: ast.AST, kind: str, name: str) -> str:
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start)
        node_id = _node_id(kind, self.path, name, start)
        graph_node = GraphNode(
            id=node_id,
            kind=kind,
            name=name,
            path=self.path,
            start_line=start,
            end_line=end,
            text=_lines_for(self.source, start, end),
            preview=name,
            parent_id=self.parent,
        )
        self.graph.add_node(graph_node)
        self.graph.add_edge(self.parent, node_id, "CONTAINS")
        return node_id

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        node_id = self._add_symbol(node, "class", node.name)
        self.stack.append(node_id)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        kind = "method" if self.graph.nodes[self.parent].kind == "class" else "function"
        node_id = self._add_symbol(node, kind, node.name)
        self.functions[node.name] = node_id
        self.stack.append(node_id)
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                called = _call_name(child.func)
                if called:
                    target = self.functions.get(called)
                    if target:
                        self.graph.add_edge(node_id, target, "CALLS")
                    else:
                        use_id = _node_id("usage", self.path, called, getattr(child, "lineno", node.lineno))
                        self.graph.add_node(
                            GraphNode(
                                id=use_id,
                                kind="usage",
                                name=called,
                                path=self.path,
                                start_line=getattr(child, "lineno", node.lineno),
                                end_line=getattr(child, "lineno", node.lineno),
                                preview=called,
                                parent_id=node_id,
                            )
                        )
                        self.graph.add_edge(node_id, use_id, "USES")
        self.generic_visit(node)
        self.stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        names = [_target_name(t) for t in node.targets]
        for name in [n for n in names if n]:
            self._add_symbol(node, "assignment", name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        name = _target_name(node.target)
        if name:
            self._add_symbol(node, "assignment", name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add_import(node, alias.asname or alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            self._add_import(node, alias.asname or f"{module}.{alias.name}".strip("."))

    def _add_import(self, node: ast.AST, name: str) -> None:
        start = getattr(node, "lineno", 1)
        node_id = _node_id("import", self.path, name, start)
        self.graph.add_node(
            GraphNode(
                id=node_id,
                kind="import",
                name=name,
                path=self.path,
                start_line=start,
                end_line=getattr(node, "end_lineno", start),
                text=_lines_for(self.source, start, getattr(node, "end_lineno", start)),
                preview=name,
                parent_id=self.parent,
            )
        )
        self.graph.add_edge(self.parent, node_id, "IMPORTS")


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _target_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Tuple):
        parts = [_target_name(elt) for elt in node.elts]
        return ",".join(part for part in parts if part)
    return None


def build_python_graph(root: Path) -> RepoGraph:
    graph = RepoGraph(root=str(root))
    repo_id = "repo"
    graph.add_node(GraphNode(id=repo_id, kind="repository", name=root.name, path="", start_line=1, end_line=1))
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if is_test_path(rel):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (UnicodeDecodeError, SyntaxError):
            continue
        line_count = max(1, len(source.splitlines()))
        file_id = _node_id("file", rel, rel, 1)
        graph.add_node(
            GraphNode(
                id=file_id,
                kind="file",
                name=rel,
                path=rel,
                start_line=1,
                end_line=line_count,
                text=None,
                preview=rel,
                parent_id=repo_id,
            )
        )
        graph.add_edge(repo_id, file_id, "CONTAINS")
        _Builder(graph, rel, source, file_id).visit(tree)
        child_ids = [e.target for e in graph.edges if e.source == file_id and e.type == "CONTAINS"]
        for i, left in enumerate(child_ids):
            for right in child_ids[i + 1 :]:
                graph.add_edge(left, right, "SIBLING")
    return graph
