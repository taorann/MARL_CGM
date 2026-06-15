from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass

from .guards import is_test_path
from .schema import GraphNode, RepoGraph


MODE_TO_EDGES = {
    "callers": {"CALLS"},
    "callees": {"CALLS"},
    "siblings": {"SIBLING"},
    "imports": {"IMPORTS"},
    "contains": {"CONTAINS"},
    "uses": {"USES"},
    "related": {"CALLS", "USES", "CONTAINS", "SIBLING", "IMPORTS"},
}

MECHANISM_MODES = {"mechanism", "owner_flow"}


@dataclass(slots=True)
class ExpandedNode:
    node: GraphNode
    relation: str
    reason: str


def expand(graph: RepoGraph, anchor: str, expand_mode: str = "related", limit: int = 12) -> list[GraphNode]:
    if anchor not in graph.nodes:
        return []
    if expand_mode == "callers":
        ids = [edge.source for edge in graph.edges if edge.type == "CALLS" and edge.target == anchor]
        return [graph.nodes[nid] for nid in ids if nid in graph.nodes and not is_test_path(graph.nodes[nid].path)][:limit]
    if expand_mode == "callees":
        ids = [edge.target for edge in graph.edges if edge.type == "CALLS" and edge.source == anchor]
        return [graph.nodes[nid] for nid in ids if nid in graph.nodes and not is_test_path(graph.nodes[nid].path)][:limit]
    modes = MODE_TO_EDGES.get(expand_mode, MODE_TO_EDGES["related"])
    return [node for node in graph.neighbors(anchor, modes) if not is_test_path(node.path)][:limit]


def expand_with_context(
    graph: RepoGraph,
    anchor: str,
    expand_mode: str = "related",
    *,
    symbol: str | None = None,
    limit: int = 12,
) -> list[ExpandedNode]:
    """Return graph/lazy-AST relation candidates around ``anchor``.

    The default modes keep the historical edge-based behavior.  The mechanism
    modes derive lightweight inheritance, override, composition, and owner-flow
    candidates from already indexed node text without rebuilding the graph.
    """
    if anchor not in graph.nodes:
        return []
    if expand_mode not in MECHANISM_MODES:
        return [ExpandedNode(node, expand_mode, f"{expand_mode} graph neighbor of {anchor}") for node in expand(graph, anchor, expand_mode, limit)]

    anchor_node = graph.nodes[anchor]
    symbol = (symbol or "").strip()
    results: list[ExpandedNode] = []
    seen: set[str] = set()

    def add(node: GraphNode | None, relation: str, reason: str) -> None:
        if node is None or is_test_path(node.path) or node.id == anchor_node.id or node.id in seen:
            return
        seen.add(node.id)
        results.append(ExpandedNode(node, relation, reason))

    class_node = _class_context(graph, anchor_node)
    anchor_method = _method_tail(anchor_node) if _is_method_like(anchor_node) else None

    if class_node is not None:
        class_name = class_node.name
        class_info = _class_info(class_node)
        base_classes = _base_classes_for(graph, class_node)
        base_lineage = _base_lineage_for(graph, class_node)
        for base in base_classes:
            add(base, "parent_class", f"{class_name} inherits from {base.name}; read parent/base behavior when local patch needs owner or pipeline context")
            if anchor_method:
                add(
                    _method_for_class(graph, base, anchor_method),
                    "overridden_method",
                    f"{class_name}.{anchor_method} overrides or bypasses {base.name}.{anchor_method}",
                )
            else:
                for method_name in _prioritized_methods(class_info.methods):
                    add(
                        _method_for_class(graph, base, method_name),
                        "overridden_method",
                        f"{class_name}.{method_name} has same-name method in parent/base class {base.name}",
                    )

        for base in base_lineage:
            if base not in base_classes:
                add(base, "ancestor_class", f"{class_name} inherits behavior from ancestor class {base.name}; read inherited pipeline context when local class has no method")
            if anchor_method:
                add(
                    _method_for_class(graph, base, anchor_method),
                    "inherited_method",
                    f"{class_name}.{anchor_method} may be inherited from {base.name}.{anchor_method}",
                )
            else:
                for method_name in _prioritized_methods(_class_info(base).methods):
                    add(
                        _method_for_class(graph, base, method_name),
                        "inherited_method",
                        f"{class_name} may inherit {base.name}.{method_name}; read this before inventing a child method that is not indexed",
                    )

        composed_classes: list[tuple[str, GraphNode]] = []
        for assignment in class_info.assignments:
            if not _looks_like_composition(assignment.target, assignment.value_name):
                continue
            for target_class in _find_classes(graph, assignment.value_name, current_path=class_node.path):
                relation = f"composition:{assignment.target}"
                add(target_class, relation, f"{class_name}.{assignment.target} points to {target_class.name}; read composed object before assuming self-owned state")
                composed_classes.append((assignment.target, target_class))

        for attr_name, composed in composed_classes:
            composed_info = _class_info(composed)
            composed_bases = _base_classes_for(graph, composed)
            for base in composed_bases:
                add(base, "composition_parent", f"{class_name}.{attr_name} -> {composed.name}, and {composed.name} inherits from {base.name}")
                for method_name in _prioritized_methods(class_info.methods):
                    add(
                        _method_for_class(graph, base, method_name),
                        "pipeline_method",
                        f"{base.name}.{method_name} is a same-name pipeline method reachable through {class_name}.{attr_name}",
                    )
                for method_name in _prioritized_methods(composed_info.methods):
                    add(
                        _method_for_class(graph, base, method_name),
                        "composition_override",
                        f"{composed.name}.{method_name} has same-name method in parent/base class {base.name}",
                    )

        if symbol:
            owner_classes = [class_node, *base_classes]
            owner_classes.extend(cls for _, cls in composed_classes)
            for _, composed in composed_classes:
                owner_classes.extend(_base_classes_for(graph, composed))
            for owner in _dedupe_nodes(owner_classes):
                for assignment in _assignments_for_class(graph, owner, symbol):
                    add(assignment, "attribute_owner", f"{owner.name} defines or assigns {symbol!r}; verify owner before using self.{symbol}")
                for method in _methods_containing_symbol(graph, owner, symbol):
                    add(method, "symbol_consumer", f"{owner.name}.{_method_tail(method)} references {symbol!r}; read consumer before patching owner flow")

    if symbol:
        for assignment in _global_assignment_candidates(graph, symbol, current_path=anchor_node.path):
            add(assignment, "attribute_owner", f"indexed assignment candidate for {symbol!r}")
        for method in _global_method_consumers(graph, symbol, current_path=anchor_node.path):
            add(method, "symbol_consumer", f"indexed method/function references {symbol!r}")

    return results[:limit]


@dataclass(slots=True)
class _AssignmentInfo:
    target: str
    value_name: str


@dataclass(slots=True)
class _ClassInfo:
    bases: list[str]
    assignments: list[_AssignmentInfo]
    methods: list[str]


def _class_context(graph: RepoGraph, node: GraphNode) -> GraphNode | None:
    if node.kind == "class":
        return node
    if node.parent_id and node.parent_id in graph.nodes and graph.nodes[node.parent_id].kind == "class":
        return graph.nodes[node.parent_id]
    class_name = _class_name_from_method_name(node.name)
    if class_name:
        for candidate in graph.nodes.values():
            if candidate.kind == "class" and candidate.path == node.path and candidate.name == class_name:
                return candidate
    for candidate in graph.nodes.values():
        if candidate.kind == "class" and candidate.path == node.path and candidate.start_line <= node.start_line <= candidate.end_line:
            return candidate
    return None


def _class_info(node: GraphNode) -> _ClassInfo:
    text = node.text or ""
    try:
        tree = ast.parse(textwrap.dedent(text).strip() + "\n")
    except SyntaxError:
        return _ClassInfo([], [], [])
    class_def = next((item for item in ast.walk(tree) if isinstance(item, ast.ClassDef)), None)
    if class_def is None:
        return _ClassInfo([], [], [])
    bases = [_tail_name(_unparse(base)) for base in class_def.bases]
    assignments: list[_AssignmentInfo] = []
    methods: list[str] = []
    for stmt in class_def.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(stmt.name)
        if isinstance(stmt, ast.Assign):
            value_name = _tail_name(_unparse(stmt.value))
            for target in stmt.targets:
                target_name = _target_name(target)
                if target_name and value_name:
                    assignments.append(_AssignmentInfo(target_name, value_name))
        if isinstance(stmt, ast.AnnAssign):
            target_name = _target_name(stmt.target)
            value_name = _tail_name(_unparse(stmt.value)) if stmt.value is not None else ""
            if target_name and value_name:
                assignments.append(_AssignmentInfo(target_name, value_name))
    return _ClassInfo([base for base in bases if base], assignments, methods)


def _base_classes_for(graph: RepoGraph, class_node: GraphNode) -> list[GraphNode]:
    bases = _class_info(class_node).bases
    nodes: list[GraphNode] = []
    for base in bases:
        nodes.extend(_find_classes(graph, base, current_path=class_node.path))
    return _dedupe_nodes(nodes)


def _base_lineage_for(graph: RepoGraph, class_node: GraphNode, *, max_depth: int = 4) -> list[GraphNode]:
    lineage: list[GraphNode] = []
    seen: set[str] = set()

    def visit(node: GraphNode, depth: int) -> None:
        if depth >= max_depth:
            return
        for base in _base_classes_for(graph, node):
            if base.id in seen:
                continue
            seen.add(base.id)
            lineage.append(base)
            visit(base, depth + 1)

    visit(class_node, 0)
    return lineage


def _find_classes(graph: RepoGraph, name: str, *, current_path: str = "") -> list[GraphNode]:
    short = _tail_name(name)
    if not short:
        return []
    candidates = [node for node in graph.nodes.values() if node.kind == "class" and node.name == short and not is_test_path(node.path)]
    current_dir = current_path.rsplit("/", 1)[0] if "/" in current_path else ""

    def rank(node: GraphNode) -> tuple[int, int, str]:
        same_file = int(node.path == current_path)
        same_dir = int(bool(current_dir) and node.path.startswith(current_dir + "/"))
        return (same_file, same_dir, -len(node.path))

    return sorted(candidates, key=rank, reverse=True)


def _method_for_class(graph: RepoGraph, class_node: GraphNode | None, method_name: str) -> GraphNode | None:
    if class_node is None or not method_name:
        return None
    for node in _contained_nodes(graph, class_node):
        if _is_method_like(node) and _method_tail(node) == method_name:
            return node
    for node in graph.nodes.values():
        if node.path != class_node.path or not _is_method_like(node):
            continue
        if _method_tail(node) == method_name and node.start_line >= class_node.start_line and node.end_line <= class_node.end_line:
            return node
    return None


def _assignments_for_class(graph: RepoGraph, class_node: GraphNode | None, symbol: str) -> list[GraphNode]:
    if class_node is None or not symbol:
        return []
    nodes: list[GraphNode] = []
    for node in _contained_nodes(graph, class_node):
        if _is_assignment(node) and _assignment_matches(node, symbol):
            nodes.append(node)
    for node in graph.nodes.values():
        if node.path == class_node.path and _is_assignment(node) and _assignment_matches(node, symbol):
            if node.start_line >= class_node.start_line and node.end_line <= class_node.end_line:
                nodes.append(node)
    return _dedupe_nodes(nodes)


def _methods_containing_symbol(graph: RepoGraph, class_node: GraphNode | None, symbol: str) -> list[GraphNode]:
    if class_node is None or not symbol:
        return []
    methods = [node for node in _contained_nodes(graph, class_node) if _is_method_like(node)]
    if not methods:
        methods = [
            node
            for node in graph.nodes.values()
            if node.path == class_node.path
            and _is_method_like(node)
            and node.start_line >= class_node.start_line
            and node.end_line <= class_node.end_line
        ]
    return _dedupe_nodes([node for node in methods if symbol in (node.text or "")])


def _global_assignment_candidates(graph: RepoGraph, symbol: str, *, current_path: str = "") -> list[GraphNode]:
    nodes = [node for node in graph.nodes.values() if _is_assignment(node) and _assignment_matches(node, symbol) and not is_test_path(node.path)]
    return _sort_by_path(nodes, current_path)[:6]


def _global_method_consumers(graph: RepoGraph, symbol: str, *, current_path: str = "") -> list[GraphNode]:
    nodes = [
        node
        for node in graph.nodes.values()
        if _is_method_like(node) and symbol in (node.text or "") and not is_test_path(node.path)
    ]
    return _sort_by_path(nodes, current_path)[:6]


def _contained_nodes(graph: RepoGraph, class_node: GraphNode) -> list[GraphNode]:
    ids = [edge.target for edge in graph.edges if edge.source == class_node.id and edge.type == "CONTAINS"]
    nodes = [graph.nodes[node_id] for node_id in ids if node_id in graph.nodes]
    if nodes:
        return nodes
    return [node for node in graph.nodes.values() if node.parent_id == class_node.id]


def _prioritized_methods(methods: list[str]) -> list[str]:
    priority = ["write", "read", "str_vals", "_set_col_formats", "__init__"]
    ordered = [name for name in priority if name in methods]
    ordered.extend(name for name in methods if name not in ordered and not name.startswith("__"))
    return ordered[:4]


def _looks_like_composition(target: str, value_name: str) -> bool:
    if not target or not value_name:
        return False
    return target.endswith("_class") or value_name[:1].isupper()


def _is_method_like(node: GraphNode) -> bool:
    return node.kind in {"method", "function"}


def _is_assignment(node: GraphNode) -> bool:
    return node.kind in {"assignment", "module_assignment", "class_assignment"}


def _assignment_matches(node: GraphNode, symbol: str) -> bool:
    name = node.name.rsplit(".", 1)[-1]
    return name == symbol


def _method_tail(node: GraphNode) -> str:
    return node.name.rsplit(".", 1)[-1]


def _class_name_from_method_name(name: str) -> str:
    parts = name.rsplit(".", 1)
    return parts[0] if len(parts) == 2 else ""


def _tail_name(name: str) -> str:
    if not name:
        return ""
    return name.split(".")[-1]


def _target_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _sort_by_path(nodes: list[GraphNode], current_path: str) -> list[GraphNode]:
    current_dir = current_path.rsplit("/", 1)[0] if "/" in current_path else ""

    def rank(node: GraphNode) -> tuple[int, int, int, str]:
        return (
            int(node.path == current_path),
            int(bool(current_dir) and node.path.startswith(current_dir + "/")),
            int(node.kind in {"method", "function"}),
            -len(node.path),
        )

    return sorted(_dedupe_nodes(nodes), key=rank, reverse=True)


def _dedupe_nodes(nodes: list[GraphNode]) -> list[GraphNode]:
    seen: set[str] = set()
    result: list[GraphNode] = []
    for node in nodes:
        if node.id in seen:
            continue
        seen.add(node.id)
        result.append(node)
    return result
