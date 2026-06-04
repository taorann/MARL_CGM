from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GraphNode:
    id: str
    kind: str
    name: str
    path: str
    start_line: int
    end_line: int
    text: str | None = None
    preview: str = ""
    parent_id: str | None = None

    @property
    def has_code(self) -> bool:
        return bool(self.text and self.text.strip())


@dataclass(slots=True, frozen=True)
class GraphEdge:
    source: str
    target: str
    type: str


@dataclass(slots=True)
class RepoGraph:
    root: str
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)
    _edge_set: set[GraphEdge] = field(default_factory=set, repr=False)

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, source: str, target: str, edge_type: str) -> None:
        edge = GraphEdge(source=source, target=target, type=edge_type.upper())
        if edge not in self._edge_set:
            self._edge_set.add(edge)
            self.edges.append(edge)

    def neighbors(self, node_id: str, edge_types: set[str] | None = None) -> list[GraphNode]:
        types = {t.upper() for t in edge_types} if edge_types else None
        ids: list[str] = []
        for edge in self.edges:
            if types and edge.type not in types:
                continue
            if edge.source == node_id:
                ids.append(edge.target)
            elif edge.target == node_id and edge.type in {"SIBLING", "CONTAINS", "USES", "CALLS"}:
                ids.append(edge.source)
        return [self.nodes[nid] for nid in ids if nid in self.nodes]

    def edges_between(self, node_ids: set[str]) -> list[GraphEdge]:
        return [e for e in self.edges if e.source in node_ids and e.target in node_ids]
