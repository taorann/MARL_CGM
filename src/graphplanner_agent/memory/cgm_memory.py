from __future__ import annotations

from dataclasses import dataclass, field

from graphplanner_agent.graph.schema import GraphEdge, GraphNode, RepoGraph


@dataclass(slots=True)
class CgmMemory:
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def commit(self, nodes: list[GraphNode], note: str | None = None) -> None:
        for node in nodes:
            self.nodes[node.id] = node
        if note:
            self.notes.append(note)

    def delete(self, delete_ids: list[str] | None = None, keep_ids: list[str] | None = None, note: str | None = None) -> None:
        if keep_ids is not None:
            keep = set(keep_ids)
            self.nodes = {nid: node for nid, node in self.nodes.items() if nid in keep}
        for node_id in delete_ids or []:
            self.nodes.pop(node_id, None)
        if note:
            self.notes.append(note)

    def graph_edges(self, graph: RepoGraph) -> list[GraphEdge]:
        return graph.edges_between(set(self.nodes))

    def summary(self) -> list[dict[str, object]]:
        return [
            {
                "id": n.id,
                "kind": _public_node_kind(n.kind),
                "name": n.name,
                "path": n.path,
                "lines": [n.start_line, n.end_line],
                "has_code": n.has_code,
            }
            for n in self.nodes.values()
        ]


def _public_node_kind(kind: str) -> str:
    return "assignment" if kind in {"assignment", "module_assignment"} else kind
