from __future__ import annotations

from dataclasses import dataclass, field

from graphplanner_agent.graph.schema import GraphNode


@dataclass(slots=True)
class WorkingEntry:
    node: GraphNode
    source: str
    score: float = 0.0
    last_step: int = 0


@dataclass(slots=True)
class WorkingMemory:
    entries: dict[str, WorkingEntry] = field(default_factory=dict)

    def add(self, node: GraphNode, source: str, score: float = 0.0, step: int = 0) -> None:
        current = self.entries.get(node.id)
        if current and current.node.has_code and not node.has_code:
            node = current.node
        self.entries[node.id] = WorkingEntry(node=node, source=source, score=score, last_step=step)

    def get(self, node_id: str) -> GraphNode | None:
        entry = self.entries.get(node_id)
        return entry.node if entry else None

    def recent_read_ids(self, limit: int = 3) -> list[str]:
        reads = [e for e in self.entries.values() if e.node.has_code]
        reads.sort(key=lambda e: e.last_step, reverse=True)
        return [e.node.id for e in reads[:limit]]

    def summary(self, limit: int = 20) -> list[dict[str, object]]:
        items = sorted(self.entries.values(), key=lambda e: (e.last_step, e.score), reverse=True)[:limit]
        return [
            {
                "id": e.node.id,
                "kind": _public_node_kind(e.node.kind),
                "name": e.node.name,
                "path": e.node.path,
                "lines": [e.node.start_line, e.node.end_line],
                "has_code": e.node.has_code,
                "code_status": code_status_for(e.source, e.node),
                "source": e.source,
            }
            for e in items
        ]


def code_status_for(source: str, node: GraphNode) -> str:
    if not node.has_code:
        return "candidate"
    if str(source or "").startswith("read:"):
        return "read"
    if str(source or "") == "hydrated_for_memory":
        return "hydrated"
    if str(source or "").startswith(("find_preview:", "relation_context:", "expand_preview:")):
        return "preview"
    return "code"


def _public_node_kind(kind: str) -> str:
    return "assignment" if kind in {"assignment", "module_assignment"} else kind
