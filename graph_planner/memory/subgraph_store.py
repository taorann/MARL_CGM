"""Subgraph store utilities (v5 semantics).

We maintain two subgraphs during an episode/trajectory:

- repo_graph (G): read-only full repo code graph (external truth).
- working_subgraph (W): planner-facing, may be noisy, nodes carry `memorized` flag.
- memory_subgraph (M): CGM-facing, high-signal; populated by projecting a small induced subgraph from W.

This module provides a lightweight in-memory representation plus helpers:
- project_to_memory: copy induced subgraph from working to memory
- prune_working: keep memorized nodes + a small recent unmemorized frontier
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import copy
import json
import os


Node = Dict[str, Any]
Edge = Dict[str, Any]


def _node_id(n: Mapping[str, Any]) -> str:
    nid = n.get("id") or n.get("node_id") or n.get("nid")
    return str(nid) if nid is not None else ""


def _edge_uv(e: Mapping[str, Any]) -> Tuple[str, str]:
    u = e.get("u") or e.get("src") or e.get("from")
    v = e.get("v") or e.get("dst") or e.get("to")
    return (str(u) if u is not None else "", str(v) if v is not None else "")


@dataclass
class Subgraph:
    """Generic subgraph structure (nodes as dict keyed by id, edges as list)."""

    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    # For working graphs, keep a recency-ordered list of node ids.
    node_ids: List[str] = field(default_factory=list)

    def to_json_obj(self) -> Dict[str, Any]:
        return {"nodes": list(self.nodes.values()), "edges": list(self.edges)}

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(str(node_id))

    def add_node(self, node: Node) -> None:
        nid = _node_id(node)
        if not nid:
            return
        if nid not in self.nodes:
            self.nodes[nid] = node
            self.node_ids.append(nid)
        else:
            # merge fields
            self.nodes[nid].update(node)

    def add_edge(self, edge: Edge) -> None:
        if not isinstance(edge, dict):
            return
        u, v = _edge_uv(edge)
        if not u or not v:
            return
        self.edges.append(edge)

    def update_node(self, node_id: str, patch: Mapping[str, Any]) -> None:
        nid = str(node_id)
        if nid not in self.nodes:
            return
        self.nodes[nid].update(dict(patch))

    def touch(self, node_id: str, step: Optional[int] = None) -> None:
        """Mark node as recently touched (move to end of node_ids)."""
        nid = str(node_id)
        if nid in self.node_ids:
            self.node_ids.remove(nid)
        self.node_ids.append(nid)
        if step is not None and nid in self.nodes:
            self.nodes[nid]["gp_last_touched_step"] = int(step)


def wrap(obj: Any) -> Subgraph:
    """Wrap a dict/list/Subgraph into our Subgraph class."""
    if isinstance(obj, Subgraph):
        return obj
    sg = Subgraph()
    if obj is None:
        return sg

    if isinstance(obj, Mapping):
        nodes = obj.get("nodes") or obj.get("Nodes") or []
        edges = obj.get("edges") or obj.get("Edges") or []
    else:
        # If list passed, treat as node list
        nodes = obj if isinstance(obj, list) else []
        edges = []

    # nodes may be dict keyed by id or list
    if isinstance(nodes, Mapping):
        for n in nodes.values():
            if isinstance(n, Mapping):
                sg.add_node(dict(n))
    elif isinstance(nodes, list):
        for n in nodes:
            if isinstance(n, Mapping):
                sg.add_node(dict(n))

    if isinstance(edges, list):
        for e in edges:
            if isinstance(e, Mapping):
                sg.add_edge(dict(e))

    return sg


def new() -> Subgraph:
    return Subgraph()


def add_nodes(sg: Subgraph, nodes: Sequence[Node]) -> None:
    if not nodes:
        return
    for n in nodes:
        if not isinstance(n, dict):
            continue
        # Ensure memorized flag exists (for working; harmless for memory).
        if "memorized" not in n:
            n["memorized"] = bool(n.get("mem", False))
        sg.add_node(n)


def add_edges(sg: Subgraph, edges: Sequence[Edge]) -> None:
    if not edges:
        return
    for e in edges:
        if isinstance(e, dict):
            sg.add_edge(e)


def stats(sg: Subgraph) -> Dict[str, int]:
    nodes = getattr(sg, "nodes", {}) or {}
    n_nodes = len(nodes) if isinstance(nodes, dict) else (len(nodes) if nodes is not None else 0)
    n_edges = len(getattr(sg, "edges", []) or [])
    n_mem = 0
    if isinstance(nodes, dict):
        for n in nodes.values():
            if isinstance(n, dict) and n.get("memorized"):
                n_mem += 1
    return {
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "n_memorized": int(n_mem),
        "n_unmemorized": int(n_nodes - n_mem),
    }


def project_to_memory(working: Subgraph, select_ids: Sequence[str]) -> Tuple[List[Node], List[Edge]]:
    """Project induced subgraph from working.

    Nodes: deep-copied; Edges: keep edges whose endpoints both in select_ids.
    """
    select_set = {str(x) for x in (select_ids or []) if x is not None}
    if not select_set:
        return [], []
    out_nodes: List[Node] = []
    for nid in select_set:
        n = working.get_node(nid)
        if isinstance(n, dict):
            out_nodes.append(copy.deepcopy(n))
    out_edges: List[Edge] = []
    for e in getattr(working, "edges", []) or []:
        if not isinstance(e, dict):
            continue
        u, v = _edge_uv(e)
        if u in select_set and v in select_set:
            out_edges.append(copy.deepcopy(e))
    return out_nodes, out_edges


def prune_working(
    working: Subgraph,
    keep_ids: Optional[Iterable[str]] = None,
    keep_recent_unmemorized: int = 20,
) -> int:
    """Prune working subgraph without clearing it.

    Always keep memorized=True nodes.
    Also keep any ids in keep_ids.
    Also keep up to `keep_recent_unmemorized` most recently touched unmemorized nodes (frontier).
    Returns number of removed (unmemorized) nodes.
    """
    keep_set = {str(x) for x in (keep_ids or []) if x is not None}

    nodes = getattr(working, "nodes", {}) or {}
    if not isinstance(nodes, dict):
        return 0

    memorized_ids = {nid for nid, n in nodes.items() if isinstance(n, dict) and n.get("memorized")}
    keep_set |= memorized_ids

    # recent unmemorized frontier by node_ids (recency-ordered)
    frontier: List[str] = []
    for nid in reversed(list(getattr(working, "node_ids", []) or [])):
        if nid in keep_set:
            continue
        n = nodes.get(nid)
        if isinstance(n, dict) and not n.get("memorized"):
            frontier.append(nid)
        if len(frontier) >= max(int(keep_recent_unmemorized), 0):
            break
    keep_set |= set(frontier)

    removed = 0
    for nid in list(nodes.keys()):
        if nid in keep_set:
            continue
        n = nodes.get(nid)
        if isinstance(n, dict) and n.get("memorized"):
            continue
        del nodes[nid]
        removed += 1

    # Filter edges to those fully within remaining nodes
    remaining = set(nodes.keys())
    new_edges: List[Edge] = []
    for e in getattr(working, "edges", []) or []:
        if not isinstance(e, dict):
            continue
        u, v = _edge_uv(e)
        if u in remaining and v in remaining:
            new_edges.append(e)
    working.edges = new_edges

    # Refresh node_ids to only remaining ids, keeping order
    working.node_ids = [nid for nid in getattr(working, "node_ids", []) or [] if nid in remaining]

    return int(removed)


def save(sg: Subgraph, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sg.to_json_obj(), f, ensure_ascii=False, indent=2)


def load(path: str) -> Subgraph:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return wrap(obj)
