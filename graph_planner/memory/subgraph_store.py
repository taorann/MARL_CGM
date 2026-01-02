"""Subgraph store utilities (protocol-aligned).

We maintain three graphs during an episode/trajectory:

- repo_graph (G): read-only full repo code graph (external truth).
- working_subgraph (W): planner-facing cache, may be noisy; nodes carry `memorized` flag.
- memory_subgraph (M): CGM-facing evidence graph, high-signal; populated by projecting an induced subgraph from W.

This module provides a lightweight in-memory representation plus helpers:
- project_to_memory: copy induced subgraph from working to memory
- prune_working: keep memorized nodes + a small recent unmemorized frontier

Protocol notes:
- WorkingSubgraph and MemorySubgraph are *distinct* types (to prevent semantic drift),
  though they share the same underlying container structure.
- WorkingSubgraph and MemorySubgraph objects must be independent (no shared dict references).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import copy
import json
import os


Node = Dict[str, Any]
Edge = Dict[str, Any]


def _node_id(n: Mapping[str, Any]) -> str:
    nid = n.get("id") or n.get("node_id") or n.get("nid")
    return str(nid) if nid is not None else ""


def _edge_uv(e: Mapping[str, Any]) -> Tuple[str, str]:
    # tolerate multiple key conventions
    u = e.get("u") or e.get("src") or e.get("from")
    v = e.get("v") or e.get("dst") or e.get("to")
    return (str(u) if u is not None else "", str(v) if v is not None else "")


@dataclass
class Subgraph:
    """Generic subgraph structure (nodes keyed by id, edges as list).

    - nodes: dict[id -> node-dict]
    - edges: list[edge-dict]
    - node_ids: recency/order cache (mainly for W pruning & prompt ordering)
    """

    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    node_ids: List[str] = field(default_factory=list)

    def to_json_obj(self) -> Dict[str, Any]:
        # Backward-compatible shape: "nodes" remains a list of node dicts.
        # We also include "node_ids" for consumers that care about recency.
        return {
            "nodes": list(self.nodes.values()),
            "edges": list(self.edges),
            "node_ids": list(self.node_ids),
        }

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

    # -------- Protocol / backward-compat helpers --------
    def iter_node_ids(self) -> Iterable[str]:
        """Iterate node ids in the subgraph.

        Many consumers treat Subgraph as a `SubgraphLike` protocol which
        requires `iter_node_ids()`. Older versions used `node_ids` as the
        recency cache; if it is present we prefer that order.
        """
        # Prefer recency order if available.
        if self.node_ids:
            for nid in list(self.node_ids):
                if nid in self.nodes:
                    yield nid
            return
        # Fallback: dictionary order.
        yield from self.nodes.keys()

    def contains(self, node_id: str) -> bool:
        """Protocol helper: return True iff node_id exists in this subgraph."""
        return str(node_id) in self.nodes


@dataclass
class WorkingSubgraph(Subgraph):
    """Planner-facing working subgraph (W)."""


@dataclass
class MemorySubgraph(Subgraph):
    """CGM-facing evidence subgraph (M)."""


def wrap(obj: Any) -> Subgraph:
    """Wrap a dict/list/Subgraph into our Subgraph class.

    Tolerates:
    - {"nodes": list[dict] | dict[id->dict], "edges": list[dict], "node_ids": list[str]}
    - list[dict] (treated as node list)
    """
    if isinstance(obj, Subgraph):
        return obj
    sg = Subgraph()
    if obj is None:
        return sg

    node_ids_in: Optional[List[str]] = None

    if isinstance(obj, Mapping):
        nodes = obj.get("nodes") or obj.get("Nodes") or []
        edges = obj.get("edges") or obj.get("Edges") or []
        node_ids_in = obj.get("node_ids") or obj.get("nodeIds") or obj.get("NodeIds")
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

    # Restore node_ids if provided (filter to existing ids, preserve order).
    if isinstance(node_ids_in, list):
        restored = []
        seen = set()
        for x in node_ids_in:
            nid = str(x) if x is not None else ""
            if nid and nid in sg.nodes and nid not in seen:
                restored.append(nid)
                seen.add(nid)
        # Append any missing ids (to keep internal invariant that all nodes appear in node_ids).
        for nid in sg.nodes.keys():
            if nid not in seen:
                restored.append(nid)
        sg.node_ids = restored

    return sg


def wrap_working(obj: Any) -> WorkingSubgraph:
    base = wrap(obj)
    # Deep copy to ensure independence from other references.
    return WorkingSubgraph(
        nodes=copy.deepcopy(base.nodes),
        edges=copy.deepcopy(base.edges),
        node_ids=list(base.node_ids),
    )


def wrap_memory(obj: Any) -> MemorySubgraph:
    base = wrap(obj)
    return MemorySubgraph(
        nodes=copy.deepcopy(base.nodes),
        edges=copy.deepcopy(base.edges),
        node_ids=list(base.node_ids),
    )


def new() -> Subgraph:
    return Subgraph()


def new_working() -> WorkingSubgraph:
    return WorkingSubgraph()


def new_memory() -> MemorySubgraph:
    return MemorySubgraph()


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


def stats(sg: Optional[Subgraph]) -> Dict[str, int]:
    if sg is None:
        return {"n_nodes": 0, "n_edges": 0, "n_memorized": 0, "n_unmemorized": 0}
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


def working_budget(working: Subgraph) -> Dict[str, int]:
    """Compute a cheap size proxy of working subgraph for prompt budgeting.

    We approximate prompt size by aggregating snippet content stored on nodes.
    Returns:
      - total_snippet_lines
      - total_snippet_chars
      - n_nodes_with_snippet
    """
    nodes = getattr(working, "nodes", {}) or {}
    if not isinstance(nodes, dict):
        return {"total_snippet_lines": 0, "total_snippet_chars": 0, "n_nodes_with_snippet": 0}

    total_lines = 0
    total_chars = 0
    with_snip = 0

    for n in nodes.values():
        if not isinstance(n, dict):
            continue
        sn_lines = n.get("snippet_lines")
        sn_text = n.get("snippet")
        if isinstance(sn_lines, list):
            with_snip += 1
            total_lines += len(sn_lines)
            total_chars += sum(len(str(x)) for x in sn_lines)
        elif isinstance(sn_text, str) and sn_text.strip():
            with_snip += 1
            total_chars += len(sn_text)
            total_lines += sn_text.count("\n") + 1

    return {
        "total_snippet_lines": int(total_lines),
        "total_snippet_chars": int(total_chars),
        "n_nodes_with_snippet": int(with_snip),
    }


def prune_working_by_budget(
    working: Subgraph,
    *,
    keep_ids: Optional[Iterable[str]] = None,
    max_total_snippet_lines: Optional[int] = None,
    max_total_snippet_chars: Optional[int] = None,
    keep_recent_unmemorized: int = 0,
) -> Dict[str, int]:
    """Prune working graph to satisfy a size budget while preserving invariants.

    Invariants:
      - Never delete memorized nodes.
      - Never delete nodes in keep_ids.
      - Optionally keep some recent unmemorized nodes.

    Budget:
      - max_total_snippet_lines / max_total_snippet_chars are *soft* targets.
        We remove oldest unmemorized nodes until under both budgets.

    Returns a dict with removed counts and post-prune budget.
    """
    keep_set = {str(x) for x in (keep_ids or []) if x is not None}

    nodes = getattr(working, "nodes", {}) or {}
    if not isinstance(nodes, dict):
        return {"removed": 0, **working_budget(working)}

    # Always keep memorized nodes.
    memorized_ids = {nid for nid, n in nodes.items() if isinstance(n, dict) and n.get("memorized")}
    keep_set |= memorized_ids

    # Also keep a small recency frontier (unmemorized) if requested.
    frontier: List[str] = []
    if keep_recent_unmemorized and keep_recent_unmemorized > 0:
        for nid in reversed(list(getattr(working, "node_ids", []) or [])):
            if nid in keep_set:
                continue
            n = nodes.get(nid)
            if isinstance(n, dict) and not n.get("memorized"):
                frontier.append(nid)
            if len(frontier) >= int(keep_recent_unmemorized):
                break
        keep_set |= set(frontier)

    def over_budget(b: Dict[str, int]) -> bool:
        if max_total_snippet_lines is not None and b.get("total_snippet_lines", 0) > int(max_total_snippet_lines):
            return True
        if max_total_snippet_chars is not None and b.get("total_snippet_chars", 0) > int(max_total_snippet_chars):
            return True
        return False

    removed = 0
    b = working_budget(working)
    if not over_budget(b):
        return {"removed": 0, **b}

    # Remove oldest unmemorized nodes first (stable order via node_ids).
    order = list(getattr(working, "node_ids", []) or [])
    for nid in order:
        if nid in keep_set:
            continue
        n = nodes.get(nid)
        if isinstance(n, dict) and n.get("memorized"):
            continue
        # delete
        if nid in nodes:
            del nodes[nid]
            removed += 1
        b = working_budget(working)
        if not over_budget(b):
            break

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

    # Refresh node_ids
    working.node_ids = [nid for nid in getattr(working, "node_ids", []) or [] if nid in remaining]

    b = working_budget(working)
    return {"removed": int(removed), **b}




def add_repo_edges_between_ids(working: Subgraph, repo_graph: Optional[Subgraph], ids: Sequence[str]) -> int:
    """Ensure edges among `ids` exist in working subgraph by pulling from repo_graph.

    This helps maintain the invariant that memory edges are also present in working
    (M ⊆ W on edges) when memory is projected from working.
    """
    if repo_graph is None or not ids:
        return 0
    id_set = {str(x) for x in ids if x is not None}
    if not id_set:
        return 0

    existing = set()
    for e in getattr(working, "edges", []) or []:
        if not isinstance(e, dict):
            continue
        u, v = _edge_uv(e)
        if u and v:
            existing.add((u, v))

    added = 0
    for e in getattr(repo_graph, "edges", []) or []:
        if not isinstance(e, dict):
            continue
        u, v = _edge_uv(e)
        if not u or not v:
            continue
        if u in id_set and v in id_set and (u, v) not in existing:
            working.add_edge(copy.deepcopy(e))
            existing.add((u, v))
            added += 1
    return int(added)


def prune_memory(memory: Subgraph, max_nodes: int, *, keep_ids: Optional[Iterable[str]] = None) -> int:
    """Prune *graph memory* to at most `max_nodes`.

    Graph memory is supposed to stay high-signal and compact for CGM.
    We prune the *oldest* nodes first (based on node_ids order), never pruning ids in keep_ids.
    Returns number of removed nodes.
    """
    try:
        max_nodes = int(max_nodes)
    except Exception:
        return 0
    if max_nodes <= 0:
        return 0

    nodes = getattr(memory, "nodes", {}) or {}
    if not isinstance(nodes, dict):
        return 0
    n = len(nodes)
    if n <= max_nodes:
        return 0

    keep_set = {str(x) for x in (keep_ids or []) if x is not None}

    # Oldest-first order from node_ids; fall back to dict order if missing
    order = list(getattr(memory, "node_ids", []) or [])
    if not order:
        order = list(nodes.keys())

    removed = 0
    for nid in order:
        if len(nodes) <= max_nodes:
            break
        if nid in keep_set:
            continue
        if nid in nodes:
            del nodes[nid]
            removed += 1

    remaining = set(nodes.keys())
    # Filter edges
    new_edges: List[Edge] = []
    for e in getattr(memory, "edges", []) or []:
        if not isinstance(e, dict):
            continue
        u, v = _edge_uv(e)
        if u in remaining and v in remaining:
            new_edges.append(e)
    memory.edges = new_edges
    memory.node_ids = [nid for nid in (getattr(memory, "node_ids", []) or []) if nid in remaining]

    return int(removed)


def save(sg: Subgraph, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sg.to_json_obj(), f, ensure_ascii=False, indent=2)


def load(path: str) -> Subgraph:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return wrap(obj)
