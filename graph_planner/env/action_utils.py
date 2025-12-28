# graph_planner/env/action_utils.py
# -*- coding: utf-8 -*-

"""Helpers for normalizing planner actions.

This module exists to keep env/planner_env.py focused on orchestration.

We keep these helpers dependency-light to avoid circular imports.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple


def normalize_explore_query_and_anchors(
    *,
    op: str,
    query: Any,
    anchors: Any,
    nodes: Any = None,
    frontier_anchor_id: Optional[str] = None,
) -> Tuple[Optional[str], List[Dict[str, Any]], Dict[str, Any]]:
    """Apply v5 protocol hard rules to explore inputs.

    Returns:
      - query_str: a single string query (or None)
      - anchors_list: a single-element anchors list (or empty)
      - trimmed: metadata about any env-side trimming/fallback

    Notes:
    - We accept legacy payload shapes (query as list; anchors as any sequence; nodes list).
    - We DO NOT modify the input object in-place.
    """

    trimmed: Dict[str, Any] = {}

    # Normalize query to a single string (or None)
    q = query
    if isinstance(q, list):
        if len(q) > 1:
            trimmed["query"] = {"from": len(q), "to": 1}
        q = q[0] if q else None
    if q is None:
        query_str: Optional[str] = None
    elif isinstance(q, str):
        query_str = q.strip() or None
    else:
        query_str = str(q).strip() or None

    # Normalize anchors
    a_list: List[Dict[str, Any]] = []
    try:
        if isinstance(anchors, list):
            a_list = [x for x in anchors if isinstance(x, Mapping)]
        elif isinstance(anchors, Mapping):
            a_list = [dict(anchors)]
        else:
            # best-effort: treat as iterable
            try:
                a_list = [x for x in list(anchors) if isinstance(x, Mapping)]
            except Exception:
                a_list = []
    except Exception:
        a_list = []

    if len(a_list) > 1:
        trimmed["anchors"] = {"from": len(a_list), "to": 1}
        a_list = a_list[:1]

    # Legacy: if nodes were provided but anchors omitted, map first node id to an anchor.
    if not a_list and nodes:
        try:
            if isinstance(nodes, list) and nodes:
                nid = nodes[0]
                if isinstance(nid, str) and nid.strip():
                    a_list = [{"id": nid.strip()}]
                    trimmed.setdefault("anchors", {"from": 0, "to": 1, "source": "nodes[0]"})
        except Exception:
            pass

    # Expand-like ops: if anchors are missing, fall back to last frontier anchor.
    if op in ("expand", "read") and not a_list and frontier_anchor_id:
        a_list = [{"id": frontier_anchor_id}]
        trimmed.setdefault(
            "anchors",
            {"from": 0, "to": 1, "source": "frontier_anchor_id"},
        )

    return query_str, a_list, trimmed
