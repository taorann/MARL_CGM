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
        # The protocol wants query_count=1, but the model may emit a list of
        # tokens/keywords. Preserve information by joining.
        if len(q) > 1:
            trimmed["query"] = {"from": len(q), "to": 1, "join": True}
        parts: List[str] = []
        for x in q:
            try:
                s = str(x).strip()
            except Exception:
                s = ""
            if s:
                parts.append(s)
        q = " ".join(parts) if parts else None
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

    # If the model provided an "anchor" id that is *likely* missing the canonical
    # type prefix (e.g. it emits `path/to/file.py:Foo.bar:123` instead of
    # `func:path/to/file.py:Foo.bar:123`), we can safely upgrade it to the
    # canonical frontier anchor when they refer to the same underlying node.
    if op in ("expand", "read") and a_list and frontier_anchor_id:
        try:
            aid = a_list[0].get("id")
            fid = frontier_anchor_id
            if isinstance(aid, str) and isinstance(fid, str):
                aid_s = aid.strip()
                fid_s = fid.strip()
                # canonical ids start with a short type prefix like "func:", "class:", ...
                canonical_prefixes = {
                    "func",
                    "class",
                    "file",
                    "dir",
                    "var",
                    "test",
                    "issue",
                    "doc",
                }
                head = aid_s.split(":", 1)[0] if aid_s else ""
                is_canonical = head in canonical_prefixes
                # If not canonical but matches the suffix of frontier id, upgrade.
                if (not is_canonical) and fid_s.endswith(aid_s) and fid_s != aid_s:
                    trimmed.setdefault(
                        "anchors",
                        {"from": 1, "to": 1, "source": "frontier_anchor_id", "reason": "missing_prefix"},
                    )
                    a_list[0]["id"] = fid_s
        except Exception:
            pass

    return query_str, a_list, trimmed
