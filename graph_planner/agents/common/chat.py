"""Shared helpers for model-driven planner agents."""

from __future__ import annotations

import json
from .text_compact import compact_issue_text
import re
from typing import Any, Dict, List, Tuple

from ...core.actions import (
    ActionUnion,
    ExploreAction,
    MemoryAction,
    NoopAction,
    RepairAction,
    SubmitAction,
)
from .contracts import PLANNER_SYSTEM_PROMPT

SYSTEM_PROMPT = PLANNER_SYSTEM_PROMPT

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})```", re.DOTALL)
FALLBACK_REASON_KEY = "fallback_reason"


def _normalize_memory_intent(value: Any) -> str:
    """Coerce memory.intent into {'commit','delete'}.

    The model sometimes emits synonyms like 'analyze' when it means committing a note.
    We normalize to keep the trajectory running (and to avoid noop fallbacks).
    """
    raw = str(value or "").strip().lower()
    if raw in {"commit", "add", "save", "store", "remember", "keep", "record", "log", "write", "analysis", "analyze"}:
        return "commit"
    if raw in {"delete", "remove", "forget", "drop", "clear", "purge"}:
        return "delete"
    return "commit"



def summarise_observation(
    obs: Dict[str, Any],
    reward: float,
    done: bool,
    info: Dict[str, Any] | None,
    *,
    include_issue: bool = True,
    issue_target_tokens: int = 320,
    working_top_k: int = 8,
    working_max_lines: int = 50,
    working_max_chars: int = 4000,
) -> Tuple[str, Dict[str, Any]]:
    """Summarise environment observation into a compact planner prompt.

    Point (1): Every turn can include the (compacted) issue text and the working subgraph,
    with strict caps (top-k nodes, max lines, max chars) to avoid blowing up context.

    Point (2): Explore op space is {find, expand}. No separate read op is surfaced to the model.
    """
    issue = obs.get("issue") or {}
    steps = obs.get("steps", 0)
    last_info = obs.get("last_info") or {}
    pack = obs.get("observation_pack") or {}

    lines: List[str] = [
        f"Issue: {issue.get('id', issue.get('issue_id', 'unknown'))} | step={steps} | reward={reward} | done={done}",
    ]

    # --- Issue context (compacted) ---
    if include_issue:
        issue_compact = compact_issue_text(
            str(issue.get("title", "")).strip(),
            str(issue.get("body", "") or ""),
            target_tokens=issue_target_tokens,
        )
        if issue_compact:
            lines.append("Issue context:")
            lines.append(issue_compact)

    # --- Working subgraph (top-k snippets) ---
    subgraph = obs.get("subgraph") or {}
    nodes = []
    if isinstance(subgraph, dict):
        nodes = subgraph.get("nodes") or []
    elif isinstance(subgraph, list):
        nodes = subgraph
    if nodes:
        # Prefer nodes that actually carry text/snippet lines.
        def _has_text(n: Dict[str, Any]) -> bool:
            if not isinstance(n, dict):
                return False
            if n.get("text") or n.get("content"):
                return True
            sl = n.get("snippet_lines")
            return isinstance(sl, list) and any(isinstance(x, str) and x.strip() for x in sl)

        filtered = [n for n in nodes if isinstance(n, dict) and _has_text(n)]
        if not filtered:
            filtered = [n for n in nodes if isinstance(n, dict)]

        # Stable top-k selection: prioritize failure_frame file, recently touched nodes, and last candidates.
        ff_path = None
        if isinstance(pack.get("failure_frame"), dict):
            ff_path = pack.get("failure_frame", {}).get("path") or pack.get("failure_frame", {}).get("file")
        last_cand_ids = set()
        if isinstance(last_info, dict):
            for c in (last_info.get("candidates") or []):
                if isinstance(c, dict) and isinstance(c.get("id"), str):
                    last_cand_ids.add(c.get("id"))

        def _node_rank(n: Dict[str, Any]) -> Tuple[int, int, int, int]:
            path = str(n.get("path") or "")
            nid = n.get("id")
            # 0 is better
            in_fail_file = 0 if (ff_path and path and (path == ff_path or path.endswith(ff_path))) else 1
            in_last_cands = 0 if (isinstance(nid, str) and nid in last_cand_ids) else 1
            touched = n.get("gp_last_touched_step") or n.get("gp_added_step") or 0
            try:
                touched_i = int(touched)
            except Exception:
                touched_i = 0
            score = n.get("score") or n.get("rank_score") or 0
            try:
                score_i = int(float(score) * 1000)
            except Exception:
                score_i = 0
            # We want: failure file first, then last candidates, then more recent, then higher score.
            return (in_fail_file, in_last_cands, -touched_i, -score_i)

        filtered = sorted(filtered, key=_node_rank)
        filtered = filtered[: max(1, int(working_top_k or 1))]


        lines.append("Working subgraph (top-k snippets):")
        for idx, n in enumerate(filtered, start=1):
            nid = n.get("id") or "?"
            path = n.get("path") or "?"
            span = n.get("span") or {}
            start = span.get("start", n.get("start", "?"))  # compat
            end = span.get("end", n.get("end", "?"))        # compat

            # Extract snippet text
            snippet_text = ""
            if isinstance(n.get("text"), str) and n.get("text").strip():
                snippet_text = n.get("text")
            elif isinstance(n.get("content"), str) and n.get("content").strip():
                snippet_text = n.get("content")
            else:
                sl = n.get("snippet_lines")
                if isinstance(sl, list):
                    snippet_text = "\n".join([x for x in sl if isinstance(x, str)])

            if snippet_text:
                # Cap by lines then by chars.
                snippet_lines = snippet_text.splitlines()
                snippet_text = "\n".join(snippet_lines[: max(1, int(working_max_lines or 1))])
                if working_max_chars and len(snippet_text) > int(working_max_chars):
                    snippet_text = snippet_text[: int(working_max_chars)] + "\n...<truncated>"

            header = f"[{idx}] {path}:{start}-{end} (id={nid})"
            lines.append(header)
            if snippet_text:
                lines.append(snippet_text)

    # --- Failure frame anchor (optional) ---
    if pack.get("failure_frame"):
        ff = pack["failure_frame"]
        file_hint = ff.get("path") or ff.get("file")
        if file_hint:
            lines.append(f"Failure frame: {file_hint}:{ff.get('lineno')}")
    # NOTE: Subgraph/memory stats intentionally omitted by default.

    # --- Last action/result echo ---
    kind = last_info.get("kind")
    op = last_info.get("op")
    if kind == "explore" and op:
        lines.append(f"Last op: explore/{op}")
        cands = last_info.get("candidates") or []
        if cands:
            lines.append("Top candidates:\n" + _format_candidates(cands))
        # Backward compat: if older code still supplies snippets, show a tiny preview.
        snippets = last_info.get("snippets") or []
        for snip in snippets[:2]:
            snippet_lines = snip.get("snippet_lines") or snip.get("snippet") or []
            if isinstance(snippet_lines, list):
                preview = " | ".join([str(x) for x in snippet_lines[:2]])
            else:
                preview = str(snippet_lines)[:200]
            lines.append(f"Snippet {snip.get('path')}@{snip.get('start')}->{snip.get('end')}: {preview}")
    elif kind:
        lines.append(f"Last op: {kind}")

    if kind == "repair":
        lines.append(f"Patch applied: {last_info.get('applied')}")
        if last_info.get("lint"):
            lines.append(f"Lint rc={last_info['lint'].get('rc')}")
        if last_info.get("tests"):
            lines.append(f"Tests passed={last_info['tests'].get('passed')}")

    summary = "\n".join(lines)
    metadata = {
        "issue": issue,
        "steps": steps,
        "last_info": last_info,
        "reward": reward,
        "done": done,
        "info": info or {},
    }
    return summary, metadata



def _normalize_plan_steps(plan, subplan=None):
    """Normalize repair plan/subplan into a List[str] with de-duplicated, non-empty steps."""
    def to_list(x):
        if x is None:
            return []
        if isinstance(x, str):
            s = x.strip()
            return [s] if s else []
        if isinstance(x, list):
            out=[]
            for v in x:
                if isinstance(v, str):
                    s=v.strip()
                    if s:
                        out.append(s)
            return out
        return []

    steps = to_list(plan)
    for s in to_list(subplan):
        if s not in steps:
            steps.append(s)
    return steps

def _format_candidates(candidates: List[Dict[str, Any]], *, limit: int = 3) -> str:
    rows = []
    for cand in candidates[:limit]:
        path = cand.get("path") or "?"
        span = cand.get("span") or {}
        start = span.get("start")
        end = span.get("end")
        score = cand.get("score")
        rows.append(f"{path}:{start}-{end} (score={score})")
    return "\n".join(rows)


def extract_json_payload(response: str) -> Dict[str, Any] | None:
    if not response:
        return None
    fence_matches = JSON_BLOCK_RE.findall(response)
    candidate = fence_matches[-1] if fence_matches else None
    if not candidate:
        stripped = response.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidate = stripped
    if not candidate:
        candidate = _first_brace_block(response)
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _first_brace_block(text: str) -> str | None:
    stack = []
    start = None
    for idx, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = idx
            stack.append(ch)
        elif ch == "}" and stack:
            stack.pop()
            if not stack and start is not None:
                return text[start : idx + 1]
    return None


def action_from_payload(payload: Dict[str, Any] | None) -> ActionUnion | None:
    if not isinstance(payload, dict):
        return None
    type_name = (payload.get("type") or payload.get("action") or payload.get("kind") or "").lower()
    if type_name == "explore":
        return ExploreAction(
            op=str(payload.get("op") or payload.get("operation") or "expand"),
            anchors=list(payload.get("anchors") or []),
            nodes=list(payload.get("nodes") or []),
            query=payload.get("query"),
            hop=int(payload.get("hop", 1)),
            limit=int(payload.get("limit", 50)),
        )
    if type_name == "memory":
        return MemoryAction(
            target=str(payload.get("target", "explore")),
            intent=_normalize_memory_intent(payload.get("intent", "commit")),
            selector=payload.get("selector"),
        )
    if type_name == "repair":
        plan_targets = payload.get("plan_targets") or payload.get("targets") or []
        if not isinstance(plan_targets, list):
            plan_targets = []
        plan_steps = _normalize_plan_steps(payload.get("plan"), payload.get("subplan"))
        patch = payload.get("patch")
        if patch is not None and not isinstance(patch, dict):
            patch = None
        return RepairAction(
            apply=bool(payload.get("apply", True)),
            issue=dict(payload.get("issue") or {}),
            plan=plan_steps,
            plan_targets=plan_targets,
            patch=patch,
        )


    if type_name == "submit":
        return SubmitAction()
    if type_name == "noop":
        return NoopAction()
    return None


def action_to_payload(action: ActionUnion) -> Dict[str, Any]:
    if isinstance(action, ExploreAction):
        return {
            "type": "explore",
            "op": action.op,
            "anchors": (action.anchors[:1] if isinstance(action.anchors, list) and len(action.anchors)>1 else action.anchors),
            "nodes": action.nodes,
            "query": (action.query[0] if isinstance(action.query, list) and action.query else action.query),
            "hop": action.hop,
            "limit": action.limit,
        }
    if isinstance(action, MemoryAction):
        return {
            "type": "memory",
            "target": action.target,
            "intent": action.intent,
            "selector": action.selector,
        }
    if isinstance(action, RepairAction):
        return {
            "type": "repair",
            "apply": action.apply,
            "issue": action.issue,
            "plan": _normalize_plan_steps(action.plan),
            "plan_targets": action.plan_targets,
            "patch": action.patch,
        }
    if isinstance(action, SubmitAction):
        return {"type": "submit"}
    return {"type": "noop"}


__all__ = [
    "SYSTEM_PROMPT",
    "FALLBACK_REASON_KEY",
    "summarise_observation",
    "extract_json_payload",
    "action_from_payload",
    "action_to_payload",
]
