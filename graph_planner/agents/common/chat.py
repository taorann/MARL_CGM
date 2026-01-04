"""Shared helpers for model-driven planner agents."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .text_compact import compact_issue_text

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


@dataclass
class ChatMessage:
    """A minimal chat message container.

    Some integrations historically imported `ChatMessage` from this module.
    The planner/runtime does not require this class at runtime, but providing
    it keeps older imports working and avoids hard failures.
    """

    role: str
    content: str
    name: str | None = None
    tool_calls: Any | None = None

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
    obs: Any,
    reward: float,
    done: bool,
    info: Dict[str, Any],
    *,
    include_issue: bool = True,
    issue_target_tokens: int = 320,
    steps_target: int = 6,
    working_top_k: int = 8,
    working_list_limit: int = 120,
    memory_list_limit: int = 30,
    text_memory_k: int = 8,
    working_snippet_k: int = 2,
    working_snippet_lines: int = 12,
    working_max_lines: int = 80,
    working_max_chars: int = 6000,
    full_working: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Convert an env observation into a compact prompt string + metadata.

    Goals:
    - keep within prompt budget (avoid dumping raw JSON)
    - show enough repo context for exploration (W), while keeping memorized set (M) visible
    - show only top-k snippets to avoid blowing token budget
    """

    def _clamp_int(x: Any, default: int) -> int:
        try:
            v = int(x)
            return v if v >= 0 else default
        except Exception:
            return default

    issue_target_tokens = _clamp_int(issue_target_tokens, 320)
    steps_target = _clamp_int(steps_target, 6)
    working_top_k = _clamp_int(working_top_k, 8)
    working_list_limit = _clamp_int(working_list_limit, 120)
    memory_list_limit = _clamp_int(memory_list_limit, 30)
    text_memory_k = _clamp_int(text_memory_k, 8)
    working_max_lines = _clamp_int(working_max_lines, 80)
    working_max_chars = _clamp_int(working_max_chars, 6000)

    if not isinstance(obs, dict):
        # Fall back gracefully; caller may decide to send raw JSON for debugging.
        summary = str(obs)
        return summary, {"reward": reward, "done": done, "info": info or {}}

    issue = obs.get("issue")
    steps = obs.get("steps")
    last_info = obs.get("last_info")
    subgraph_stats = obs.get("subgraph_stats") or {}
    memory_stats = obs.get("memory_stats") or {}
    text_memory = obs.get("text_memory") or {}
    working_subgraph = obs.get("working_subgraph") or {}

    # Normalize working nodes
    if isinstance(working_subgraph, dict):
        nodes = working_subgraph.get("nodes") or []
    elif isinstance(working_subgraph, list):
        nodes = working_subgraph
    else:
        nodes = []

    # Normalize steps
    if not isinstance(steps, list):
        steps = []

    # Rank nodes: memorize-marked first, then last_candidates, then higher score, then stable id
    def _node_rank(n: Dict[str, Any]) -> Tuple[int, int, float, str]:
        memorized = 1 if bool(n.get("memorized")) else 0
        in_last = 1 if bool(n.get("in_last_candidates")) else 0
        try:
            score = float(n.get("score") or 0.0)
        except Exception:
            score = 0.0
        nid = str(n.get("id") or "")
        # sort descending for memorized/in_last/score
        return (-memorized, -in_last, -score, nid)

    nodes_sorted = sorted([n for n in nodes if isinstance(n, dict)], key=_node_rank)

    # Helpers to format nodes/snippets
    def _node_loc(n: Dict[str, Any]) -> str:
        path = n.get("path") or n.get("file") or ""
        span = n.get("span") or {}
        if isinstance(span, dict):
            s = span.get("start")
            e = span.get("end")
            if isinstance(s, int) and isinstance(e, int):
                return f"{path}:{s}-{e}"
            if isinstance(s, int):
                return f"{path}:{s}"
        return str(path)

    def _node_title(n: Dict[str, Any]) -> str:
        return str(n.get("name") or n.get("symbol") or n.get("title") or n.get("kind") or n.get("id") or "")

    def _node_line(n: Dict[str, Any]) -> str:
        nid = str(n.get("id") or "")
        kind = str(n.get("kind") or "")
        loc = _node_loc(n)
        title = _node_title(n)
        mem = "M" if bool(n.get("memorized")) else "-"
        return f"- [{mem}] {nid} | {kind} | {title} | {loc}"

    def _snippet_text(n: Dict[str, Any]) -> str:
        # Prefer embedded snippet_lines
        sl = n.get("snippet_lines")
        if isinstance(sl, list) and sl:
            try:
                joined = "\n".join(str(x) for x in sl)
            except Exception:
                joined = str(sl)
            return joined
        sn = n.get("snippet")
        if isinstance(sn, str) and sn.strip():
            return sn
        return ""

    # Compose summary
    out: List[str] = []

    def _issue_to_title_body(v: Any) -> Tuple[str, str]:
        """Coerce env-provided issue payload into (title, body) strings.

        SWE-bench tasks often provide a dict with keys like `problem_statement`.
        We intentionally **ignore** any potential solution fields (e.g., `patch`,
        `test_patch`) if present in the payload.
        """
        if v is None:
            return "", ""
        if isinstance(v, str):
            return "", v.strip()
        if not isinstance(v, dict):
            return "", str(v).strip()

        instance_id = str(v.get("instance_id") or v.get("id") or v.get("issue_id") or "").strip()
        repo = str(v.get("repo") or v.get("repo_id") or "").strip()
        title = str(v.get("title") or "").strip()

        body = (
            v.get("problem_statement")
            or v.get("body")
            or v.get("description")
            or v.get("text")
            or ""
        )
        body = str(body).strip()

        if not title and body:
            for line in body.splitlines():
                line = line.strip()
                if line:
                    title = line[:120]
                    break

        header_lines: List[str] = []
        if instance_id:
            header_lines.append(f"Instance: {instance_id}")
        if repo:
            header_lines.append(f"Repo: {repo}")
        header = ("\n".join(header_lines) + "\n\n") if header_lines else ""

        return title, (header + body).strip()

    if include_issue:
        issue_title, issue_body = _issue_to_title_body(issue)
        if issue_title or issue_body:
            out.append("## Issue")
            out.append(compact_issue_text(issue_title, issue_body, target_tokens=issue_target_tokens))
            out.append("")

    if steps:
        out.append("## Recent steps")
        # Keep last steps_target steps
        for s in steps[-steps_target:]:
            if isinstance(s, str) and s.strip():
                out.append(f"- {s.strip()}")
        out.append("")

    # Text memory notes (these are summaries, not chain-of-thought)
    notes = text_memory.get("notes")
    if isinstance(notes, list) and notes:
        out.append("## Notes")
        for s in notes[-text_memory_k:]:
            if isinstance(s, str) and s.strip():
                out.append(f"- {s.strip()}")
        out.append("")

    # A small, high-signal summary of the last env result.
    if isinstance(last_info, dict) and last_info:
        out.append("## Last result")
        kind = str(last_info.get("kind") or last_info.get("type") or "").strip()
        op = str(last_info.get("op") or last_info.get("intent") or "").strip()
        if kind or op:
            out.append(f"- kind/op: {kind or '-'} / {op or '-'}")
        frontier = last_info.get("frontier_anchor_id")
        if isinstance(frontier, str) and frontier.strip():
            out.append(f"- frontier_anchor_id: {frontier.strip()}")

        def _stats_line(label: str, stats: Any) -> str | None:
            if not isinstance(stats, dict):
                return None
            try:
                nodes = int(stats.get("n_nodes") or 0)
                edges = int(stats.get("n_edges") or 0)
            except Exception:
                return None
            return f"- {label}: {nodes}/{edges}"

        w_line = _stats_line("W(nodes/edges)", subgraph_stats)
        m_line = _stats_line("M(nodes/edges)", memory_stats)
        if w_line:
            out.append(w_line)
        if m_line:
            out.append(m_line)

        # Deltas are injected by the env step wrapper (so the planner can detect no-progress loops).
        def _d(k: str) -> str:
            v = last_info.get(k)
            if isinstance(v, (int, float)):
                return str(int(v))
            return "?"

        if any(k in last_info for k in ("delta_working_nodes", "delta_working_edges", "delta_memory_nodes", "delta_memory_edges")):
            out.append(
                f"- ΔW(nodes/edges): {_d('delta_working_nodes')}/{_d('delta_working_edges')} | "
                f"ΔM(nodes/edges): {_d('delta_memory_nodes')}/{_d('delta_memory_edges')}"
            )

        # Quick hints for exploration effectiveness.
        c = last_info.get("candidates")
        if isinstance(c, list):
            out.append(f"- candidates: {len(c)}")
        seeded = last_info.get("seeded_working_ids")
        if isinstance(seeded, list) and seeded:
            out.append(f"- seeded_into_W: {len(seeded)}")
        pruned = last_info.get("pruned_working")
        if isinstance(pruned, list) and pruned:
            out.append(f"- pruned_from_W: {len(pruned)}")
        out.append("")

    # Memorized nodes are a high-signal subset of W, but we do not show a separate
    # M section to avoid duplication. Memorized nodes are marked with [M] inside W.
    mem_nodes = [n for n in nodes_sorted if bool(n.get("memorized"))]

    # Guidance: keep expands small and populate memory early so CGM has signal.
    if nodes_sorted and not mem_nodes:
        suggest_ids = [str(n.get("id") or "").strip() for n in nodes_sorted[:3]]
        suggest_ids = [x for x in suggest_ids if x]
        out.append("## Guidance")
        out.append("- Expand is capped (default 20). Prefer multiple focused expands.")
        if suggest_ids:
            out.append(f"- Memory (M) is empty. Commit 1–3 high-signal nodes now, e.g. memory_commit(select_ids={suggest_ids}).")
        else:
            out.append("- Memory (M) is empty. Commit 1–3 high-signal nodes now via memory_commit(select_ids=[...]).")
        out.append("")

    # Candidates (from the most recent find). Always guard against missing
    # candidates to avoid crashing and falling back to raw JSON prompts.
    cands = last_info.get("candidates") if isinstance(last_info, dict) else None
    if isinstance(cands, list):
        q = str(last_info.get("query") or "").strip()  # type: ignore[union-attr]
        out.append(f"## Candidates — {len(cands)}")
        if q:
            out.append(f"- query: `{q}`")
        if not cands:
            out.append(
                "_No candidates matched. Do **not** repeat the same query; broaden it (e.g., drop `symbol:` / split terms) or change the anchor._"
            )
            out.append("")
        else:
            for c in cands[: max(1, working_top_k)]:
                if not isinstance(c, dict):
                    continue
                nid = str(c.get("id") or "").strip()
                if not nid:
                    continue
                kind = str(c.get("kind") or "")
                path = str(c.get("path") or "")
                span = c.get("span")
                if isinstance(span, dict):
                    s = span.get("start")
                    e = span.get("end")
                    if s is not None and e is not None:
                        path = f"{path}:{s}-{e}"
                score = c.get("score")
                if isinstance(score, (int, float)):
                    out.append(f"- {nid} ({kind}) {path}  score={float(score):.3f}")
                else:
                    out.append(f"- {nid} ({kind}) {path}")
            if len(cands) > max(1, working_top_k):
                out.append(f"- ... (+{len(cands) - max(1, working_top_k)} more)")
            out.append("")

    # Working nodes list (IDs only), then snippets
    if nodes_sorted:
        out.append(f"## Working subgraph (W) — {len(nodes_sorted)} nodes")

        # High-signal snapshot so humans (and the planner) can quickly see what W contains.
        # Keep this extremely compact to avoid blowing the prompt budget.
        try:
            w_edges = None
            if isinstance(working_subgraph, dict):
                es = working_subgraph.get("edges")
                if isinstance(es, list):
                    w_edges = len(es)
            if isinstance(last_info, dict):
                st = last_info.get("subgraph_stats")
                if isinstance(st, dict) and isinstance(st.get("n_edges"), int):
                    w_edges = int(st.get("n_edges"))

            file_counts: Counter[str] = Counter()
            kind_counts: Counter[str] = Counter()
            for n in nodes_sorted:
                if not isinstance(n, dict):
                    continue
                k = str(n.get("kind") or "").lower().strip() or "?"
                kind_counts[k] += 1
                p = str(n.get("path") or "").strip()
                if p:
                    file_counts[p] += 1

            mem_n = sum(1 for n in nodes_sorted if isinstance(n, dict) and bool(n.get("memorized")))
            top_files = ", ".join(
                f"{p}({c})" for p, c in file_counts.most_common(5)
            )
            top_kinds = ", ".join(
                f"{k}({c})" for k, c in kind_counts.most_common(6)
            )
            w_edge_s = f"/{w_edges} edges" if isinstance(w_edges, int) else ""
            out.append(f"- W: {len(nodes_sorted)} nodes{w_edge_s}; M: {mem_n} nodes; files: {len(file_counts)}")
            if top_files:
                out.append(f"- top files: {top_files}")
            if top_kinds:
                out.append(f"- kinds: {top_kinds}")
        except Exception:
            pass

        # Optionally include a compact *full* index of W so the planner can make decisions based on the
        # entire working set (not only the top-K preview). This is intentionally snippet-free.
        if full_working:
            try:
                import os as _os
                full_limit = int(_os.environ.get("GP_FULL_W_LIMIT", "800"))
            except Exception:
                full_limit = 800
            out.append("### Working index (full, compact)")
            out.append(f"(Up to {full_limit} nodes; fields: [M]=memorized, kind, path, name, id)")
            shown = 0
            for n in nodes_sorted:
                if shown >= full_limit:
                    break
                nid = str(n.get("id") or "")
                kind = str(n.get("kind") or n.get("type") or "")
                name = str(n.get("name") or n.get("symbol") or "")
                path = str(n.get("path") or n.get("file") or n.get("filepath") or "")
                mem = "[M]" if bool(n.get("memorized")) else "   "
                # Keep each line short and stable.
                line = f"- {mem} {kind[:18]:18} {path[:64]:64} {name[:40]:40} {nid[:32]}"
                out.append(line.rstrip())
                shown += 1
            if len(nodes_sorted) > shown:
                out.append(f"- ... (+{len(nodes_sorted)-shown} more)")
            out.append("")

        out.append(
            f"(List truncated to {working_list_limit}; snippets include candidate previews, memorized nodes, "
            f"and recently touched working nodes.)"
        )
        for n in nodes_sorted[:working_list_limit]:
            out.append(_node_line(n))
        if len(nodes_sorted) > working_list_limit:
            out.append(f"- ... (+{len(nodes_sorted)-working_list_limit} more)")
        out.append("")
        out.append("### Snippets (candidates + memorized)")

        # 1) candidate previews (find results)
        entries: List[Tuple[str, str]] = []
        seen: set[str] = set()
        for p in (last_info.get("candidate_previews") or []):
            pid = str(p.get("id") or "").strip()
            if not pid or pid in seen:
                continue
            sn_lines = p.get("snippet_lines") or []
            snip = "\n".join(sn_lines).strip()
            if snip:
                entries.append((pid, snip))
                seen.add(pid)

        # 2) memorized nodes (subset of W)
        for n in nodes_sorted:
            if not bool(n.get("memorized")):
                continue
            nid = str(n.get("id") or "").strip()
            if not nid or nid in seen:
                continue
            snip = _snippet_text(n)
            if snip:
                entries.append((nid, snip))
                seen.add(nid)

        # 3) recently touched working nodes (even if not memorized)
        # This makes sure the planner can see the code it just explored.
        max_snippets = max(working_top_k, working_snippet_k)
        if working_snippet_k > 0:
            for n in nodes_sorted:
                nid = str(n.get("id") or "").strip()
                if not nid or nid in seen:
                    continue
                # Avoid file nodes (they are summaries, not code bodies)
                if str(n.get("kind") or "").lower() == "file":
                    continue
                snip = _snippet_text(n)
                if snip:
                    entries.append((nid, snip))
                    seen.add(nid)
                if len(entries) >= max_snippets:
                    break

        # Limit total snippets we show
        entries = entries[:max_snippets]

        used = 0
        chars = 0
        for nid, snip in entries:
            # clamp lines/chars per snippet
            sn_lines = snip.splitlines()
            line_cap = working_snippet_lines if working_snippet_lines > 0 else working_max_lines
            if line_cap and len(sn_lines) > line_cap:
                sn_lines = sn_lines[:line_cap] + ["..."]
            snip2 = "\n".join(sn_lines)
            if working_max_chars and len(snip2) > working_max_chars:
                snip2 = snip2[:working_max_chars] + "..."

            # global char budget per section
            if working_max_chars and chars + len(snip2) > working_max_chars:
                break
            out.append(f"#### {nid}")
            out.append("```")
            out.append(snip2)
            out.append("```")
            out.append("")
            used += 1
            chars += len(snip2)

    summary = "\n".join(out).strip()

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
        # v5 simplified protocol: the planner should ONLY provide a high-level plan.
        # The env will always run CGM and apply edits; ignore any patch/targets the model emits.
        plan_steps = _normalize_plan_steps(payload.get("plan"), payload.get("subplan"))
        return RepairAction(
            apply=True,
            issue={},
            plan=plan_steps,
            plan_targets=[],
            patch=None,
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
        # Only serialise the minimal planner-facing payload.
        return {
            "type": "repair",
            "plan": _normalize_plan_steps(action.plan),
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
