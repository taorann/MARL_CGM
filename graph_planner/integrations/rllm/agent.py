"""GraphPlanner rLLM agent + (optional) OpenAI tool-call execution engine.

This module provides:
  - GraphPlannerRLLMAgent: the rLLM Agent implementation.
  - GraphPlannerToolExecutionEngine: a thin subclass of rLLM's AgentExecutionEngine
    that uses Chat Completions + tools (official tool-use) *without modifying rLLM*.

Why the engine subclass?
  rLLM's default AgentExecutionEngine calls the legacy Completions endpoint by
  serialising chat messages into a single prompt string.
  For Qwen3 (and many modern chat models), the most reliable structured action
  output is the official tool-calling interface.

Protocol mapping
  We expose fine-grained tools to the planner model:
    - explore_find(query, anchor?)
    - explore_expand(anchor?)
    - memory_commit(select_ids, keep_ids, note, tag)
    - memory_delete(delete_ids, note, tag)
    - memory_commit_note(note, tag)
    - repair(plan)
    - submit()
    - noop(reason)

  But the environment contract is still the *internal* action union:
    explore(op=...), memory(intent=...), repair, submit, noop.
  Therefore we translate tool_calls -> internal actions inside the agent.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ...core.actions import ActionUnion


class _ActionResult:
    """Minimal wrapper expected by rLLM engine.

    rLLM's AgentExecutionEngine expects ``agent.update_from_model`` to return an
    object with an ``.action`` attribute.

    IMPORTANT: ``ActionUnion`` is a *typing alias* (``typing.Union[...]``), so
    it cannot be instantiated. We therefore return this wrapper.
    """

    def __init__(self, action: ActionUnion):
        self.action = action

from ...agents.common.chat import summarise_observation
from ...agents.common.contracts import (
    PLANNER_CONTRACT,
    ProtocolError,
    parse_action_block,
    validate_planner_action,
)
from ...infra import telemetry as telemetry_mod

# rLLM types (vendored) are optional for tooling contexts.
# IMPORTANT: AgentExecutionEngine asserts isinstance(agent, BaseAgent) where BaseAgent comes
# from rllm.agents.agent in the upstream rllm package. Import from there to match the engine.
try:  # pragma: no cover
    from rllm.agents.agent import BaseAgent  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    try:
        from rllm.rllm.agents.agent import BaseAgent  # type: ignore[attr-defined]
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Cannot import BaseAgent from rllm (tried rllm.agents.agent and rllm.rllm.agents.agent). "
            "Your rllm install layout is unexpected."
        ) from e



# -----------------------------------------------------------------------------
# OpenAI tool schema (Qwen3 / OpenAI-compat) for the planner.
# -----------------------------------------------------------------------------

_NODE_ID_LIST = {
    "type": "array",
    "items": {"type": "string"},
}


OPENAI_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "explore_find",
            "description": "Search the repository graph and return a ranked list of candidates. The env sets frontier_anchor_id to the top candidate and may seed the top-k candidates into working_subgraph (W) for visibility. Do NOT repeat the exact same find(query) if the target already appears in candidates/W; instead expand or commit memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A SINGLE query string (do not split into arrays). You may use a lightweight DSL: symbol:<term> (prefer id/name matches), path:<term> (prefer path matches), +term (must include), -term (must NOT include), and quoted phrases (\"...\"). Keep it short (about 6-12 tokens) and include 1-2 strong identifier-like terms when possible.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explore_expand",
            "description": "Expand around the selected anchor (prefer frontier_anchor_id or a node id from W); merge expanded nodes into working_subgraph. Keep expansions small (env cap is typically 20).",
            "parameters": {
                "type": "object",
                "properties": {
                    "anchor": {
                        "type": "string",
                        "description": "Anchor id to expand (exactly ONE).",
                    }
                },
                "required": ["anchor"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_commit",
            "description": "Commit a small induced subgraph from working_subgraph into memory_subgraph (CGM evidence).",
            "parameters": {
                "type": "object",
                "properties": {
                    "select_ids": {
                        **_NODE_ID_LIST,
                        "description": "Ids from working_subgraph to write into memory_subgraph.",
                    },
                    "keep_ids": {
                        **_NODE_ID_LIST,
                        "description": "Ids to keep in working_subgraph during pruning (NOT written into memory).",
                    },
                    "note": {
                        "type": "string",
                        "description": "Optional planner-only text memory note to append.",
                    },
                    "tag": {"type": "string", "description": "Optional tag for this commit."},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_delete",
            "description": "Delete nodes from memory_subgraph (and incident edges); unmark in working_subgraph if present.",
            "parameters": {
                "type": "object",
                "properties": {
                    "delete_ids": {**_NODE_ID_LIST, "description": "Ids to delete from memory_subgraph."},
                    "note": {"type": "string", "description": "Optional planner-only note to append."},
                    "tag": {"type": "string", "description": "Optional tag."},
                },
                "required": ["delete_ids"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_commit_note",
            "description": "Append a planner-only text memory note (does not touch graphs).",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "The note to append."},
                    "tag": {"type": "string", "description": "Optional tag."},
                },
                "required": ["note"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repair",
            "description": "Ask the environment to run CGM repair using memory_subgraph evidence + a high-level plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Short step plan; do NOT include code patches.",
                    },
                },
                "required": ["plan"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Finalize the task (run tests / submit patch).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "noop",
            "description": "Do nothing this step.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why noop."},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return repr(obj)




def _safe_int(x: Any, default: int = 0) -> int:
    """Best-effort int parsing for env/config values."""
    if x is None:
        return int(default)
    try:
        # Accept strings like "120" or "120.0"
        if isinstance(x, str):
            s = x.strip()
            if s == "":
                return int(default)
            try:
                return int(s)
            except ValueError:
                return int(float(s))
        return int(x)
    except Exception:
        return int(default)
def _truncate(s: str, n: int = 500) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n] + "...<truncated>"


def _parse_json_maybe(raw: Any) -> Any:
    """Parse JSON string if needed; otherwise return as-is."""
    if isinstance(raw, str):
        txt = raw.strip()
        if txt.startswith("{") or txt.startswith("["):
            try:
                return json.loads(txt)
            except Exception:
                return raw
    return raw


def _tool_call_to_internal_action(name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
    """Translate external tool name -> internal planner action schema."""

    # Contracts expect:
    #   - explore.anchors: List[{"id": str}]
    #   - explore.query: Optional[str]
    # Tool calls are not always well-typed (e.g. select_ids could be a string).
    # Normalise aggressively to keep the env execution path stable.

    def _norm_ids(v: Any) -> List[str]:
        """None -> [], str -> [str], list/tuple -> filtered str list."""
        if v is None:
            return []
        if isinstance(v, str):
            s = v.strip()
            return [s] if s else []
        if isinstance(v, (list, tuple)):
            out: List[str] = []
            for x in v:
                if isinstance(x, str):
                    s = x.strip()
                    if s:
                        out.append(s)
            return out
        return []

    def _norm_anchors(v: Any) -> List[Dict[str, Any]]:
        ids = _norm_ids(v)
        return [{"id": s} for s in ids]

    tool = str(name or "")
    args = dict(arguments or {})

    # explore
    if tool == "explore_find":
        query = args.get("query")
        # Query is a single free-form string (may contain multiple keywords / directives).
        if isinstance(query, (list, tuple)):
            query = " ".join([str(x) for x in query if x is not None]).strip()
        params: Dict[str, Any] = {"op": "find", "query": query, "anchors": []}
        return {"name": "explore", "params": params}

    if tool == "explore_expand":
        anchor = args.get("anchor")
        anchors = _norm_anchors(anchor)
        params = {"op": "expand", "anchors": anchors}
        return {"name": "explore", "params": params}

    # memory
    if tool == "memory_commit":
        selector: Dict[str, Any] = {
            "select_ids": _norm_ids(args.get("select_ids")),
        }
        keep_ids = args.get("keep_ids")
        if keep_ids is not None:
            selector["keep_ids"] = _norm_ids(keep_ids)
        note = args.get("note")
        if isinstance(note, str) and note.strip():
            selector["note"] = note
        tag = args.get("tag")
        if isinstance(tag, str) and tag.strip():
            selector["tag"] = tag
        return {"name": "memory", "params": {"intent": "commit", "selector": selector}}

    if tool == "memory_delete":
        selector: Dict[str, Any] = {"delete_ids": _norm_ids(args.get("delete_ids"))}
        note = args.get("note")
        if isinstance(note, str) and note.strip():
            selector["note"] = note
        tag = args.get("tag")
        if isinstance(tag, str) and tag.strip():
            selector["tag"] = tag
        return {"name": "memory", "params": {"intent": "delete", "selector": selector}}

    if tool == "memory_commit_note":
        selector: Dict[str, Any] = {"note": str(args.get("note") or "").strip()}
        tag = args.get("tag")
        if isinstance(tag, str) and tag.strip():
            selector["tag"] = tag
        return {
            "name": "memory",
            "params": {"intent": "commit", "target": "note", "selector": selector},
        }

    # direct tools that match internal action names
    if tool in {"repair", "submit", "noop"}:
        return {"name": tool, "params": args}

    # Unknown tool: fall back to noop.
    return {"name": "noop", "params": {"reason": f"unknown_tool:{tool}"}}


def _maybe_parse_openai_tool_wrapper(model_response: str) -> Optional[Dict[str, Any]]:
    """Parse wrapper JSON produced by GraphPlannerToolExecutionEngine.

    Wrapper format (string):
      {"content": "...", "tool_calls": [{"name": "explore_find", "arguments": {...}}, ...]}
    """
    if not isinstance(model_response, str):
        return None
    txt = model_response.strip()
    if not txt.startswith("{"):
        return None
    try:
        obj = json.loads(txt)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    tool_calls = obj.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    first = tool_calls[0]
    if not isinstance(first, dict):
        return None
    name = first.get("name")
    args = _parse_json_maybe(first.get("arguments") or {})
    if not isinstance(args, dict):
        args = {}
    return _tool_call_to_internal_action(str(name or ""), args)


@dataclass
class _StepState:
    observation: Any
    reward: float
    done: bool
    info: Dict[str, Any]
    model_response: str | None = None
    action: Dict[str, Any] | None = None


class GraphPlannerRLLMAgent(BaseAgent):
    """GraphPlanner decision agent for rLLM."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        use_rule_fallback: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.system_prompt = system_prompt or PLANNER_CONTRACT.SYSTEM_PROMPT
        self.use_rule_fallback = bool(use_rule_fallback)

        # Tool-use knobs.
        #  - required: backend enforces at least one tool call per turn (preferred)
        #  - auto: model may choose tool or text
        #  - none: disable tool use
        self.openai_tool_choice: Any = os.environ.get("GP_TOOL_CHOICE", "required")

        # Internal message buffer (do NOT assign to BaseAgent.chat_completions property).
        self._messages: List[Dict[str, str]] = []

        self._steps: List[_StepState] = []
        self.reset()

    # ----- rLLM API -----
    def reset(self) -> None:
        self._steps = []
        # Keep a fixed-size message buffer (no unbounded growth):
        #   0) system: contract + tool usage rules
        #   1) assistant: last-step preamble (<=1 sentence)
        #   2) user: latest summarized observation
        self._messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": ""},
        ]

    def update_from_env(self, observation: Any, reward: float, done: bool, info: Dict[str, Any]) -> None:
        """Receive the latest env transition and refresh the user observation message."""
        self._steps.append(_StepState(observation=observation, reward=float(reward), done=bool(done), info=dict(info or {})))

        # The planner sees a compact, human-readable observation. Raw JSON tends to blow up the prompt.
        send_raw = bool(os.environ.get("GP_SEND_RAW_OBS")) or bool(os.environ.get("DEBUG_SEND_RAW_OBS"))
        if isinstance(observation, str):
            obs_text = observation
        elif send_raw:
            obs_text = _safe_json(observation)
        else:
            issue_tokens = _safe_int(os.environ.get("GP_ISSUE_TOKENS"), default=320)
            working_top_k = _safe_int(os.environ.get("GP_WORKING_TOP_K"), default=8)
            working_list_limit = _safe_int(os.environ.get("GP_WORKING_LIST_LIMIT"), default=120)
            memory_list_limit = _safe_int(os.environ.get("GP_MEMORY_LIST_LIMIT"), default=30)
            text_memory_k = _safe_int(os.environ.get("GP_TEXT_MEMORY_K"), default=8)
            working_snippet_k = _safe_int(os.environ.get("GP_WORKING_SNIPPET_K"), default=2)
            working_snippet_lines = _safe_int(os.environ.get("GP_WORKING_SNIPPET_LINES"), default=12)
            max_lines = _safe_int(os.environ.get("GP_WORKING_MAX_LINES"), default=80)
            max_chars = _safe_int(os.environ.get("GP_WORKING_MAX_CHARS"), default=6000)
            try:
                obs_text, _meta = summarise_observation(
                    observation,
                    reward=float(reward),
                    done=bool(done),
                    info=info or {},
                    include_issue=True,
                    issue_target_tokens=issue_tokens,
                    steps_target=_safe_int(os.environ.get("GP_STEPS_TARGET"), default=0),
                    working_top_k=working_top_k,
                    working_list_limit=working_list_limit,
                    memory_list_limit=memory_list_limit,
                    text_memory_k=text_memory_k,
                    working_snippet_k=working_snippet_k,
                    working_snippet_lines=working_snippet_lines,
                    working_max_lines=max_lines,
                    working_max_chars=max_chars,
                )
            except Exception:
                obs_text = _safe_json(observation)

        # Replace the latest user message (keep message count fixed).
        if not self._messages:
            self.reset()
        # Ensure structure: [system, assistant, user]
        while len(self._messages) < 3:
            self._messages.append({"role": "user", "content": ""})
        self._messages[2] = {"role": "user", "content": obs_text}

        if bool(os.environ.get("DEBUG_ACTION_RESULT")):
            prefix = f"[gp-agent] step={len(self._steps)-1} update_from_env"
            print(prefix, "reward=", reward, "done=", done, "kind=", info.get("kind"), "op=", info.get("op"))

        # Verbose per-step state print: working/memory node ids + memorized marks.
        # Enabled by default; set GP_PRINT_STATE=0 to turn off.
        _ps = os.environ.get("GP_PRINT_STATE", "1").strip().lower()
        if _ps not in {"0", "false", "no", "off"}:
            try:
                if not isinstance(observation, dict):
                    return
                ws = observation.get("working_subgraph") or observation.get("subgraph") or {}
                ms = observation.get("memory_subgraph") or {}
                w_nodes = []
                m_nodes = []
                if isinstance(ws, dict):
                    w_nodes = ws.get("nodes") or []
                elif isinstance(ws, list):
                    w_nodes = ws
                if isinstance(ms, dict):
                    m_nodes = ms.get("nodes") or []
                elif isinstance(ms, list):
                    m_nodes = ms

                m_ids = set()
                for n in m_nodes:
                    if isinstance(n, dict):
                        m_ids.add(str(n.get("id") or ""))
                    else:
                        m_ids.add(str(n))

                # Print a tail slice to keep logs manageable.
                tail = w_nodes[-12:] if len(w_nodes) > 12 else w_nodes
                out = []
                for n in tail:
                    if isinstance(n, dict):
                        nid = str(n.get("id") or "")
                    else:
                        nid = str(n)
                    mark = "[M]" if nid in m_ids else "[-]"
                    out.append(f"{mark}{nid}")
                last_info = obs.get("last_info") if isinstance(obs, dict) else None
                cand_count = None
                last_op = None
                frontier = None
                try:
                    if isinstance(last_info, dict):
                        c = last_info.get("candidates")
                        if isinstance(c, list):
                            cand_count = len(c)
                        last_op = last_info.get("op")
                        frontier = last_info.get("frontier_anchor_id")
                    if isinstance(obs, dict) and obs.get("frontier_anchor_id"):
                        frontier = obs.get("frontier_anchor_id")
                except Exception:
                    pass
                print(f"[gp-agent] W(n={len(w_nodes)}) M(n={len(m_nodes)}) frontier={frontier} last_op={last_op} cand={cand_count} W.tail=" + ", ".join(out))
            except Exception:
                pass

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Full dialogue history for the engine (OpenAI chat format)."""
        return list(self._messages)

    def update_from_model(self, model_response: str) -> _ActionResult:
        # Record model response.
        if self._steps:
            self._steps[-1].model_response = model_response

        # Append assistant message for history.
        # When tool-calling is enabled, model_response is a wrapper JSON; do NOT
        # feed the wrapper back to the model (it pollutes the dialogue context).
        wrapper_obj: Optional[Dict[str, Any]] = None
        if isinstance(model_response, str):
            txt = model_response.strip()
            if txt.startswith("{"):
                try:
                    obj = json.loads(txt)
                    if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
                        wrapper_obj = obj
                except Exception:
                    wrapper_obj = None

        def _first_sentence(s: str) -> str:
            s = (s or "").strip()
            if not s:
                return ""
            # Prefer newline sentence boundary.
            head = s.split("\n", 1)[0].strip()
            # If still long, cut at the first period-like boundary.
            for sep in ("。", ".", "!", "?", "！", "？"):
                if sep in head:
                    head = head.split(sep, 1)[0].strip()
                    break
            return head[:240]

        # Keep a fixed-size dialogue context: update the "preamble" slot
        # instead of appending to history.
        if wrapper_obj is not None:
            preamble = _first_sentence(str(wrapper_obj.get("content") or ""))
        else:
            preamble = _first_sentence(str(model_response or ""))
        if len(self._messages) >= 2 and isinstance(self._messages[1], dict):
            self._messages[1]["content"] = preamble

        action: Dict[str, Any]
        try:
            # 1) Preferred: tool wrapper JSON (produced by our engine).
            if wrapper_obj is not None:
                tool_calls = wrapper_obj.get("tool_calls") or []
                first = tool_calls[0] if tool_calls else {}
                name = first.get("name") if isinstance(first, dict) else None
                args = _parse_json_maybe(first.get("arguments") if isinstance(first, dict) else {})
                if not isinstance(args, dict):
                    args = {}
                internal = _tool_call_to_internal_action(str(name or ""), args)
                action = validate_planner_action(internal)
            else:
                # 2) Legacy text protocol (<function=...> blocks or bare JSON).
                parsed_block = parse_action_block(model_response)
                action = validate_planner_action(parsed_block)
        except ProtocolError as exc:
            # NOTE: When tool-calling is enabled we still sometimes see the model
            # emit plain text (no tool_calls, no <function=...> block). In that
            # case `parse_action_block` raises `missing-function-tag`.
            #
            # We must not crash the whole trajectory: instead fall back to a
            # conservative noop (with a short excerpt for debugging).
            code = str(getattr(exc, "code", ""))
            if ("missing-function-tag" in code) or ("MISSING_FUNCTION_TAG" in code):
                excerpt = _truncate(str(model_response or ""), 240)
                fallback = {"name": "noop", "params": {"reason": "missing_function_tag", "excerpt": excerpt}}
                action = validate_planner_action(fallback)
            elif self.use_rule_fallback:
                fallback = {"name": "noop", "params": {"reason": f"protocol_error:{exc.code}"}}
                action = validate_planner_action(fallback)
            else:
                # Best-effort diagnostics to help spot prompt/protocol issues.
                try:
                    print("[gp-agent] protocol_error:", code)
                    print("[gp-agent] raw_model_excerpt=", _truncate(str(model_response or ""), 400))
                except Exception:
                    pass
                raise

        # --- Anti-loop guard: prevent looping on identical explore/find ---
        try:
            if os.environ.get("GP_AVOID_REPEAT_FIND", "1").strip().lower() not in {"0", "false", "no", "off"} and len(self._steps) >= 2:
                def _is_explore_find(a: Any) -> bool:
                    try:
                        if a is None:
                            return False
                        if isinstance(a, Mapping):
                            if a.get("name") == "explore":
                                p = a.get("params") or {}
                                return isinstance(p, dict) and str(p.get("op") or "").lower() == "find"
                            return str(a.get("type") or "").lower() == "explore" and str(a.get("op") or "").lower() == "find"
                        return str(getattr(a, "type", "")).lower() == "explore" and str(getattr(a, "op", "")).lower() == "find"
                    except Exception:
                        return False

                def _get_query(a: Any) -> str:
                    if a is None:
                        return ""
                    if isinstance(a, Mapping):
                        if a.get("name") == "explore":
                            p = a.get("params") or {}
                            if isinstance(p, dict):
                                return str(p.get("query") or "").strip()
                        return str(a.get("query") or "").strip()
                    return str(getattr(a, "query", "") or "").strip()

                def _get_anchors(a: Any) -> Any:
                    if a is None:
                        return []
                    if isinstance(a, Mapping):
                        if a.get("name") == "explore":
                            p = a.get("params") or {}
                            if isinstance(p, dict):
                                return p.get("anchors") or []
                        return a.get("anchors") or []
                    return getattr(a, "anchors", []) or []

                prev_action = self._steps[-2].action
                if _is_explore_find(action) and _is_explore_find(prev_action):
                    q = _get_query(action)
                    pq = _get_query(prev_action)
                    if q and pq and q == pq and _get_anchors(action) == _get_anchors(prev_action):
                        obs = (self._steps[-1].observation or {}) if self._steps else {}
                        last_info = obs.get("last_info") or {}
                        frontier = obs.get("frontier_anchor_id") or last_info.get("frontier_anchor_id") or None

                        # A) If a frontier anchor exists, expand it instead of repeating the find.
                        if isinstance(frontier, str) and frontier.strip():
                            rewritten = {
                                "name": "explore",
                                "params": {
                                    "op": "expand",
                                    "anchors": [{"id": frontier}],
                                    "hop": 1,
                                    "limit": int(os.environ.get("GP_REPEAT_FIND_EXPAND_LIMIT", "20") or 20),
                                },
                            }
                            action = validate_planner_action(rewritten)
                            if os.environ.get("GP_DEBUG_REPEAT_FIND", "0").strip().lower() in {"1", "true", "yes", "y"}:
                                print(f"[gp-agent] anti-loop: repeated find({q!r}) -> expand({frontier})")
                        else:
                            # B) If the repeated find returned 0 candidates, broaden the query once.
                            cands = last_info.get("candidates")
                            if isinstance(cands, list) and len(cands) == 0:
                                def _fallback_query(qs: str) -> str:
                                    qs = str(qs or "").strip()
                                    if not qs:
                                        return ""
                                    parts = []
                                    for token in qs.split():
                                        if ":" in token:
                                            parts.append(token.split(":", 1)[1])
                                        else:
                                            parts.append(token)
                                    qs = " ".join(parts).strip()
                                    qs = re.sub(r"[\._/\-:]+", " ", qs)
                                    qs = re.sub(r"\s+", " ", qs).strip()
                                    return qs

                                fb = _fallback_query(q)
                                if fb and fb != q:
                                    rewritten = {
                                        "name": "explore",
                                        "params": {
                                            "op": "find",
                                            "query": fb,
                                            "hop": 1,
                                            "limit": int(os.environ.get("GP_FIND_LIMIT", "50") or 50),
                                        },
                                    }
                                    action = validate_planner_action(rewritten)
                                    if os.environ.get("GP_DEBUG_REPEAT_FIND", "0").strip().lower() in {"1", "true", "yes", "y"}:
                                        print(f"[gp-agent] anti-loop: repeated find({q!r}) w/0 candidates -> find({fb!r})")
        except Exception:
            pass

        if self._steps:
            self._steps[-1].action = action

        if bool(os.environ.get("DEBUG_FULL_MODEL_OUTPUT")):
            print("[gp-agent] raw_model=", _truncate(model_response, 1200))
        if bool(os.environ.get("DEBUG_ACTION_RESULT")):
            print("[gp-agent] parsed_action=", _truncate(_safe_json(action), 1200))

        telemetry_mod.emit_event("planner_action", {"action": action})
        # IMPORTANT: return a wrapper with `.action` for rLLM engine
        return _ActionResult(action=action)

    def get_current_state(self) -> Optional[_StepState]:
        return self._steps[-1] if self._steps else None

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Messages passed to the OpenAI-compatible chat completions API.

        Note: BaseAgent may define chat_completions as a read-only property. We
        override it to expose our mutable fixed-size buffer.
        """

        # Keep shape stable: system, assistant(preamble), user(observation)
        return list(self._messages)

    # ----- Tool-use helpers (read by GraphPlannerToolExecutionEngine) -----
    def get_openai_tools(self) -> List[Dict[str, Any]]:
        # Allow disabling via env for debugging.
        if os.environ.get("GP_DISABLE_TOOL_USE") in {"1", "true", "yes"}:
            return []
        return OPENAI_TOOLS


# -----------------------------------------------------------------------------
# Optional: tool-call engine adapter (no changes to vendored rLLM).
# -----------------------------------------------------------------------------

try:  # pragma: no cover
    import asyncio
    import contextvars
    import openai
    from openai.types.chat import ChatCompletion
    from openai.types.completion import Completion

    from rllm.engine.agent_execution_engine import AgentExecutionEngine

    _CURRENT_AGENT: contextvars.ContextVar[Any] = contextvars.ContextVar("_gp_current_agent", default=None)


    class GraphPlannerToolExecutionEngine(AgentExecutionEngine):
        """An AgentExecutionEngine that uses chat.completions + tools.

        This class keeps rLLM untouched by:
          - setting a ContextVar in run_agent_trajectory_async
          - overriding _get_openai_async to call chat.completions with tools
        """

        async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Text", **kwargs):
            token = _CURRENT_AGENT.set(self.agents[idx])
            try:
                return await super().run_agent_trajectory_async(idx, application_id, seed=seed, mode=mode, **kwargs)
            finally:
                _CURRENT_AGENT.reset(token)

        async def _get_openai_async(self, prompt, _, **kwargs):
            """Override the OpenAI request to use tools.

            Input `prompt` is usually a chat message list (role/content dicts).
            We return either:
              - wrapper JSON string containing tool_calls, or
              - plain assistant content (fallback)
            """

            agent = _CURRENT_AGENT.get()
            tools = []
            tool_choice: Any = "auto"
            if agent is not None:
                try:
                    tools = list(getattr(agent, "get_openai_tools")())  # type: ignore[misc]
                except Exception:
                    tools = []
                tool_choice = getattr(agent, "openai_tool_choice", "auto")

            # If prompt is not in chat format, fall back to the base completions path.
            if not (isinstance(prompt, list) and all(isinstance(m, dict) for m in prompt)):
                return await super()._get_openai_async(prompt, _, **kwargs)

            async def get_response(messages: List[Dict[str, Any]]):
                retries = self.api_retries
                while retries > 0:
                    try:
                        params: Dict[str, Any] = {}
                        params.update(self.sampling_params or {})
                        params.update(kwargs or {})

                        # Avoid passing timeout twice (rLLM passes timeout via kwargs).
                        timeout_s = params.pop("timeout", None)
                        if timeout_s is None:
                            timeout_s = params.pop("request_timeout", 3600)
                        # Some backends reject tools=[]; omit when empty.
                        if tools:
                            params["tools"] = tools
                            params["tool_choice"] = tool_choice
                        response = await self.client.chat.completions.create(
                            messages=messages,
                            timeout=timeout_s,
                            **params,
                        )
                        return response
                    except openai.RateLimitError:
                        retries -= 1
                        if retries == 0:
                            return "Error: Rate limit reached and retries exhausted."
                        await asyncio.sleep(5)
                    except Exception as e:
                        return f"Error processing content: {e}"

            response = await get_response(prompt)
            if isinstance(response, str):
                return response

            # OpenAI python types.
            if isinstance(response, ChatCompletion):
                msg = response.choices[0].message
                content = msg.content or ""

                # tool_calls (new) OR function_call (legacy)
                tool_calls_out: List[Dict[str, Any]] = []
                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls or []:
                        try:
                            tool_calls_out.append(
                                {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                }
                            )
                        except Exception:
                            continue
                elif getattr(msg, "function_call", None):
                    fc = msg.function_call
                    try:
                        tool_calls_out.append({"name": fc.name, "arguments": fc.arguments})
                    except Exception:
                        pass

                if tool_calls_out:
                    wrapper = {"content": content, "tool_calls": tool_calls_out}
                    return json.dumps(wrapper, ensure_ascii=False)
                return content

            # Legacy completion type (should not happen with chat.completions)
            if isinstance(response, Completion):
                return response.choices[0].text

            return str(response)


except Exception:  # pragma: no cover
    # rLLM / openai not importable in certain tooling contexts.
    GraphPlannerToolExecutionEngine = None  # type: ignore[assignment]
