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
    - explore_find(query)
    - explore_expand(anchor?)
    - memory_commit(select_ids, keep_ids?, note?, tag?)
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
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ...core.actions import ActionUnion
from ...agents.common.chat import ChatMessage
from ...agents.common.contracts import (
    PLANNER_CONTRACT,
    ProtocolError,
    parse_action_block,
    validate_planner_action,
)
from ...infra import telemetry as telemetry_mod

# rLLM types (vendored) are optional for tooling contexts.
try:  # pragma: no cover
    from rllm.agents.agent import BaseAgent, Action as RLLMAction, Step as RLLMStep, Trajectory as RLLMTrajectory
except Exception:  # pragma: no cover
    BaseAgent = object  # type: ignore
    RLLMAction = object  # type: ignore
    RLLMStep = object  # type: ignore
    RLLMTrajectory = object  # type: ignore


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
            "description": "Find exactly one next anchor in repo_graph by query; add it into working_subgraph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query; must be a single string."},
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
            "description": "Expand candidates around the current/selected anchor; merge into working_subgraph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "anchor": {
                        "type": "string",
                        "description": "Anchor id to expand. If empty, env uses frontier_anchor_id.",
                    },
                },
                "required": [],
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
                    "select_ids": {**_NODE_ID_LIST, "description": "Ids from working_subgraph to write into memory_subgraph."},
                    "keep_ids": {**_NODE_ID_LIST, "description": "Ids to keep in working_subgraph during pruning (NOT written into memory)."},
                    "note": {"type": "string", "description": "Optional text memory note to append (a short summary, not reasoning)."},
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
                    "note": {"type": "string", "description": "Optional text memory note to append (a short summary, not reasoning)."},
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
            "description": "Append a text memory note (does not touch graphs).",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "The note to append (a short summary, not reasoning)."},
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
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "noop",
            "description": "Do nothing this step.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
    },
]



def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return repr(obj)


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
    """Translate external tool name -> internal planner action schema.

    Notes
    - Tool schemas intentionally do NOT include `thought`/CoT fields.
    - `select_ids`/`keep_ids`/`delete_ids` are normalised to `List[str]`.
    - `repair` always sets `apply=True` (the model does not decide apply-vs-preview).
    """

    def _norm_str_list(x: Any) -> List[str]:
        if x is None:
            return []
        if isinstance(x, str):
            s = x.strip()
            return [s] if s else []
        if isinstance(x, (list, tuple)):
            out: List[str] = []
            for v in x:
                if v is None:
                    continue
                if isinstance(v, str):
                    s = v.strip()
                else:
                    try:
                        s = str(v).strip()
                    except Exception:
                        continue
                if s:
                    out.append(s)
            return out
        return []

    tool = str(name or "").strip()
    args = dict(arguments or {})

    # explore
    if tool == "explore_find":
        query = args.get("query")
        params: Dict[str, Any] = {"op": "find", "query": query, "anchors": []}
        return {"name": "explore", "params": params}

    if tool == "explore_expand":
        anchor = args.get("anchor")
        anchors: List[Dict[str, Any]] = []
        if isinstance(anchor, str) and anchor.strip():
            anchors = [{"id": anchor.strip()}]
        params = {"op": "expand", "anchors": anchors}
        return {"name": "explore", "params": params}

    # memory
    if tool == "memory_commit":
        selector: Dict[str, Any] = {"select_ids": _norm_str_list(args.get("select_ids"))}
        keep_ids = _norm_str_list(args.get("keep_ids"))
        if keep_ids:
            selector["keep_ids"] = keep_ids
        note = args.get("note")
        if isinstance(note, str) and note.strip():
            selector["note"] = note.strip()
        tag = args.get("tag")
        if isinstance(tag, str) and tag.strip():
            selector["tag"] = tag.strip()
        return {"name": "memory", "params": {"intent": "commit", "selector": selector}}

    if tool == "memory_delete":
        selector: Dict[str, Any] = {"delete_ids": _norm_str_list(args.get("delete_ids"))}
        note = args.get("note")
        if isinstance(note, str) and note.strip():
            selector["note"] = note.strip()
        tag = args.get("tag")
        if isinstance(tag, str) and tag.strip():
            selector["tag"] = tag.strip()
        return {"name": "memory", "params": {"intent": "delete", "selector": selector}}

    if tool == "memory_commit_note":
        selector: Dict[str, Any] = {"note": str(args.get("note") or "").strip()}
        tag = args.get("tag")
        if isinstance(tag, str) and tag.strip():
            selector["tag"] = tag.strip()
        return {"name": "memory", "params": {"intent": "commit", "target": "observation", "selector": selector}}

    # repair / submit / noop
    if tool == "repair":
        plan = args.get("plan")
        if isinstance(plan, str):
            plan = [plan]
        if not isinstance(plan, list):
            plan = []
        plan = [str(x).strip() for x in plan if str(x).strip()]
        return {"name": "repair", "params": {"apply": True, "plan": plan}}

    if tool == "submit":
        return {"name": "submit", "params": {}}

    if tool == "noop":
        return {"name": "noop", "params": {}}

    # Unknown tool: fall back to noop.
    return {"name": "noop", "params": {}}



def _maybe_parse_openai_tool_wrapper(model_response: str) -> Optional[Tuple[Dict[str, Any], str]]:
    """Parse wrapper JSON produced by GraphPlannerToolExecutionEngine.

    Wrapper format (string):
      {"content": "...", "tool_calls": [{"name": "explore_find", "arguments": {...}}, ...]}

    Returns:
      (internal_action_dict, assistant_content)
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
    content = obj.get("content")
    content_str = content if isinstance(content, str) else ""
    return _tool_call_to_internal_action(str(name or ""), args), content_str



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

        self._steps: List[_StepState] = []
        self._chat_completions: List[ChatMessage] = []
        self.reset()

    # ----- rLLM API -----
    def reset(self) -> None:
        self._steps = []
        self._chat_completions = [
            {"role": "system", "content": self.system_prompt},
        ]

    @property
    def chat_completions(self) -> List[ChatMessage]:
        return self._chat_completions

    @chat_completions.setter
    def chat_completions(self, value: List[ChatMessage]) -> None:
        self._chat_completions = value

    @property
    def trajectory(self) -> Any:
        """Convert internal state to rLLM Trajectory."""
        try:
            steps = []
            for st in self._steps:
                steps.append(
                    RLLMStep(
                        chat_completions=list(self._chat_completions),
                        action=st.action,
                        observation=st.observation,
                        model_response=st.model_response or "",
                        info=st.info,
                        reward=float(st.reward),
                        done=bool(st.done),
                    )
                )
            return RLLMTrajectory(task=None, steps=steps, reward=0.0)
        except Exception:
            return {"steps": [st.__dict__ for st in self._steps], "reward": 0.0}

    def update_from_env(self, observation: Any, reward: float, done: bool, info: Dict[str, Any], **kwargs: Any) -> None:
        self._steps.append(_StepState(observation=observation, reward=float(reward), done=bool(done), info=dict(info or {})))

        # The planner sees the observation (already preformatted by env).
        obs_text = observation if isinstance(observation, str) else _safe_json(observation)
        self.chat_completions.append({"role": "user", "content": obs_text})

        if bool(os.environ.get("DEBUG_ACTION_RESULT")):
            prefix = f"[gp-agent] step={len(self._steps)-1} update_from_env"
            print(prefix, "reward=", reward, "done=", done, "kind=", info.get("kind"), "op=", info.get("op"))

    def update_from_model(self, model_response: str, **kwargs: Any) -> "RLLMAction":
        # Record model response.
        if self._steps:
            self._steps[-1].model_response = model_response

        assistant_content = model_response or ""

        action_obj: Any
        try:
            # 1) Preferred: tool wrapper JSON (produced by our engine).
            parsed = _maybe_parse_openai_tool_wrapper(model_response)
            if parsed is not None:
                internal_action, content = parsed
                # Keep history clean: only store the (optional) one-sentence preamble.
                assistant_content = (content or "").strip()

                # Fill required repair.issue from the latest observation, not from the model.
                if internal_action.get("name") == "repair":
                    try:
                        obs = self._steps[-1].observation if self._steps else None
                        issue = (obs or {}).get("issue") if isinstance(obs, dict) else None
                        if isinstance(issue, dict):
                            internal_action.setdefault("params", {}).setdefault("issue", issue)
                        else:
                            internal_action.setdefault("params", {}).setdefault("issue", {})
                    except Exception:
                        internal_action.setdefault("params", {}).setdefault("issue", {})

                action_obj = validate_planner_action(internal_action)
            else:
                # 2) Legacy text protocol (<function=...> blocks or bare JSON).
                parsed_block = parse_action_block(model_response)
                action_obj = validate_planner_action(parsed_block)
        except ProtocolError as exc:
            # Optional rule-fallback (very conservative).
            if self.use_rule_fallback:
                fallback = {"name": "noop", "params": {}}
                action_obj = validate_planner_action(fallback)
            else:
                raise

        # Append assistant message to history. For tool-calls, this is only the short preamble.
        if assistant_content:
            self.chat_completions.append({"role": "assistant", "content": assistant_content})

        if self._steps:
            self._steps[-1].action = getattr(action_obj, "model_dump", lambda **_: action_obj)()  # debug-friendly

        # Always compute a safe, serialisable representation for logging/telemetry.
        try:
            dumped = action_obj.model_dump() if hasattr(action_obj, "model_dump") else action_obj
        except Exception:
            dumped = action_obj

        if bool(os.environ.get("DEBUG_FULL_MODEL_OUTPUT")):
            print("[gp-agent] raw_model=", _truncate(model_response, 1200))
        if bool(os.environ.get("DEBUG_ACTION_RESULT")):
            print("[gp-agent] parsed_action=", _truncate(_safe_json(dumped), 1200))

        try:
            telemetry_mod.emit_event("planner_action", {"action": dumped})
        except Exception:
            pass

        # rLLM engine expects an Action dataclass with an `.action` attribute.
        return RLLMAction(action=action_obj)


    def get_current_state(self) -> Optional[_StepState]:
        return self._steps[-1] if self._steps else None

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
