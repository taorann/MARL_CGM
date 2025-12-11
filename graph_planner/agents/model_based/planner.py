"""Planner agent that delegates decision making to a locally hosted LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

from ..common import text_protocol
from ..common.contracts import ProtocolError, parse_action_block, validate_planner_action
from ..common.chat import (
    FALLBACK_REASON_KEY,
    SYSTEM_PROMPT,
    action_to_payload,
    summarise_observation,
)
from ...core.actions import (
    ActionUnion,
    ExploreAction,
    MemoryAction,
    NoopAction,
    RepairAction,
    SubmitAction,
)
from ...infra.config import Config, load as load_config
from ...integrations.local_llm import LocalLLMError, build_planner_client


@dataclass
class _AgentState:
    issue: Dict[str, Any] = field(default_factory=dict)
    phase: str = "expand"
    last_candidates: List[Dict[str, Any]] = field(default_factory=list)
    last_snippets: List[Dict[str, Any]] = field(default_factory=list)
    last_memory: Dict[str, Any] = field(default_factory=dict)
    last_repair: Dict[str, Any] = field(default_factory=dict)
    plan_targets: List[Dict[str, Any]] = field(default_factory=list)
    plan_text: str = ""


class _ChatClient(Protocol):
    def chat(
        self,
        messages: Iterable[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        ...


class LocalLLMPlannerAgent:
    """Agent that mirrors the rule-based flow but relies on a local chat model."""

    def __init__(
        self,
        *,
        client: Optional[_ChatClient] = None,
        system_prompt: Optional[str] = None,
        use_rule_fallback: bool = False,
    ) -> None:
        """Create a planner agent backed by a local LLM.

        Args:
            client: Pre-initialised chat client. If None, it is built from config.
            system_prompt: Optional system prompt override.
            use_rule_fallback: Deprecated; kept only for backwards-compatible signature.
        """
        self.cfg: Config = load_config()
        pm_cfg = getattr(self.cfg, "planner_model", None)
        if client is None:
            if pm_cfg is None:
                raise RuntimeError("planner_model section missing in configuration")
            try:
                client = build_planner_client(pm_cfg)
            except Exception as exc:  # pragma: no cover - configuration error
                raise RuntimeError(
                    "planner model client could not be initialised; ensure local endpoint is configured"
                ) from exc
        self._client = client
        self.state = _AgentState()
        # Rule-based fallback is deprecated; keep attribute for backwards compatibility.
        self._rule_agent = None
        prompt = system_prompt or getattr(pm_cfg, "system_prompt", None) or SYSTEM_PROMPT
        self._messages: List[Dict[str, str]] = [{"role": "system", "content": prompt}]
        self._last_reward: float = 0.0
        self._last_done: bool = False
        self._last_info: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Single planner step.

        Strategy
        --------
        1. Summarise observation and ask the local LLM for an action.
        2. If the action JSON cannot be parsed / validated, send a brief
           clarification message and ask the model *once more*.
        3. If the second attempt still fails (or the client errors), fall back
           to a simple safe action so the episode can proceed.
        """
        if not self.state.issue or obs.get("reset") or obs.get("steps") == 0:
            self._on_reset(obs)
        self._update_state(obs)

        summary, metadata = summarise_observation(
            obs, self._last_reward, self._last_done, self._last_info
        )
        metadata = self._normalise_metadata(metadata)
        self._messages.append({"role": "user", "content": summary})

        # ------------------------------------------------------------------
        # First attempt
        # ------------------------------------------------------------------
        try:
            response = self._client.chat(self._messages, extra={"metadata": metadata})
        except LocalLLMError as exc:
            return self._fallback_decision(
                obs,
                summary,
                "client_error",
                error=str(exc),
                raw_response=None,
                raw_action=None,
            )

        try:
            thought, action_obj, assistant_msg, parser_meta = self._parse_model_response(
                response, obs
            )
        except (ProtocolError, ValueError) as exc:
            # ------------------------------------------------------------------
            # Second attempt: explicitly ask model to fix the format.
            # ------------------------------------------------------------------
            retry_hint = (
                "Your previous reply could not be parsed as a valid planner action. "
                "Please reply again with a single JSON action block that matches "
                "the planner action schema."
            )
            self._messages.append({"role": "user", "content": retry_hint})
            try:
                retry_response = self._client.chat(
                    self._messages, extra={"metadata": metadata}
                )
            except LocalLLMError as exc2:
                # Both the original call and the retry failed at the client level.
                return self._fallback_decision(
                    obs,
                    summary,
                    "client_error_retry",
                    error=str(exc2),
                    raw_response=response,
                    raw_action=None,
                )

            try:
                thought, action_obj, assistant_msg, parser_meta = self._parse_model_response(
                    retry_response, obs
                )
                parser_meta["used_retry"] = True
            except (ProtocolError, ValueError) as exc2:
                # Parsing failed twice; fall back to a simple safe action.
                return self._fallback_decision(
                    obs,
                    summary,
                    "invalid-action",
                    error=f"initial: {exc}; retry: {exc2}",
                    raw_response=retry_response,
                    raw_action=None,
                )

        self._messages.append({"role": "assistant", "content": assistant_msg})
        self.state.phase = getattr(action_obj, "type", self.state.phase)
        result = {
            "prompt": summary,
            "response": assistant_msg,
            "thought": thought,
            "action_obj": action_obj,
            "metadata": parser_meta,
        }
        return result

    def observe_outcome(self, reward: float, done: bool, info: Optional[Dict[str, Any]]) -> None:
        self._last_reward = reward
        self._last_done = done
        self._last_info = info or {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _on_reset(self, obs: Dict[str, Any]) -> None:
        self.state = _AgentState(issue=obs.get("issue") or {})
        self._messages = [self._messages[0]]  # keep system prompt
        self._last_reward = 0.0
        self._last_done = False
        self._last_info = {}

    def _update_state(self, obs: Dict[str, Any]) -> None:
        info = obs.get("last_info") or {}
        kind = info.get("kind")
        if kind == "explore" and info.get("op") == "expand":
            self.state.last_candidates = info.get("candidates", [])
            self.state.phase = "memory"
        elif kind == "memory":
            self.state.last_memory = info
            self.state.phase = "read"
        elif kind == "explore" and info.get("op") == "read":
            self.state.last_snippets = info.get("snippets", [])
            self.state.phase = "plan"
        elif kind == "repair":
            self.state.last_repair = info
            if info.get("applied"):
                self.state.phase = "submit"
            else:
                self.state.phase = "expand"

    def _parse_model_response(
        self, response: str, obs: Dict[str, Any]
    ) -> tuple[str, ActionUnion, str, Dict[str, Any]]:
        parsed: Dict[str, Any] | None = None
        raw_params: Dict[str, Any] = {}
        try:
            parsed = parse_action_block(response)
            raw_params = dict(parsed.get("params") or {})
            thought = str(raw_params.get("thought", "")).strip()
            params = dict(raw_params)
            params.pop("thought", None)
            action_obj = validate_planner_action(parsed)
        except ProtocolError:
            # Let the caller decide whether to retry or fall back.
            raise
        except Exception as exc:
            # Wrap any other parsing/validation error so callers can distinguish
            # an invalid action from transport/client errors.
            raise ValueError(f"invalid-action: {exc}") from exc

        if isinstance(action_obj, RepairAction) and self.state.issue:
            action_obj = action_obj.copy(update={"issue": dict(self.state.issue)})

        meta = {
            "used_fallback": False,
            "raw_action": parsed or {},
        }
        assistant_msg = response or text_protocol.format_action_block(
            str(parsed.get("name") if parsed else "noop"),
            raw_params if parsed else {},
        )
        return thought, action_obj, assistant_msg, meta

    def _fallback_decision(
        self,
        obs: Dict[str, Any],
        summary: str,
        reason: str,
        *,
        error: Optional[str] = None,
        raw_response: Optional[str] = None,
        raw_action: Any | None = None,
    ) -> Dict[str, Any]:
        thought, action, assistant_msg, meta = self._fallback_tuple(
            obs, reason, raw_response or "", raw_action, error=error
        )
        self._messages.append({"role": "assistant", "content": assistant_msg})
        return {
            "prompt": summary,
            "response": assistant_msg,
            "thought": thought,
            "action_obj": action,
            "metadata": meta,
        }

    def _fallback_tuple(
        self,
        obs: Dict[str, Any],  # kept for signature compatibility; not used
        reason: str,
        response: str,
        raw_action: Any | None = None,
        *,
        error: Optional[str] = None,
    ) -> tuple[str, Any, str, Dict[str, Any]]:
        """Return a simple safe fallback action (no rule-based planner)."""
        # Compose a minimal "thought" explaining why we fell back.
        thought = f"[fallback] {reason}"
        if error:
            thought += f" | error={error}"

        # Default to a SubmitAction so the episode can terminate gracefully.
        action: ActionUnion = SubmitAction()
        payload = action_to_payload(action)
        params: Dict[str, Any] = {
            "thought": thought,
            **{k: v for k, v in payload.items() if k != "type"},
        }
        if error:
            params["error"] = error
        params[FALLBACK_REASON_KEY] = reason

        assistant_msg = text_protocol.format_action_block(payload.get("type", "noop"), params)
        meta: Dict[str, Any] = {
            "used_fallback": True,
            FALLBACK_REASON_KEY: reason,
            "raw_action": raw_action,
            "model_response": response,
        }
        if error:
            meta["error"] = error
        return thought, action, assistant_msg, meta

    def _normalise_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return json.loads(json.dumps(metadata, default=str))
        except Exception:
            return {k: str(v) for k, v in (metadata or {}).items()}


__all__ = ["LocalLLMPlannerAgent"]
