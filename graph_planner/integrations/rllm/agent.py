"""Graph Planner rLLM agent 封装。

English summary
    Provides a thin wrapper around the rLLM ``BaseAgent`` so PPO training can
    interact with Graph Planner while keeping JSON parsing, fallback logic and
    CGM patch synthesis encapsulated in Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import os
import json
import sys

from ...infra.vendor import ensure_rllm_importable

ensure_rllm_importable()

try:
    from rllm.agents.agent import Action, BaseAgent, Step, Trajectory  # type: ignore[attr-defined]
except ModuleNotFoundError:
    try:
        from rllm.rllm.agents.agent import Action, BaseAgent, Step, Trajectory  # type: ignore[attr-defined]
    except ModuleNotFoundError as _exc:  # pragma: no cover - optional dependency
        Action = None  # type: ignore[assignment]
        BaseAgent = None  # type: ignore[assignment]
        Step = None  # type: ignore[assignment]
        Trajectory = None  # type: ignore[assignment]
        _AGENT_IMPORT_ERROR = _exc
    else:
        _AGENT_IMPORT_ERROR = None
else:
    _AGENT_IMPORT_ERROR = None

RuleFallbackAgent = None

from ...agents.common import text_protocol
from ...agents.common.contracts import ProtocolError, parse_action_block, validate_planner_action
from ...agents.common.chat import (
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


if BaseAgent is None:

    class GraphPlannerRLLMAgent:  # type: ignore[misc]
        """Placeholder that surfaces an actionable import error when rLLM is missing."""

        def __init__(self, *args, **kwargs):
            raise ImportError("rLLM is required to use GraphPlannerRLLMAgent") from _AGENT_IMPORT_ERROR

else:

    @dataclass
    class _AgentState:
        """保存最近一次交互的上下文信息，供 fallback 与补丁生成复用。"""

        issue: Dict[str, Any] = field(default_factory=dict)
        phase: str = "expand"
        last_candidates: List[Dict[str, Any]] = field(default_factory=list)
        last_snippets: List[Dict[str, Any]] = field(default_factory=list)
        last_memory: Dict[str, Any] = field(default_factory=dict)
        last_repair: Dict[str, Any] = field(default_factory=list)
        plan_targets: List[Dict[str, Any]] = field(default_factory=list)
        plan_text: str = ""

    DEBUG = bool(os.environ.get("DEBUG"))

    @dataclass
    class GraphPlannerRLLMAgent(BaseAgent):
        """面向 rLLM 的 Graph Planner 代理封装。"""

        system_prompt: str = SYSTEM_PROMPT
        # 默认关闭规则 fallback，保持“纯模型”行为
        use_rule_fallback: bool = False

        def _trace_id(self) -> str:
            """Concise trace tag for logs: <runner>/<run_id>.

            - runner is 2-digit runner_id when available (00/01/02...).
            - run_id is truncated to first 6 chars to avoid log spam.
            """
            obs = self._last_env_observation or {}
            runner_id = None
            run_id = None
            if isinstance(obs, dict):
                runner_id = obs.get("runner_id")
                run_id = obs.get("run_id")
                issue = obs.get("issue") or {}
                if isinstance(issue, dict):
                    runner_id = runner_id if runner_id is not None else issue.get("runner_id")
                    run_id = run_id or issue.get("run_id") or issue.get("id")
            try:
                runner = int(runner_id) if runner_id is not None else 0
            except Exception:
                runner = 0
            rs = str(run_id or "no_run")
            return f"{runner:02d}/{rs[:6]}"

        def __post_init__(self) -> None:
            """初始化轨迹、消息列表以及可选的规则后备代理。"""

            self._trajectory = Trajectory()
            self._messages: List[Dict[str, str]] = []
            self._rule_agent = None
            self._last_env_observation: Dict[str, Any] | None = None
            self._step_index = 0
            self._state = _AgentState()
            self._config: Config = load_config()
            if self.use_rule_fallback:
                self._maybe_init_rule_fallback()
            self.reset()

        # ------------------------------------------------------------------
        # BaseAgent interface
        # ------------------------------------------------------------------
        def reset(self):
            """重置内部状态与交互历史。"""

            self._trajectory = Trajectory()
            self._messages = [{"role": "system", "content": self.system_prompt}]
            self._cur_step: Step | None = None
            self._last_env_observation = None
            self._step_index = 0
            self._state = _AgentState()

        def _maybe_init_rule_fallback(self) -> None:
            """Lazy initialiser for optional rule-based fallback agent."""

            global RuleFallbackAgent

            if not self.use_rule_fallback:
                self._rule_agent = None
                return

            if RuleFallbackAgent is None:
                from ...agents.rule_based.planner import PlannerAgent as RuleFallbackAgent  # type: ignore

            self._rule_agent = RuleFallbackAgent()

        def update_from_env(self, observation: Any, reward: float, done: bool, info: Dict[str, Any] | None, **kwargs):
            """根据环境返回的观察值更新轨迹和内部状态。"""

            # Ensure trace_id can read current observation
            self._last_env_observation = observation
            info = info or {}
            if DEBUG and os.environ.get("DEBUG_ACTION_RESULT") == "1":
                trace = self._trace_id()
                try:
                    max_chars = int(os.environ.get("DEBUG_MAX_CHARS", "50000"))
                except Exception:
                    max_chars = 50000
                try:
                    dumped_info = json.dumps(info, ensure_ascii=False, default=str)
                except Exception:
                    dumped_info = repr(info)
                try:
                    dumped_obs = json.dumps(observation, ensure_ascii=False, default=str)
                except Exception:
                    dumped_obs = repr(observation)
                if max_chars and len(dumped_info) > max_chars:
                    dumped_info = dumped_info[:max_chars] + "...<truncated>"
                if max_chars and len(dumped_obs) > max_chars:
                    dumped_obs = dumped_obs[:max_chars] + "...<truncated>"
                print(f"[gp-agent {trace}] env_info={dumped_info}")
                print(f"[gp-agent {trace}] env_observation={dumped_obs}")

            if DEBUG:
                trace = self._trace_id()
                kind = info.get("kind")
                op = info.get("op")
                print(
                    f"[gp-agent {trace}] step={self._step_index} update_from_env: "
                    f"reward={reward} done={done} kind={kind} op={op}"
                )
            text, metadata = summarise_observation(observation, reward, done, info)
            if self._trajectory.steps:
                prior = self._trajectory.steps[-1]
                prior.next_observation = text
                prior.reward = reward
                prior.done = done
                prior.info = {**prior.info, **metadata}
            self._messages.append({"role": "user", "content": text})
            self._cur_step = Step(observation=text, info=metadata)
            self._cur_step.chat_completions = list(self._messages)
            self._last_env_observation = observation
            self._update_state(observation)

        def update_from_model(self, response: str, **kwargs) -> Action:
            """解析模型输出，更新当前步骤并返回 rLLM ``Action``。"""

            if self._cur_step is None:
                raise RuntimeError("update_from_env must be called before update_from_model")
            thought, action_obj, assistant_msg, parser_meta = self._parse_model_response(response)
            if DEBUG and os.environ.get("DEBUG_FULL_MODEL_OUTPUT") == "1":
                trace = self._trace_id()
                print(f"[gp-agent {trace}] raw_model_output:\n{response}\n[/raw_model_output]")

            if DEBUG:
                trace = self._trace_id()
                tp = getattr(action_obj, "type", None) or getattr(action_obj, "kind", None)
                try:
                    if hasattr(action_obj, "model_dump"):
                        payload = action_obj.model_dump(exclude_none=True)  # pydantic v2
                    elif hasattr(action_obj, "dict"):
                        payload = action_obj.dict(exclude_none=True)  # pydantic v1
                    else:
                        payload = getattr(action_obj, "__dict__", {})
                    action_json = json.dumps(payload, ensure_ascii=False, default=str)
                except Exception:
                    action_json = repr(action_obj)
                no_trunc = (os.environ.get("DEBUG_NO_TRUNCATION") == "1") or (os.environ.get("DEBUG_FULL_MODEL_OUTPUT") == "1")
                if (not no_trunc) and len(action_json) > 600:
                    action_json = action_json[:600] + "...<truncated>"
                thought_preview = thought
                if (not no_trunc) and len(thought_preview) > 400:
                    thought_preview = thought_preview[:400] + "...<truncated>"
                print(
                    f"[gp-agent {trace}] step={self._step_index} parsed_from_model: "
                    f"type={tp} thought={thought_preview!r} action={action_json}"
                )
            self._messages.append({"role": "assistant", "content": assistant_msg})

            self._cur_step.thought = thought
            self._cur_step.action = action_to_payload(action_obj)
            self._cur_step.model_response = response
            self._cur_step.info.update(parser_meta)
            self._trajectory.steps.append(self._cur_step)
            self._step_index += 1
            return Action(action=action_obj)

        @property
        def trajectory(self) -> Trajectory:
            """训练过程中累计的步骤轨迹。"""

            return self._trajectory

        @property
        def chat_completions(self) -> List[Dict[str, str]]:
            """以聊天消息形式返回历史对话。"""

            return list(self._messages)

        def get_current_state(self) -> Step | None:
            """返回最近一次 ``Step``，如不存在则返回 ``None``。"""

            if not self._trajectory.steps:
                return None
            return self._trajectory.steps[-1]

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def _parse_model_response(
            self, response: str
        ) -> tuple[str, ActionUnion, str, Dict[str, Any]]:
            """尝试从模型回复中解析 Thought 与 Action，失败时触发 fallback。"""

            parsed: Dict[str, Any] | None = None
            raw_params: Dict[str, Any] = {}
            try:
                parsed = parse_action_block(response)
                raw_params = dict(parsed.get("params") or {})
                thought = str(raw_params.get("thought", "")).strip()
                action_obj = validate_planner_action(parsed)
            except ProtocolError as exc:
                return self._fallback_action(exc.code, response, raw_action=parsed, error=exc.detail)
            except Exception as exc:
                return self._fallback_action("invalid-action", response, raw_action=parsed, error=str(exc))

            if isinstance(action_obj, RepairAction) and self._state.issue:
                action_obj = action_obj.copy(update={"issue": dict(self._state.issue)})

            meta = {
                "used_fallback": False,
                "raw_action": parsed or {},
            }
            assistant_msg = response or text_protocol.format_action_block(
                str(parsed.get("name") if parsed else "noop"),
                raw_params if parsed else {},
            )
            thought = str(raw_params.get("thought", "")).strip()
            return thought, action_obj, assistant_msg, meta

        def _fallback_action(
            self,
            reason: str,
            response: str,
            raw_action: Any | None = None,
            *,
            error: str | None = None,
        ) -> tuple[str, ActionUnion, str, Dict[str, Any]]:
            """在模型输出无法解析时的 fallback 行为。

            - 若启用规则 fallback（use_rule_fallback=True），则调用规则 agent 给出一个保底动作；
            - 若未启用规则 fallback，则直接抛出异常，让上层 driver 决定是否重试生成。
            """

            # 1) 未启用规则 fallback：直接把错误抛给上层
            # If rule fallback is disabled, do not hard-fail the whole eval.
            # Instead return a safe NoopAction and make the error visible in logs.
            if not self.use_rule_fallback or self._rule_agent is None:
                # Do not crash the whole eval just because the model produced a
                # slightly malformed block. Return a safe no-op action and emit
                # a visible debug log so you can spot prompt / parser issues.
                if os.environ.get("DEBUG") or os.environ.get("EBUG"):
                    print(
                        f"[gp-agent {self._trace_id()}] rule fallback disabled; returning noop: "
                        f"reason={reason!r} error={error!r} "
                        f"response_prefix={response[:200]!r}",
                        file=sys.stderr,
                    )
                action_obj = NoopAction(reason=f"fallback:{reason}")
                assistant_msg = text_protocol.format_action_block(
                    "noop",
                    {"reason": reason, "error": str(error or "")},
                )
                meta = {
                    "used_fallback": True,
                    "fallback": "noop",
                    "reason": reason,
                    "error": error,
                    "raw_action": raw_action,
                    "model_response": response,
                }
                return "", action_obj, assistant_msg, meta

            # 2) 启用了规则 fallback：沿用原来的“保底动作”逻辑
            observation = self._last_env_observation or {}
            fallback = self._rule_agent.step(observation)
            action = fallback.get("action_obj") or SubmitAction()
            thought = fallback.get("plan", fallback.get("prompt", ""))
            payload = action_to_payload(action)
            params = {"thought": thought, **{k: v for k, v in payload.items() if k != "type"}}
            params[FALLBACK_REASON_KEY] = reason
            if error:
                params["error"] = error
            assistant_msg = text_protocol.format_action_block(payload.get("type", "noop"), params)
            meta = {
                "used_fallback": True,
                FALLBACK_REASON_KEY: reason,
                "raw_action": raw_action,
                "model_response": response,
            }
            if error:
                meta["error"] = error
            return thought, action, assistant_msg, meta

        # ------------------------------------------------------------------
        # Patch helpers
        # ------------------------------------------------------------------
        def _update_state(self, observation: Dict[str, Any]) -> None:
            """保存最近一次环境信息，供补丁生成器使用。"""

            issue = observation.get("issue") or {}
            if issue and not self._state.issue:
                self._state.issue = issue
            info = observation.get("last_info") or {}
            kind = info.get("kind")
            if kind == "explore" and info.get("op") == "expand":
                self._state.last_candidates = info.get("candidates", [])
                self._state.phase = "memory"
            elif kind == "memory":
                self._state.last_memory = info
                self._state.phase = "read"
            elif kind == "explore" and info.get("op") == "read":
                self._state.last_snippets = info.get("snippets", [])
                self._state.phase = "plan"
            elif kind == "repair":
                self._state.last_repair = info
                if info.get("applied"):
                    self._state.phase = "submit"
                else:
                    self._state.phase = "expand"

__all__ = ["GraphPlannerRLLMAgent"]
