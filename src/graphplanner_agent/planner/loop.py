from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from graphplanner_agent.config import AgentConfig
from graphplanner_agent.env.stepper import CodeRepairEnv
from graphplanner_agent.planner.prompt import build_messages
from graphplanner_agent.planner.response_parser import parse_planner_message, parse_planner_response
from graphplanner_agent.telemetry.console import info, summarize_action_result, summarize_action_status
from graphplanner_agent.telemetry.events import TraceWriter


@dataclass(slots=True)
class PlannerLoopResult:
    status: str
    steps: int
    reason: str | None = None


class PlannerLoop:
    def __init__(
        self,
        env: CodeRepairEnv,
        planner_client,
        config: AgentConfig,
        trace: TraceWriter | None = None,
        console: bool = False,
        on_step: Callable[[dict[str, object]], None] | None = None,
    ):
        self.env = env
        self.planner_client = planner_client
        self.config = config
        self.trace = trace
        self.console = console
        self.on_step = on_step

    def run(self) -> PlannerLoopResult:
        started = time.monotonic()
        status = "not_pass"
        for step in range(1, self.config.max_steps + 1):
            observation = self.env.observe()
            messages = build_messages(observation, tool_calling=self.config.planner_tool_calling)
            parsed = None
            last_error: Exception | None = None
            for policy_attempt in range(3):
                parsed, last_error = self._parse_action(messages, step)
                if parsed is None:
                    break
                disabled_reason = self.env.action_disabled_reason(parsed.action.tool)
                if not disabled_reason:
                    break
                diagnostic = {
                    "step": step,
                    "attempt": policy_attempt + 1,
                    "error": "repair action is unavailable in the current environment state",
                    "reason": disabled_reason,
                    "action": {"tool": parsed.action.tool, "params": parsed.action.params},
                }
                self.env.record_planner_diagnostic(diagnostic)
                if self.trace:
                    self.trace.event("planner_policy_rejected_action", diagnostic)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The {parsed.action.tool} tool is unavailable right now: "
                            f"{disabled_reason}. Call exactly one currently available tool. "
                            "If search candidates already exist, read one candidate by node id."
                        ),
                    }
                )
            if parsed is None:
                self.env.done = True
                self.env.status = "bug"
                return PlannerLoopResult(status="bug", steps=step, reason=f"planner output parse failed: {last_error}")
            step_started = time.monotonic()
            result = self.env.step(parsed.action)
            step_elapsed = time.monotonic() - step_started
            if self.console:
                action_status = summarize_action_status(result)
                summary = summarize_action_result(result)
                suffix = f" | {summary}" if summary else ""
                info(f"[step {step:02d}] {parsed.action.tool} {action_status} {step_elapsed:.1f}s{suffix}")
            else:
                action_status = summarize_action_status(result)
                summary = summarize_action_result(result)
            if self.on_step:
                self.on_step(
                    {
                        "step": step,
                        "tool": parsed.action.tool,
                        "status": action_status,
                        "elapsed": step_elapsed,
                        "summary": summary,
                    }
                )
            if self.trace:
                self.trace.event(
                    "planner_step",
                    {
                        "step": step,
                        "visible_thinking": parsed.visible_thinking,
                        "action": {"tool": parsed.action.tool, "params": parsed.action.params},
                        "result": result,
                    },
                )
            if self.env.done:
                status = self.env.status
                return PlannerLoopResult(status=status, steps=step, reason="env_done")
        elapsed = time.monotonic() - started
        return PlannerLoopResult(status=status, steps=self.config.max_steps, reason=f"max_steps after {elapsed:.1f}s")

    def _parse_action(self, messages: list[dict[str, str]], step: int):
        parsed = None
        last_error: Exception | None = None
        for attempt in range(self.config.planner_max_parse_retries + 1):
            raw_for_diagnostics = None
            try:
                if self.config.planner_tool_calling and hasattr(self.planner_client, "complete_message"):
                    raw_message = self.planner_client.complete_message(
                        messages,
                        tools=self.env.planner_tool_schemas(),
                        tool_choice="auto",
                    )
                    raw_for_diagnostics = raw_message
                    parsed = parse_planner_message(raw_message)
                else:
                    raw = self.planner_client.complete(messages)
                    raw_for_diagnostics = raw
                    parsed = parse_planner_response(raw)
                return parsed, None
            except Exception as exc:
                last_error = exc
                diagnostic = {
                    "step": step,
                    "attempt": attempt + 1,
                    "error": str(exc),
                    "raw_response": _compact_raw(raw_for_diagnostics),
                }
                self.env.record_planner_diagnostic(diagnostic)
                if self.trace:
                    self.trace.event("planner_malformed_response", diagnostic)
                messages.append({"role": "user", "content": _malformed_retry_message(str(exc), self.config.planner_tool_calling)})
        return None, last_error


def _compact_raw(value, limit: int = 3000):
    if value is None:
        return None
    try:
        import json

        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(value)
    if len(text) <= limit:
        return value
    return text[:limit] + f"...<truncated {len(text) - limit} chars>"


def _malformed_retry_message(error: str, tool_calling: bool) -> str:
    if tool_calling:
        return (
            "Previous response was malformed: "
            f"{error}. Call exactly one provided tool via tool_calls. Do not answer in prose or JSON text."
        )
    return f"Previous response was malformed: {error}. Emit exactly one JSON action."
