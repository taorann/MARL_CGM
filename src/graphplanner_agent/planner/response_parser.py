from __future__ import annotations

from dataclasses import dataclass
import json
import re

from .protocol import PlannerAction


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


@dataclass(slots=True)
class ParsedPlannerResponse:
    action: PlannerAction
    visible_thinking: str | None
    formal_text: str


def _extract_json(text: str) -> str:
    fenced = FENCED_JSON_RE.search(text)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    raise ValueError("planner response did not contain a JSON action")


def parse_planner_response(text: str) -> ParsedPlannerResponse:
    think_match = THINK_RE.search(text)
    visible = think_match.group(1).strip() if think_match else None
    formal = THINK_RE.sub("", text).strip()
    payload = json.loads(_extract_json(formal))
    return ParsedPlannerResponse(
        action=PlannerAction.from_obj(payload),
        visible_thinking=visible,
        formal_text=formal,
    )


def parse_planner_message(message: dict) -> ParsedPlannerResponse:
    reasoning_parts: list[str] = []
    for key in ("reasoning_content", "reasoning", "thinking"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            reasoning_parts.append(value.strip())
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "\n".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
    think_match = THINK_RE.search(str(content))
    if think_match:
        reasoning_parts.append(think_match.group(1).strip())

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        call = tool_calls[0]
        function = call.get("function", {}) if isinstance(call, dict) else {}
        name = function.get("name")
        arguments = function.get("arguments") or "{}"
        if isinstance(arguments, str):
            params = json.loads(arguments or "{}")
        elif isinstance(arguments, dict):
            params = arguments
        else:
            raise ValueError("tool call arguments must be JSON string or object")
        action = PlannerAction(tool=str(name), params=dict(params))
        action.validate()
        formal = json.dumps({"tool": action.tool, "params": action.params}, sort_keys=True)
        return ParsedPlannerResponse(
            action=action,
            visible_thinking="\n\n".join(reasoning_parts) or None,
            formal_text=formal,
        )

    parsed = parse_planner_response(str(content))
    visible = "\n\n".join([part for part in reasoning_parts + ([parsed.visible_thinking] if parsed.visible_thinking else []) if part]) or None
    return ParsedPlannerResponse(action=parsed.action, visible_thinking=visible, formal_text=parsed.formal_text)
