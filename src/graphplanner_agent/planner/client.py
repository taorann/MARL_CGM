from __future__ import annotations

import json
import time
from urllib.error import HTTPError, URLError
import urllib.request

from graphplanner_agent.config import AgentConfig
from graphplanner_agent.infra.http import urlopen_no_proxy_for_localhost

RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}


class OpenAIPlannerClient:
    def __init__(self, config: AgentConfig):
        if not config.planner_endpoint:
            raise ValueError("planner_endpoint is required for OpenAIPlannerClient")
        self.config = config

    def complete_message(self, messages: list[dict[str, str]], tools: list[dict] | None = None, tool_choice: str | dict | None = None) -> dict:
        body = {
            "model": self.config.planner_model,
            "messages": messages,
            "temperature": self.config.planner_temperature,
        }
        if self.config.planner_enable_thinking is not None:
            body["enable_thinking"] = self.config.planner_enable_thinking
            if self.config.planner_enable_thinking:
                body["stream"] = True
        if tools is not None:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
        headers = {"Content-Type": "application/json"}
        if self.config.planner_api_key:
            headers["Authorization"] = f"Bearer {self.config.planner_api_key}"
        req = urllib.request.Request(
            self.config.planner_endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urlopen_no_proxy_for_localhost(req, timeout=self.config.planner_request_timeout) as resp:
                    if body.get("stream"):
                        deadline = time.monotonic() + max(1, int(self.config.planner_request_timeout))
                        return _parse_streaming_chat_message(resp, deadline=deadline)
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]
            except HTTPError as exc:
                response_body = exc.read().decode("utf-8", "replace")
                if exc.code not in RETRYABLE_HTTP_STATUS or attempt == 2:
                    raise RuntimeError(f"planner HTTP {exc.code}: {response_body[:1000]}") from exc
                last_error = RuntimeError(f"planner HTTP {exc.code}: {response_body[:1000]}")
            except URLError as exc:
                if attempt == 2:
                    raise RuntimeError(f"planner request failed: {exc}") from exc
                last_error = exc
            except TimeoutError as exc:
                raise RuntimeError(f"planner request timed out: {exc}") from exc
            time.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"planner request failed: {last_error}")

    def complete(self, messages: list[dict[str, str]]) -> str:
        message = self.complete_message(messages)
        return message.get("content") or ""


class StaticPlannerClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not self.responses:
            return '{"tool":"memory_commit_note","params":{"note":"static planner exhausted scripted responses"}}'
        return self.responses.pop(0)


def make_planner_client(config: AgentConfig, scripted_responses: list[str] | None = None):
    if scripted_responses is not None:
        return StaticPlannerClient(scripted_responses)
    return OpenAIPlannerClient(config)


def _parse_streaming_chat_message(resp, *, deadline: float | None = None) -> dict:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_call_parts: dict[int, dict] = {}
    role = "assistant"
    for raw_line in _iter_stream_lines(resp, deadline=deadline, label="planner"):
        line = raw_line.decode("utf-8", "replace").strip()
        if not line or not line.startswith("data:"):
            continue
        data_text = line[5:].strip()
        if data_text == "[DONE]":
            break
        try:
            event = json.loads(data_text)
        except json.JSONDecodeError:
            continue
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            continue
        if isinstance(delta.get("role"), str):
            role = delta["role"]
        for key in ("reasoning_content", "reasoning", "thinking"):
            value = delta.get(key)
            if isinstance(value, str):
                reasoning_parts.append(value)
        value = delta.get("content")
        if isinstance(value, str):
            content_parts.append(value)
        calls = delta.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, dict):
                    continue
                index = int(call.get("index", len(tool_call_parts)))
                current = tool_call_parts.setdefault(
                    index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if isinstance(call.get("id"), str):
                    current["id"] += call["id"]
                if isinstance(call.get("type"), str):
                    current["type"] = call["type"]
                function = call.get("function")
                if isinstance(function, dict):
                    cur_func = current.setdefault("function", {"name": "", "arguments": ""})
                    if isinstance(function.get("name"), str):
                        cur_func["name"] += function["name"]
                    if isinstance(function.get("arguments"), str):
                        cur_func["arguments"] += function["arguments"]
    message = {
        "role": role,
        "content": "".join(content_parts),
    }
    reasoning = "".join(reasoning_parts).strip()
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_call_parts:
        message["tool_calls"] = [tool_call_parts[index] for index in sorted(tool_call_parts)]
    return message


def _iter_stream_lines(resp, *, deadline: float | None, label: str):
    reader = getattr(resp, "read", None)
    if not callable(reader):
        for raw_line in resp:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"{label} streaming response exceeded wall-clock timeout")
            yield raw_line
        return
    buffer = bytearray()
    while True:
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(f"{label} streaming response exceeded wall-clock timeout")
        chunk = reader(1)
        if not chunk:
            if buffer:
                yield bytes(buffer)
            break
        buffer.extend(chunk)
        if chunk == b"\n":
            yield bytes(buffer)
            buffer.clear()
