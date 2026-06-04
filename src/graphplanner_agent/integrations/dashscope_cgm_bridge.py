from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
import urllib.request

import uvicorn
from fastapi import FastAPI, HTTPException

from graphplanner_agent.infra.http import urlopen_no_proxy_for_localhost


LOGGER = logging.getLogger("graphplanner_agent.dashscope_cgm_bridge")
DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_MODEL = "qwen3-235b-a22b-thinking-2507"
RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}


@dataclass(slots=True)
class BridgeConfig:
    endpoint: str
    api_key: str
    model: str
    temperature: float = 0.0
    max_output_tokens: int = 1536
    timeout: int = 450
    enable_thinking: bool = True


def _issue_section(issue: dict[str, Any]) -> str:
    title = str(issue.get("title") or "").strip()
    body = str(issue.get("body") or "").strip()
    parts = []
    if title:
        parts.append(title)
    if body:
        parts.append(body)
    return "\n\n".join(parts).strip()


def _target_section(plan: dict[str, Any]) -> str:
    targets = plan.get("targets")
    if not isinstance(targets, list) or not targets:
        return ""
    lines: list[str] = []
    for target in targets[:8]:
        if not isinstance(target, dict):
            continue
        path = str(target.get("path") or "").strip()
        start = target.get("start")
        end = target.get("end")
        node_id = str(target.get("id") or "").strip()
        if not path:
            continue
        lines.append(f"- {path}:{start}-{end} ({node_id})")
    return "\n".join(lines)


def _snippet_section(snippets: Any, *, limit: int = 12) -> str:
    if not isinstance(snippets, list) or not snippets:
        return ""
    blocks: list[str] = []
    for snippet in snippets[:limit]:
        if not isinstance(snippet, dict):
            continue
        path = str(snippet.get("path") or "").strip()
        start = snippet.get("start")
        end = snippet.get("end")
        numbered = str(snippet.get("numbered_text") or "").strip()
        if not path or not numbered:
            continue
        role = "Target" if snippet.get("is_target") else "Context"
        blocks.append(f"[{role} code: {path}:{start}-{end}]\n{numbered}")
    return "\n\n".join(blocks)


def _graph_section(graph: dict[str, Any], *, max_nodes: int = 12, max_edges: int = 24) -> str:
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    lines: list[str] = []
    if isinstance(nodes, list):
        lines.append("Nodes:")
        for node in nodes[:max_nodes]:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or "").strip()
            kind = str(node.get("kind") or node.get("type") or "").strip()
            name = str(node.get("name") or "").strip()
            path = str(node.get("path") or "").strip()
            start = node.get("start_line")
            end = node.get("end_line")
            lines.append(f"- {node_id} | {kind} | {name} | {path}:{start}-{end}")
    if isinstance(edges, list) and edges:
        lines.append("\nEdges:")
        for edge in edges[:max_edges]:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source") or "").strip()
            target = str(edge.get("target") or "").strip()
            edge_type = str(edge.get("type") or edge.get("edgeType") or "").strip()
            if source and target:
                lines.append(f"- {source} --{edge_type or 'rel'}--> {target}")
    return "\n".join(lines).strip()


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def _message_text(message: dict[str, Any], keys: tuple[str, ...]) -> str:
    parts: list[str] = []
    for key in keys:
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n\n".join(parts)


def _parse_streaming_chat_message(resp, *, deadline: float | None = None) -> dict[str, Any]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    role = "assistant"
    for raw_line in _iter_stream_lines(resp, deadline=deadline, label="CGM"):
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
        value = delta.get("content")
        if isinstance(value, str):
            content_parts.append(value)
        for key in ("reasoning_content", "reasoning", "thinking"):
            value = delta.get(key)
            if isinstance(value, str):
                reasoning_parts.append(value)
    message: dict[str, Any] = {"role": role, "content": "".join(content_parts)}
    reasoning = "".join(reasoning_parts).strip()
    if reasoning:
        message["reasoning_content"] = reasoning
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


def _repair_prompt(payload: dict[str, Any]) -> str:
    issue = _issue_section(payload.get("issue") if isinstance(payload.get("issue"), dict) else {})
    targets = _target_section(payload.get("plan") if isinstance(payload.get("plan"), dict) else {})
    snippets = _snippet_section(payload.get("snippets"))
    graph_summary = _graph_section(payload.get("graph") if isinstance(payload.get("graph"), dict) else {})
    plan_text = str(payload.get("plan_text") or "").strip()
    repo = str(payload.get("repo") or payload.get("issue", {}).get("repo") or "").strip()
    parts = [
        "You are fixing a real repository bug.",
        "Return exactly one valid JSON object and nothing else.",
        "Do not return unified diff, Markdown fences, explanations, tests, shell commands, new files, deletes, or renames.",
        'The JSON schema is {"summary": string, "edits": [{"path": string, "start": integer, "end": integer, "new_text": string}]}.',
        "The new_text value must be a valid JSON string containing replacement source only; use standard JSON newline escapes and do not double-escape them as literal \\\\n text.",
        "Use source snippets as evidence. Edit only necessary implementation files; start/end must refer to complete original source lines from the numbered snippets.",
        "Context snippets explain surrounding mechanisms and are not a request to edit those files.",
        "If an edit range includes an if/elif/else/for/while/try/except/with header, new_text must include the complete header and complete block.",
        "Keep the patch minimal and only modify implementation files that are necessary.",
        "Preserve existing public API behavior and user-facing message style unless the issue explicitly asks for new wording.",
        "When fixing message formatting, prefer a minimal local change to the existing template over inventing an unrelated message.",
        "Treat Planner Guidance Notes as advisory. Issue text and visible code evidence are authoritative when they conflict.",
        "The issue description and code evidence are authoritative. Planner guidance is advisory only.",
    ]
    if repo:
        parts.append(f"Repository: {repo}")
    if issue:
        parts.append(f"[Issue]\n{issue}")
    if snippets:
        parts.append(f"[Relevant Code]\n{snippets}")
    if graph_summary:
        parts.append(f"[Graph Summary]\n{graph_summary}")
    if targets:
        parts.append(f"[Planner Guidance]\n{targets}")
    if plan_text:
        parts.append(f"[Planner Guidance Notes]\n{plan_text}")
    return "\n\n".join(parts).strip()


def _review_prompt(payload: dict[str, Any]) -> str:
    issue = _issue_section(payload.get("issue") if isinstance(payload.get("issue"), dict) else {})
    targets = _target_section(payload.get("plan") if isinstance(payload.get("plan"), dict) else {})
    snippets = _snippet_section(payload.get("snippets"))
    graph_summary = _graph_section(payload.get("graph") if isinstance(payload.get("graph"), dict) else {})
    plan_text = str(payload.get("plan_text") or "").strip()
    prior = str(payload.get("prior_feedback") or "").strip()
    request = payload.get("review_request") if isinstance(payload.get("review_request"), dict) else {}
    previous_review = request.get("previous_review")
    previous_review_text = json.dumps(previous_review, ensure_ascii=False, indent=2, sort_keys=True) if isinstance(previous_review, dict) else ""
    review_focus = str(request.get("planner_review_focus") or "").strip()
    repo = str(payload.get("repo") or payload.get("issue", {}).get("repo") or "").strip()
    parts = [
        "You are reviewing a proposed repair intent for a real repository bug.",
        "Do not write a patch. Do not output diff text. Do not invent tests.",
        "Benchmark test source is unavailable; never request it.",
        "Do not suggest reading test files, test functions, hidden assertions, or expected values from tests.",
        "Use only issue text, runtime output summaries, implementation snippets, graph facts, and prior patch feedback.",
        "If the missing evidence would be hidden test source, translate it into implementation evidence or runtime-output evidence to collect.",
        "Return exactly one valid JSON object and nothing else.",
        (
            'The JSON schema is {"verdict": "ready|needs_more_evidence|change_target|avoid_patch", '
            '"confidence": number, "mechanism_assessment": string, "target_assessment": string, '
            '"evidence_gaps": [string], "suggested_next_action": string, "adoption_advice": string}.'
        ),
        "Judge whether the planner intent is supported by the issue text and visible source snippets.",
        "Do not output contract-style fields such as fix_contract or acceptance_criteria.",
        "The review is advice, not a binding repair contract. adoption_advice should say whether to adopt, revise, or reject the planner intent.",
        "Only name a method, attribute, helper, or keyword as an evidence gap if it appears in the supplied issue/runtime/code. Otherwise label it as a hypothesis in adoption_advice, not required evidence.",
        "For a ready verdict, evidence_gaps should be empty and suggested_next_action should not ask to read/search/inspect more implementation evidence.",
        "If implementation/API/signature evidence is still missing, or if your next action would be to inspect more code, use needs_more_evidence rather than ready.",
        "If the plan proposes an API/attribute/keyword not visible in the snippets, mention that uncertainty in adoption_advice.",
        "If a previous patch failed, explain what that failure says about the intent rather than repeating the patch.",
        "If a previous review is provided with planner_review_focus, resolve that disagreement and produce revised/deeper adoption_advice.",
        "evidence_gaps and suggested_next_action should name implementation code or runtime-output facts, never benchmark tests.",
        "The issue description and code evidence are authoritative. Planner guidance is advisory only.",
    ]
    if repo:
        parts.append(f"Repository: {repo}")
    if issue:
        parts.append(f"[Issue]\n{issue}")
    if snippets:
        parts.append(f"[Relevant Code]\n{snippets}")
    if graph_summary:
        parts.append(f"[Graph Summary]\n{graph_summary}")
    if targets:
        parts.append(f"[Planner Target Hypothesis]\n{targets}")
    if plan_text:
        parts.append(f"[Planner Intent To Review]\n{plan_text}")
    if prior:
        parts.append(f"[Prior Repair Feedback]\n{prior}")
    if previous_review_text:
        parts.append(f"[Previous CGM Review]\n{previous_review_text}")
    if review_focus:
        parts.append(f"[Planner Review Focus]\n{review_focus}")
    return "\n\n".join(parts).strip()

def _strip_json_wrapping(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
    return stripped


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = _strip_json_wrapping(text)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(stripped[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("CGM output JSON must be an object")
    return data


def _patch_from_json_output(text: str) -> dict[str, Any]:
    data = _extract_json_object(text)
    patch = data.get("patch") if isinstance(data.get("patch"), dict) else data
    if not isinstance(patch, dict):
        raise ValueError("CGM output must contain a patch object or top-level edits")
    edits = patch.get("edits")
    if not isinstance(edits, list) or not edits:
        raise ValueError("CGM output JSON patch must contain a non-empty edits list")
    normalized_edits: list[dict[str, Any]] = []
    for edit in edits:
        if not isinstance(edit, dict):
            raise ValueError("CGM edit entries must be objects")
        path = str(edit.get("path") or "").strip()
        new_text = edit.get("new_text")
        if not path:
            raise ValueError("CGM edit is missing path")
        if not isinstance(new_text, str) or not new_text.strip():
            raise ValueError(f"CGM edit for {path} is missing non-empty new_text")
        normalized_edits.append(
            {
                "path": path,
                "start": int(edit["start"]),
                "end": int(edit["end"]),
                "new_text": new_text,
            }
        )
    return {
        "summary": str(patch.get("summary") or data.get("summary") or "dashscope-cgm-json"),
        "edits": normalized_edits,
    }


def _review_from_json_output(text: str) -> dict[str, Any]:
    data = _extract_json_object(text)
    verdict = str(data.get("verdict") or "").strip()
    if verdict not in {"ready", "needs_more_evidence", "change_target", "avoid_patch"}:
        verdict = "needs_more_evidence"
    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    def strings(key: str) -> list[str]:
        value = data.get(key)
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    return {
        "verdict": verdict,
        "confidence": confidence,
        "mechanism_assessment": str(data.get("mechanism_assessment") or data.get("issue_mechanism") or "").strip(),
        "target_assessment": str(data.get("target_assessment") or "").strip(),
        "evidence_gaps": strings("evidence_gaps") or strings("missing_evidence"),
        "suggested_next_action": str(data.get("suggested_next_action") or "").strip(),
        "adoption_advice": str(data.get("adoption_advice") or data.get("feedback_for_planner") or "").strip(),
    }


class DashScopeCgmBridge:
    def __init__(self, config: BridgeConfig):
        self.config = config

    def _call_model(self, prompt: str) -> dict[str, str]:
        body = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "enable_thinking": self.config.enable_thinking,
        }
        if self.config.enable_thinking:
            body["stream"] = True
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        req = urllib.request.Request(
            self.config.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urlopen_no_proxy_for_localhost(req, timeout=self.config.timeout) as resp:
                    if body.get("stream"):
                        deadline = time.monotonic() + max(1, int(self.config.timeout))
                        message = _parse_streaming_chat_message(resp, deadline=deadline)
                    else:
                        data = json.loads(resp.read().decode("utf-8"))
                        message = data["choices"][0]["message"]
                reasoning = _message_text(message, ("reasoning_content", "reasoning", "thinking"))
                return {
                    "content": str(message.get("content") or ""),
                    "reasoning_content": reasoning,
                }
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                if exc.code not in RETRYABLE_HTTP_STATUS or attempt == 2:
                    raise RuntimeError(f"dashscope HTTP {exc.code}: {detail[:1200]}") from exc
                last_error = RuntimeError(f"dashscope HTTP {exc.code}: {detail[:1200]}")
            except URLError as exc:
                if attempt == 2:
                    raise RuntimeError(f"dashscope request failed: {exc}") from exc
                last_error = exc
            except TimeoutError as exc:
                raise RuntimeError(f"dashscope request timed out: {exc}") from exc
            LOGGER.warning("retrying DashScope CGM request after transient error: %s", last_error)
            time.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"dashscope request failed: {last_error}")

    def generate_patch(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = _repair_prompt(payload)
        model_output = self._call_model(prompt)
        raw_text = model_output["content"]
        reasoning = model_output.get("reasoning_content") or ""
        patch = _patch_from_json_output(raw_text)
        return {
            "summary": "dashscope-cgm-bridge",
            "patch": patch,
            "raw_preview": raw_text[:800],
            "reasoning_content": reasoning,
            "reasoning_preview": reasoning[:1200],
            "reasoning_chars": len(reasoning),
            "thinking_enabled": self.config.enable_thinking,
            "model": self.config.model,
            "output_format": "json_patch",
        }

    def review_intent(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = _review_prompt(payload)
        model_output = self._call_model(prompt)
        raw_text = model_output["content"]
        reasoning = model_output.get("reasoning_content") or ""
        review = _review_from_json_output(raw_text)
        return {
            "summary": "dashscope-cgm-bridge-review",
            "review": review,
            "raw_preview": raw_text[:800],
            "reasoning_content": reasoning,
            "reasoning_preview": reasoning[:1200],
            "reasoning_chars": len(reasoning),
            "thinking_enabled": self.config.enable_thinking,
            "model": self.config.model,
            "output_format": "intent_review_json",
        }


def create_app(bridge: DashScopeCgmBridge, *, route: str = "/generate") -> FastAPI:
    app = FastAPI()

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": "dashscope",
            "model": bridge.config.model,
            "endpoint": bridge.config.endpoint,
            "thinking_enabled": bridge.config.enable_thinking,
        }

    @app.post(route)
    def generate(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            if str(payload.get("mode") or "").strip() == "intent_review":
                return bridge.review_intent(payload)
            return bridge.generate_patch(payload)
        except Exception as exc:
            LOGGER.exception("dashscope cgm bridge failure")
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Expose a DashScope-backed CGM /generate bridge.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30002)
    parser.add_argument("--route", default="/generate")
    parser.add_argument("--endpoint", default=os.getenv("DASHSCOPE_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--model", default=os.getenv("CGM_DASHSCOPE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.getenv("CGM_DASHSCOPE_API_KEY") or os.getenv("PLANNER_API_KEY"))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("CGM_DASHSCOPE_TEMPERATURE", "0")))
    parser.add_argument("--max-output-tokens", type=int, default=int(os.getenv("CGM_DASHSCOPE_MAX_TOKENS", "1536")))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("CGM_DASHSCOPE_TIMEOUT", "450")))
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=_parse_bool(os.getenv("CGM_DASHSCOPE_ENABLE_THINKING"), default=True),
    )
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("CGM_DASHSCOPE_API_KEY or PLANNER_API_KEY is required")

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    config = BridgeConfig(
        endpoint=str(args.endpoint),
        api_key=str(args.api_key),
        model=str(args.model),
        temperature=float(args.temperature),
        max_output_tokens=int(args.max_output_tokens),
        timeout=int(args.timeout),
        enable_thinking=bool(args.enable_thinking),
    )
    app = create_app(DashScopeCgmBridge(config), route=str(args.route))
    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level=str(args.log_level).lower())


if __name__ == "__main__":
    main()
