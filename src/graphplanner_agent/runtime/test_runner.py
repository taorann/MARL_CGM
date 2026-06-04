from __future__ import annotations

import ast
import re

from graphplanner_agent.runtime.sandbox_base import TestResult


TRACEBACK_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in ([^\n]+)')
PYTEST_STATUS_LINE_RE = re.compile(r"^(?P<status>FAILED|ERROR|PASSED|SKIPPED)\s+(?P<selector>\S+::\S+)(?:\s+-\s+(?P<detail>.*))?$")
EXCEPTION_LINE_RE = re.compile(r"^\s*E\s+(?P<type>[A-Za-z_][\w.]*?(?:Error|Exception|Failure)):\s*(?P<message>.*)$")
PLAIN_EXCEPTION_LINE_RE = re.compile(r"^(?P<type>[A-Za-z_][\w.]*?(?:Error|Exception|Failure)):\s*(?P<message>.*)$")


def behavior_summary(result: TestResult, limit: int = 1600) -> dict[str, object]:
    text = (result.stderr + "\n" + result.stdout).strip()
    frames = [
        {"path": path, "line": int(line), "function": func.strip()}
        for path, line, func in TRACEBACK_FRAME_RE.findall(text)
        if "/test" not in path and not path.rsplit("/", 1)[-1].startswith("test_")
    ]
    observations = actual_runtime_observations(text)
    excerpt = _actual_only_excerpt(observations, frames[-5:], limit)
    command, command_omitted = _safe_command_summary(result.command)
    return {
        "status": result.status,
        "command": command,
        "command_omitted_for_benchmark_hygiene": command_omitted,
        "returncode": result.returncode,
        "resolved": result.resolved,
        "tests_status": result.tests_status,
        "parser_error": result.parser_error,
        "implementation_frames": frames[-5:],
        "runtime_observations": observations,
        "excerpt": excerpt,
    }


def actual_runtime_observations(output: str, *, limit: int = 8) -> dict[str, object]:
    """Extract behavior facts from runtime output without exposing test-source expectations."""
    failed_selectors: list[str] = []
    exception_types: list[str] = []
    actual_messages: list[str] = []
    actual_assertion_values: list[str] = []
    omitted_expected = False

    for raw in (output or "").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if "INFRA_WRONG_PYTHON_ENV" in stripped:
            _append_unique(exception_types, "WrongPythonEnvironment", limit)
            _append_unique(actual_messages, stripped, limit)
            continue
        status = PYTEST_STATUS_LINE_RE.match(stripped)
        if status:
            if status.group("status") in {"FAILED", "ERROR"}:
                _append_unique(failed_selectors, status.group("selector"), limit)
            detail = status.group("detail") or ""
            exc_type, message, hidden = _safe_exception_detail(detail)
            omitted_expected = omitted_expected or hidden
            if exc_type:
                _append_unique(exception_types, exc_type, limit)
            if message:
                _append_unique(actual_messages, message, limit)
            value, hidden = _actual_value_from_assertion(detail)
            omitted_expected = omitted_expected or hidden
            if value:
                _append_unique(actual_assertion_values, value, limit)
            continue

        exc = EXCEPTION_LINE_RE.match(line)
        if exc:
            exc_type, message, hidden = _safe_exception_detail(f"{exc.group('type')}: {exc.group('message')}")
            omitted_expected = omitted_expected or hidden
            if exc_type:
                _append_unique(exception_types, exc_type, limit)
            if message:
                _append_unique(actual_messages, message, limit)
            value, hidden = _actual_value_from_assertion(exc.group("message") or "")
            omitted_expected = omitted_expected or hidden
            if value:
                _append_unique(actual_assertion_values, value, limit)
            continue

        plain_exc = PLAIN_EXCEPTION_LINE_RE.match(stripped)
        if plain_exc:
            exc_type, message, hidden = _safe_exception_detail(
                f"{plain_exc.group('type')}: {plain_exc.group('message')}"
            )
            omitted_expected = omitted_expected or hidden
            if exc_type:
                _append_unique(exception_types, exc_type, limit)
            if message:
                _append_unique(actual_messages, message, limit)

    return {
        "policy": "actual runtime output only; benchmark test source and hidden expected values are omitted",
        "failed_selectors": failed_selectors,
        "exception_types": exception_types,
        "actual_messages": actual_messages,
        "actual_assertion_values": actual_assertion_values,
        "omitted_hidden_expected_values": omitted_expected,
    }


def _safe_exception_detail(detail: str) -> tuple[str | None, str | None, bool]:
    text = (detail or "").strip()
    if not text:
        return None, None, False
    match = re.match(r"(?P<type>[A-Za-z_][\w.]*?(?:Error|Exception|Failure)):\s*(?P<message>.*)", text)
    if not match:
        return None, None, False
    exc_type = match.group("type")
    message = match.group("message").strip()
    value, hidden = _actual_value_from_assertion(message)
    if value:
        return exc_type, f"assertion comparison failed; actual={value}", True
    return exc_type, _truncate(message, 600) if message else None, hidden


def _actual_value_from_assertion(text: str) -> tuple[str | None, bool]:
    marker = "assert "
    if marker not in text or " == " not in text:
        return None, False
    tail = text.split(marker, 1)[1]
    left = tail.split(" == ", 1)[0].strip()
    if not left:
        return None, True
    try:
        value = ast.literal_eval(left)
    except Exception:
        if not _looks_like_literal(left):
            return None, True
        value = left
    return _truncate(repr(value), 700), True


def _looks_like_literal(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith(("'", '"', "[", "(", "{")) or stripped in {"None", "True", "False"} or bool(re.match(r"^-?\d", stripped))


def _actual_only_excerpt(observations: dict[str, object], frames: list[dict[str, object]], limit: int) -> str:
    lines = ["Actual runtime failure summary (benchmark expected values omitted):"]
    selectors = observations.get("failed_selectors")
    if selectors:
        lines.append("failed_selectors: " + ", ".join(str(item) for item in selectors))
    exception_types = observations.get("exception_types")
    if exception_types:
        lines.append("exception_types: " + ", ".join(str(item) for item in exception_types))
    messages = observations.get("actual_messages")
    if messages:
        lines.append("actual_messages:")
        lines.extend(f"- {item}" for item in messages)
    values = observations.get("actual_assertion_values")
    if values:
        lines.append("actual_assertion_values:")
        lines.extend(f"- {item}" for item in values)
    if frames:
        lines.append("implementation_frames:")
        lines.extend(f"- {frame['path']}:{frame['line']} in {frame['function']}" for frame in frames)
    if observations.get("omitted_hidden_expected_values"):
        lines.append("note: pytest equality expected/wanted side was omitted to avoid benchmark leakage")
    return _truncate("\n".join(lines), limit)


def _append_unique(items: list[str], value: str | None, limit: int) -> None:
    if not value:
        return
    value = _truncate(str(value).strip(), 700)
    if value and value not in items and len(items) < limit:
        items.append(value)


def _safe_command_summary(command: str) -> tuple[str, bool]:
    text = command or ""
    if any(marker in text for marker in ["gp_swebench_eval.sh", "PYTESTPATCH", "base64.b64decode"]):
        return "<official SWE-bench eval command omitted; contains benchmark harness setup/test patch>", True
    return text, False


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"...<truncated {len(text) - limit} chars>"
