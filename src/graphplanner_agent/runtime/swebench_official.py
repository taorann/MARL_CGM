from __future__ import annotations

import base64
import re
import tempfile
from pathlib import Path
from typing import Any

from graphplanner_agent.datasets import TaskSpec
from graphplanner_agent.runtime.sandbox_base import CommandResult, TestResult


START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"


def swebench_spec(task: TaskSpec) -> dict[str, Any] | None:
    spec = task.sandbox.get("swebench_spec") if isinstance(task.sandbox, dict) else None
    if not isinstance(spec, dict):
        spec = task.metadata.get("swebench_spec") if isinstance(task.metadata, dict) else None
    return spec if isinstance(spec, dict) else None


def official_eval_script_lines(task: TaskSpec) -> list[str]:
    spec = swebench_spec(task)
    if not spec:
        return []
    scripts = spec.get("eval_script_list")
    if not isinstance(scripts, list):
        return []
    return [str(line).rstrip() for line in scripts if str(line).strip()]


def official_eval_script(task: TaskSpec) -> str | None:
    lines = official_eval_script_lines(task)
    if not lines:
        return None
    return "\n".join(["#!/bin/bash", "set -uxo pipefail", *lines]) + "\n"


def official_eval_command(task: TaskSpec, script_path: str = "/tmp/gp_swebench_eval.sh") -> str | None:
    script = official_eval_script(task)
    if script is None:
        return None
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    return (
        "python - <<'PY'\n"
        "import base64, pathlib, subprocess, sys\n"
        f"path = pathlib.Path({script_path!r})\n"
        f"path.write_bytes(base64.b64decode({encoded!r}))\n"
        "path.chmod(0o755)\n"
        "sys.exit(subprocess.call(['/bin/bash', str(path)]))\n"
        "PY"
    )


def result_from_official_run(task: TaskSpec, command: str, result: CommandResult) -> TestResult:
    report, parser_error = parse_official_report(task, result.stdout + "\n" + result.stderr)
    if result.timed_out:
        status = "timeout"
    elif report and report.get("resolved") is True:
        status = "passed"
    elif report:
        status = "failed"
    else:
        status = "failed" if result.returncode else "passed"
    tests_status = report.get("tests_status", {}) if isinstance(report, dict) else {}
    resolved = bool(report.get("resolved")) if isinstance(report, dict) and "resolved" in report else None
    return TestResult(status, command, result.stdout, result.stderr, result.returncode, tests_status, resolved, parser_error)


def parse_official_report(task: TaskSpec, output: str) -> tuple[dict[str, Any] | None, str | None]:
    spec = swebench_spec(task)
    if not spec:
        return None, None
    repo = str(spec.get("repo") or task.metadata.get("repo") or "")
    version = str(spec.get("version") or task.metadata.get("version") or "")
    if repo and not version:
        version = _infer_swebench_version(repo, official_eval_script_lines(task))
    if not repo or not version:
        generic = _generic_pytest_report(task, output)
        if generic is not None:
            return generic, None
        missing = "repo/version" if not repo and not version else ("repo" if not repo else "version")
        return None, f"missing {missing} for SWE-bench log parser"
    try:
        from swebench.harness.constants import KEY_INSTANCE_ID, KEY_PREDICTION
        from swebench.harness.grading import get_eval_report
        from swebench.harness.test_spec.test_spec import TestSpec
    except Exception as exc:
        return None, f"swebench harness import failed: {exc}"

    test_spec = TestSpec(
        instance_id=task.task_id,
        repo=repo,
        version=version,
        repo_script_list=[],
        eval_script_list=official_eval_script_lines(task),
        env_script_list=[],
        arch=str(spec.get("arch") or "x86_64"),
        FAIL_TO_PASS=task.fail_to_pass,
        PASS_TO_PASS=task.pass_to_pass,
        language=str(spec.get("language") or "py"),
        docker_specs=dict(spec.get("docker_specs") or {}),
        namespace=None,
    )
    prediction = {KEY_INSTANCE_ID: task.task_id, KEY_PREDICTION: "__already_applied__"}
    with tempfile.TemporaryDirectory() as raw:
        log_path = Path(raw) / "test_output.txt"
        log_path.write_text(output, encoding="utf-8", errors="replace")
        try:
            full = get_eval_report(test_spec, prediction, str(log_path), include_tests_status=True)
        except Exception as exc:
            generic = _generic_pytest_report(task, output)
            if generic is not None:
                return generic, None
            return None, f"swebench report parse failed: {exc}"
    report = full.get(task.task_id) if isinstance(full, dict) else None
    if isinstance(report, dict) and report.get("patch_successfully_applied"):
        return report, None
    generic = _generic_pytest_report(task, output)
    if generic is not None:
        return generic, None
    return (report if isinstance(report, dict) else None), None


def _infer_swebench_version(repo: str, eval_script_lines: list[str]) -> str:
    try:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    except Exception:
        return ""
    versions = MAP_REPO_VERSION_TO_SPECS.get(repo)
    if not isinstance(versions, dict) or not versions:
        return ""
    script_text = "\n".join(eval_script_lines)
    matches: list[str] = []
    for version, repo_spec in versions.items():
        if not isinstance(repo_spec, dict):
            continue
        test_cmd = repo_spec.get("test_cmd")
        candidates = test_cmd if isinstance(test_cmd, list) else [test_cmd]
        if any(isinstance(cmd, str) and cmd and cmd in script_text for cmd in candidates):
            matches.append(str(version))
    if len(matches) == 1:
        return matches[0]
    if len(versions) == 1:
        return str(next(iter(versions)))
    return ""


PYTEST_STATUS_RE = re.compile(
    r"^(?P<status>PASSED|FAILED|ERROR|SKIPPED|XFAILED|XPASSED|XFAIL|XPASS)\s+(?P<test>\S+::\S+)\s*$"
)


def _generic_pytest_report(task: TaskSpec, output: str) -> dict[str, Any] | None:
    status_map = _generic_pytest_status_map(output)
    if not status_map:
        return None
    fail_success, fail_failure = _classify_selectors(task.fail_to_pass, status_map)
    pass_success, pass_failure = _classify_selectors(task.pass_to_pass, status_map)
    tests_status = {
        "FAIL_TO_PASS": {"success": fail_success, "failure": fail_failure},
        "PASS_TO_PASS": {"success": pass_success, "failure": pass_failure},
        "FAIL_TO_FAIL": {"success": [], "failure": []},
        "PASS_TO_FAIL": {"success": [], "failure": []},
    }
    resolved = bool(fail_success or not task.fail_to_pass) and not fail_failure and not pass_failure
    return {
        "patch_is_None": False,
        "patch_exists": True,
        "patch_successfully_applied": True,
        "resolved": resolved,
        "tests_status": tests_status,
        "parser": "generic_pytest_summary",
    }


def _generic_pytest_status_map(output: str) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for raw in (output or "").splitlines():
        line = raw.strip()
        match = PYTEST_STATUS_RE.match(line)
        if not match:
            continue
        status = match.group("status")
        test = match.group("test")
        if status in {"PASSED", "XFAILED", "XFAIL"}:
            statuses[test] = "PASSED"
        elif status in {"FAILED", "ERROR", "XPASSED", "XPASS"}:
            statuses[test] = "FAILED"
        elif status == "SKIPPED":
            statuses[test] = "SKIPPED"
    return statuses


def _classify_selectors(selectors: list[str], status_map: dict[str, str]) -> tuple[list[str], list[str]]:
    success: list[str] = []
    failure: list[str] = []
    for selector in selectors:
        related = _matching_statuses(selector, status_map)
        if related and all(status == "PASSED" for status in related):
            success.append(selector)
        elif any(status == "FAILED" for status in related):
            failure.append(selector)
        elif selector in status_map and status_map[selector] == "PASSED":
            success.append(selector)
        else:
            # SWE-bench treats missing pass/fail details as silent success when
            # no failure was observed for the selector in the parsed output.
            success.append(selector)
    return success, failure


def _matching_statuses(selector: str, status_map: dict[str, str]) -> list[str]:
    prefix = str(selector or "").strip()
    if not prefix:
        return []
    exact = status_map.get(prefix)
    if exact is not None:
        return [exact]
    bracket_prefix = prefix + "["
    child_prefix = prefix + "::"
    return [
        status
        for test, status in status_map.items()
        if test.startswith(bracket_prefix) or test.startswith(child_prefix)
    ]
