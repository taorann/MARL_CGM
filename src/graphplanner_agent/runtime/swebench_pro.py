from __future__ import annotations

import base64
import json
import os
import re
from typing import Any

from graphplanner_agent.datasets import TaskSpec
from graphplanner_agent.runtime.sandbox_base import CommandResult, TestResult


START_PRO_JSON = ">>>>> Start SWE-bench Pro Parsed Output"
END_PRO_JSON = ">>>>> End SWE-bench Pro Parsed Output"


def swebench_pro_spec(task: TaskSpec) -> dict[str, Any] | None:
    for root in (task.sandbox, task.metadata):
        if isinstance(root, dict) and isinstance(root.get("swebench_pro"), dict):
            return dict(root["swebench_pro"])
    return None


def is_swebench_pro_task(task: TaskSpec) -> bool:
    return swebench_pro_spec(task) is not None


def setup_command(task: TaskSpec) -> str | None:
    spec = swebench_pro_spec(task)
    if not spec:
        return None
    before = str(spec.get("before_repo_set_cmd") or "").strip()
    if not before:
        return None
    payload = _b64_json({"before_repo_set_cmd": before})
    return (
        "python - <<'PY'\n"
        "import base64, json, subprocess, sys\n"
        f"payload = json.loads(base64.b64decode({payload!r}).decode('utf-8'))\n"
        "cmd = payload.get('before_repo_set_cmd') or ''\n"
        "proc = subprocess.run(cmd, cwd='/app', shell=True, executable='/bin/bash', text=True)\n"
        "sys.exit(proc.returncode)\n"
        "PY"
    )


def test_command(task: TaskSpec, *, workspace: str = "/workspace/graphplanner_swebench_pro") -> str:
    spec = swebench_pro_spec(task)
    if not spec:
        raise ValueError("task is not a SWE-bench Pro task")
    run_all_for_p2p = _run_all_for_p2p(task, spec)
    payload = _b64_json(
        {
            "workspace": workspace,
            "run_script": str(spec.get("run_script") or ""),
            "parser": str(spec.get("parser") or ""),
            "selected_tests": _as_list(spec.get("selected_test_files_to_run")),
            "pass_to_pass": list(task.pass_to_pass or []),
            "run_all_for_p2p": run_all_for_p2p,
        }
    )
    return (
        "python - <<'PY'\n"
        "import base64, json, pathlib, subprocess, sys\n"
        f"payload = json.loads(base64.b64decode({payload!r}).decode('utf-8'))\n"
        "workspace = pathlib.Path(payload['workspace'])\n"
        "workspace.mkdir(parents=True, exist_ok=True)\n"
        "run_script = workspace / 'run_script.sh'\n"
        "parser = workspace / 'parser.py'\n"
        "run_script.write_text(payload.get('run_script') or '', encoding='utf-8')\n"
        "parser.write_text(payload.get('parser') or '', encoding='utf-8')\n"
        "run_script.chmod(0o755)\n"
        "selected = [str(item) for item in payload.get('selected_tests') or [] if str(item).strip()]\n"
        "pass_to_pass = [str(item) for item in payload.get('pass_to_pass') or [] if str(item).strip()]\n"
        "runs = []\n"
        "combined_tests = []\n"
        "def run_and_parse(label, args):\n"
        "    stdout_path = workspace / (label + '_stdout.txt')\n"
        "    stderr_path = workspace / (label + '_stderr.txt')\n"
        "    output_path = workspace / (label + '_output.json')\n"
        "    with stdout_path.open('w', encoding='utf-8', errors='replace') as out, stderr_path.open('w', encoding='utf-8', errors='replace') as err:\n"
        "        run_proc = subprocess.run(['bash', str(run_script), *args], cwd='/app', text=True, stdout=out, stderr=err)\n"
        "    parser_proc = subprocess.run([sys.executable, str(parser), str(stdout_path), str(stderr_path), str(output_path)], cwd=str(workspace), text=True, capture_output=True)\n"
        "    print(label.upper() + '_RUN_SCRIPT_RETURN_CODE=' + str(run_proc.returncode))\n"
        "    print(label.upper() + '_PARSER_RETURN_CODE=' + str(parser_proc.returncode))\n"
        "    if parser_proc.stdout:\n"
        "        print(parser_proc.stdout[-4000:])\n"
        "    if parser_proc.stderr:\n"
        "        print(parser_proc.stderr[-4000:], file=sys.stderr)\n"
        "    report = {}\n"
        "    if output_path.exists():\n"
        "        try:\n"
        "            report = json.loads(output_path.read_text(encoding='utf-8', errors='replace') or '{}')\n"
        "        except json.JSONDecodeError:\n"
        "            report = {}\n"
        "    tests = report.get('tests') if isinstance(report, dict) else []\n"
        "    if not isinstance(tests, list):\n"
        "        tests = []\n"
        "    combined_tests.extend(tests)\n"
        "    runs.append({'label': label, 'args': args, 'run_script_returncode': run_proc.returncode, 'parser_returncode': parser_proc.returncode, 'test_count': len(tests)})\n"
        "    print('>>>>> Start ' + label + ' Test Output')\n"
        "    print(stdout_path.read_text(encoding='utf-8', errors='replace')[-12000:] if stdout_path.exists() else '')\n"
        "    print('>>>>> End ' + label + ' Test Output')\n"
        "    if stderr_path.exists():\n"
        "        sys.stderr.write(stderr_path.read_text(encoding='utf-8', errors='replace')[-12000:])\n"
        "    return run_proc.returncode, parser_proc.returncode\n"
        "def run_regression_setup():\n"
        "    package_json = pathlib.Path('/app/package.json')\n"
        "    if not package_json.exists():\n"
        "        return\n"
        "    try:\n"
        "        package = json.loads(package_json.read_text(encoding='utf-8'))\n"
        "    except json.JSONDecodeError:\n"
        "        return\n"
        "    scripts = package.get('scripts') if isinstance(package, dict) else None\n"
        "    if not isinstance(scripts, dict) or 'reskindex' not in scripts:\n"
        "        return\n"
        "    setup_proc = subprocess.run(['yarn', '--silent', 'reskindex'], cwd='/app', text=True, capture_output=True)\n"
        "    runs.append({'label': 'regression_setup', 'args': ['yarn --silent reskindex'], 'run_script_returncode': setup_proc.returncode, 'parser_returncode': 0, 'test_count': 0})\n"
        "    print('REGRESSION_SETUP_RETURN_CODE=' + str(setup_proc.returncode))\n"
        "    if setup_proc.stdout:\n"
        "        print(setup_proc.stdout[-4000:])\n"
        "    if setup_proc.stderr:\n"
        "        print(setup_proc.stderr[-4000:], file=sys.stderr)\n"
        "if selected:\n"
        "    run_and_parse('selected', selected)\n"
        "else:\n"
        "    run_and_parse('all', [])\n"
        "if pass_to_pass:\n"
        "    run_and_parse('pass_to_pass', pass_to_pass)\n"
        "elif selected and payload.get('run_all_for_p2p'):\n"
        "    run_regression_setup()\n"
        "    run_and_parse('regression', [])\n"
        f"print({START_PRO_JSON!r})\n"
        "print(json.dumps({'tests': combined_tests, 'runs': runs}, ensure_ascii=False))\n"
        f"print({END_PRO_JSON!r})\n"
        "sys.exit(0 if all(int(run.get('parser_returncode') or 0) == 0 for run in runs) else 1)\n"
        "PY"
    )


def result_from_run(task: TaskSpec, result: CommandResult) -> TestResult:
    safe_command = "swebench_pro: run official per-instance run_script.sh and parser.py; classify FAIL_TO_PASS and PASS_TO_PASS"
    if result.timed_out:
        return TestResult("timeout", safe_command, result.stdout, result.stderr, result.returncode, resolved=False)
    report, parser_error = _extract_report(result.stdout)
    if report is None:
        return TestResult("failed", safe_command, result.stdout, result.stderr, result.returncode, resolved=False, parser_error=parser_error)
    status_map = _status_map(report)
    runs = _runs(report)
    fail_success, fail_failure = _classify_selectors(task.fail_to_pass, status_map)
    pass_source = "explicit"
    if task.pass_to_pass:
        pass_success, pass_failure = _classify_selectors(task.pass_to_pass, status_map)
    else:
        pass_source = "not_provided"
        pass_success, pass_failure = [], []
    raw_failed = _failed_names_from_output(result.stdout + "\n" + result.stderr)
    selected_failed, regression_failed = _run_failures_by_label(runs)
    if selected_failed and not fail_failure and not fail_success:
        fail_failure.extend(raw_failed or ["selected test run failed without parsed failed selector"])
    if regression_failed and task.pass_to_pass:
        observed_failures = [
            name for name in raw_failed if not _matches_any_selector(name, task.fail_to_pass)
        ]
        if not observed_failures:
            observed_failures = ["regression test run failed without parsed failed selector"]
        for failure in observed_failures:
            if failure not in pass_failure:
                pass_failure.append(failure)
        pass_source = "inferred_from_regression_run"
    p2p_required = bool(task.pass_to_pass)
    tests_status = {
        "FAIL_TO_PASS": {"success": fail_success, "failure": fail_failure, "required": bool(task.fail_to_pass)},
        "PASS_TO_PASS": {
            "success": pass_success,
            "failure": pass_failure,
            "required": p2p_required,
            "source": pass_source,
        },
        "FAIL_TO_FAIL": {"success": [], "failure": []},
        "PASS_TO_FAIL": {"success": [], "failure": []},
    }
    if task.fail_to_pass:
        resolved = (
            len(fail_success) == len(task.fail_to_pass)
            and not fail_failure
            and not pass_failure
            and (not p2p_required or len(pass_success) == len(task.pass_to_pass))
        )
    else:
        resolved = bool(status_map) and not any(status in {"FAILED", "ERROR"} for status in status_map.values())
    return TestResult(
        "passed" if resolved else "failed",
        safe_command,
        result.stdout,
        result.stderr,
        result.returncode,
        tests_status,
        resolved,
        parser_error,
    )


def _extract_report(stdout: str) -> tuple[dict[str, Any] | None, str | None]:
    text = stdout or ""
    if START_PRO_JSON not in text or END_PRO_JSON not in text:
        return None, "missing SWE-bench Pro parsed output markers"
    body = text.split(START_PRO_JSON, 1)[1].split(END_PRO_JSON, 1)[0].strip()
    if not body:
        return None, "empty SWE-bench Pro parser output"
    try:
        report = json.loads(body)
    except json.JSONDecodeError as exc:
        return None, f"SWE-bench Pro parser output is not JSON: {exc}"
    if not isinstance(report, dict) or not isinstance(report.get("tests"), list):
        return None, "SWE-bench Pro parser output missing tests list"
    return report, None


def _classify_selectors(selectors: list[str], status_map: dict[str, str]) -> tuple[list[str], list[str]]:
    success: list[str] = []
    failure: list[str] = []
    for selector in selectors:
        statuses = _matching_statuses(selector, status_map)
        if statuses and all(status == "PASSED" for status in statuses):
            success.append(selector)
        elif any(status in {"FAILED", "ERROR"} for status in statuses):
            failure.append(selector)
        elif not statuses:
            failure.append(selector)
    return success, failure


def _classify_observed_pass_to_pass(fail_to_pass: list[str], status_map: dict[str, str]) -> tuple[list[str], list[str]]:
    success: list[str] = []
    failure: list[str] = []
    for name, status in status_map.items():
        if _matches_any_selector(name, fail_to_pass):
            continue
        if status == "PASSED":
            success.append(name)
        elif status in {"FAILED", "ERROR"}:
            failure.append(name)
    return success, failure


def _matches_any_selector(name: str, selectors: list[str]) -> bool:
    if not selectors:
        return False
    candidate = _normalize_test_name(name)
    for selector in selectors:
        needle = _normalize_test_name(selector)
        if candidate == needle or candidate in needle or needle in candidate:
            return True
    return False


def _status_map(report: dict[str, Any]) -> dict[str, str]:
    priority = {"ERROR": 4, "FAILED": 3, "SKIPPED": 2, "PASSED": 1}
    out: dict[str, str] = {}
    for item in report.get("tests", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        status = str(item.get("status") or "").upper()
        if not name:
            continue
        if priority.get(status, 0) >= priority.get(out.get(name, ""), 0):
            out[name] = status
    return out


def _runs(report: dict[str, Any]) -> list[dict[str, Any]]:
    runs = report.get("runs")
    if not isinstance(runs, list):
        return []
    return [dict(item) for item in runs if isinstance(item, dict)]


def _has_regression_run(runs: list[dict[str, Any]]) -> bool:
    return any(str(run.get("label") or "") == "regression" for run in runs)


def _regression_run_passed(runs: list[dict[str, Any]]) -> bool:
    regression = [run for run in runs if str(run.get("label") or "") == "regression"]
    if not regression:
        return False
    return all(int(run.get("run_script_returncode") or 0) == 0 for run in regression)


def _run_failures_by_label(runs: list[dict[str, Any]]) -> tuple[bool, bool]:
    selected_failed = False
    regression_failed = False
    for run in runs:
        failed = int(run.get("run_script_returncode") or 0) != 0
        label = str(run.get("label") or "")
        if label in {"selected", "all"} and failed:
            selected_failed = True
        elif label in {"regression", "regression_setup"} and failed:
            regression_failed = True
    return selected_failed, regression_failed


def _failed_names_from_output(output: str) -> list[str]:
    failures: list[str] = []
    current_file = ""
    current_suite = ""
    for raw in (output or "").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("FAIL "):
            parts = stripped.split()
            current_file = parts[1] if len(parts) > 1 else current_file
            current_suite = ""
            continue
        if not current_file:
            continue
        if stripped.startswith(("\u2715", "\u00d7", "\u2716", "x ")):
            test_name = stripped[1:].strip() if not stripped.startswith("x ") else stripped[2:].strip()
            test_name = re.sub(r"\s+\(\d+(?:\.\d+)?\s*ms\)$", "", test_name).strip()
            if test_name:
                parts = [current_file]
                if current_suite:
                    parts.append(current_suite)
                parts.append(test_name)
                failure = " | ".join(parts)
                if failure not in failures:
                    failures.append(failure)
            continue
        if stripped.startswith("● "):
            failure = stripped[2:].replace(" › ", " | ").strip()
            if current_file and not failure.startswith(current_file):
                failure = f"{current_file} | {failure}"
            if failure and failure not in failures:
                failures.append(failure)
            continue
        if stripped and not stripped.startswith(("\u2713", "\u2714", "\u25cb", "\u270e", "\u25cf", ">", "$")):
            current_suite = stripped
    return failures


def _matching_statuses(selector: str, status_map: dict[str, str]) -> list[str]:
    needle = _normalize_test_name(selector)
    out: list[str] = []
    for name, status in status_map.items():
        candidate = _normalize_test_name(name)
        if candidate == needle or candidate in needle or needle in candidate:
            out.append(status)
    return out


def _normalize_test_name(value: str) -> str:
    text = " ".join(str(value or "").split())
    text = re.sub(r"^\./", "", text)
    return text


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _run_all_for_p2p(task: TaskSpec, spec: dict[str, Any]) -> bool:
    explicit = os.getenv("GRAPHPLANNER_SWEBENCH_PRO_RUN_ALL_FOR_P2P")
    if explicit is not None:
        return _truthy(explicit)
    if "run_all_for_p2p" in spec:
        return _truthy(spec.get("run_all_for_p2p"))
    return False


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _b64_json(value: dict[str, Any]) -> str:
    return base64.b64encode(json.dumps(value, ensure_ascii=False).encode("utf-8")).decode("ascii")
