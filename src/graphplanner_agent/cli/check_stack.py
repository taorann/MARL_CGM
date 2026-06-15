from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from graphplanner_agent.config import AgentConfig
from graphplanner_agent.datasets import TaskSpec
from graphplanner_agent.planner.client import OpenAIPlannerClient
from graphplanner_agent.planner.prompt import build_messages
from graphplanner_agent.planner.protocol import PLANNER_TOOL_SCHEMAS
from graphplanner_agent.planner.response_parser import parse_planner_message, parse_planner_response
from graphplanner_agent.repair.cgm_client import make_cgm_client
from graphplanner_agent.repair.patch_apply import apply_patch
from graphplanner_agent.repair.patch_quality import syntax_check_python
from graphplanner_agent.repair.patch_schema import Patch, PatchEdit
from graphplanner_agent.runtime.remote_swe import RemoteSweRuntime
from graphplanner_agent.runtime.remote_swe_session import RemoteSweSession, infer_sif_dir_from_ref, normalize_sif_image_ref
from graphplanner_agent.telemetry.console import compact_json, info, summarize_cgm_response


def _remote_layout_summary(layout: dict[str, object]) -> str:
    stdout = str(layout.get("stdout") or "")
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    repo = next((line.split("=", 1)[1] for line in lines if line.startswith("remote_repo=")), "")
    user = lines[1] if len(lines) > 1 else ""
    host = lines[2] if len(lines) > 2 else ""
    proxy = "yes" if "swe_proxy_ok" in lines else "no"
    runner_manager = "yes" if "runner_manager_ok" in lines else "no"
    python = next((line for line in lines if line.startswith("Python ")), "")
    ok = "ok" if layout.get("ok") else "error"
    parts = [
        f"status={ok}",
        f"rc={layout.get('returncode')}",
        f"user={user or '?'}",
        f"host={host or '?'}",
        f"repo={repo or '?'}",
        f"proxy={proxy}",
        f"runner_manager={runner_manager}",
    ]
    if python:
        parts.append(f"python={python}")
    stderr = str(layout.get("stderr") or "").strip()
    if stderr:
        parts.append(f"stderr={stderr[:500]}")
    return " ".join(parts)


def _remote_op_summary(name: str, response: dict[str, object]) -> str:
    ok = bool(response.get("ok"))
    parts = [
        name,
        "ok" if ok else "error",
        f"rc={response.get('returncode')}",
        f"time={float(response.get('runtime_sec') or 0.0):.1f}s",
    ]
    stdout = str(response.get("stdout") or "").strip()
    if stdout:
        parts.append("stdout=" + " | ".join(line.strip() for line in stdout.splitlines() if line.strip())[:300])
    stderr = _display_stderr(str(response.get("stderr") or ""))
    if stderr:
        parts.append(f"stderr={stderr[:500]}")
    error = response.get("error")
    if error:
        parts.append(f"error={error}")
    return " ".join(parts)


def _display_stderr(stderr: str) -> str:
    lines = []
    for line in (stderr or "").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("INFO:") and (
            "Instance stats will not be available" in text
            or "instance started successfully" in text
            or text.startswith("INFO:    Stopping gp-")
        ):
            continue
        lines.append(text)
    return " | ".join(lines)


def _full_smoke_line(name: str, started: float, detail: str = "ok") -> str:
    return f"[check] remote_swe_full {name} ok time={time.perf_counter() - started:.1f}s {detail}".rstrip()


def _run_remote_swe_full_smoke(config: AgentConfig, image_ref: str, *, quiet: bool = False) -> None:
    sandbox_key = "sif_path" if image_ref.endswith(".sif") else "image"
    task = TaskSpec(
        task_id="remote-swe-full-smoke",
        repo_path=Path("."),
        issue_title="remote_swe full smoke",
        issue_body="Exercise all container-backed runtime operations.",
        sandbox={sandbox_key: image_ref},
        test_command="python - <<'PY'\nprint('remote-swe-test-ok')\nPY",
    )
    runtime = RemoteSweRuntime(config)
    tmp_path = "gp_remote_swe_full_smoke_tmp.py"
    try:
        t0 = time.perf_counter()
        runtime.start(task)
        if not quiet:
            info(_full_smoke_line("start", t0, f"run_id={runtime.run_id}"))

        t0 = time.perf_counter()
        cmd = runtime.run("pwd && python -V && test -d /testbed && echo testbed_ok", timeout=120)
        if cmd.returncode != 0:
            raise RuntimeError(f"exec failed rc={cmd.returncode}: {cmd.stderr or cmd.stdout}")
        if not quiet:
            info(_full_smoke_line("exec", t0, "stdout=" + " | ".join(cmd.stdout.strip().splitlines())))

        t0 = time.perf_counter()
        before = runtime.snapshot([tmp_path])
        if not quiet:
            info(_full_smoke_line("snapshot", t0, f"entries={len(before)}"))

        t0 = time.perf_counter()
        runtime.write_file(tmp_path, "VALUE = 1\n")
        if runtime.read_file(tmp_path).strip() != "VALUE = 1":
            raise RuntimeError("read/write verification failed")
        if not quiet:
            info(_full_smoke_line("read_write", t0, tmp_path))

        t0 = time.perf_counter()
        patch = Patch([PatchEdit(tmp_path, 1, 1, "VALUE = 2\n")], "full smoke edit")
        apply_patch(runtime, patch)
        if runtime.read_file(tmp_path).strip() != "VALUE = 2":
            raise RuntimeError("patch verification failed")
        if not quiet:
            info(_full_smoke_line("apply_patch", t0, tmp_path))

        t0 = time.perf_counter()
        syntax = syntax_check_python(runtime, patch)
        if syntax is None or not syntax.passed:
            raise RuntimeError(f"syntax check failed: {syntax.summary() if syntax else 'no result'}")
        if not quiet:
            info(_full_smoke_line("syntax", t0, syntax.status))

        t0 = time.perf_counter()
        runtime.rollback(before)
        removed = runtime.run(f"test ! -e {tmp_path} && echo rollback_removed", timeout=60)
        if removed.returncode != 0:
            raise RuntimeError(f"rollback did not restore original state: {removed.stderr or removed.stdout}")
        if not quiet:
            info(_full_smoke_line("rollback", t0, removed.stdout.strip()))

        t0 = time.perf_counter()
        test = runtime.run_fail_to_pass(task)
        if not test.passed:
            raise RuntimeError(f"run_fail_to_pass failed rc={test.returncode}: {test.summary()}")
        if not quiet:
            info(_full_smoke_line("run_fail_to_pass", t0, test.status))

        t0 = time.perf_counter()
        graph = runtime.build_graph()
        cache = "hit" if runtime.last_graph_cache_hit else "miss"
        if not graph.nodes or not graph.edges:
            raise RuntimeError(f"build_graph returned empty graph nodes={len(graph.nodes)} edges={len(graph.edges)}")
        if not quiet:
            info(_full_smoke_line("build_graph", t0, f"nodes={len(graph.nodes)} edges={len(graph.edges)} cache={cache}"))
    finally:
        try:
            runtime.run(f"rm -f {tmp_path}", timeout=60)
        except Exception:
            pass
        t0 = time.perf_counter()
        runtime.stop()
        if not quiet:
            info(_full_smoke_line("stop", t0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check planner and CGM connectivity for the rebuild.")
    parser.add_argument("--planner-endpoint")
    parser.add_argument("--planner-model")
    parser.add_argument("--cgm-backend", choices=["mock", "http", "dashscope"])
    parser.add_argument("--cgm-endpoint")
    parser.add_argument("--sandbox-backend", choices=["local", "remote_swe"])
    parser.add_argument("--sandbox-ssh-target")
    parser.add_argument("--sandbox-remote-repo")
    parser.add_argument("--sandbox-num-runners", type=int)
    parser.add_argument("--sandbox-workdir")
    parser.add_argument("--sandbox-sif-dir")
    parser.add_argument("--remote-swe-smoke", action="store_true")
    parser.add_argument("--remote-swe-full-smoke", action="store_true")
    parser.add_argument("--remote-swe-image", help="Optional image key or .sif path/name for start -> python -V -> stop smoke.")
    parser.add_argument("--skip-planner", action="store_true")
    parser.add_argument("--planner-tool-smoke", action="store_true", help="Ask the planner through OpenAI tool calling and print raw/parsed tool use.")
    parser.add_argument("--print-raw", action="store_true", help="Print raw planner message.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = AgentConfig.from_env()
    if args.planner_endpoint:
        config.planner_endpoint = args.planner_endpoint
    if args.planner_model:
        config.planner_model = args.planner_model
    if args.cgm_backend:
        config.cgm_backend = args.cgm_backend
    if args.cgm_endpoint:
        config.cgm_endpoint = args.cgm_endpoint
    if args.sandbox_backend:
        config.sandbox_backend = args.sandbox_backend
    if args.sandbox_ssh_target:
        config.sandbox_ssh_target = args.sandbox_ssh_target
    if args.sandbox_remote_repo:
        config.sandbox_remote_repo = args.sandbox_remote_repo
    if args.sandbox_num_runners:
        config.sandbox_num_runners = args.sandbox_num_runners
    if args.sandbox_workdir:
        config.sandbox_workdir = args.sandbox_workdir
    if args.sandbox_sif_dir:
        config.sandbox_sif_dir = args.sandbox_sif_dir
    config.finalize()
    if not args.quiet:
        planner = "skip" if args.skip_planner else (config.planner_endpoint or "<unset>")
        info(f"[check] config planner={planner} cgm={config.cgm_backend} sandbox={config.sandbox_backend}")

    if not args.skip_planner:
        try:
            client = OpenAIPlannerClient(config)
            if args.planner_tool_smoke:
                messages = [
                    {
                        "role": "system",
                        "content": "Use exactly one provided tool. You may include visible thinking if the model supports it.",
                    },
                    {
                        "role": "user",
                        "content": "For this stack check, call run_failed_test now. Do not answer in prose.",
                    },
                ]
                raw_message = client.complete_message(messages, tools=PLANNER_TOOL_SCHEMAS, tool_choice="auto")
                if args.print_raw:
                    info("[raw] planner_message=" + json.dumps(raw_message, ensure_ascii=False, sort_keys=True))
                parsed = parse_planner_message(raw_message)
            else:
                raw = client.complete(
                    build_messages(
                        '{"issue":{"title":"stack check"},"runtime_facts":{"fail_to_pass_behavior_present":false}}'
                    )
                )
                if args.print_raw:
                    info(f"[raw] planner_text={raw}")
                parsed = parse_planner_response(raw)
            if not args.quiet:
                info(
                    "[check] planner ok "
                    f"tool={parsed.action.tool} params={json.dumps(parsed.action.params, ensure_ascii=False, sort_keys=True)} "
                    f"visible_thinking={'yes' if parsed.visible_thinking else 'no'}"
                )
            if parsed.visible_thinking and not args.quiet:
                info("[check] planner thinking excerpt=" + parsed.visible_thinking[:800].replace("\n", "\\n"))
        except Exception as exc:
            info(f"[check] planner error {type(exc).__name__}: {exc}")
            return 1

    cgm = make_cgm_client(config)
    payload = {
        "issue": {
            "id": "stack-check",
            "title": "stack check",
            "body": "Title: stack check\n\nFunction add returns subtraction; change it to addition.",
            "repo": "smoke",
            "language": "python",
        },
        "plan": {"targets": [{"path": "pkg/calc.py", "start": 1, "end": 2, "id": "func:pkg/calc.py:add", "why": "stack check"}]},
        "plan_text": "Edit pkg/calc.py so add(a, b) returns a + b.\n\nUse the exact numbered snippet lines when choosing edit start/end.",
        "repo": "smoke",
        "language": "python",
        "graph": {
            "nodes": [
                {
                    "id": "func:pkg/calc.py:add",
                    "type": "function",
                    "nodeType": "Function",
                    "name": "add",
                    "path": "pkg/calc.py",
                    "start_line": 1,
                    "end_line": 2,
                    "text": "def add(a, b):\n    return a - b\n",
                    "is_memory_target": True,
                }
            ],
            "edges": [],
            "reponame": "smoke",
            "language": "python",
        },
        "snippets": [
            {
                "path": "pkg/calc.py",
                "start": 1,
                "end": 2,
                "text": "def add(a, b):\n    return a - b\n",
                "lines": ["def add(a, b):", "    return a - b"],
            }
        ],
        "metadata": {"constraints": {"max_edits": 1, "implementation_only": True, "no_test_changes": True}},
    }
    cgm_resp = cgm.generate_patch(payload)
    if not args.quiet:
        info(f"[check] cgm ok {summarize_cgm_response(cgm_resp)}")
    if args.remote_swe_smoke or args.remote_swe_full_smoke or config.sandbox_backend == "remote_swe":
        image_ref = args.remote_swe_image or "remote-swe-layout-check"
        image = normalize_sif_image_ref(image_ref)
        sif_dir = config.sandbox_sif_dir or infer_sif_dir_from_ref(image_ref)
        session = RemoteSweSession(
            ssh_target=config.sandbox_ssh_target,
            remote_repo=config.sandbox_remote_repo,
            image=image,
            run_id="gp-stack-check",
            remote_python=config.sandbox_remote_python,
            swe_proxy_path=config.sandbox_swe_proxy_path,
            runner_manager_path=config.sandbox_runner_manager_path,
            num_runners=config.sandbox_num_runners,
            ensure_runners=False,
            ssh_args=config.sandbox_ssh_args,
            sif_dir=sif_dir,
        )
        layout = session.check_remote_layout()
        if args.print_raw:
            info("[raw] remote_swe_layout=" + json.dumps(layout, ensure_ascii=False, sort_keys=True))
        elif not args.quiet:
            info("[check] remote_swe " + _remote_layout_summary(layout))
        if not layout.get("ok"):
            return 2
        if args.remote_swe_full_smoke:
            if not args.remote_swe_image:
                info("[check] remote_swe_full error --remote-swe-image is required")
                return 2
            try:
                _run_remote_swe_full_smoke(config, args.remote_swe_image, quiet=args.quiet)
            except Exception as exc:
                info(f"[check] remote_swe_full error {type(exc).__name__}: {exc}")
                return 2
            return 0
        if args.remote_swe_image:
            session.ensure_runners = True
            start = session.start(timeout=300.0, cwd=config.sandbox_workdir)
            if args.print_raw:
                info("[raw] remote_swe_start=" + compact_json(start, limit=1000))
            elif not args.quiet:
                info("[check] remote_swe " + _remote_op_summary("start", start))
            run = session.exec("pwd && python -V", cwd=config.sandbox_workdir, timeout=120.0)
            if args.print_raw:
                info("[raw] remote_swe_exec=" + compact_json(run, limit=1000))
            elif not args.quiet:
                info("[check] remote_swe " + _remote_op_summary("exec", run))
            stop = session.stop(timeout=60.0)
            if args.print_raw:
                info("[raw] remote_swe_stop=" + compact_json(stop, limit=1000))
            elif not args.quiet:
                info("[check] remote_swe " + _remote_op_summary("stop", stop))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
