from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from datetime import datetime, timezone

from graphplanner_agent.config import AgentConfig
from graphplanner_agent.datasets import dump_results, load_tasks
from graphplanner_agent.env import CodeRepairEnv
from graphplanner_agent.planner.client import make_planner_client
from graphplanner_agent.planner.loop import PlannerLoop
from graphplanner_agent.repair.cgm_client import make_cgm_client
from graphplanner_agent.runtime import make_runtime
from graphplanner_agent.runtime.swebench_official import official_eval_script_lines
from graphplanner_agent.telemetry import ProgressTracker, TraceWriter
from graphplanner_agent.telemetry.console import compact_json, info

from .remote_preflight import normalize_remote_preflight_mode, run_remote_swe_preflight


def _load_scripted_actions(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [line for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return [json.dumps(item) if isinstance(item, dict) else str(item) for item in data]
    raise ValueError("scripted actions must be a JSON array or JSONL action stream")


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value.strip())
    return value.strip("_") or "run"


def _make_run_dir(tasks, runs_root: Path, label: str | None) -> Path:
    task_part = _safe_name(tasks[0].task_id if len(tasks) == 1 else f"{len(tasks)}tasks")
    label_part = _safe_name(label or "eval")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")
    return runs_root / f"{task_part}__{label_part}__{stamp}"


def _write_run_metadata(path: Path, args: argparse.Namespace, config: AgentConfig, task_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "task_count": task_count,
        "planner_model": config.planner_model,
        "planner_endpoint": config.planner_endpoint,
        "planner_tool_calling": config.planner_tool_calling,
        "cgm_backend": config.cgm_backend,
        "cgm_endpoint": config.cgm_endpoint,
        "sandbox_backend": config.sandbox_backend,
        "remote_preflight": getattr(args, "remote_preflight", "auto"),
        "max_steps": config.max_steps,
        "observation_mode": config.observation_mode,
        "tasks_path": str(args.tasks),
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _validate_task_inputs(tasks, tasks_path: Path) -> None:
    errors: list[str] = []
    for idx, task in enumerate(tasks):
        issue_empty = not (task.issue_body or "").strip()
        has_behavior_selector = bool(task.fail_to_pass or task.pass_to_pass)
        has_official_script = bool(official_eval_script_lines(task))
        looks_like_swe = bool(task.docker_image) or "__" in task.task_id
        if looks_like_swe and issue_empty:
            errors.append(f"task[{idx}] {task.task_id}: missing issue body/problem_statement")
        if looks_like_swe and not task.test_command and not has_behavior_selector and not has_official_script:
            errors.append(
                f"task[{idx}] {task.task_id}: missing fail_to_pass/PASS_TO_PASS and SWE-bench eval_script_list; "
                "would fall back to broad pytest and can produce infra failures"
            )
    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(f"invalid task input file {tasks_path}:\n  - {joined}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the train-free GraphPlanner rebuild.")
    parser.add_argument("--tasks", required=True, type=Path, help="Task JSON or JSONL file.")
    parser.add_argument("--run-dir", type=Path, help="Directory for one eval run: results.jsonl, progress.md, traces/, metadata.json.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs/tmp"), help="Root used with --run-label to create a timestamped run directory.")
    parser.add_argument("--run-label", help="Create a timestamped run directory under --runs-root using this label.")
    parser.add_argument("--results-path", type=Path)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--progress-path", type=Path)
    parser.add_argument("--planner-endpoint")
    parser.add_argument("--planner-model")
    parser.add_argument("--planner-tool-calling", action="store_true")
    parser.add_argument("--cgm-backend", choices=["mock", "http"])
    parser.add_argument("--cgm-endpoint")
    parser.add_argument("--sandbox-backend", choices=["local", "remote_swe"])
    parser.add_argument("--sandbox-ssh-target")
    parser.add_argument("--sandbox-remote-repo")
    parser.add_argument("--sandbox-num-runners", type=int)
    parser.add_argument("--sandbox-workdir")
    parser.add_argument("--sandbox-sif-dir")
    parser.add_argument(
        "--remote-preflight",
        choices=["auto", "off", "cleanup", "full"],
        default="auto",
        help="For remote_swe, validate the remote layout, cleanup stale sandboxes, and ensure runners before tasks. full also start/stops a smoke sandbox.",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--observation-mode", choices=["json", "text"], help="Planner observation format. Defaults to GRAPHPLANNER_OBSERVATION_MODE or json.")
    parser.add_argument("--scripted-actions", type=Path, help="Use static planner actions instead of the planner endpoint.")
    parser.add_argument("--quiet", action="store_true", help="Only print final JSON records.")
    parser.add_argument("--verbose", action="store_true", help="Print compact per-step progress in addition to traces.")
    args = parser.parse_args()

    config = AgentConfig.from_env()
    if args.planner_endpoint:
        config.planner_endpoint = args.planner_endpoint
    if args.planner_model:
        config.planner_model = args.planner_model
    if args.planner_tool_calling:
        config.planner_tool_calling = True
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
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.observation_mode:
        config.observation_mode = args.observation_mode
    if args.verbose:
        config.console_verbose = True
    config.finalize()

    tasks = load_tasks(args.tasks)
    _validate_task_inputs(tasks, args.tasks)
    scripted = _load_scripted_actions(args.scripted_actions)
    run_dir = args.run_dir
    if run_dir is None and args.run_label:
        run_dir = _make_run_dir(tasks, args.runs_root, args.run_label)
    results_path = args.results_path or (run_dir / "results.jsonl" if run_dir else Path("runs/results.jsonl"))
    trace_dir = args.trace_dir or (run_dir / "traces" if run_dir else Path("runs/traces"))
    progress_path = args.progress_path or (run_dir / "progress.md" if run_dir else Path("runs/progress.md"))
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_run_metadata(run_dir / "metadata.json", args, config, len(tasks))
        if not args.quiet:
            info(f"[run] dir={run_dir}")
            info(f"[run] results={results_path} progress={progress_path} traces={trace_dir}")
            info(f"[run] shell log suggestion: {run_dir / 'run.log'}")
    preflight_mode = normalize_remote_preflight_mode(args.remote_preflight, backend=config.sandbox_backend)
    if preflight_mode != "off" and config.sandbox_backend == "remote_swe" and tasks:
        preflight_info = run_remote_swe_preflight(config, tasks[0], mode=preflight_mode)
        args.remote_preflight = preflight_mode
        if bool(preflight_info.get("ok")):
            config.sandbox_ensure_runners_before_start = False
            config.sandbox_cleanup_pool_before_start = False
        if run_dir is not None:
            _write_run_metadata(run_dir / "metadata.json", args, config, len(tasks))
            (run_dir / "remote_preflight.json").write_text(
                json.dumps(preflight_info, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if not args.quiet:
            info("[remote] preflight " + compact_json(preflight_info, limit=900))
        if not bool(preflight_info.get("ok")):
            if run_dir is not None:
                (run_dir / "remote_preflight_failed.json").write_text(
                    json.dumps(preflight_info, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            return 2
    progress = ProgressTracker(progress_path)
    records: list[dict[str, object]] = []

    for task in tasks:
        trace = TraceWriter(
            trace_dir / f"{task.task_id}.jsonl",
            trace_dir / f"{task.task_id}.md",
        )
        runtime = None
        progress.start_task(
            task.task_id,
            {
                "backend": config.sandbox_backend,
                "cgm": config.cgm_backend,
                "max_steps": config.max_steps,
            },
        )
        try:
            if not args.quiet:
                info(
                    f"[task] {task.task_id} backend={config.sandbox_backend} "
                    f"cgm={config.cgm_backend} max_steps={config.max_steps}"
                )
            planner = make_planner_client(config, scripted_responses=list(scripted) if scripted is not None else None)
            cgm = make_cgm_client(config)
            runtime = make_runtime(task, config)
            progress.update_task(task.task_id, "building_graph")
            env = CodeRepairEnv.create(task, runtime, cgm, config)
            progress.update_task(task.task_id, "graph_built", {"graph_nodes": len(env.graph.nodes), "graph_edges": len(env.graph.edges)})
            if not args.quiet:
                info(f"[task] graph nodes={len(env.graph.nodes)} edges={len(env.graph.edges)}")
            result = PlannerLoop(
                env,
                planner,
                config,
                trace,
                console=(config.console_verbose and not args.quiet),
                on_step=lambda step_record, task_id=task.task_id: progress.record_step(
                    task_id,
                    int(step_record["step"]),
                    str(step_record["tool"]),
                    str(step_record["status"]),
                    float(step_record["elapsed"]),
                    str(step_record["summary"]),
                ),
            ).run()
            record = {
                "task_id": task.task_id,
                "status": result.status,
                "steps": result.steps,
                "reason": result.reason,
                "verified": env.verified,
                "latest_result": env.latest_result,
            }
        except Exception as exc:
            record = {"task_id": task.task_id, "status": "bug", "steps": 0, "reason": f"{type(exc).__name__}: {exc}"}
            trace.event("bug", record)
        finally:
            if runtime is not None:
                try:
                    runtime.stop()
                except Exception as exc:
                    stop_record = {"task_id": task.task_id, "error": f"{type(exc).__name__}: {exc}"}
                    trace.event("runtime_stop_error", stop_record)
                    if not args.quiet:
                        info(f"[runtime] stop warning: {stop_record['error']}")
        records.append(record)
        dump_results(results_path, records)
        summary = progress.record(str(record["status"]), task_id=task.task_id, reason=str(record.get("reason") or ""))
        if args.quiet:
            print(json.dumps(record, sort_keys=True))
        else:
            info(
                f"[result] {record['task_id']} status={record['status']} "
                f"steps={record.get('steps')} verified={record.get('verified')} reason={record.get('reason')}"
            )
            if config.console_verbose:
                info("[result-detail] " + compact_json(record.get("latest_result"), limit=600))
            info(
                f"[progress] total={summary['total']} pass={summary['pass']} "
                f"not_pass={summary['not_pass']} bug={summary['bug']} "
                f"acc={summary['accuracy']:.3f} bug_excl={summary['bug_excluded_accuracy']:.3f}"
            )

    dump_results(results_path, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
