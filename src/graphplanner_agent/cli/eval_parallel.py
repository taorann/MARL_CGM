from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading

from graphplanner_agent.config import AgentConfig
from graphplanner_agent.datasets import TaskSpec, load_tasks
from graphplanner_agent.env import CodeRepairEnv
from graphplanner_agent.memory import CgmMemory, TextNotes, WorkingMemory
from graphplanner_agent.planner.client import make_planner_client
from graphplanner_agent.planner.loop import PlannerLoop
from graphplanner_agent.repair.cgm_client import make_cgm_client
from graphplanner_agent.runtime import make_runtime
from graphplanner_agent.telemetry import ProgressTracker, TraceWriter
from graphplanner_agent.telemetry.console import compact_json, info

from .eval import _load_scripted_actions, _validate_task_inputs
from .remote_preflight import normalize_remote_preflight_mode, run_remote_swe_preflight


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value.strip())
    return value.strip("_") or "run"


def _make_run_dir(tasks: list[TaskSpec], runs_root: Path, label: str | None) -> Path:
    task_part = _safe_name(tasks[0].task_id if len(tasks) == 1 else f"{len(tasks)}tasks")
    label_part = _safe_name(label or "parallel_eval")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")
    return runs_root / f"{task_part}__{label_part}__{stamp}"


def _write_run_metadata(path: Path, args: argparse.Namespace, config: AgentConfig, task_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "task_count": task_count,
        "parallel": args.parallel,
        "resume_from_results": str(args.resume_from_results) if args.resume_from_results else None,
        "retry_bug_results": bool(args.retry_bug_results),
        "planner_model": config.planner_model,
        "planner_endpoint": config.planner_endpoint,
        "planner_tool_calling": config.planner_tool_calling,
        "planner_enable_thinking": config.planner_enable_thinking,
        "cgm_backend": config.cgm_backend,
        "cgm_endpoint": config.cgm_endpoint,
        "sandbox_backend": config.sandbox_backend,
        "sandbox_num_runners": config.sandbox_num_runners,
        "worker_ensure_runners_disabled": not config.sandbox_ensure_runners_before_start,
        "worker_cleanup_disabled": True,
        "remote_cleanup_once": bool(args.remote_cleanup_once),
        "remote_preflight": args.remote_preflight,
        "baseline_results": [str(path) for path in (args.baseline_results or [])],
        "max_steps": config.max_steps,
        "observation_mode": config.observation_mode,
        "tasks_path": str(args.tasks),
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _apply_config_args(config: AgentConfig, args: argparse.Namespace) -> AgentConfig:
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
    if args.parallel and config.sandbox_backend == "remote_swe":
        config.sandbox_num_runners = max(int(config.sandbox_num_runners or 1), int(args.parallel))
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
    config.sandbox_cleanup_pool_before_start = False
    config.finalize()
    return config


def _resume_skip_ids(path: Path, *, retry_bugs: bool) -> tuple[set[str], dict[str, object]]:
    skip: set[str] = set()
    status_counts: dict[str, int] = {}
    skipped_status_counts: dict[str, int] = {}
    malformed = 0
    total = 0
    if not path.exists():
        return skip, {"path": str(path), "exists": False, "skipped_task_count": 0}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(record, dict):
            malformed += 1
            continue
        task_id = str(record.get("task_id") or "").strip()
        if not task_id:
            malformed += 1
            continue
        status = str(record.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        if retry_bugs and status == "bug":
            continue
        skip.add(task_id)
        skipped_status_counts[status] = skipped_status_counts.get(status, 0) + 1
    return skip, {
        "path": str(path),
        "exists": True,
        "records": total,
        "malformed_records": malformed,
        "status_counts": status_counts,
        "skipped_status_counts": skipped_status_counts,
        "retry_bugs": retry_bugs,
        "skipped_task_count": len(skip),
    }


def _baseline_counts_from_results(paths: list[Path]) -> tuple[dict[str, int], dict[str, object]]:
    latest_by_task: dict[str, dict[str, object]] = {}
    malformed = 0
    records = 0
    missing = 0
    for path in paths:
        if not path.exists():
            missing += 1
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(record, dict):
                malformed += 1
                continue
            task_id = str(record.get("task_id") or "").strip()
            if not task_id:
                malformed += 1
                continue
            latest_by_task[task_id] = record
    counts = {"pass": 0, "not_pass": 0, "bug": 0}
    skipped_bug = 0
    skipped_infra = 0
    skipped_other = 0
    for record in latest_by_task.values():
        status = str(record.get("status") or "")
        if record.get("infra_contaminated"):
            skipped_infra += 1
            continue
        if status == "bug":
            skipped_bug += 1
            continue
        if status not in counts:
            skipped_other += 1
            continue
        counts[status] += 1
    return counts, {
        "paths": [str(path) for path in paths],
        "records": records,
        "missing_paths": missing,
        "malformed_records": malformed,
        "unique_task_count": len(latest_by_task),
        "counts": counts,
        "skipped_bug": skipped_bug,
        "skipped_infra_contaminated": skipped_infra,
        "skipped_other_status": skipped_other,
    }


def _run_one_task(
    task: TaskSpec,
    *,
    base_config: AgentConfig,
    scripted: list[str] | None,
    trace_dir: Path,
    progress: ProgressTracker,
    progress_lock: threading.Lock,
    remote_start_lock: threading.Lock | None,
    worker_id: int,
    quiet: bool,
) -> dict[str, object]:
    config = replace(base_config)
    config.sandbox_cleanup_pool_before_start = False
    config.finalize()
    trace = TraceWriter(
        trace_dir / f"{task.task_id}.jsonl",
        trace_dir / f"{task.task_id}.md",
    )
    runtime = None
    with progress_lock:
        progress.start_task(
            task.task_id,
            {
                "worker": worker_id,
                "backend": config.sandbox_backend,
                "cgm": config.cgm_backend,
                "max_steps": config.max_steps,
            },
        )
    try:
        if not quiet:
            info(f"[task:{worker_id}] start {task.task_id}")
        planner = make_planner_client(config, scripted_responses=list(scripted) if scripted is not None else None)
        cgm = make_cgm_client(config)
        runtime = make_runtime(task, config)
        with progress_lock:
            progress.update_task(task.task_id, "building_graph")
        if remote_start_lock is not None:
            with remote_start_lock:
                runtime.start(task)
        else:
            runtime.start(task)
        graph = runtime.build_graph()
        env = CodeRepairEnv(
            task=task,
            runtime=runtime,
            cgm=cgm,
            config=config,
            graph=graph,
            working=WorkingMemory(),
            memory=CgmMemory(),
            notes=TextNotes(),
        )
        with progress_lock:
            progress.update_task(
                task.task_id,
                "graph_built",
                {"graph_nodes": len(env.graph.nodes), "graph_edges": len(env.graph.edges)},
            )

        def on_step(step_record: dict[str, object]) -> None:
            with progress_lock:
                progress.record_step(
                    task.task_id,
                    int(step_record["step"]),
                    str(step_record["tool"]),
                    str(step_record["status"]),
                    float(step_record["elapsed"]),
                    str(step_record["summary"]),
                )

        result = PlannerLoop(
            env,
            planner,
            config,
            trace,
            console=(config.console_verbose and not quiet),
            on_step=on_step,
        ).run()
        record = {
            "task_id": task.task_id,
            "status": result.status,
            "steps": result.steps,
            "reason": result.reason,
            "verified": env.verified,
            "latest_result": env.latest_result,
            "worker": worker_id,
        }
        _annotate_infra_contamination(record, trace.jsonl_path)
        return record
    except Exception as exc:
        record = {
            "task_id": task.task_id,
            "status": "bug",
            "steps": 0,
            "reason": f"{type(exc).__name__}: {exc}",
            "worker": worker_id,
        }
        trace.event("bug", record)
        _annotate_infra_contamination(record, trace.jsonl_path)
        return record
    finally:
        if runtime is not None:
            try:
                runtime.stop()
            except Exception as exc:
                trace.event("runtime_stop_error", {"task_id": task.task_id, "error": f"{type(exc).__name__}: {exc}"})


def _append_result(path: Path, record: dict[str, object], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _is_planner_network_bug(record: dict[str, object]) -> bool:
    if str(record.get("status") or "") != "bug":
        return False
    reason = str(record.get("reason") or "").lower()
    if "planner" not in reason:
        return False
    return any(
        marker in reason
        for marker in (
            "connection refused",
            "remote end closed connection",
            "timed out",
            "urlopen error",
            "planner http 429",
            "planner http 500",
            "planner http 502",
            "planner http 503",
            "planner http 504",
        )
    )


def _is_remote_runner_pool_bug(record: dict[str, object]) -> bool:
    if str(record.get("status") or "") != "bug":
        return False
    reason = str(record.get("reason") or "").lower()
    return (
        "remote swe_proxy failed" in reason
        and "op='start'" in reason
        and "timed out waiting for an idle runner" in reason
    )


def _text_from_record_and_trace(record: dict[str, object], trace_path: Path | None = None) -> str:
    chunks = [json.dumps(record, ensure_ascii=False, default=str)]
    if trace_path is not None and trace_path.exists():
        try:
            chunks.append(trace_path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
    return "\n".join(chunks).lower()


def _infra_contamination_reasons(record: dict[str, object], trace_path: Path | None = None) -> list[str]:
    text = _text_from_record_and_trace(record, trace_path)
    reasons: list[str] = []
    markers = [
        ("runner_pool_start_timeout", "timed out waiting for an idle runner"),
        ("no_active_instance", "no active instance on this runner"),
        ("remote_swe_proxy_failed", "remote swe_proxy failed"),
        ("remote_runtime_error", "remotesweerror"),
        ("cgm_unavailable", "cgm_unavailable"),
        ("cgm_timeout", "cgm unavailable"),
        ("infra_retryable", "infra_retryable"),
    ]
    for reason, marker in markers:
        if marker in text and reason not in reasons:
            reasons.append(reason)
    return reasons


def _annotate_infra_contamination(record: dict[str, object], trace_path: Path | None = None) -> None:
    reasons = _infra_contamination_reasons(record, trace_path)
    if not reasons:
        return
    record["infra_contaminated"] = True
    record["infra_contamination_reasons"] = reasons


def _is_remote_sandbox_invalid(record: dict[str, object]) -> bool:
    reasons = record.get("infra_contamination_reasons")
    if not isinstance(reasons, list):
        reasons = _infra_contamination_reasons(record)
    severe = {"runner_pool_start_timeout", "no_active_instance"}
    return any(str(reason) in severe for reason in reasons)


def _mark_remote_sandbox_invalid_bug(record: dict[str, object]) -> None:
    original_status = record.get("status")
    original_reason = str(record.get("reason") or "")
    record.setdefault("original_status", original_status)
    record["status"] = "bug"
    record["verified"] = False
    record["reason"] = (
        "infra_bug: remote_swe sandbox instance was unavailable; "
        "skipping this issue and continuing. "
        f"original_status={original_status} original_reason={original_reason[:500]}"
    )


def _write_infra_contamination_report(path: Path, records: list[dict[str, object]]) -> None:
    contaminated = [
        {
            "task_id": record.get("task_id"),
            "status": record.get("status"),
            "reasons": record.get("infra_contamination_reasons", []),
        }
        for record in records
        if record.get("infra_contaminated")
    ]
    path.write_text(
        json.dumps(
            {
                "count": len(contaminated),
                "records": contaminated,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GraphPlanner eval with multiple independent issue workers.")
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/tmp"))
    parser.add_argument("--run-label")
    parser.add_argument("--results-path", type=Path)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--progress-path", type=Path)
    parser.add_argument("--planner-endpoint")
    parser.add_argument("--planner-model")
    parser.add_argument("--planner-tool-calling", action="store_true")
    parser.add_argument("--cgm-backend", choices=["mock", "http", "dashscope"])
    parser.add_argument("--cgm-endpoint")
    parser.add_argument("--sandbox-backend", choices=["local", "remote_swe"])
    parser.add_argument("--sandbox-ssh-target")
    parser.add_argument("--sandbox-remote-repo")
    parser.add_argument("--sandbox-num-runners", type=int)
    parser.add_argument("--sandbox-workdir")
    parser.add_argument("--sandbox-sif-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--observation-mode", choices=["json", "text"])
    parser.add_argument("--scripted-actions", type=Path)
    parser.add_argument("--remote-cleanup-once", action="store_true", help="Deprecated alias for --remote-preflight cleanup.")
    parser.add_argument(
        "--remote-preflight",
        choices=["auto", "off", "cleanup", "full"],
        default="auto",
        help="For remote_swe, validate the remote layout, cleanup stale sandboxes, and ensure runners before tasks. full also start/stops a smoke sandbox.",
    )
    parser.add_argument(
        "--resume-from-results",
        type=Path,
        help="Skip task ids already present in a previous results.jsonl. By default bug records are skipped too.",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        action="append",
        default=[],
        help="Seed progress counters from previous clean results without skipping any tasks. May be passed multiple times.",
    )
    parser.add_argument(
        "--baseline-label",
        help="Human-readable label for --baseline-results in progress.md.",
    )
    parser.add_argument(
        "--retry-bug-results",
        action="store_true",
        help="When --resume-from-results is set, retry previous records whose status is bug.",
    )
    parser.add_argument(
        "--stop-after-planner-network-bugs",
        type=int,
        default=0,
        help="Cancel queued tasks after this many consecutive planner network bugs. Default: parallel*2.",
    )
    parser.add_argument(
        "--stop-after-remote-runner-bugs",
        type=int,
        default=1,
        help="Cancel queued tasks after this many consecutive remote_swe idle-runner start failures. Default: 1.",
    )
    parser.add_argument(
        "--stop-after-remote-sandbox-invalid",
        type=int,
        default=1,
        help="Cancel queued tasks after this many consecutive severe remote sandbox failures such as no active instance. Default: 1.",
    )
    parser.add_argument(
        "--remote-sandbox-invalid-policy",
        choices=["continue", "stop"],
        default="continue",
        help="When a task loses its remote sandbox instance, record it as an infra bug and continue by default; use stop to halt the round.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = _apply_config_args(AgentConfig.from_env(), args)
    tasks = load_tasks(args.tasks)
    resume_info: dict[str, object] | None = None
    if args.resume_from_results:
        skip_ids, resume_info = _resume_skip_ids(args.resume_from_results, retry_bugs=args.retry_bug_results)
        before = len(tasks)
        tasks = [task for task in tasks if task.task_id not in skip_ids]
        resume_info.update(
            {
                "original_task_count": before,
                "remaining_task_count": len(tasks),
                "filtered_task_count": before - len(tasks),
            }
        )
    _validate_task_inputs(tasks, args.tasks)
    scripted = _load_scripted_actions(args.scripted_actions)

    run_dir = args.run_dir or (_make_run_dir(tasks, args.runs_root, args.run_label) if args.run_label else None)
    results_path = args.results_path or (run_dir / "results.jsonl" if run_dir else Path("runs/results.jsonl"))
    trace_dir = args.trace_dir or (run_dir / "traces" if run_dir else Path("runs/traces"))
    progress_path = args.progress_path or (run_dir / "progress.md" if run_dir else Path("runs/progress.md"))
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("", encoding="utf-8")

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_run_metadata(run_dir / "metadata.json", args, config, len(tasks))
        if resume_info is not None:
            (run_dir / "resume_info.json").write_text(
                json.dumps(resume_info, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if not args.quiet:
            info(f"[run] dir={run_dir}")
            info(f"[run] parallel={args.parallel} results={results_path} progress={progress_path} traces={trace_dir}")
            info(f"[run] shell log suggestion: {run_dir / 'run.log'}")
            if resume_info is not None:
                info("[resume] " + compact_json(resume_info, limit=600))

    preflight_mode = normalize_remote_preflight_mode(args.remote_preflight, backend=config.sandbox_backend)
    if args.remote_cleanup_once and preflight_mode == "off":
        preflight_mode = "cleanup"
    if preflight_mode != "off" and config.sandbox_backend == "remote_swe" and tasks:
        preflight_info = run_remote_swe_preflight(config, tasks[0], mode=preflight_mode)
        args.remote_preflight = preflight_mode
        if bool(preflight_info.get("ok")):
            config.sandbox_ensure_runners_before_start = False
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
    baseline_info: dict[str, object] | None = None
    if args.baseline_results:
        baseline_counts, baseline_info = _baseline_counts_from_results(list(args.baseline_results))
        progress.seed_counts(
            baseline_counts,
            label=args.baseline_label or "clean previous results",
        )
        if run_dir is not None:
            (run_dir / "baseline_info.json").write_text(
                json.dumps(baseline_info, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if not args.quiet:
            info("[baseline] " + compact_json(baseline_info, limit=600))
    if resume_info is not None:
        skipped_counts = resume_info.get("skipped_status_counts")
        if isinstance(skipped_counts, dict):
            if baseline_info is not None and not args.quiet:
                info("[baseline] resume skipped counts ignored because --baseline-results was supplied")
            if baseline_info is not None:
                skipped_counts = None
        if isinstance(skipped_counts, dict):
            progress.seed_counts(
                {
                    "pass": int(skipped_counts.get("pass", 0) or 0),
                    "not_pass": int(skipped_counts.get("not_pass", 0) or 0),
                    "bug": int(skipped_counts.get("bug", 0) or 0),
                },
                label=f"skipped results from {args.resume_from_results}",
            )
    progress_lock = threading.Lock()
    result_lock = threading.Lock()
    remote_start_lock = threading.Lock() if config.sandbox_backend == "remote_swe" else None
    completed = 0
    worker_slots = max(1, int(args.parallel or 1))
    network_bug_limit = int(args.stop_after_planner_network_bugs or max(4, worker_slots * 2))
    consecutive_planner_network_bugs = 0
    remote_runner_bug_limit = int(args.stop_after_remote_runner_bugs or 0)
    consecutive_remote_runner_bugs = 0
    remote_sandbox_invalid_limit = int(args.stop_after_remote_sandbox_invalid or 0)
    consecutive_remote_sandbox_invalid = 0
    submitted = 0
    task_iter = iter(enumerate(tasks))
    run_records: list[dict[str, object]] = []

    def submit_next(executor: ThreadPoolExecutor, futures: dict) -> bool:
        nonlocal submitted
        try:
            idx, task = next(task_iter)
        except StopIteration:
            return False
        futures[
            executor.submit(
                _run_one_task,
                task,
                base_config=config,
                scripted=scripted,
                trace_dir=trace_dir,
                progress=progress,
                progress_lock=progress_lock,
                remote_start_lock=remote_start_lock,
                worker_id=(idx % worker_slots) + 1,
                quiet=args.quiet,
            )
        ] = task
        submitted += 1
        return True

    with ThreadPoolExecutor(max_workers=worker_slots) as executor:
        futures: dict = {}
        for _ in range(worker_slots):
            if not submit_next(executor, futures):
                break
        while futures:
            future = next(as_completed(tuple(futures)))
            task = futures.pop(future)
            try:
                record = future.result()
            except Exception as exc:
                record = {"task_id": task.task_id, "status": "bug", "steps": 0, "reason": f"{type(exc).__name__}: {exc}"}
            if args.remote_sandbox_invalid_policy == "continue" and _is_remote_sandbox_invalid(record):
                _mark_remote_sandbox_invalid_bug(record)
            _append_result(results_path, record, result_lock)
            run_records.append(record)
            with progress_lock:
                summary = progress.record(str(record.get("status") or "bug"), task_id=task.task_id, reason=str(record.get("reason") or ""))
            completed += 1
            if args.quiet:
                print(json.dumps(record, sort_keys=True, default=str))
            else:
                info(
                    f"[result] {completed}/{len(tasks)} {record.get('task_id')} "
                    f"status={record.get('status')} steps={record.get('steps')} "
                    f"verified={record.get('verified')} reason={record.get('reason')}"
                )
                if config.console_verbose:
                    info("[result-detail] " + compact_json(record.get("latest_result"), limit=600))
                info(
                    f"[progress] total={summary['total']} pass={summary['pass']} "
                    f"not_pass={summary['not_pass']} bug={summary['bug']} "
                    f"acc={summary['accuracy']:.3f} bug_excl={summary['bug_excluded_accuracy']:.3f}"
                )
            if _is_planner_network_bug(record):
                consecutive_planner_network_bugs += 1
            else:
                consecutive_planner_network_bugs = 0
            if _is_remote_runner_pool_bug(record):
                consecutive_remote_runner_bugs += 1
            else:
                consecutive_remote_runner_bugs = 0
            if args.remote_sandbox_invalid_policy == "stop" and _is_remote_sandbox_invalid(record):
                consecutive_remote_sandbox_invalid += 1
            else:
                consecutive_remote_sandbox_invalid = 0
            if network_bug_limit > 0 and consecutive_planner_network_bugs >= network_bug_limit:
                cancelled = 0
                for pending in futures:
                    if pending is not future and not pending.done() and pending.cancel():
                        cancelled += 1
                stop_record = {
                    "reason": "stopping eval after consecutive planner network bugs",
                    "consecutive_planner_network_bugs": consecutive_planner_network_bugs,
                    "cancelled_queued_tasks": cancelled,
                    "completed": completed,
                    "task_count": len(tasks),
                }
                if run_dir is not None:
                    (run_dir / "stopped_after_planner_network_bugs.json").write_text(
                        json.dumps(stop_record, ensure_ascii=False, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                if not args.quiet:
                    info("[stop] " + compact_json(stop_record, limit=600))
                break
            if remote_runner_bug_limit > 0 and consecutive_remote_runner_bugs >= remote_runner_bug_limit:
                cancelled = 0
                for pending in futures:
                    if pending is not future and not pending.done() and pending.cancel():
                        cancelled += 1
                stop_record = {
                    "reason": "stopping eval after remote_swe runner pool appears unavailable",
                    "consecutive_remote_runner_bugs": consecutive_remote_runner_bugs,
                    "cancelled_queued_tasks": cancelled,
                    "completed": completed,
                    "task_count": len(tasks),
                    "last_task_id": record.get("task_id"),
                    "last_reason": str(record.get("reason") or "")[:1200],
                }
                if run_dir is not None:
                    (run_dir / "stopped_after_remote_runner_bugs.json").write_text(
                        json.dumps(stop_record, ensure_ascii=False, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                if not args.quiet:
                    info("[stop] " + compact_json(stop_record, limit=800))
                break
            if remote_sandbox_invalid_limit > 0 and consecutive_remote_sandbox_invalid >= remote_sandbox_invalid_limit:
                cancelled = 0
                for pending in futures:
                    if pending is not future and not pending.done() and pending.cancel():
                        cancelled += 1
                stop_record = {
                    "reason": "stopping eval after remote_swe sandbox appears invalid",
                    "consecutive_remote_sandbox_invalid": consecutive_remote_sandbox_invalid,
                    "cancelled_queued_tasks": cancelled,
                    "completed": completed,
                    "task_count": len(tasks),
                    "last_task_id": record.get("task_id"),
                    "last_contamination_reasons": record.get("infra_contamination_reasons", []),
                    "last_reason": str(record.get("reason") or "")[:1200],
                }
                if run_dir is not None:
                    (run_dir / "stopped_after_remote_sandbox_invalid.json").write_text(
                        json.dumps(stop_record, ensure_ascii=False, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                if not args.quiet:
                    info("[stop] " + compact_json(stop_record, limit=800))
                break
            submit_next(executor, futures)
    if run_dir is not None:
        _write_infra_contamination_report(run_dir / "infra_contaminated_results.json", run_records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
