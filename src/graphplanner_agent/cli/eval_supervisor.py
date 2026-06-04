from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable

from graphplanner_agent.telemetry.console import compact_json, info

from .eval_parallel import _baseline_counts_from_results, _safe_name


def _task_id(record: dict[str, object]) -> str:
    return str(record.get("task_id") or record.get("instance_id") or "").strip()


def _load_jsonl_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            records.append(item)
    return records


def _write_jsonl_records(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) for record in records]
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def _clean_result_records(paths: Iterable[Path]) -> dict[str, dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    for path in paths:
        for record in _load_jsonl_records(path):
            task_id = _task_id(record)
            if not task_id:
                continue
            if record.get("infra_contaminated"):
                continue
            if str(record.get("status") or "") not in {"pass", "not_pass"}:
                continue
            latest[task_id] = record
    return latest


def _remaining_task_records(tasks_path: Path, clean_records: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    clean_ids = set(clean_records)
    remaining: list[dict[str, object]] = []
    for record in _load_jsonl_records(tasks_path):
        task_id = _task_id(record)
        if task_id and task_id in clean_ids:
            continue
        remaining.append(record)
    return remaining


def _run_dir(runs_root: Path, label: str | None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")
    return runs_root / f"{_safe_name(label or 'supervised_eval')}__{stamp}"


def _stop_marker(round_dir: Path) -> str | None:
    markers = [
        "stopped_after_remote_sandbox_invalid.json",
        "stopped_after_remote_runner_bugs.json",
        "stopped_after_planner_network_bugs.json",
    ]
    for name in markers:
        path = round_dir / name
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return name
            return str(payload.get("reason") or name)
    return None


def _status_counts(records: Iterable[dict[str, object]]) -> dict[str, int]:
    counts = {"pass": 0, "not_pass": 0, "bug": 0}
    for record in records:
        status = str(record.get("status") or "bug")
        if status not in counts:
            status = "bug"
        counts[status] += 1
    return counts


def _write_supervisor_summary(
    path: Path,
    *,
    rounds: list[dict[str, object]],
    clean_baseline_path: Path,
    clean_records: dict[str, dict[str, object]],
    remaining_count: int,
    max_rounds: int,
) -> None:
    counts, baseline_info = _baseline_counts_from_results([clean_baseline_path])
    total = sum(counts.values())
    valid = total - counts.get("bug", 0)
    summary = {
        "rounds": rounds,
        "max_rounds": max_rounds,
        "clean_baseline_path": str(clean_baseline_path),
        "clean_record_count": len(clean_records),
        "remaining_count": remaining_count,
        "counts": counts,
        "accuracy": counts.get("pass", 0) / total if total else 0.0,
        "bug_excluded_accuracy": counts.get("pass", 0) / valid if valid else 0.0,
        "baseline_info": baseline_info,
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run eval_parallel in recovery rounds until all tasks have clean results.")
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/tmp"))
    parser.add_argument("--run-label", default="supervised_eval")
    parser.add_argument("--baseline-results", type=Path, action="append", default=[])
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--recovery-sleep", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--observation-mode", choices=["json", "text"])
    parser.add_argument("--cgm-backend", choices=["mock", "http"])
    parser.add_argument("--cgm-endpoint")
    parser.add_argument("--sandbox-backend", choices=["local", "remote_swe"])
    parser.add_argument("--sandbox-num-runners", type=int)
    parser.add_argument("--remote-preflight", choices=["auto", "off", "cleanup", "full"], default="auto")
    parser.add_argument("--stop-after-remote-runner-bugs", type=int, default=1)
    parser.add_argument("--stop-after-remote-sandbox-invalid", type=int, default=1)
    parser.add_argument("--remote-sandbox-invalid-policy", choices=["continue", "stop"], default="continue")
    parser.add_argument("--stop-after-planner-network-bugs", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args, extra_args = parser.parse_known_args()

    run_dir = args.run_dir or _run_dir(args.runs_root, args.run_label)
    run_dir.mkdir(parents=True, exist_ok=True)
    supervisor_log = run_dir / "supervisor.log"
    rounds: list[dict[str, object]] = []
    clean_sources = list(args.baseline_results)
    clean_records = _clean_result_records(clean_sources)
    clean_baseline_path = run_dir / "clean_baseline.jsonl"
    _write_jsonl_records(clean_baseline_path, clean_records.values())

    if not args.quiet:
        info(f"[supervisor] dir={run_dir}")
        info(f"[supervisor] initial_clean={len(clean_records)} tasks={args.tasks}")

    for round_idx in range(1, max(1, int(args.max_rounds)) + 1):
        remaining = _remaining_task_records(args.tasks, clean_records)
        _write_supervisor_summary(
            run_dir / "summary.json",
            rounds=rounds,
            clean_baseline_path=clean_baseline_path,
            clean_records=clean_records,
            remaining_count=len(remaining),
            max_rounds=args.max_rounds,
        )
        if not remaining:
            if not args.quiet:
                info("[supervisor] complete: no remaining tasks")
            return 0

        round_dir = run_dir / f"round_{round_idx:02d}"
        round_tasks = run_dir / f"round_{round_idx:02d}_tasks.jsonl"
        _write_jsonl_records(round_tasks, remaining)
        cmd = [
            sys.executable,
            "-m",
            "graphplanner_agent.cli.eval_parallel",
            "--tasks",
            str(round_tasks),
            "--parallel",
            str(args.parallel),
            "--run-dir",
            str(round_dir),
            "--baseline-results",
            str(clean_baseline_path),
            "--baseline-label",
            f"supervisor_clean_baseline_round_{round_idx}",
            "--remote-preflight",
            args.remote_preflight,
            "--stop-after-remote-runner-bugs",
            str(args.stop_after_remote_runner_bugs),
            "--stop-after-remote-sandbox-invalid",
            str(args.stop_after_remote_sandbox_invalid),
            "--remote-sandbox-invalid-policy",
            args.remote_sandbox_invalid_policy,
        ]
        if args.stop_after_planner_network_bugs:
            cmd.extend(["--stop-after-planner-network-bugs", str(args.stop_after_planner_network_bugs)])
        if args.max_steps:
            cmd.extend(["--max-steps", str(args.max_steps)])
        if args.observation_mode:
            cmd.extend(["--observation-mode", args.observation_mode])
        if args.cgm_backend:
            cmd.extend(["--cgm-backend", args.cgm_backend])
        if args.cgm_endpoint:
            cmd.extend(["--cgm-endpoint", args.cgm_endpoint])
        if args.sandbox_backend:
            cmd.extend(["--sandbox-backend", args.sandbox_backend])
        if args.sandbox_num_runners:
            cmd.extend(["--sandbox-num-runners", str(args.sandbox_num_runners)])
        cmd.extend(extra_args)

        if not args.quiet:
            info(f"[supervisor] round={round_idx} remaining={len(remaining)} cmd={' '.join(cmd)}")
        with supervisor_log.open("a", encoding="utf-8") as log:
            log.write(f"\n## round {round_idx}\n")
            log.write(" ".join(cmd) + "\n")
            log.flush()
            started = time.monotonic()
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
            elapsed = time.monotonic() - started

        round_results = round_dir / "results.jsonl"
        round_clean = _clean_result_records([round_results])
        clean_records.update(round_clean)
        _write_jsonl_records(clean_baseline_path, clean_records.values())
        round_records = _load_jsonl_records(round_results)
        stop_reason = _stop_marker(round_dir)
        round_summary = {
            "round": round_idx,
            "returncode": proc.returncode,
            "elapsed": round(elapsed, 3),
            "round_dir": str(round_dir),
            "input_task_count": len(remaining),
            "result_count": len(round_records),
            "clean_added_count": len(round_clean),
            "status_counts": _status_counts(round_records),
            "stop_reason": stop_reason,
        }
        rounds.append(round_summary)
        _write_supervisor_summary(
            run_dir / "summary.json",
            rounds=rounds,
            clean_baseline_path=clean_baseline_path,
            clean_records=clean_records,
            remaining_count=len(_remaining_task_records(args.tasks, clean_records)),
            max_rounds=args.max_rounds,
        )
        if not args.quiet:
            info("[supervisor] " + compact_json(round_summary, limit=800))
        if proc.returncode != 0 and not stop_reason:
            return proc.returncode
        if args.recovery_sleep > 0:
            time.sleep(args.recovery_sleep)

    remaining = _remaining_task_records(args.tasks, clean_records)
    if not args.quiet:
        info(f"[supervisor] stopped after max rounds; remaining={len(remaining)}")
    return 0 if not remaining else 2


if __name__ == "__main__":
    raise SystemExit(main())
