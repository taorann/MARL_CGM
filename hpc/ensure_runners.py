#!/usr/bin/env python
"""
Ensure there are at least N gp_runner_* Slurm jobs for GraphPlanner remote_swe.

Key fixes:
- Avoid duplicate submissions via a simple filesystem lock under QUEUE_ROOT.
- Use a robust USER detection (USER env may be empty under non-interactive SSH).
- Detect and scancel duplicate jobs with the same runner id (e.g., two gp_runner_3).
- Write per-runner stdout/stderr to QUEUE_ROOT/logs via sbatch -o/-e.

Usage (on login24):
    cd $HOME/MARL_CGM
    python hpc/ensure_runners.py --target 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_DEF_NAME_PREFIX = "gp_runner_"


@dataclass(frozen=True)
class SlurmJob:
    job_id: str
    name: str
    state: str  # RUNNING / PENDING / etc.


def _whoami() -> str:
    u = (os.environ.get("USER") or "").strip()
    if u:
        return u
    try:
        return subprocess.check_output(["whoami"], text=True).strip()
    except Exception:
        return ""


def _queue_root() -> Path:
    return Path(os.environ.get("QUEUE_ROOT", str(Path.home() / "gp_queue"))).expanduser().resolve()


def _lock_path(queue_root: Path) -> Path:
    # directory lock works well over NFS; no fcntl required
    return queue_root / ".ensure_runners.lock"


class _DirLock:
    def __init__(self, path: Path, *, timeout_sec: float = 120.0, poll_sec: float = 0.5) -> None:
        self.path = path
        self.timeout_sec = timeout_sec
        self.poll_sec = poll_sec

    def __enter__(self) -> None:
        t0 = time.time()
        while True:
            try:
                self.path.mkdir(parents=True, exist_ok=False)
                meta = {
                    "pid": os.getpid(),
                    "host": os.uname().nodename,
                    "time": time.time(),
                }
                (self.path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
                return
            except FileExistsError:
                if time.time() - t0 > self.timeout_sec:
                    raise SystemExit(f"[ensure_runners] lock busy too long: {self.path}")
                time.sleep(self.poll_sec)

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            for p in self.path.glob("*"):
                try:
                    p.unlink()
                except Exception:
                    pass
            self.path.rmdir()
        except Exception:
            pass


def _squeue_jobs(user: str) -> List[SlurmJob]:
    # %T gives long state names
    out = subprocess.check_output(["squeue", "-u", user, "-h", "-o", "%i %j %T"], text=True)
    jobs: List[SlurmJob] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 2:
            continue
        job_id = parts[0]
        name = parts[1]
        state = parts[2] if len(parts) >= 3 else ""
        jobs.append(SlurmJob(job_id=job_id, name=name, state=state))
    return jobs


def _parse_runner_jobs(jobs: List[SlurmJob], name_prefix: str) -> Dict[int, List[SlurmJob]]:
    by_rid: Dict[int, List[SlurmJob]] = {}
    for j in jobs:
        if not j.name.startswith(name_prefix):
            continue
        suffix = j.name[len(name_prefix):]
        try:
            rid = int(suffix)
        except ValueError:
            continue
        by_rid.setdefault(rid, []).append(j)
    return by_rid


def _state_rank(state: str) -> int:
    s = (state or "").upper()
    # prefer RUNNING over PENDING over everything else
    if s == "RUNNING" or s == "R":
        return 0
    if s == "PENDING" or s == "PD":
        return 1
    return 2


def _dedupe_runner_jobs(by_rid: Dict[int, List[SlurmJob]]) -> Dict[int, SlurmJob]:
    """If multiple jobs exist for the same RID, keep the best one and scancel the rest."""
    keep: Dict[int, SlurmJob] = {}
    for rid, lst in by_rid.items():
        if not lst:
            continue
        # choose best by state rank then job_id (smallest = oldest)
        lst_sorted = sorted(lst, key=lambda j: (_state_rank(j.state), int(j.job_id) if j.job_id.isdigit() else 10**18))
        winner = lst_sorted[0]
        keep[rid] = winner
        losers = lst_sorted[1:]
        if losers:
            loser_ids = [j.job_id for j in losers]
            print(f"[ensure_runners] DEDUPE: rid={rid} keep={winner.job_id} ({winner.state}) cancel={loser_ids}")
            subprocess.run(["scancel", *loser_ids], check=False)
    return keep


def _submit_runner(
    runner_id: int,
    runner_script: Path,
    *,
    name_prefix: str,
    partition: Optional[str],
    qos: Optional[str],
    queue_root: Path,
) -> None:
    runner_script = runner_script.expanduser().resolve()
    if not runner_script.exists():
        raise SystemExit(f"[ensure_runners] runner script not found: {runner_script}")

    job_name = f"{name_prefix}{runner_id}"
    logs_dir = (queue_root / "logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(logs_dir / f"{job_name}_%j.out")
    err_path = str(logs_dir / f"{job_name}_%j.err")

    cmd: List[str] = ["sbatch", "-J", job_name, "-o", out_path, "-e", err_path, f"--export=ALL,RUNNER_ID={runner_id}"]

    if partition:
        cmd.extend(["-p", partition])
    if qos:
        cmd.extend(["--qos", qos])

    cmd.append(str(runner_script))

    print(f"[ensure_runners] SUBMIT rid={runner_id} cmd={' '.join(cmd)}")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise SystemExit(f"[ensure_runners] sbatch failed for rid={runner_id}: {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensure there are at least N gp_runner_* jobs.")
    parser.add_argument("--target", type=int, required=True, help="Target number of gp_runner_* jobs (minimum).")
    parser.add_argument(
        "--runner-script",
        type=str,
        default="hpc_jobs/run_runner_cn_nl.tpl.sh",
        help="Path to the Slurm script used to launch a runner.",
    )
    parser.add_argument(
        "--name-prefix",
        type=str,
        default=_DEF_NAME_PREFIX,
        help="Job name prefix used to identify runner jobs.",
    )
    parser.add_argument("--partition", type=str, default=None, help="Override Slurm partition (-p).")
    parser.add_argument("--qos", type=str, default=None, help="Override Slurm QoS (--qos).")
    args = parser.parse_args()

    target = int(args.target)
    if target <= 0:
        print("[ensure_runners] target <= 0, nothing to do.")
        return

    user = _whoami()
    if not user:
        raise SystemExit("[ensure_runners] cannot determine USER (USER env empty and whoami failed)")

    queue_root = _queue_root()
    queue_root.mkdir(parents=True, exist_ok=True)

    runner_script = Path(args.runner_script)
    name_prefix = str(args.name_prefix)
    partition = args.partition or os.environ.get("RUNNER_PARTITION") or None
    qos = args.qos or os.environ.get("RUNNER_QOS") or None

    with _DirLock(_lock_path(queue_root)):
        jobs = _squeue_jobs(user)
        by_rid = _parse_runner_jobs(jobs, name_prefix)
        keep = _dedupe_runner_jobs(by_rid)
        existing_ids = set(keep.keys())

        candidates = [rid for rid in range(target) if rid not in existing_ids]
        if not candidates:
            print(f"[ensure_runners] OK: have {len(existing_ids)} runner(s) already (target={target}).")
            return

        print(f"[ensure_runners] existing={sorted(existing_ids)} missing={candidates} (target={target})")
        for rid in candidates:
            _submit_runner(
                rid,
                runner_script,
                name_prefix=name_prefix,
                partition=partition,
                qos=qos,
                queue_root=queue_root,
            )

    print("[ensure_runners] Done.")


if __name__ == "__main__":
    main()
