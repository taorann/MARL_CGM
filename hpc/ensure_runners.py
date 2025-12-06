#!/usr/bin/env python
"""
Ensure there are at least N gp_runner_* Slurm jobs running the runner script.

Usage (on login24):
    cd $HOME/MARL_CGM
    python hpc/ensure_runners.py --target 4
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Set


_DEF_NAME_PREFIX = "gp_runner_"


def _parse_existing_runner_ids(name_prefix: str = _DEF_NAME_PREFIX) -> Set[int]:
    """Parse current runner IDs from squeue output."""
    try:
        out = subprocess.check_output(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %j"],
            text=True,
        )
    except Exception as exc:
        raise SystemExit(f"[ensure_runners] Failed to call squeue: {exc}") from exc

    ids: Set[int] = set()
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        _job_id, job_name = parts
        if not job_name.startswith(name_prefix):
            continue
        suffix = job_name[len(name_prefix) :]
        try:
            rid = int(suffix)
        except ValueError:
            continue
        ids.add(rid)
    return ids


def _submit_runner(
    runner_id: int,
    runner_script: Path,
    name_prefix: str = _DEF_NAME_PREFIX,
    partition: str | None = None,
    qos: str | None = None,
) -> None:
    """Submit a runner job with the given runner ID."""
    runner_script = runner_script.expanduser().resolve()
    if not runner_script.exists():
        raise SystemExit(f"[ensure_runners] runner script not found: {runner_script}")

    job_name = f"{name_prefix}{runner_id}"

    cmd: list[str] = ["sbatch"]

    # If a partition is provided, override the script's default #SBATCH partition.
    if partition:
        cmd.extend(["-p", partition])

    # If a QoS is provided, override the script's default #SBATCH QoS.
    if qos:
        cmd.extend(["--qos", qos])

    cmd.extend(
        [
            "-J",
            job_name,
            f"--export=ALL,RUNNER_ID={runner_id}",
            str(runner_script),
        ]
    )

    print(f"[ensure_runners] Submitting runner RID={runner_id}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip()
        raise SystemExit(f"[ensure_runners] sbatch failed for RID={runner_id}: {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensure there are at least N gp_runner_* jobs.")
    parser.add_argument(
        "--target", type=int, required=True, help="Target number of gp_runner_* jobs (minimum)."
    )
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
    parser.add_argument(
        "--partition",
        type=str,
        default=None,
        help=(
            "Override Slurm partition (-p) for runner jobs. "
            "If not set, use the partition specified in the runner script "
            "(currently '#SBATCH -p cn_nl')."
        ),
    )
    parser.add_argument(
        "--qos",
        type=str,
        default=None,
        help=(
            "Override Slurm QOS (--qos=...) for runner jobs. "
            "If not set, use the QOS specified in the runner script."
        ),
    )
    args = parser.parse_args()

    target = int(args.target)
    if target <= 0:
        print("[ensure_runners] target <= 0, nothing to do.")
        return

    runner_script = Path(args.runner_script)
    name_prefix = str(args.name_prefix)

    # Partition precedence: CLI flag > RUNNER_PARTITION env var > script default.
    partition = args.partition or os.environ.get("RUNNER_PARTITION")

    # QoS precedence: CLI flag > RUNNER_QOS env var > script default.
    qos = args.qos or os.environ.get("RUNNER_QOS")

    existing_ids = _parse_existing_runner_ids(name_prefix=name_prefix)
    current = len(existing_ids)
    print(f"[ensure_runners] Existing runner count = {current}, target = {target}")

    if current >= target:
        print("[ensure_runners] Enough runners are already running; nothing to do.")
        return

    missing = target - current
    candidates = []
    rid = 0
    while len(candidates) < missing:
        if rid not in existing_ids:
            candidates.append(rid)
        rid += 1

    for rid in candidates:
        _submit_runner(
            rid,
            runner_script,
            name_prefix=name_prefix,
            partition=partition,
            qos=qos,
        )

    print("[ensure_runners] Done submitting missing runners.")


if __name__ == "__main__":
    main()
