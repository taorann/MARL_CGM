from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphplanner_agent.datasets.swebench_sample import prepare_swebench_sample


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare an enriched SWE-bench sample for GraphPlanner eval.")
    parser.add_argument("--source", type=Path, required=True, help="Raw SWE-bench/R2E JSONL task file.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL with target eval scripts/selectors.")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--keep-order", action="store_true", help="Use the first N tasks instead of a random sample.")
    parser.add_argument("--ssh-target", default="", help="Remote host used to read r2e_ds_json when it is not local.")
    parser.add_argument("--ssh-args", default="", help="Extra ssh args, e.g. '-i key -p 40022'.")
    parser.add_argument("--require-remote-sif", action="store_true", help="Sample only tasks whose .sif exists remotely.")
    parser.add_argument("--remote-sif-dir", default="", help="Remote directory containing SWE-bench .sif images.")
    args = parser.parse_args()

    stats = prepare_swebench_sample(
        args.source,
        args.output,
        sample_size=args.sample_size,
        seed=args.seed,
        ssh_target=args.ssh_target,
        ssh_args=args.ssh_args,
        require_remote_sif=args.require_remote_sif,
        remote_sif_dir=args.remote_sif_dir,
        keep_order=args.keep_order,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
