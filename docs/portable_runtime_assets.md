# Portable Runtime Assets

This repository is intended to be the source of truth for the train-free
GraphPlanner agent. A fresh clone should not need sibling directories from the
old `MARL_CGM-main` checkout.

## In-Repo Assets

- `src/graphplanner_agent/`: local agent implementation.
- `remote_runtime/`: minimal code copied to the remote repo before each eval.
  - `remote_runtime/hpc/`
  - `remote_runtime/hpc_jobs/`
  - `remote_runtime/graph_planner/runtime/`
  - `remote_runtime/graph_planner/tools/`
- `scripts/sync_remote_runtime_code.sh`: pushes `remote_runtime/` to the
  remote home repo and removes stale remote runtime code.
- `scripts/sync_remote_graph_code.sh`: pushes the remote `graph_planner` copy
  into lustre share roots used inside Apptainer containers.
- `scripts/prepare_sif_from_dataset.py`: helper for building SIF images from
  dataset metadata.
- `datasets/swebench/`: lightweight SWE-bench metadata used by local sampling
  and SIF preparation scripts.
- `datasets/swebench_pro/image_only.jsonl`: lightweight Pro image list used by
  the SIF preparation script.
- `.planner_dashscope.env.example`: secret-free local configuration template.

## External Assets

These are intentionally not committed:

- `.planner_dashscope.env`: contains API keys.
- SSH private keys and local tunnel configuration.
- SIF images and Apptainer build caches under lustre3.
- Eval output under `runs/tmp/` and graph caches.
- Large benchmark archives beyond the lightweight metadata under
  `datasets/swebench/`.

## Bootstrap Checklist On A New Machine

1. Clone this repository.
2. Copy `.planner_dashscope.env.example` to `.planner_dashscope.env` and fill
   in `PLANNER_API_KEY`.
3. Ensure SSH access to the remote login node works.
4. Ensure lustre3 paths in `.planner_dashscope.env` are mounted/reachable.
5. Run:

```bash
set -a
source .planner_dashscope.env
set +a

scripts/sync_remote_runtime_code.sh
scripts/sync_remote_graph_code.sh
```

6. Run a stack check:

```bash
PYTHONPATH=src python -m graphplanner_agent.cli.check_stack \
  --skip-planner \
  --cgm-backend dashscope \
  --sandbox-backend remote_swe \
  --remote-swe-full-smoke \
  --remote-swe-image 'slimshetty/swebench-verified:sweb.eval.x86_64.astropy__astropy-12907'
```
