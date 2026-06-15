#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: scripts/run_swebench_pro_eval.sh TASKS_JSONL [eval args...]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASKS_PATH="$1"
shift

if [[ "$TASKS_PATH" != /* ]]; then
  TASKS_PATH="$ROOT_DIR/$TASKS_PATH"
fi
if [[ ! -f "$TASKS_PATH" ]]; then
  echo "task file not found: $TASKS_PATH" >&2
  exit 2
fi

PRO_REMOTE_ROOT="${GRAPHPLANNER_PRO_REMOTE_ROOT:-/lustre3/chongbin_pkuhpc/chongbin_cls/graphplanner_sif}"
SIF_DIR="${GRAPHPLANNER_PRO_SIF_DIR:-${GRAPHPLANNER_SANDBOX_SIF_DIR:-$PRO_REMOTE_ROOT/sweb_pro_probe}}"
RUNS_ROOT="${GRAPHPLANNER_RUNS_ROOT:-$ROOT_DIR/runs/tmp}"
COMMAND_TIMEOUT="${GRAPHPLANNER_COMMAND_TIMEOUT:-1800}"

if [[ "$RUNS_ROOT" != /* ]]; then
  RUNS_ROOT="$ROOT_DIR/$RUNS_ROOT"
fi
if [[ "$SIF_DIR" != /* ]]; then
  SIF_DIR="$ROOT_DIR/$SIF_DIR"
fi

export GRAPHPLANNER_SANDBOX_QUEUE_ROOT="${GRAPHPLANNER_SANDBOX_QUEUE_ROOT:-$PRO_REMOTE_ROOT/gp_queue}"
export GRAPHPLANNER_SANDBOX_SHARE_ROOT="${GRAPHPLANNER_SANDBOX_SHARE_ROOT:-$PRO_REMOTE_ROOT/share}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

cd "$ROOT_DIR/src"
GRAPHPLANNER_COMMAND_TIMEOUT="$COMMAND_TIMEOUT" \
python -m graphplanner_agent.cli.eval \
  --tasks "$TASKS_PATH" \
  --runs-root "$RUNS_ROOT" \
  --sandbox-backend remote_swe \
  --sandbox-workdir /app \
  --sandbox-sif-dir "$SIF_DIR" \
  --remote-preflight "${GRAPHPLANNER_REMOTE_PREFLIGHT:-cleanup}" \
  "$@"
