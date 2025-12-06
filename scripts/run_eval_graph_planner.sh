#!/usr/bin/env bash

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "run_eval_graph_planner.sh must be executed with bash. Try 'bash $0' instead." >&2
  exit 1
fi

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CONFIG="${ROOT_DIR}/configs/eval/graph_planner_eval_defaults.yaml"

# ======== remote_swe backend configuration ========
SSH_TARGET="chongbin_cls@login24"
REMOTE_REPO="/appsnew/home/chongbin_pkuhpc/chongbin_cls/MARL_CGM"
# ==================================================

CONFIG_FLAG_PRESENT=0
for arg in "$@"; do
  if [[ "$arg" == --config || "$arg" == --config=* ]]; then
    CONFIG_FLAG_PRESENT=1
    break
  fi
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    CONFIG_FLAG_PRESENT=1
    break
  fi
done

if [[ $CONFIG_FLAG_PRESENT -eq 0 ]]; then
  set -- --config "${GRAPH_PLANNER_EVAL_CONFIG:-$DEFAULT_CONFIG}" "$@"
fi

PYTHONPATH="${PYTHONPATH:-${ROOT_DIR}}" \
TOKENIZERS_PARALLELISM="false" \
python "${ROOT_DIR}/scripts/eval_graph_planner_engine.py" \
  "$@" \
  --sandbox-backend remote_swe \
  --sandbox-ssh-target "${SSH_TARGET}" \
  --sandbox-remote-repo "${REMOTE_REPO}"
