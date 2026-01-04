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

# If using remote_swe backend and required fields are missing, inject defaults.
NEED_REMOTE_SWE=0
HAS_SSH_TARGET=0
HAS_REMOTE_REPO=0
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  arg="${args[$i]}"
  next_arg="${args[$((i + 1))]:-}"

  case "$arg" in
    --sandbox-backend=*)
      backend_value="${arg#--sandbox-backend=}"
      [[ "$backend_value" == "remote_swe" ]] && NEED_REMOTE_SWE=1
      ;;
    --sandbox-backend)
      [[ "$next_arg" == "remote_swe" ]] && NEED_REMOTE_SWE=1
      ;;
  esac

  case "$arg" in
    --sandbox-ssh-target|--sandbox-ssh-target=*)
      HAS_SSH_TARGET=1
      ;;
    --sandbox-remote-repo|--sandbox-remote-repo=*)
      HAS_REMOTE_REPO=1
      ;;
  esac
done

if [[ $NEED_REMOTE_SWE -eq 1 ]]; then
  if [[ $HAS_SSH_TARGET -eq 0 ]]; then
    set -- --sandbox-ssh-target "${GP_SANDBOX_SSH_TARGET:-$SSH_TARGET}" "$@"
  fi
  if [[ $HAS_REMOTE_REPO -eq 0 ]]; then
    set -- --sandbox-remote-repo "${GP_SANDBOX_REMOTE_REPO:-$REMOTE_REPO}" "$@"
  fi
fi

set -x
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
TOKENIZERS_PARALLELISM="false" \
python "${ROOT_DIR}/scripts/eval_graph_planner_engine.py" \
  "$@"
