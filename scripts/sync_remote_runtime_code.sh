#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_RUNTIME_ROOT="${GRAPHPLANNER_LOCAL_RUNTIME_ROOT:-$ROOT_DIR/remote_runtime}"
REMOTE_REPO="${GRAPHPLANNER_SANDBOX_REMOTE_REPO:-${GP_SANDBOX_REMOTE_REPO:-/appsnew/home/chongbin_pkuhpc/chongbin_cls/MARL_CGM}}"
SSH_TARGET="${GRAPHPLANNER_SANDBOX_SSH_TARGET:-${GP_SANDBOX_SSH_TARGET:-chongbin_cls@127.0.0.1}}"
REMOTE_BACKUP_ROOT="${GRAPHPLANNER_REMOTE_RUNTIME_BACKUP_ROOT:-/lustre3/chongbin_pkuhpc/chongbin_cls/graphplanner_sif/remote_runtime_backups}"
QUEUE_ROOT="${GRAPHPLANNER_SANDBOX_QUEUE_ROOT:-${GP_QUEUE_ROOT:-${QUEUE_ROOT:-}}}"

if [[ -z "${GP_REMOTE_SWE_SSH_ARGS:-}" && -z "${GRAPHPLANNER_SANDBOX_SSH_ARGS:-}" ]]; then
  if [[ -f /root/.ssh/id_ed25519_login24 ]]; then
    export GP_REMOTE_SWE_SSH_ARGS="-i /root/.ssh/id_ed25519_login24 -p 40022 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=20 -o ConnectionAttempts=2 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
  else
    export GP_REMOTE_SWE_SSH_ARGS="-o BatchMode=yes -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
  fi
fi
SSH_ARGS="${GRAPHPLANNER_SANDBOX_SSH_ARGS:-${GP_REMOTE_SWE_SSH_ARGS:-}}"

if [[ -z "${GRAPHPLANNER_REMOTE_SCP_ARGS:-}" && -z "${GP_REMOTE_SWE_SCP_ARGS:-}" ]]; then
  if [[ -f /root/.ssh/id_ed25519_login24 ]]; then
    export GP_REMOTE_SWE_SCP_ARGS="-i /root/.ssh/id_ed25519_login24 -P 40022 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=20 -o ConnectionAttempts=2"
  else
    export GP_REMOTE_SWE_SCP_ARGS="-o BatchMode=yes -o StrictHostKeyChecking=no"
  fi
fi
SCP_ARGS="${GRAPHPLANNER_REMOTE_SCP_ARGS:-${GP_REMOTE_SWE_SCP_ARGS:-}}"

for required in hpc hpc_jobs graph_planner/runtime graph_planner/tools graph_planner/__init__.py; do
  if [[ ! -e "$LOCAL_RUNTIME_ROOT/$required" ]]; then
    echo "missing local runtime asset: $LOCAL_RUNTIME_ROOT/$required" >&2
    exit 2
  fi
done

TMP_DIR="$ROOT_DIR/runs/tmp"
mkdir -p "$TMP_DIR"
BUNDLE="$TMP_DIR/graphplanner_remote_runtime_$$.tgz"
tar -C "$LOCAL_RUNTIME_ROOT" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  --exclude='runs' \
  --exclude='datasets' \
  -czf "$BUNDLE" \
  hpc hpc_jobs graph_planner/__init__.py graph_planner/runtime graph_planner/tools

REMOTE_BUNDLE="/tmp/graphplanner_remote_runtime_${USER:-user}_$$.tgz"
scp $SCP_ARGS "$BUNDLE" "$SSH_TARGET:$REMOTE_BUNDLE" >/dev/null

REMOTE_SCRIPT='
set -euo pipefail
remote_repo="$1"
bundle="$2"
backup_root="$3"
queue_root="${4:-}"
stamp="$(date -u +%Y-%m-%d_%H-%M-%S_UTC)"
mkdir -p "$remote_repo" "$backup_root"
cd "$remote_repo"
backup_dir="$backup_root/runtime_${stamp}"
mkdir -p "$backup_dir"
for item in hpc hpc_jobs graph_planner; do
  if [[ -e "$item" ]]; then
    cp -a "$item" "$backup_dir/$item"
    rm -rf "$item"
  fi
done
tar -xzf "$bundle" -C "$remote_repo"
find hpc hpc_jobs graph_planner -type d -name __pycache__ -prune -exec rm -rf {} +
if [[ -n "$queue_root" && -d "$queue_root" ]]; then
  find "$queue_root" -path "*/in/*.json" -mmin +60 -print -delete || true
fi
python - <<PY
from pathlib import Path
checks = {
    "swe_proxy_existing_runner_route": "choose_runner_for_existing" in Path("hpc/swe_proxy.py").read_text(encoding="utf-8"),
    "ensure_stale_heartbeat": "STALE_HEARTBEAT" in Path("hpc/ensure_runners.py").read_text(encoding="utf-8"),
    "runner_template_queue_root": "QUEUE_ROOT" in Path("hpc_jobs/run_runner_cn_nl.tpl.sh").read_text(encoding="utf-8"),
    "graph_builder": Path("graph_planner/tools/swe_build_graph.py").is_file(),
}
print(checks)
missing = [name for name, ok in checks.items() if not ok]
if missing:
    raise SystemExit("remote runtime sync verification failed: " + ", ".join(missing))
PY
rm -f "$bundle"
echo "synced runtime into $remote_repo"
echo "backup: $backup_dir"
'

ssh $SSH_ARGS "$SSH_TARGET" "bash -s -- '$REMOTE_REPO' '$REMOTE_BUNDLE' '$REMOTE_BACKUP_ROOT' '$QUEUE_ROOT'" <<<"$REMOTE_SCRIPT"
rm -f "$BUNDLE"
