#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: scripts/launch_swebench_pro_eval.sh TASKS_JSONL [eval args...]" >&2
  echo "env: GRAPHPLANNER_ENV_FILE=.planner_dashscope.env GRAPHPLANNER_RUN_LABEL=label GRAPHPLANNER_LAUNCH_DRY_RUN=1" >&2
}

if [[ $# -lt 1 ]]; then
  usage
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

ENV_FILE="${GRAPHPLANNER_ENV_FILE:-$ROOT_DIR/.planner_dashscope.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

safe_name() {
  python - "$1" <<'PY'
import re, sys
value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", sys.argv[1].strip()).strip("_")
print(value or "run")
PY
}

first_task_id() {
  python - "$TASKS_PATH" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
text = p.read_text(encoding="utf-8")
if p.suffix.lower() == ".jsonl":
    for line in text.splitlines():
        if line.strip():
            data = json.loads(line)
            break
    else:
        data = {}
else:
    loaded = json.loads(text)
    data = loaded[0] if isinstance(loaded, list) and loaded else loaded
print(data.get("task_id") or data.get("instance_id") or p.stem)
PY
}

has_arg() {
  local wanted="$1"
  shift
  for arg in "$@"; do
    [[ "$arg" == "$wanted" || "$arg" == "$wanted="* ]] && return 0
  done
  return 1
}

ensure_min_int() {
  local name="$1"
  local minimum="$2"
  local current="${!name:-}"
  if [[ ! "$current" =~ ^[0-9]+$ || "$current" -lt "$minimum" ]]; then
    if [[ "${GRAPHPLANNER_ALLOW_SHORT_TIMEOUT:-0}" != "1" ]]; then
      printf -v "$name" '%s' "$minimum"
      export "$name"
    fi
  fi
}

PRO_REMOTE_ROOT="${GRAPHPLANNER_PRO_REMOTE_ROOT:-/lustre3/chongbin_pkuhpc/chongbin_cls/graphplanner_sif}"
export GRAPHPLANNER_PRO_REMOTE_ROOT="$PRO_REMOTE_ROOT"
export GRAPHPLANNER_PRO_SIF_DIR="${GRAPHPLANNER_PRO_SIF_DIR:-${GRAPHPLANNER_SANDBOX_SIF_DIR:-$PRO_REMOTE_ROOT/sweb_pro_probe}}"
export GRAPHPLANNER_SANDBOX_SIF_DIR="${GRAPHPLANNER_SANDBOX_SIF_DIR:-$GRAPHPLANNER_PRO_SIF_DIR}"
export GRAPHPLANNER_SANDBOX_QUEUE_ROOT="${GRAPHPLANNER_SANDBOX_QUEUE_ROOT:-$PRO_REMOTE_ROOT/gp_queue}"
export GRAPHPLANNER_SANDBOX_SHARE_ROOT="${GRAPHPLANNER_SANDBOX_SHARE_ROOT:-$PRO_REMOTE_ROOT/share}"
export GRAPHPLANNER_SANDBOX_NUM_RUNNERS="${GRAPHPLANNER_SANDBOX_NUM_RUNNERS:-4}"
export GRAPHPLANNER_SANDBOX_SSH_TARGET="${GRAPHPLANNER_SANDBOX_SSH_TARGET:-chongbin_cls@127.0.0.1}"
if [[ -z "${GP_REMOTE_SWE_SSH_ARGS:-}" && -z "${GRAPHPLANNER_SANDBOX_SSH_ARGS:-}" ]]; then
  if [[ -f /root/.ssh/id_ed25519_login24 ]]; then
    export GP_REMOTE_SWE_SSH_ARGS="-i /root/.ssh/id_ed25519_login24 -p 40022 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=20 -o ConnectionAttempts=2 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
  else
    export GP_REMOTE_SWE_SSH_ARGS="-o BatchMode=yes -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
  fi
fi
export GRAPHPLANNER_REMOTE_PREFLIGHT="${GRAPHPLANNER_REMOTE_PREFLIGHT:-full}"
export GRAPHPLANNER_MAX_STEPS="${GRAPHPLANNER_MAX_STEPS:-48}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

ensure_min_int GRAPHPLANNER_COMMAND_TIMEOUT 1800
ensure_min_int GRAPHPLANNER_PLANNER_TIMEOUT 450
export CGM_HTTP_TIMEOUT="${CGM_HTTP_TIMEOUT:-1200}"
export CGM_DASHSCOPE_TIMEOUT="${CGM_DASHSCOPE_TIMEOUT:-1200}"
export CGM_HTTP_MAX_ATTEMPTS="${CGM_HTTP_MAX_ATTEMPTS:-1}"
export CGM_DASHSCOPE_MODEL="${CGM_DASHSCOPE_MODEL:-qwen3-235b-a22b-thinking-2507}"
export CGM_DASHSCOPE_ENABLE_THINKING="${CGM_DASHSCOPE_ENABLE_THINKING:-1}"
export CGM_DASHSCOPE_MAX_TOKENS="${CGM_DASHSCOPE_MAX_TOKENS:-1536}"

if [[ "${GRAPHPLANNER_SYNC_REMOTE_RUNTIME:-1}" != "0" ]]; then
  "$ROOT_DIR/scripts/sync_remote_runtime_code.sh"
fi

if [[ "${GRAPHPLANNER_SYNC_REMOTE_GRAPH:-1}" != "0" ]]; then
  "$ROOT_DIR/scripts/sync_remote_graph_code.sh" "$GRAPHPLANNER_SANDBOX_SHARE_ROOT"
fi

RUNS_ROOT="${GRAPHPLANNER_RUNS_ROOT:-$ROOT_DIR/runs/tmp}"
if [[ "$RUNS_ROOT" != /* ]]; then
  RUNS_ROOT="$ROOT_DIR/$RUNS_ROOT"
fi
LABEL="$(safe_name "${GRAPHPLANNER_RUN_LABEL:-pro_eval}")"
TASK_PART="$(safe_name "$(first_task_id)")"
STAMP="$(date -u '+%Y-%m-%d_%H-%M-%S_UTC')"
RUN_DIR="${GRAPHPLANNER_RUN_DIR:-$RUNS_ROOT/${TASK_PART}__${LABEL}__${STAMP}}"
mkdir -p "$RUN_DIR"

EXTRA_ARGS=("$@")
if ! has_arg "--max-steps" "${EXTRA_ARGS[@]}"; then
  EXTRA_ARGS+=(--max-steps "$GRAPHPLANNER_MAX_STEPS")
fi
CGM_BACKEND_EFFECTIVE="${GRAPHPLANNER_CGM_BACKEND:-${CGM_BACKEND:-dashscope}}"
export CGM_BACKEND="$CGM_BACKEND_EFFECTIVE"
if ! has_arg "--cgm-backend" "${EXTRA_ARGS[@]}"; then
  EXTRA_ARGS+=(--cgm-backend "$CGM_BACKEND_EFFECTIVE")
fi
CGM_ENDPOINT_EFFECTIVE="${GRAPHPLANNER_CGM_ENDPOINT:-${CGM_ENDPOINT:-}}"
if [[ -z "$CGM_ENDPOINT_EFFECTIVE" && "$CGM_BACKEND_EFFECTIVE" == "http" ]]; then
  CGM_ENDPOINT_EFFECTIVE="http://127.0.0.1:30003/generate"
fi
if [[ -n "$CGM_ENDPOINT_EFFECTIVE" ]] && ! has_arg "--cgm-endpoint" "${EXTRA_ARGS[@]}"; then
  EXTRA_ARGS+=(--cgm-endpoint "$CGM_ENDPOINT_EFFECTIVE")
fi

cat > "$RUN_DIR/launch.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
ENV_FILE='$ENV_FILE'
if [[ -f "\$ENV_FILE" ]]; then
  set -a
  source "\$ENV_FILE"
  set +a
fi
export GRAPHPLANNER_PRO_REMOTE_ROOT='$GRAPHPLANNER_PRO_REMOTE_ROOT'
export GRAPHPLANNER_PRO_SIF_DIR='$GRAPHPLANNER_PRO_SIF_DIR'
export GRAPHPLANNER_SANDBOX_SIF_DIR='$GRAPHPLANNER_SANDBOX_SIF_DIR'
export GRAPHPLANNER_SANDBOX_QUEUE_ROOT='$GRAPHPLANNER_SANDBOX_QUEUE_ROOT'
export GRAPHPLANNER_SANDBOX_SHARE_ROOT='$GRAPHPLANNER_SANDBOX_SHARE_ROOT'
export GRAPHPLANNER_SANDBOX_NUM_RUNNERS='$GRAPHPLANNER_SANDBOX_NUM_RUNNERS'
export GRAPHPLANNER_SANDBOX_SSH_TARGET='$GRAPHPLANNER_SANDBOX_SSH_TARGET'
export GP_REMOTE_SWE_SSH_ARGS='$GP_REMOTE_SWE_SSH_ARGS'
export GRAPHPLANNER_REMOTE_PREFLIGHT='$GRAPHPLANNER_REMOTE_PREFLIGHT'
export GRAPHPLANNER_MAX_STEPS='$GRAPHPLANNER_MAX_STEPS'
export GRAPHPLANNER_COMMAND_TIMEOUT='$GRAPHPLANNER_COMMAND_TIMEOUT'
export GRAPHPLANNER_PLANNER_TIMEOUT='$GRAPHPLANNER_PLANNER_TIMEOUT'
export CGM_HTTP_TIMEOUT='$CGM_HTTP_TIMEOUT'
export CGM_DASHSCOPE_TIMEOUT='$CGM_DASHSCOPE_TIMEOUT'
export CGM_HTTP_MAX_ATTEMPTS='$CGM_HTTP_MAX_ATTEMPTS'
export CGM_BACKEND='$CGM_BACKEND_EFFECTIVE'
export CGM_DASHSCOPE_MODEL='$CGM_DASHSCOPE_MODEL'
export CGM_DASHSCOPE_ENABLE_THINKING='$CGM_DASHSCOPE_ENABLE_THINKING'
export CGM_DASHSCOPE_MAX_TOKENS='$CGM_DASHSCOPE_MAX_TOKENS'
export PYTHONUNBUFFERED='$PYTHONUNBUFFERED'
cd '$ROOT_DIR'
exec scripts/run_swebench_pro_eval.sh '$TASKS_PATH' --run-dir '$RUN_DIR' ${EXTRA_ARGS[*]@Q}
EOF
chmod +x "$RUN_DIR/launch.sh"

python - "$RUN_DIR/launch_metadata.json" "$TASKS_PATH" "$ENV_FILE" "$ROOT_DIR" <<'PY'
import json, os, sys
from pathlib import Path
out, tasks, env_file, root = map(Path, sys.argv[1:5])
safe_env = {
    key: os.environ.get(key)
    for key in [
        "PLANNER_MODEL",
        "PLANNER_ENDPOINT",
        "GRAPHPLANNER_COMMAND_TIMEOUT",
        "GRAPHPLANNER_PLANNER_TIMEOUT",
        "GRAPHPLANNER_MAX_STEPS",
        "GRAPHPLANNER_REMOTE_PREFLIGHT",
        "GRAPHPLANNER_SANDBOX_NUM_RUNNERS",
        "GRAPHPLANNER_SANDBOX_SSH_TARGET",
        "GRAPHPLANNER_SANDBOX_SIF_DIR",
        "GRAPHPLANNER_SANDBOX_QUEUE_ROOT",
        "GRAPHPLANNER_SANDBOX_SHARE_ROOT",
        "CGM_HTTP_TIMEOUT",
        "CGM_DASHSCOPE_TIMEOUT",
        "CGM_HTTP_MAX_ATTEMPTS",
        "CGM_BACKEND",
        "GRAPHPLANNER_CGM_BACKEND",
        "CGM_DASHSCOPE_MODEL",
        "CGM_DASHSCOPE_ENABLE_THINKING",
        "CGM_DASHSCOPE_MAX_TOKENS",
    ]
}
out.write_text(json.dumps({
    "root": str(root),
    "tasks": str(tasks),
    "env_file": str(env_file) if env_file.exists() else None,
    "env": safe_env,
}, indent=2, sort_keys=True), encoding="utf-8")
PY

echo "run_dir: $RUN_DIR"
echo "log: $RUN_DIR/run.log"
echo "pid: $RUN_DIR/run.pid"
echo "progress: $RUN_DIR/progress.md"
echo "results: $RUN_DIR/results.jsonl"
echo "traces: $RUN_DIR/traces"

if [[ "${GRAPHPLANNER_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
  echo "dry_run: launch script written but not started"
  exit 0
fi

setsid bash "$RUN_DIR/launch.sh" > "$RUN_DIR/run.log" 2>&1 < /dev/null &
PID="$!"
echo "$PID" > "$RUN_DIR/run.pid"
echo "started_pid: $PID"
