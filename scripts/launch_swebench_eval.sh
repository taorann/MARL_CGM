#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: scripts/launch_swebench_eval.sh TASKS_JSONL [eval/supervisor args...]" >&2
  echo "env: GRAPHPLANNER_ENV_FILE=.planner_dashscope.env GRAPHPLANNER_RUN_LABEL=label GRAPHPLANNER_EVAL_MODE=supervisor|parallel GRAPHPLANNER_LAUNCH_DRY_RUN=1" >&2
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
for line in p.read_text(encoding="utf-8").splitlines():
    if line.strip():
        data = json.loads(line)
        print(data.get("task_id") or data.get("instance_id") or p.stem)
        break
else:
    print(p.stem)
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

REMOTE_ROOT="${GRAPHPLANNER_SWEB_REMOTE_ROOT:-/lustre3/chongbin_pkuhpc/chongbin_cls/graphplanner_sif}"
export GRAPHPLANNER_SANDBOX_SIF_DIR="${GRAPHPLANNER_SANDBOX_SIF_DIR:-${GRAPHPLANNER_SWEB_SIF_DIR:-$REMOTE_ROOT/sweb}}"
export GRAPHPLANNER_SANDBOX_QUEUE_ROOT="${GRAPHPLANNER_SANDBOX_QUEUE_ROOT:-$REMOTE_ROOT/gp_queue_sweb}"
export GRAPHPLANNER_SANDBOX_SHARE_ROOT="${GRAPHPLANNER_SANDBOX_SHARE_ROOT:-$REMOTE_ROOT/share_sweb}"
export GRAPHPLANNER_SANDBOX_NUM_RUNNERS="${GRAPHPLANNER_SANDBOX_NUM_RUNNERS:-4}"
export GRAPHPLANNER_SANDBOX_SSH_TARGET="${GRAPHPLANNER_SANDBOX_SSH_TARGET:-chongbin_cls@127.0.0.1}"
if [[ -z "${GP_REMOTE_SWE_SSH_ARGS:-}" && -z "${GRAPHPLANNER_SANDBOX_SSH_ARGS:-}" ]]; then
  if [[ -f /root/.ssh/id_ed25519_login24 ]]; then
    export GP_REMOTE_SWE_SSH_ARGS="-i /root/.ssh/id_ed25519_login24 -p 40022 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=20 -o ConnectionAttempts=2 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
  else
    export GP_REMOTE_SWE_SSH_ARGS="-o BatchMode=yes -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
  fi
fi

export GRAPHPLANNER_MAX_STEPS="${GRAPHPLANNER_MAX_STEPS:-48}"
export GRAPHPLANNER_REMOTE_PREFLIGHT="${GRAPHPLANNER_REMOTE_PREFLIGHT:-cleanup}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
ensure_min_int GRAPHPLANNER_COMMAND_TIMEOUT 1800
ensure_min_int GRAPHPLANNER_PLANNER_TIMEOUT 600
export CGM_HTTP_TIMEOUT="${CGM_HTTP_TIMEOUT:-1200}"
export CGM_DASHSCOPE_TIMEOUT="${CGM_DASHSCOPE_TIMEOUT:-1200}"
export CGM_HTTP_MAX_ATTEMPTS="${CGM_HTTP_MAX_ATTEMPTS:-2}"
export CGM_MAX_PATCH_EDITS="${CGM_MAX_PATCH_EDITS:-12}"
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
LABEL="$(safe_name "${GRAPHPLANNER_RUN_LABEL:-swebench_eval}")"
TASK_PART="$(safe_name "$(first_task_id)")"
STAMP="$(date -u '+%Y-%m-%d_%H-%M-%S_UTC')"
RUN_DIR="${GRAPHPLANNER_RUN_DIR:-$RUNS_ROOT/${TASK_PART}__${LABEL}__${STAMP}}"
mkdir -p "$RUN_DIR"

EXTRA_ARGS=("$@")
if ! has_arg "--max-steps" "${EXTRA_ARGS[@]}"; then
  EXTRA_ARGS+=(--max-steps "$GRAPHPLANNER_MAX_STEPS")
fi
if ! has_arg "--observation-mode" "${EXTRA_ARGS[@]}"; then
  EXTRA_ARGS+=(--observation-mode "${GRAPHPLANNER_OBSERVATION_MODE:-text}")
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
if ! has_arg "--sandbox-backend" "${EXTRA_ARGS[@]}"; then
  EXTRA_ARGS+=(--sandbox-backend remote_swe)
fi
if ! has_arg "--sandbox-num-runners" "${EXTRA_ARGS[@]}"; then
  EXTRA_ARGS+=(--sandbox-num-runners "$GRAPHPLANNER_SANDBOX_NUM_RUNNERS")
fi

MODE="${GRAPHPLANNER_EVAL_MODE:-supervisor}"
cat > "$RUN_DIR/launch.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd '$ROOT_DIR/src'
ENV_FILE='$ENV_FILE'
if [[ -f "\$ENV_FILE" ]]; then
  set -a
  source "\$ENV_FILE"
  set +a
fi
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
export CGM_MAX_PATCH_EDITS='$CGM_MAX_PATCH_EDITS'
export CGM_BACKEND='$CGM_BACKEND_EFFECTIVE'
export CGM_DASHSCOPE_MODEL='$CGM_DASHSCOPE_MODEL'
export CGM_DASHSCOPE_ENABLE_THINKING='$CGM_DASHSCOPE_ENABLE_THINKING'
export CGM_DASHSCOPE_MAX_TOKENS='$CGM_DASHSCOPE_MAX_TOKENS'
export PYTHONUNBUFFERED='$PYTHONUNBUFFERED'
if [[ '$MODE' == 'parallel' ]]; then
  python -u -m graphplanner_agent.cli.eval_parallel \\
    --tasks '$TASKS_PATH' \\
    --parallel '$GRAPHPLANNER_SANDBOX_NUM_RUNNERS' \\
    --run-dir '$RUN_DIR' \\
    --remote-preflight '$GRAPHPLANNER_REMOTE_PREFLIGHT' \\
    ${EXTRA_ARGS[*]@Q}
else
  python -u -m graphplanner_agent.cli.eval_supervisor \\
    --tasks '$TASKS_PATH' \\
    --run-dir '$RUN_DIR' \\
    --parallel '$GRAPHPLANNER_SANDBOX_NUM_RUNNERS' \\
    --max-rounds '${GRAPHPLANNER_SUPERVISOR_MAX_ROUNDS:-3}' \\
    --remote-preflight '$GRAPHPLANNER_REMOTE_PREFLIGHT' \\
    --stop-after-remote-runner-bugs '${GRAPHPLANNER_STOP_AFTER_REMOTE_RUNNER_BUGS:-2}' \\
    --stop-after-remote-sandbox-invalid '${GRAPHPLANNER_STOP_AFTER_REMOTE_SANDBOX_INVALID:-4}' \\
    --remote-sandbox-invalid-policy '${GRAPHPLANNER_REMOTE_SANDBOX_INVALID_POLICY:-continue}' \\
    --stop-after-planner-network-bugs '${GRAPHPLANNER_STOP_AFTER_PLANNER_NETWORK_BUGS:-8}' \\
    ${EXTRA_ARGS[*]@Q}
fi
rc=\$?
echo "\$rc" > '$RUN_DIR/exit_code.txt'
echo "[launch] exit_code=\$rc"
exit "\$rc"
EOF
chmod +x "$RUN_DIR/launch.sh"

python - "$RUN_DIR/launch_metadata.json" "$TASKS_PATH" "$ENV_FILE" "$ROOT_DIR" "$MODE" <<'PY'
import json, os, sys
from pathlib import Path
out, tasks, env_file, root, mode = sys.argv[1:6]
safe_keys = [
    "PLANNER_MODEL", "PLANNER_ENDPOINT", "GRAPHPLANNER_COMMAND_TIMEOUT",
    "GRAPHPLANNER_PLANNER_TIMEOUT", "GRAPHPLANNER_MAX_STEPS",
    "GRAPHPLANNER_REMOTE_PREFLIGHT", "GRAPHPLANNER_SANDBOX_NUM_RUNNERS",
    "GRAPHPLANNER_SANDBOX_SSH_TARGET", "GRAPHPLANNER_SANDBOX_SIF_DIR",
    "GRAPHPLANNER_SANDBOX_QUEUE_ROOT", "GRAPHPLANNER_SANDBOX_SHARE_ROOT",
    "CGM_HTTP_TIMEOUT", "CGM_DASHSCOPE_TIMEOUT", "CGM_HTTP_MAX_ATTEMPTS",
    "CGM_MAX_PATCH_EDITS", "CGM_BACKEND", "GRAPHPLANNER_CGM_BACKEND",
    "CGM_DASHSCOPE_MODEL", "CGM_DASHSCOPE_ENABLE_THINKING",
    "CGM_DASHSCOPE_MAX_TOKENS",
]
Path(out).write_text(json.dumps({
    "root": root,
    "tasks": tasks,
    "env_file": env_file if Path(env_file).exists() else None,
    "mode": mode,
    "env": {key: os.environ.get(key) for key in safe_keys},
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
