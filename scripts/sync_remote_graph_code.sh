#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REMOTE_REPO="${GRAPHPLANNER_SANDBOX_REMOTE_REPO:-${GP_SANDBOX_REMOTE_REPO:-/appsnew/home/chongbin_pkuhpc/chongbin_cls/MARL_CGM}}"
SSH_TARGET="${GRAPHPLANNER_SANDBOX_SSH_TARGET:-${GP_SANDBOX_SSH_TARGET:-chongbin_cls@127.0.0.1}}"

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

if [[ $# -gt 0 ]]; then
  SHARE_ROOTS=("$@")
elif [[ -n "${GRAPHPLANNER_SANDBOX_SHARE_ROOT:-}" ]]; then
  SHARE_ROOTS=("$GRAPHPLANNER_SANDBOX_SHARE_ROOT")
else
  SHARE_ROOTS=(
    "/lustre3/chongbin_pkuhpc/chongbin_cls/graphplanner_sif/share_sweb"
    "/lustre3/chongbin_pkuhpc/chongbin_cls/graphplanner_sif/share"
    "/lustre3/chongbin_pkuhpc/chongbin_cls/graphplanner_sif/share_pro"
  )
fi

python - "$ROOT_DIR/runs/tmp/sync_remote_graph_code_payload.json" "$REMOTE_REPO" "${SHARE_ROOTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
payload = {"remote_repo": sys.argv[2], "share_roots": sys.argv[3:]}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload), encoding="utf-8")
print(out)
PY

PAYLOAD="$ROOT_DIR/runs/tmp/sync_remote_graph_code_payload.json"
REMOTE_PY='
import json, pathlib, shutil, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
src = pathlib.Path(payload["remote_repo"]) / "graph_planner"
if not src.is_dir():
    raise SystemExit(f"missing remote graph_planner package: {src}")
for raw in payload["share_roots"]:
    share = pathlib.Path(raw)
    dst_parent = share / "MARL_CGM"
    dst = dst_parent / "graph_planner"
    dst_parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    probe = dst / "tools" / "swe_build_graph.py"
    if not probe.is_file():
        raise SystemExit(f"sync failed, missing {probe}")
    print(f"synced {src} -> {dst}")
'

REMOTE_PAYLOAD="/tmp/graphplanner_sync_graph_payload_${USER:-user}_$$.json"
scp $SCP_ARGS "$PAYLOAD" "$SSH_TARGET:$REMOTE_PAYLOAD" >/dev/null
ssh $SSH_ARGS "$SSH_TARGET" "python - '$REMOTE_PAYLOAD' <<'PY'
$REMOTE_PY
PY
rm -f '$REMOTE_PAYLOAD'"
