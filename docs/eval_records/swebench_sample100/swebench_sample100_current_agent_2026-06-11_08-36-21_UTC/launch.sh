#!/usr/bin/env bash
set -uo pipefail
cd /root/private_data/MARL_CGM-main/agent_rebuild/src
set -a
source ../.planner_dashscope.env
set +a
export CGM_BACKEND=http
export CGM_ENDPOINT=http://127.0.0.1:30003/generate
export CGM_HTTP_MAX_ATTEMPTS="${CGM_HTTP_MAX_ATTEMPTS:-2}"
export CGM_MAX_PATCH_EDITS="${CGM_MAX_PATCH_EDITS:-12}"
export GRAPHPLANNER_MAX_STEPS="${GRAPHPLANNER_MAX_STEPS:-48}"
export GRAPHPLANNER_COMMAND_TIMEOUT=1800
export GRAPHPLANNER_PLANNER_TIMEOUT="${GRAPHPLANNER_PLANNER_TIMEOUT:-600}"
export GRAPHPLANNER_SANDBOX_BACKEND=remote_swe
export GRAPHPLANNER_SANDBOX_SSH_TARGET=chongbin_cls@127.0.0.1
export GP_REMOTE_SWE_SSH_ARGS='-i /root/.ssh/id_ed25519_login24 -p 40022 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=20 -o ConnectionAttempts=2 -o ServerAliveInterval=30 -o ServerAliveCountMax=4'
export GRAPHPLANNER_SANDBOX_SIF_DIR=/appsnew/home/chongbin_pkuhpc/chongbin_cls/sif/sweb
export GRAPHPLANNER_SANDBOX_QUEUE_ROOT=/lustre3/chongbin_pkuhpc/chongbin_cls/graphplanner_sif/gp_queue_sweb
export GRAPHPLANNER_SANDBOX_SHARE_ROOT=/lustre3/chongbin_pkuhpc/chongbin_cls/graphplanner_sif/share_sweb
export GRAPHPLANNER_SANDBOX_NUM_RUNNERS=4
export GRAPHPLANNER_REMOTE_GRAPH_TIMEOUT=1800
python -u -m graphplanner_agent.cli.eval_supervisor \
  --tasks ../runs/tmp/swebench_sample100_available_enriched_seed20260601.jsonl \
  --run-dir /root/private_data/MARL_CGM-main/agent_rebuild/runs/tmp/swebench_sample100_current_agent_2026-06-11_08-36-21_UTC \
  --parallel 4 \
  --max-rounds 3 \
  --max-steps "${GRAPHPLANNER_MAX_STEPS}" \
  --observation-mode text \
  --cgm-backend http \
  --cgm-endpoint "$CGM_ENDPOINT" \
  --sandbox-backend remote_swe \
  --sandbox-num-runners 4 \
  --remote-preflight cleanup \
  --stop-after-remote-runner-bugs 2 \
  --stop-after-remote-sandbox-invalid 4 \
  --remote-sandbox-invalid-policy continue \
  --stop-after-planner-network-bugs 8
rc=$?
echo "$rc" > /root/private_data/MARL_CGM-main/agent_rebuild/runs/tmp/swebench_sample100_current_agent_2026-06-11_08-36-21_UTC/exit_code.txt
echo "[launch] exit_code=$rc"
exit "$rc"
