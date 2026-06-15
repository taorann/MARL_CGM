#!/usr/bin/env bash
set -euo pipefail
cd /root/private_data/MARL_CGM-main/agent_rebuild/src
source ../.planner_dashscope.env
export CGM_BACKEND=http
export CGM_ENDPOINT="http://127.0.0.1:30003/generate"
export GRAPHPLANNER_SANDBOX_BACKEND=remote_swe
export GRAPHPLANNER_OBSERVATION_MODE=text
export GRAPHPLANNER_SANDBOX_NUM_RUNNERS=4
export GP_NUM_RUNNERS=4
exec python -m graphplanner_agent.cli.eval_supervisor \
  --tasks /root/private_data/MARL_CGM-main/agent_rebuild/runs/tmp/swebench_sample100_available_enriched_seed20260601.jsonl \
  --parallel 4 \
  --run-dir "/root/private_data/MARL_CGM-main/agent_rebuild/runs/tmp/swebench_supervised_continue_on_sandbox_bug_2026-06-02_17-07-06_UTC" \
  --baseline-results /root/private_data/MARL_CGM-main/agent_rebuild/runs/tmp/swebench_sample100_clean_baseline_after_retry23_2026-06-02.jsonl \
  --max-rounds 3 \
  --recovery-sleep 3 \
  --cgm-backend http \
  --cgm-endpoint http://127.0.0.1:30003/generate \
  --sandbox-backend remote_swe \
  --sandbox-num-runners 4 \
  --observation-mode text \
  --max-steps 48 \
  --remote-sandbox-invalid-policy continue \
  --stop-after-remote-runner-bugs 1 \
  --stop-after-remote-sandbox-invalid 1
