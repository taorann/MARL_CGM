#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?usage: monitor_eval_run.sh RUN_DIR [INTERVAL_SECONDS]}"
INTERVAL="${2:-1800}"
LOG_PATH="$RUN_DIR/monitor.log"

while true; do
  {
    echo
    echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ====="
    if [[ -f "$RUN_DIR/run.pid" ]]; then
      PID="$(cat "$RUN_DIR/run.pid")"
      echo "pid=$PID"
      ps -p "$PID" -o pid,stat,etime,cmd || true
    else
      echo "pid_file_missing"
    fi
    echo "--- summary ---"
    if [[ -f "$RUN_DIR/summary.json" ]]; then
      python - "$RUN_DIR/summary.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
data = json.loads(p.read_text())
print(json.dumps({
    "clean_record_count": data.get("clean_record_count"),
    "remaining_count": data.get("remaining_count"),
    "counts": data.get("counts"),
    "accuracy": data.get("accuracy"),
    "bug_excluded_accuracy": data.get("bug_excluded_accuracy"),
    "rounds": data.get("rounds", [])[-3:],
}, ensure_ascii=False, indent=2, sort_keys=True))
PY
    else
      echo "summary_missing"
    fi
    echo "--- single eval ---"
    if [[ -f "$RUN_DIR/results.jsonl" ]]; then
      echo "results_lines=$(wc -l < "$RUN_DIR/results.jsonl")"
      python - "$RUN_DIR/results.jsonl" <<'PY'
import json, sys
from pathlib import Path
counts = {"pass": 0, "not_pass": 0, "bug": 0}
p = Path(sys.argv[1])
for line in p.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    status = rec.get("status")
    if status not in counts:
        status = "bug"
    counts[status] += 1
print(json.dumps({"counts": counts}, sort_keys=True))
PY
    else
      echo "single_results_missing"
    fi
    if [[ -f "$RUN_DIR/progress.md" ]]; then
      sed -n '1,40p' "$RUN_DIR/progress.md"
    else
      echo "single_progress_missing"
    fi
    echo "--- latest round ---"
    latest_round="$(find "$RUN_DIR" -maxdepth 1 -type d -name 'round_*' | sort | tail -n 1 || true)"
    if [[ -n "$latest_round" ]]; then
      echo "round_dir=$latest_round"
      if [[ -f "$latest_round/results.jsonl" ]]; then
        echo "results_lines=$(wc -l < "$latest_round/results.jsonl")"
        python - "$latest_round/results.jsonl" <<'PY'
import json, sys
from pathlib import Path
counts = {"pass": 0, "not_pass": 0, "bug": 0}
infra = 0
p = Path(sys.argv[1])
for line in p.read_text().splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    status = rec.get("status")
    if status not in counts:
        status = "bug"
    counts[status] += 1
    if rec.get("infra_contaminated"):
        infra += 1
print(json.dumps({"counts": counts, "infra_contaminated": infra}, sort_keys=True))
PY
      fi
      if [[ -f "$latest_round/progress.md" ]]; then
        sed -n '1,24p' "$latest_round/progress.md"
      fi
    else
      echo "round_missing"
    fi
  } >> "$LOG_PATH" 2>&1

  if [[ -f "$RUN_DIR/run.pid" ]]; then
    PID="$(cat "$RUN_DIR/run.pid")"
    if ! ps -p "$PID" >/dev/null 2>&1; then
      echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') monitor exiting: process ended =====" >> "$LOG_PATH"
      exit 0
    fi
  fi
  sleep "$INTERVAL"
done
