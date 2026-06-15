#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  scripts/remote_sif_download.sh start
  scripts/remote_sif_download.sh status REMOTE_JOB_DIR
  scripts/remote_sif_download.sh stop REMOTE_JOB_DIR

env:
  GRAPHPLANNER_SWEB_DATASET=/path/to/swebench/test.jsonl
  GRAPHPLANNER_PRO_DATASET=/path/to/pro_image_only.jsonl
  GRAPHPLANNER_SIF_REMOTE_ROOT=/lustre3/.../graphplanner_sif
  GRAPHPLANNER_EXISTING_SWEB_SIF_DIR=/optional/existing/sweb/sif/dir
  GRAPHPLANNER_DOCKER_MIRROR=docker.1panel.live
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="${1:-}"
shift || true

REMOTE_ROOT="${GRAPHPLANNER_SIF_REMOTE_ROOT:-/lustre3/chongbin_pkuhpc/chongbin_cls/graphplanner_sif}"
REMOTE_REPO="${GRAPHPLANNER_SANDBOX_REMOTE_REPO:-${GP_SANDBOX_REMOTE_REPO:-/appsnew/home/chongbin_pkuhpc/chongbin_cls/MARL_CGM}}"
EXISTING_SWEB_SIF_DIR="${GRAPHPLANNER_EXISTING_SWEB_SIF_DIR:-}"
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

remote_py() {
  ssh $SSH_ARGS "$SSH_TARGET" "python - '$@'"
}

choose_mirror() {
  if [[ -n "${GRAPHPLANNER_DOCKER_MIRROR:-}" ]]; then
    echo "$GRAPHPLANNER_DOCKER_MIRROR"
    return
  fi
  local candidates=(
    "docker.1panel.live"
    "docker.m.daocloud.io"
    "docker.1ms.run"
    "dockerproxy.com"
  )
  local mirror line
  for mirror in "${candidates[@]}"; do
    line="$(ssh $SSH_ARGS "$SSH_TARGET" "timeout 15 bash -lc 'curl -k -sSI --connect-timeout 8 https://$mirror/v2/ | grep -m1 \"^HTTP/\"'" 2>/dev/null || true)"
    if [[ "$line" == HTTP/* ]]; then
      echo "$mirror"
      return
    fi
  done
  echo "docker.1panel.live"
}

status_job() {
  local job="${1:?status requires REMOTE_JOB_DIR}"
  ssh $SSH_ARGS "$SSH_TARGET" "JOB='$job' python - <<'PY'
import json, os, pathlib, signal, subprocess
job = pathlib.Path(os.environ['JOB'])
print(f'job={job}')
for name in ['swebench_prepare.pid', 'pro_prepare.pid']:
    path = job / name
    pid = path.read_text().strip() if path.exists() else ''
    alive = False
    if pid.isdigit():
        try:
            os.kill(int(pid), 0)
            alive = True
        except OSError:
            alive = False
    print(f'{name}={pid or \"missing\"} alive={alive}')
print('--- processes ---')
user = os.environ.get('USER') or ''
proc = subprocess.run(['ps', '-fu', user], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for line in proc.stdout.splitlines():
    if 'remote_sif_download.sh status' in line or ('JOB=' in line and 'python -' in line):
        continue
    if any(token in line for token in ['prepare_sif_from_dataset.py', 'singularity build', 'apptainer build']):
        print(line)
print('--- counts ---')
for label, path in [('sweb', pathlib.Path('$REMOTE_ROOT')/'sweb'), ('pro', pathlib.Path('$REMOTE_ROOT')/'sweb_pro_probe')]:
    count = len(list(path.glob('*.sif'))) if path.exists() else 0
    print(f'{label}_sif_count={count}')
print('--- manifests ---')
from collections import Counter
for name in ['swebench_manifest.jsonl', 'pro_manifest.jsonl']:
    path = job / name
    c = Counter()
    if path.exists():
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            if line.strip():
                try:
                    c[json.loads(line).get('status', '?')] += 1
                except Exception:
                    c['malformed'] += 1
    print(f'{name}: {dict(c)} total={sum(c.values())}')
print('--- log tails ---')
for name in ['swebench_prepare.log', 'swebench_prepare_1panel.log', 'pro_prepare.log', 'pro_prepare_1panel.log']:
    path = job / name
    if path.exists():
        print(f'--- {path} ---')
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()[-12:]
        print('\\n'.join(lines))
PY"
}

stop_job() {
  local job="${1:?stop requires REMOTE_JOB_DIR}"
  ssh $SSH_ARGS "$SSH_TARGET" "JOB='$job' python - <<'PY'
import os, pathlib, signal, time
job = pathlib.Path(os.environ['JOB'])
pids = []
for name in ['swebench_prepare.pid', 'pro_prepare.pid']:
    path = job / name
    if path.exists():
        text = path.read_text().strip()
        if text.isdigit():
            pids.append(int(text))
for sig in [signal.SIGTERM, signal.SIGKILL]:
    for pid in pids:
        try:
            os.killpg(pid, sig)
            print(f'sent {sig.name} to process group {pid}')
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            print(f'cannot signal {pid}: {exc}')
    time.sleep(3)
print('stopped')
PY"
}

start_job() {
  local sweb_dataset="${GRAPHPLANNER_SWEB_DATASET:-$ROOT_DIR/datasets/swebench/test.jsonl}"
  local pro_dataset="${GRAPHPLANNER_PRO_DATASET:-$ROOT_DIR/datasets/swebench_pro/image_only.jsonl}"
  if [[ ! -f "$sweb_dataset" ]]; then
    echo "missing SWE-bench dataset: $sweb_dataset" >&2
    exit 2
  fi
  if [[ ! -f "$pro_dataset" ]]; then
    echo "missing Pro image-only dataset: $pro_dataset" >&2
    exit 2
  fi

  local stamp job mirror
  stamp="$(date -u '+%Y-%m-%d_%H-%M-%S_UTC')"
  job="$REMOTE_ROOT/download_jobs/full_$stamp"
  mirror="$(choose_mirror)"

  ssh $SSH_ARGS "$SSH_TARGET" "mkdir -p '$job'"
  scp $SCP_ARGS "$sweb_dataset" "$SSH_TARGET:$job/swebench_full_test.jsonl" >/dev/null
  scp $SCP_ARGS "$pro_dataset" "$SSH_TARGET:$job/swebench_pro_all_image_only.jsonl" >/dev/null
  scp $SCP_ARGS "$ROOT_DIR/scripts/prepare_sif_from_dataset.py" "$SSH_TARGET:$job/prepare_sif_from_dataset.py" >/dev/null

  ssh $SSH_ARGS "$SSH_TARGET" "JOB='$job' ROOT='$REMOTE_ROOT' REMOTE_REPO='$REMOTE_REPO' MIRROR='$mirror' EXISTING_SWEB_SIF_DIR='$EXISTING_SWEB_SIF_DIR' python - <<'PY'
import os, pathlib, textwrap
job = pathlib.Path(os.environ['JOB'])
root = pathlib.Path(os.environ['ROOT'])
remote_repo = pathlib.Path(os.environ['REMOTE_REPO'])
mirror = os.environ['MIRROR']
for sub in ['sweb', 'sweb_pro_probe', 'tmp_sweb', 'tmp_pro', 'cache_sweb', 'cache_pro']:
    (root / sub).mkdir(parents=True, exist_ok=True)
existing_raw = os.environ.get('EXISTING_SWEB_SIF_DIR', '').strip()
existing_sweb = pathlib.Path(existing_raw) if existing_raw else None
seeded = 0
if existing_sweb and existing_sweb.exists():
    target_sweb = root / 'sweb'
    for path in existing_sweb.glob('*.sif'):
        target = target_sweb / path.name
        if not target.exists() and not target.is_symlink():
            target.symlink_to(path)
            seeded += 1
script = job / 'prepare_sif_from_dataset.py'
for name, dataset, sif, tmp, cache, log, manifest in [
    ('swebench', 'swebench_full_test.jsonl', 'sweb', 'tmp_sweb', 'cache_sweb', 'swebench_prepare.log', 'swebench_manifest.jsonl'),
    ('pro', 'swebench_pro_all_image_only.jsonl', 'sweb_pro_probe', 'tmp_pro', 'cache_pro', 'pro_prepare.log', 'pro_manifest.jsonl'),
]:
    run = job / f'run_{name}.sh'
    run.write_text(textwrap.dedent(f'''\
        #!/usr/bin/env bash
        set -euo pipefail
        cd '{remote_repo}'
        env DKMR0='{mirror}' DKMR1='docker.m.daocloud.io' python '{script}' \\
          --dataset '{job / dataset}' \\
          --sif-dir '{root / sif}' \\
          --apptainer-bin singularity \\
          --apptainer-tmpdir '{root / tmp}' \\
          --apptainer-cachedir '{root / cache}' \\
          --mirror '{mirror}' \\
          --mirror docker.m.daocloud.io \\
          --continue-on-error \\
          --manifest '{job / manifest}'
        '''), encoding='utf-8')
    run.chmod(0o755)
print(f'{job} seeded_sweb_symlinks={seeded}')
PY
cd '$job'
nohup setsid bash ./run_swebench.sh > swebench_prepare.log 2>&1 < /dev/null & echo \$! > swebench_prepare.pid
nohup setsid bash ./run_pro.sh > pro_prepare.log 2>&1 < /dev/null & echo \$! > pro_prepare.pid
sleep 2
cat > metadata.json <<EOF
{\"job_dir\":\"$job\",\"mirror\":\"$mirror\",\"remote_root\":\"$REMOTE_ROOT\"}
EOF"

  echo "job_dir: $job"
  echo "mirror: $mirror"
  echo "swebench_log: $job/swebench_prepare.log"
  echo "pro_log: $job/pro_prepare.log"
  echo "swebench_manifest: $job/swebench_manifest.jsonl"
  echo "pro_manifest: $job/pro_manifest.jsonl"
  status_job "$job"
}

case "$CMD" in
  start)
    start_job
    ;;
  status)
    status_job "${1:-${GRAPHPLANNER_SIF_JOB_DIR:-}}"
    ;;
  stop)
    stop_job "${1:-${GRAPHPLANNER_SIF_JOB_DIR:-}}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
