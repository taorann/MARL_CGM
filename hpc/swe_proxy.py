#!/usr/bin/env python
"""swe_proxy.py

A thin JSON stdin/stdout proxy for GraphPlanner's remote_swe sandbox backend.

It forwards requests to ApptainerQueueRuntime (file-queue IPC) which dispatches
work to long-running runners on compute nodes. The runner executes commands
inside a .sif image via apptainer/singularity.

This proxy exists because the evaluation process runs on a login node (or a
different host) and needs to communicate with Apptainer runners via a shared
file-queue root.

Request formats (stdin JSON):

1) Legacy one-shot exec (no "op"):
   {
     "run_id": "...",
     "image": "<docker image string>",
     "cmd": "<shell command>",
     "timeout": 900,
     "cwd": "/repo",
     "env": {"K": "V"}
   }

2) Instance-mode ops (preferred):
   {
     "run_id": "...",
     "op": "start" | "exec" | "stop" | "build_repo_graph" | "build_graph",
     "image": "<docker image string>",
     "cmd": "<shell command>",               # required for op=exec
     "timeout": 900,
     "cwd": "/repo",
     "env": {"K": "V"},
     "repo": "/repo"                         # optional for build_* (defaults to /repo)
   }

Output (stdout JSON):
  {"ok": bool, "returncode": int, "stdout": str, "stderr": str, "runtime_sec": float, "error": str | null}
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Mapping

from graph_planner.runtime.apptainer_queue_runtime import ApptainerQueueRuntime


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _dbg(msg: str) -> None:
    if os.environ.get("DEBUG") or os.environ.get("EBUG"):
        ts = _now()
        print(f"[swe_proxy {ts}] {msg}", file=sys.stderr, flush=True)


def _exec_result_to_dict(res: Any, *, max_stdout_bytes: Optional[int] = None) -> Dict[str, Any]:
    # ApptainerQueueRuntime returns ExecResult (dataclass-like)
    stdout = getattr(res, "stdout", "") or ""
    stderr = getattr(res, "stderr", "") or ""
    if max_stdout_bytes is not None:
        stdout = stdout[:max_stdout_bytes]
        stderr = stderr[:max_stdout_bytes]
    return {
        "ok": bool(getattr(res, "ok", True)),
        "returncode": int(getattr(res, "returncode", 0) or 0),
        "stdout": stdout,
        "stderr": stderr,
        "runtime_sec": float(getattr(res, "runtime_sec", 0.0) or 0.0),
        "error": getattr(res, "error", None),
    }


def _get_env_dict(env: Any) -> Dict[str, str]:
    if not env:
        return {}
    if isinstance(env, dict):
        return {str(k): str(v) for k, v in env.items()}
    return {}


def _resolve_cwd(req_cwd: Optional[str]) -> Path:
    if not req_cwd:
        return Path("/repo")
    # keep it simple: container-side cwd string
    return Path(str(req_cwd))


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        print(json.dumps({"ok": False, "error": "empty stdin"}, ensure_ascii=False))
        return 1

    try:
        payload = json.loads(raw)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"invalid json: {e}"}, ensure_ascii=False))
        return 1

    run_id = str(payload.get("run_id") or "__default__")
    op = str(payload.get("op") or "").strip().lower()  # empty => legacy one-shot exec

    image = payload.get("image")
    cmd = payload.get("cmd")
    timeout = float(payload.get("timeout") or 600.0)
    env = _get_env_dict(payload.get("env"))
    cwd = _resolve_cwd(payload.get("cwd"))
    repo = str(payload.get("repo") or "/repo")

    # Allow callers to raise stdout cap (repo_graph can be large)
    max_stdout_bytes = payload.get("max_stdout_bytes")
    try:
        max_stdout_bytes = int(max_stdout_bytes) if max_stdout_bytes is not None else None
    except Exception:
        max_stdout_bytes = None

    queue_root = Path(os.environ.get("GP_QUEUE_ROOT", os.environ.get("QUEUE_ROOT", "gp_queue")))
    sif_dir = Path(os.environ.get("GP_SIF_DIR", os.environ.get("SIF_DIR", "sif")))
    num_runners = int(os.environ.get("GP_NUM_RUNNERS", os.environ.get("NUM_RUNNERS", "1")))
    default_timeout_sec = float(os.environ.get("GP_DEFAULT_TIMEOUT_SEC", "600"))

    aq = ApptainerQueueRuntime(
        queue_root=queue_root,
        sif_dir=sif_dir,
        num_runners=num_runners,
        default_timeout_sec=default_timeout_sec,
        max_stdout_bytes=(max_stdout_bytes if max_stdout_bytes is not None else int(os.environ.get("GP_MAX_STDOUT_BYTES", "20000000"))),
    )

    _dbg(f"recv op={op!r} run_id={run_id!r} image={image!r} cwd={str(cwd)!r} timeout={timeout} queue_root={str(queue_root)!r} sif_dir={str(sif_dir)!r} num_runners={num_runners}")

    try:
        # ---------- legacy one-shot exec ----------
        if not op:
            if not image or not cmd:
                print(json.dumps({"ok": False, "error": "image and cmd are required for legacy exec"}, ensure_ascii=False))
                return 1
            res = aq.exec(
                run_id=run_id,
                docker_image=str(image),
                cmd=["bash", "-lc", str(cmd)],
                cwd=cwd,
                env=env,
                timeout_sec=timeout,
            )
            print(json.dumps(_exec_result_to_dict(res, max_stdout_bytes=max_stdout_bytes), ensure_ascii=False))
            return 0

        # ---------- instance mode ----------
        if op == "start":
            if not image:
                print(json.dumps({"ok": False, "error": "image is required for op=start"}, ensure_ascii=False))
                return 1
            res = aq.start_instance(run_id=run_id, docker_image=str(image), cwd=cwd, env=env, timeout_sec=timeout)
            print(json.dumps(_exec_result_to_dict(res, max_stdout_bytes=max_stdout_bytes), ensure_ascii=False))
            return 0

        if op == "exec":
            if not image or not cmd:
                print(json.dumps({"ok": False, "error": "image and cmd are required for op=exec"}, ensure_ascii=False))
                return 1
            res = aq.exec_in_instance(
                run_id=run_id,
                docker_image=str(image),
                cmd=["bash", "-lc", str(cmd)],
                cwd=cwd,
                env=env,
                timeout_sec=timeout,
            )
            print(json.dumps(_exec_result_to_dict(res, max_stdout_bytes=max_stdout_bytes), ensure_ascii=False))
            return 0

        if op == "stop":
            # image is optional for stop; keep for compatibility
            # stop_instance does not need image/sif mapping; it only stops the named instance.
            res = aq.stop_instance(run_id=run_id, cwd=cwd, env=env, timeout_sec=timeout)
            print(json.dumps(_exec_result_to_dict(res, max_stdout_bytes=max_stdout_bytes), ensure_ascii=False))
            return 0

        if op in {"build_repo_graph", "build_graph"}:
            if not image:
                print(json.dumps({"ok": False, "error": "image is required for build_*"}, ensure_ascii=False))
                return 1
            # Build repo-level graph; emit base64(gzip(JSONL)) to keep stdout transport-safe.
            py = "PYTHONPATH=$PYTHONPATH:/mnt/share/MARL_CGM:/mnt/share/MARL_CGM-main:/gp"
            build_cmd = (
                f"{py} python -m graph_planner.tools.swe_build_graph "
                f"--repo {repo} --format jsonl --emit-base64-gzip"
            )
            res = aq.exec_in_instance(
                run_id=run_id,
                docker_image=str(image),
                cmd=["bash", "-lc", build_cmd],
                cwd=cwd,
                env=env,
                timeout_sec=timeout,
            )
            # Ensure repo_graph payload isn't truncated: caller can also set max_stdout_bytes
            out = _exec_result_to_dict(res, max_stdout_bytes=max_stdout_bytes)
            print(json.dumps(out, ensure_ascii=False))
            return 0

        print(json.dumps({"ok": False, "error": f"unknown op: {op}"}, ensure_ascii=False))
        return 1

    except Exception as e:
        print(json.dumps({"ok": False, "error": repr(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
