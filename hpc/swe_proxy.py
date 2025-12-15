#!/usr/bin/env python
"""swe_proxy.py

A thin JSON stdin/stdout proxy for GraphPlanner's remote_swe sandbox backend.

It forwards requests to ApptainerQueueRuntime (file-queue IPC) which dispatches
work to long-running runners on compute nodes. The runner executes commands
inside a .sif image via apptainer/singularity.

Supported request formats:

1) Legacy one-shot exec (no "op"):
   {
     "run_id": "...",
     "image": "<docker image string>",
     "cmd": "<shell command>",
     "cwd": "/repo",           # optional
     "timeout": 600,            # optional
     "env": {"K": "V"}         # optional
   }

2) Trajectory / instance mode (explicit "op"):
   - start: {"op":"start", "run_id":"...", "image":"...", "cwd":"/repo", "timeout":600, "env":{...}}
   - exec : {"op":"exec",  "run_id":"...", "image":"...", "cmd":"...", "cwd":"/repo", "timeout":600, "env":{...}}
   - stop : {"op":"stop",  "run_id":"...", "timeout":600}

3) Helpers used by SandboxRuntime (remote_swe):
   - build_graph:
     {"op":"build_graph", "run_id":"...", "image":"...", "issue_id":"...", "repo":"/repo", "cwd":"/repo"}
   - build_repo_graph:
     {"op":"build_repo_graph", "run_id":"...", "image":"...", "repo":"/repo", "cwd":"/repo"}

Env:
  - GP_NUM_RUNNERS (default 1)
  - QUEUE_ROOT (default ~/gp_queue)
  - GP_SIF_DIR or SIF_DIR (default ~/sif/sweb)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List

from graph_planner.runtime.apptainer_queue_runtime import ApptainerQueueRuntime
from graph_planner.runtime.queue_protocol import QueueResponse


def _dbg(msg: str) -> None:
    if os.environ.get("DEBUG") or os.environ.get("EBUG"):
        print(f"[swe_proxy] {msg}", file=sys.stderr)


def _resp_to_dict(resp: QueueResponse) -> Dict[str, Any]:
    return {
        "ok": bool(resp.ok),
        "returncode": int(resp.returncode or 0),
        "stdout": resp.stdout or "",
        "stderr": resp.stderr or "",
        "runtime_sec": float(resp.runtime_sec or 0.0),
        "error": resp.error,
    }


def _instance_roundtrip(
    aq: ApptainerQueueRuntime,
    *,
    op_type: str,  # instance_start | instance_exec | instance_stop
    run_id: str,
    image: Optional[str],
    cmd: Optional[str],
    timeout: float,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Path] = None,
) -> QueueResponse:
    """Construct a QueueRequest and send it to the mapped runner."""

    runner_id = aq._choose_runner(run_id)  # type: ignore[attr-defined]
    sif_path: Optional[Path] = None
    if image:
        sif_path = aq._image_to_sif(image)  # type: ignore[attr-defined]

    cmd_list: List[str]
    if cmd is None:
        cmd_list = []
    else:
        cmd_list = ["bash", "-lc", cmd]

    workdir = cwd or Path("/repo")

    # Build raw QueueRequest without importing it here (avoid duplicate protocol deps)
    req = {
        "req_id": aq._new_req_id(),  # type: ignore[attr-defined]
        "runner_id": runner_id,
        "run_id": run_id,
        "type": op_type,
        "image": image,
        "sif_path": str(sif_path) if sif_path is not None else None,
        "cmd": cmd_list,
        "cwd": str(workdir),
        "env": dict(env or {}),
        "timeout_sec": float(timeout or aq.default_timeout_sec),
        "src": None,
        "dst": None,
        "meta": {},
    }

    # Use internal roundtrip for instance ops
    resp = aq._roundtrip(type("_Q", (), req)())  # type: ignore[attr-defined]
    return resp


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        print(json.dumps({"ok": False, "error": "empty stdin"}))
        return 1

    try:
        payload = json.loads(raw)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"invalid json: {e}"}))
        return 1

    run_id = str(payload.get("run_id") or "__default__")
    op = str(payload.get("op") or "").strip().lower()
    image = payload.get("image")
    cmd = payload.get("cmd")
    timeout = float(payload.get("timeout") or 600.0)
    env = payload.get("env") or {}
    cwd = Path(payload.get("cwd") or "/repo")

    num_runners = int(os.environ.get("GP_NUM_RUNNERS", "1") or "1")
    queue_root = Path(os.environ.get("QUEUE_ROOT", str(Path.home() / "gp_queue"))).expanduser().resolve()
    sif_dir = Path(os.environ.get("GP_SIF_DIR") or os.environ.get("SIF_DIR") or str(Path.home() / "sif" / "sweb")).expanduser().resolve()

    aq = ApptainerQueueRuntime(
        queue_root=queue_root,
        sif_dir=sif_dir,
        num_runners=num_runners,
        default_timeout_sec=timeout,
    )

    _dbg(f"recv op={op!r} run_id={run_id!r} image={image!r} cwd={str(cwd)!r} queue_root={str(queue_root)!r} sif_dir={str(sif_dir)!r} num_runners={num_runners}")

    try:
        # ---------- instance mode ----------
        if op in {"start", "exec", "stop", "build_graph", "build_repo_graph"}:
            if op in {"start", "exec", "build_graph", "build_repo_graph"} and not image:
                print(json.dumps({"ok": False, "error": f"image is required for op={op}"}))
                return 1

            if op == "start":
                resp = _instance_roundtrip(
                    aq,
                    op_type="instance_start",
                    run_id=run_id,
                    image=str(image),
                    cmd=None,
                    timeout=timeout,
                    env=env,
                    cwd=cwd,
                )
            elif op == "exec":
                if not isinstance(cmd, str) or not cmd:
                    print(json.dumps({"ok": False, "error": "cmd is required for op=exec"}))
                    return 1
                resp = _instance_roundtrip(
                    aq,
                    op_type="instance_exec",
                    run_id=run_id,
                    image=str(image),
                    cmd=str(cmd),
                    timeout=timeout,
                    env=env,
                    cwd=cwd,
                )
            elif op == "build_graph":
                issue_id = str(payload.get("issue_id") or "").strip()
                if not issue_id:
                    print(json.dumps({"ok": False, "error": "issue_id is required for op=build_graph"}))
                    return 1
                repo = str(payload.get("repo") or "/repo")
                py = "PYTHONPATH=$PYTHONPATH:/mnt/share/MARL_CGM:/gp"
                build_cmd = (
                    f"{py} python -m graph_planner.tools.swe_build_graph "
                    f"--repo {repo} --issue-id {issue_id}"
                )
                resp = _instance_roundtrip(
                    aq,
                    op_type="instance_exec",
                    run_id=run_id,
                    image=str(image),
                    cmd=build_cmd,
                    timeout=timeout,
                    env=env,
                    cwd=cwd,
                )
            elif op == "build_repo_graph":
                repo = str(payload.get("repo") or "/repo")
                py = "PYTHONPATH=$PYTHONPATH:/mnt/share/MARL_CGM:/gp"
                build_cmd = (
                    f"{py} python -m graph_planner.tools.swe_build_graph "
                    f"--repo {repo} --emit-base64-gzip"
                )
                resp = _instance_roundtrip(
                    aq,
                    op_type="instance_exec",
                    run_id=run_id,
                    image=str(image),
                    cmd=build_cmd,
                    timeout=timeout,
                    env=env,
                    cwd=cwd,
                )
            else:  # stop
                resp = _instance_roundtrip(
                    aq,
                    op_type="instance_stop",
                    run_id=run_id,
                    image=None,
                    cmd=None,
                    timeout=timeout,
                    env=env,
                    cwd=cwd,
                )

            print(json.dumps(_resp_to_dict(resp), ensure_ascii=False))
            return 0

        # ---------- legacy one-shot exec ----------
        if not image or not cmd:
            print(json.dumps({"ok": False, "error": "missing image/cmd"}, ensure_ascii=False))
            return 1

        res = aq.exec(
            run_id=run_id,
            docker_image=str(image),
            cmd=["bash", "-lc", str(cmd)],
            cwd=cwd,
            env=env,
            timeout_sec=timeout,
        )
        print(json.dumps({
            "ok": bool(res.returncode == 0),
            "returncode": int(res.returncode),
            "stdout": res.stdout or "",
            "stderr": res.stderr or "",
            "runtime_sec": float(getattr(res, "runtime_sec", 0.0) or 0.0),
            "error": getattr(res, "error", None),
        }, ensure_ascii=False))
        return 0

    except Exception as e:
        print(json.dumps({"ok": False, "error": repr(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
