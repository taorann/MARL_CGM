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
     "cwd": "/testbed",           # optional
     "timeout": 600,            # optional
     "env": {"K": "V"}         # optional
   }

2) Trajectory / instance mode (explicit "op"):
   - start: {"op":"start", "run_id":"...", "image":"...", "cwd":"/testbed", "timeout":600, "env":{...}}
   - exec : {"op":"exec",  "run_id":"...", "image":"...", "cmd":"...", "cwd":"/testbed", "timeout":600, "env":{...}}
   - stop : {"op":"stop",  "run_id":"...", "timeout":600}

3) Helpers used by SandboxRuntime (remote_swe):
   - build_graph:
     {"op":"build_graph", "run_id":"...", "image":"...", "issue_id":"...", "repo":"/testbed", "cwd":"/testbed"}
   - build_repo_graph:
     {"op":"build_repo_graph", "run_id":"...", "image":"...", "repo":"/testbed", "cwd":"/testbed"}

Env:
  - GP_NUM_RUNNERS (default 1)
  - QUEUE_ROOT (default ~/gp_queue)
  - GP_SIF_DIR or SIF_DIR (default ~/sif/sweb)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

from graph_planner.runtime.apptainer_queue_runtime import ApptainerQueueRuntime
from graph_planner.runtime.queue_protocol import QueueRequest, QueueResponse


def _canon_pwd(val: Any) -> str | None:
    """Canonicalize a container working directory.

    We keep this intentionally conservative because `--pwd` failures are noisy
    in Apptainer/Singularity.

    Rules:
    - Non-string / empty -> None (caller should default)
    - Normalize path separators; collapse '.' and '..'
    - Force absolute paths
    - Map common legacy '/repo' to '/testbed'
    - Reject path traversal (anything that still contains '..')
    """

    if not isinstance(val, str):
        return None
    s = val.strip()
    if not s:
        return None

    # Normalize Windows-style separators just in case.
    s = s.replace("\\", "/")

    # Force absolute.
    if not s.startswith("/"):
        s = "/" + s

    # Collapse '/a/../b' etc.
    s = os.path.normpath(s)

    # Some SWE-bench tooling uses /repo; our images use /testbed.
    if s == "/repo":
        s = "/testbed"

    # Basic traversal guard.
    if ".." in Path(s).parts:
        return None

    return s


def _dbg(msg: str) -> None:
    if os.environ.get("DEBUG") or os.environ.get("EBUG"):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[swe_proxy {ts}] {msg}", file=sys.stderr, flush=True)


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

    # Prefer a free/reusable runner when possible (reduces "runner busy").
    runner_id = aq.choose_runner(run_id, image=image)  # type: ignore[attr-defined]
    sif_path: Optional[Path] = None
    if image:
        sif_path = aq._image_to_sif(image)  # type: ignore[attr-defined]

    cmd_list: List[str]
    if cmd is None:
        cmd_list = []
    else:
        cmd_list = ["bash", "-lc", cmd]

    workdir = cwd or Path("/testbed")

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

    _dbg(f"roundtrip op={op_type} run_id={run_id} runner={runner_id} timeout={timeout} cwd={workdir}")

    # Use internal roundtrip for instance ops
    t0 = time.perf_counter()
    resp = aq._roundtrip(QueueRequest(**req))  # type: ignore[attr-defined]
    dt = time.perf_counter() - t0
    _dbg(f"roundtrip done op={op_type} run_id={run_id} ok={getattr(resp, 'ok', None)!r} rc={getattr(resp, 'returncode', None)!r} dt={dt:.2f}s")
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
    cwd = Path(_canon_pwd(payload.get("cwd")) or "/testbed")

    max_stdout_bytes = payload.get("max_stdout_bytes")
    try:
        max_stdout_bytes = int(max_stdout_bytes) if max_stdout_bytes is not None else None
    except Exception:
        max_stdout_bytes = None

    num_runners = int(os.environ.get("GP_NUM_RUNNERS", "1") or "1")
    queue_root = Path(os.environ.get("QUEUE_ROOT", str(Path.home() / "gp_queue"))).expanduser().resolve()
    sif_dir = Path(os.environ.get("GP_SIF_DIR") or os.environ.get("SIF_DIR") or str(Path.home() / "sif" / "sweb")).expanduser().resolve()

    aq = ApptainerQueueRuntime(
        queue_root=queue_root,
        sif_dir=sif_dir,
        num_runners=num_runners,
        default_timeout_sec=timeout,
        max_stdout_bytes=(max_stdout_bytes if max_stdout_bytes is not None else int(os.environ.get("GP_MAX_STDOUT_BYTES", "20000000"))),
    )

    # Ensure repo_graph payload isn't truncated by an overly small stdout cap
    if op == "build_repo_graph":
        aq.max_stdout_bytes = max(int(getattr(aq, "max_stdout_bytes", 0) or 0), 200_000_000)


    _dbg(f"recv op={op!r} run_id={run_id!r} image={image!r} cwd={str(cwd)!r} queue_root={str(queue_root)!r} sif_dir={str(sif_dir)!r} num_runners={num_runners}")


    try:
        # ---------- pool cleanup ----------
        if op == "cleanup_pool":
            # Best-effort: stop any lingering instances on each runner to free slots.
            per_stop_timeout = float(os.environ.get("GP_CLEANUP_STOP_TIMEOUT", "30") or 30.0)
            per_stop_timeout = max(1.0, min(per_stop_timeout, float(timeout or 30.0)))
            results = []
            for rid in range(max(int(num_runners), 1)):
                hb = None
                try:
                    hb = aq._read_heartbeat(int(rid))  # type: ignore[attr-defined]
                except Exception:
                    hb = None
                cur_run = ""
                try:
                    if isinstance(hb, dict):
                        cur_run = str(hb.get("current_run_id") or "")
                except Exception:
                    cur_run = ""
                stop_run_id = cur_run or (str(run_id) if run_id else f"cleanup-{rid}")
                try:
                    r = aq.stop_instance_on_runner(
                        runner_id=int(rid),
                        run_id=stop_run_id,
                        cwd=cwd,
                        env=env,
                        timeout_sec=per_stop_timeout,
                        meta={"cleanup_pool": "1", "prev_run_id": cur_run or ""},
                    )
                    results.append(
                        {
                            "runner_id": int(rid),
                            "prev_run_id": cur_run,
                            "ok": bool(getattr(r, "ok", True)),
                            "returncode": int(getattr(r, "returncode", 0) or 0),
                            "stderr": str(getattr(r, "stderr", "") or ""),
                        }
                    )
                except Exception as e:
                    results.append(
                        {
                            "runner_id": int(rid),
                            "prev_run_id": cur_run,
                            "ok": False,
                            "returncode": 1,
                            "stderr": f"{type(e).__name__}: {e}",
                        }
                    )
            print(json.dumps({"ok": True, "results": results}, ensure_ascii=False))
            return 0

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
                repo = str(_canon_pwd(payload.get("repo")) or "/testbed")
                py = "PYTHONPATH=$PYTHONPATH:/mnt/share/MARL_CGM:/mnt/share/MARL_CGM-main:/gp"
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
                repo = str(_canon_pwd(payload.get("repo")) or "/testbed")
                py = "PYTHONPATH=$PYTHONPATH:/mnt/share/MARL_CGM:/mnt/share/MARL_CGM-main:/gp"
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
