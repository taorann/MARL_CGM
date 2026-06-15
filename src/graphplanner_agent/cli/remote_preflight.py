from __future__ import annotations

from datetime import datetime, timezone
import json
import subprocess
import time
from typing import Any

from graphplanner_agent.config import AgentConfig
from graphplanner_agent.datasets import TaskSpec
from graphplanner_agent.runtime.remote_swe import _image_ref_from_task
from graphplanner_agent.runtime.remote_swe_session import RemoteSweSession, infer_sif_dir_from_ref, normalize_sif_image_ref


def normalize_remote_preflight_mode(value: str | None, *, backend: str) -> str:
    mode = str(value or "auto").strip().lower()
    if mode == "auto":
        return "cleanup" if str(backend or "").lower() == "remote_swe" else "off"
    if mode not in {"off", "cleanup", "full"}:
        raise ValueError(f"invalid remote preflight mode: {value}; expected auto, off, cleanup, or full")
    return mode


def _runner_pool_status(session: RemoteSweSession, *, timeout: float = 45.0) -> dict[str, Any]:
    if not hasattr(session, "simple_ssh_cmd") or not hasattr(session, "_remote_env_prefix"):
        kwargs = getattr(session, "kwargs", {})
        runner_count = int(getattr(session, "num_runners", None) or (kwargs.get("num_runners") if isinstance(kwargs, dict) else None) or 1)
        return {
            "ok": True,
            "skipped": True,
            "reason": "session object does not expose SSH status probe",
            "runner_count": runner_count,
            "fresh_count": runner_count,
            "pending_request_count": 0,
            "runners": [],
        }
    script = r'''
import json
import os
import time
from pathlib import Path

queue_root = Path(os.environ.get("QUEUE_ROOT") or str(Path.home() / "gp_queue")).expanduser()
runner_count = int(os.environ.get("GP_NUM_RUNNERS") or "1")
ttl = float(os.environ.get("GRAPHPLANNER_REMOTE_HEARTBEAT_TTL_SEC") or "120")
runners = []
for rid in range(runner_count):
    runner_dir = queue_root / f"runner-{rid}"
    hb_path = runner_dir / "heartbeat.json"
    heartbeat_age = None
    heartbeat_ok = False
    if hb_path.exists():
        try:
            data = json.loads(hb_path.read_text(encoding="utf-8"))
            ts = float(data.get("ts", 0.0) or 0.0)
        except Exception:
            ts = hb_path.stat().st_mtime
        if ts > 0:
            heartbeat_age = max(0.0, time.time() - ts)
            heartbeat_ok = heartbeat_age <= ttl
    current_run = ""
    current_path = runner_dir / "current_run_id"
    if current_path.exists():
        current_run = current_path.read_text(encoding="utf-8", errors="replace").strip()
    runners.append(
        {
            "runner_id": rid,
            "heartbeat_age_sec": None if heartbeat_age is None else round(heartbeat_age, 3),
            "heartbeat_ok": heartbeat_ok,
            "current_run_id": current_run,
        }
    )
pending = sorted(str(p) for p in queue_root.glob("runner-*/in/*.json"))
print(json.dumps({
    "ok": True,
    "queue_root": str(queue_root),
    "runner_count": runner_count,
    "fresh_count": sum(1 for item in runners if item["heartbeat_ok"]),
    "pending_request_count": len(pending),
    "pending_request_preview": pending[:5],
    "runners": runners,
}, sort_keys=True))
'''
    command = f"{session._remote_env_prefix()} {session.remote_python or 'python'} - <<'PY'\n{script}\nPY"
    proc = subprocess.run(
        session.simple_ssh_cmd(command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "ok": False,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"ok": False, "reason": f"runner status JSON parse failed: {exc}", "stdout": proc.stdout}
    return data


def _wait_for_fresh_runners(
    session: RemoteSweSession,
    *,
    expected_count: int,
    timeout: float,
    poll_sec: float = 5.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while True:
        last = _runner_pool_status(session, timeout=45.0)
        if bool(last.get("ok")) and int(last.get("fresh_count") or 0) >= expected_count:
            return last
        if time.monotonic() >= deadline:
            last = dict(last)
            last["ok"] = False
            last["reason"] = f"runner heartbeats did not become fresh before {timeout:.1f}s"
            return last
        time.sleep(poll_sec)


def run_remote_swe_preflight(config: AgentConfig, first_task: TaskSpec, *, mode: str = "cleanup") -> dict[str, Any]:
    mode = normalize_remote_preflight_mode(mode, backend=config.sandbox_backend)
    started = time.monotonic()
    image_ref = _image_ref_from_task(first_task)
    image = normalize_sif_image_ref(image_ref)
    record: dict[str, Any] = {
        "ok": False,
        "mode": mode,
        "task_id": first_task.task_id,
        "image_ref": image_ref,
        "image": image,
        "runner_count": int(config.sandbox_num_runners or 1),
        "remote_repo": config.sandbox_remote_repo,
        "ssh_target": config.sandbox_ssh_target,
        "steps": [],
    }
    if mode == "off":
        record.update({"ok": True, "skipped": True, "reason": "remote preflight disabled"})
        return record
    if not image:
        record.update({"reason": "first task has no remote image"})
        return record

    session = RemoteSweSession(
        ssh_target=config.sandbox_ssh_target,
        remote_repo=config.sandbox_remote_repo,
        image=image,
        run_id=f"gp-preflight__{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        remote_python=config.sandbox_remote_python,
        swe_proxy_path=config.sandbox_swe_proxy_path,
        runner_manager_path=config.sandbox_runner_manager_path,
        num_runners=config.sandbox_num_runners,
        ensure_runners=True,
        ssh_args=config.sandbox_ssh_args,
        sif_dir=config.sandbox_sif_dir or infer_sif_dir_from_ref(image_ref),
    )

    def step(name: str, payload: dict[str, Any]) -> None:
        record["steps"].append({"name": name, **payload})

    try:
        layout = session.check_remote_layout(timeout=45.0)
        step("layout", layout)
        if not bool(layout.get("ok")):
            record.update({"reason": "remote layout check failed"})
            return record

        before = _runner_pool_status(session)
        step("runner_pool_before_cleanup", before)

        cleanup = session.cleanup_pool(timeout=180.0, cwd=config.sandbox_workdir)
        step("cleanup_pool", cleanup)
        if not bool(cleanup.get("ok", True)):
            record.update({"reason": "remote cleanup_pool reported not ok"})
            return record
        cleanup_results = cleanup.get("results")
        if isinstance(cleanup_results, list):
            failed = [item for item in cleanup_results if isinstance(item, dict) and not bool(item.get("ok", True))]
            if failed:
                record.update(
                    {
                        "reason": "remote cleanup_pool has failed runner result",
                        "failed_cleanup_results": failed[:3],
                    }
                )
                return record

        after_cleanup = _runner_pool_status(session)
        step("runner_pool_after_cleanup", after_cleanup)
        if bool(after_cleanup.get("ok")) and int(after_cleanup.get("pending_request_count") or 0) > 0:
            record.update(
                {
                    "reason": "remote queue still has pending runner requests after cleanup",
                    "runner_pool": after_cleanup,
                }
            )
            return record

        session.ensure_remote_runners(timeout=max(180.0, 60.0 * int(config.sandbox_num_runners or 1)))
        step("ensure_runners", {"ok": True, "runner_count": int(config.sandbox_num_runners or 1)})
        fresh = _wait_for_fresh_runners(
            session,
            expected_count=int(config.sandbox_num_runners or 1),
            timeout=max(120.0, 45.0 * int(config.sandbox_num_runners or 1)),
        )
        step("runner_pool_after_ensure", fresh)
        if not bool(fresh.get("ok")):
            record.update({"reason": "remote runners did not become fresh", "runner_pool": fresh})
            return record

        if mode == "full":
            start = session.start(timeout=max(float(config.command_timeout or 0), 300.0), cwd=config.sandbox_workdir)
            step("smoke_start", start)
            if not bool(start.get("ok", True)) or int(start.get("returncode") or 0) != 0:
                record.update({"reason": "remote smoke start failed"})
                return record
            stop = session.stop(timeout=60.0)
            step("smoke_stop", stop)
            if not bool(stop.get("ok", True)) or int(stop.get("returncode") or 0) != 0:
                record.update({"reason": "remote smoke stop failed"})
                return record

        record.update({"ok": True, "elapsed": round(time.monotonic() - started, 3)})
        return record
    except Exception as exc:
        record.update(
            {
                "ok": False,
                "elapsed": round(time.monotonic() - started, 3),
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
        return record
