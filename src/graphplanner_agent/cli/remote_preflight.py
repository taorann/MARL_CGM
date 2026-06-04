from __future__ import annotations

from datetime import datetime, timezone
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

        cleanup = session.cleanup_pool(timeout=180.0, cwd=config.sandbox_workdir)
        step("cleanup_pool", cleanup)
        if not bool(cleanup.get("ok", True)):
            record.update({"reason": "remote cleanup_pool reported not ok"})
            return record

        session.ensure_remote_runners(timeout=max(180.0, 60.0 * int(config.sandbox_num_runners or 1)))
        step("ensure_runners", {"ok": True, "runner_count": int(config.sandbox_num_runners or 1)})

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
