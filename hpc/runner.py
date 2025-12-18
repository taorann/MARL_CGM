from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from graph_planner.runtime.queue_protocol import (
    QueueRequest,
    QueueResponse,
    runner_inbox,
    runner_outbox,
)

# ----------------- 全局配置（通过环境变量注入） -----------------

APPTAINER_BIN = os.environ.get("APPTAINER_BIN", "apptainer")

QUEUE_ROOT = Path(os.environ["QUEUE_ROOT"])
RUNNER_ID = int(os.environ["RUNNER_ID"])
SHARE_ROOT = Path(os.environ["SHARE_ROOT"])
POLL_INTERVAL_SEC = float(os.environ.get("RUNNER_POLL_INTERVAL", "0.5"))

# 一个 runner ⇔ 一个 apptainer instance（名字固定为 gp-00/gp-01/...）
INSTANCE_NAME = f"gp-{RUNNER_ID:02d}"

# Runner liveness markers (used by proxy-side routing)
RUNNER_ROOT = QUEUE_ROOT / f"runner-{RUNNER_ID}"
READY_PATH = RUNNER_ROOT / "ready.json"
HEARTBEAT_PATH = RUNNER_ROOT / "heartbeat.json"

# 绑定状态（不允许 handle_one_request 无脑覆盖）
CURRENT_RUN_ID: Optional[str] = None
CURRENT_TASK_ID: Optional[str] = None
CURRENT_IMAGE: Optional[str] = None
CURRENT_SIF: Optional[str] = None
CURRENT_BOUND_AT: Optional[float] = None

LOCK_TTL_SEC = float(os.environ.get("RUNNER_LOCK_TTL_SEC", "7200"))  # 2h 默认


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now() -> float:
    return time.time()


def _write_ready() -> None:
    meta = {"rid": RUNNER_ID, "pid": os.getpid(), "host": os.uname().nodename, "ts": _now()}
    _ensure_dir(RUNNER_ROOT)
    READY_PATH.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def _write_heartbeat(extra: Optional[Dict[str, Any]] = None) -> None:
    meta: Dict[str, Any] = {
        "rid": RUNNER_ID,
        "pid": os.getpid(),
        "host": os.uname().nodename,
        "ts": _now(),
        "current_run_id": CURRENT_RUN_ID,
        "current_task_id": CURRENT_TASK_ID,
        "current_image": CURRENT_IMAGE,
        "instance": INSTANCE_NAME,
        "bound_at": CURRENT_BOUND_AT,
    }
    if extra:
        meta.update(extra)
    _ensure_dir(RUNNER_ROOT)
    HEARTBEAT_PATH.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def _parse_task_id(run_id: str) -> str:
    # 形如 gp-<task_id>__<uuid>；兜底返回 run_id
    if not isinstance(run_id, str):
        return ""
    s = run_id
    if s.startswith("gp-"):
        s = s[3:]
    if "__" in s:
        return s.split("__", 1)[0]
    return s


def _instance_exists() -> bool:
    try:
        proc = subprocess.run(
            [APPTAINER_BIN, "instance", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return False
        # 输出通常包含 NAME 一列；粗暴匹配即可
        return INSTANCE_NAME in (proc.stdout or "")
    except Exception:
        return False


def _is_lock_stale() -> bool:
    if CURRENT_BOUND_AT is None:
        return False
    return (_now() - float(CURRENT_BOUND_AT)) > float(LOCK_TTL_SEC)


def _clear_lock(reason: str = "") -> None:
    global CURRENT_RUN_ID, CURRENT_TASK_ID, CURRENT_IMAGE, CURRENT_SIF, CURRENT_BOUND_AT
    CURRENT_RUN_ID = None
    CURRENT_TASK_ID = None
    CURRENT_IMAGE = None
    CURRENT_SIF = None
    CURRENT_BOUND_AT = None
    _write_heartbeat({"lock_cleared": reason})


def _resolve_cwd(req_cwd: str | None, run_id: str) -> tuple[Path, str | None]:
    """Split cwd into a safe host-side cwd and an optional container-side pwd.

    remote_swe passes container paths like "/testbed". Runner must NOT treat them as host paths.
    We always execute on a host-side per-run directory under $SHARE_ROOT/gp_work, and pass
    the container path via apptainer --pwd.
    """
    container_pwd: str | None = None
    if isinstance(req_cwd, str) and req_cwd.startswith("/"):
        container_pwd = req_cwd

    host_base = Path(os.environ.get("RUNNER_WORKDIR_HOST", str(SHARE_ROOT / "gp_work")))
    host_cwd = host_base / run_id / INSTANCE_NAME
    _ensure_dir(host_cwd)
    return host_cwd, container_pwd


def _job_sig_from_req(req: QueueRequest) -> Tuple[str, str]:
    task_id = ""
    if isinstance(req.meta, dict) and req.meta.get("task_id"):
        task_id = str(req.meta.get("task_id") or "")
    if not task_id:
        task_id = _parse_task_id(req.run_id)
    image = str(req.image or "")
    return task_id, image


def _job_sig_current() -> Tuple[str, str]:
    return str(CURRENT_TASK_ID or ""), str(CURRENT_IMAGE or "")


def main() -> None:
    inbox = runner_inbox(QUEUE_ROOT, RUNNER_ID)
    outbox = runner_outbox(QUEUE_ROOT, RUNNER_ID)
    _ensure_dir(inbox)
    _ensure_dir(outbox)

    _write_ready()
    _write_heartbeat()

    last_hb = 0.0
    while True:
        now = _now()
        if now - last_hb >= 2.0:
            # stale lock auto-heal：锁过期且 instance 不在了
            if CURRENT_RUN_ID and _is_lock_stale() and (not _instance_exists()):
                _clear_lock("stale_lock_instance_missing")
            _write_heartbeat()
            last_hb = now

        handled = False
        for req_path in sorted(inbox.glob("*.json")):
            handled = True
            handle_one_request(req_path, outbox)
            try:
                req_path.unlink()
            except FileNotFoundError:
                pass

        if not handled:
            time.sleep(POLL_INTERVAL_SEC)


def handle_one_request(req_path: Path, outbox: Path) -> None:
    """Always write a response JSON, even if execution fails."""
    req_id = req_path.stem
    try:
        data: Dict[str, Any] = json.loads(req_path.read_text(encoding="utf-8"))
        req = QueueRequest(**data)
        _write_heartbeat({"last_req_id": req.req_id, "type": req.type, "seen_run_id": req.run_id})

        if req.type == "exec":
            resp = handle_exec(req)
        elif req.type == "instance_start":
            resp = handle_instance_start(req)
        elif req.type == "instance_exec":
            resp = handle_instance_exec(req)
        elif req.type == "instance_stop":
            resp = handle_instance_stop(req)
        elif req.type == "put":
            resp = handle_put(req)
        elif req.type == "get":
            resp = handle_get(req)
        elif req.type == "cleanup":
            resp = handle_cleanup(req)
        else:
            resp = QueueResponse(
                req_id=req.req_id,
                runner_id=req.runner_id,
                run_id=req.run_id,
                type=req.type,
                ok=True,
                returncode=0,
                stdout="",
                stderr="noop",
                runtime_sec=0.0,
            )
    except Exception as e:
        resp = QueueResponse(
            req_id=req_id,
            runner_id=RUNNER_ID,
            run_id=CURRENT_RUN_ID or "",
            type="noop",
            ok=False,
            returncode=1,
            stdout="",
            stderr="",
            runtime_sec=0.0,
            error=f"runner_exception: {repr(e)}",
        )

    resp_path = outbox / f"{resp.req_id}.json"
    resp_path.write_text(json.dumps(resp.__dict__, ensure_ascii=False), encoding="utf-8")
    _write_heartbeat({"last_resp_id": resp.req_id, "ok": str(resp.ok)})


# ----------------- 一次性 exec（直接对 SIF 调用） -----------------


def handle_exec(req: QueueRequest) -> QueueResponse:
    assert req.sif_path and req.cmd

    workdir_host, container_pwd = _resolve_cwd(req.cwd, req.run_id)

    env = os.environ.copy()
    if req.env:
        env.update(req.env)

    cmd = [APPTAINER_BIN, "exec", "--cleanenv", "--bind", f"{SHARE_ROOT}:/mnt/share"]
    if container_pwd:
        cmd.extend(["--pwd", container_pwd])
    cmd.extend([req.sif_path, *req.cmd])

    t0 = _now()
    proc = subprocess.run(
        cmd,
        cwd=str(workdir_host),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=req.timeout_sec or None,
    )
    t1 = _now()

    return QueueResponse(
        req_id=req.req_id,
        runner_id=req.runner_id,
        run_id=req.run_id,
        type=req.type,
        ok=(proc.returncode == 0),
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        runtime_sec=t1 - t0,
        error=None if proc.returncode == 0 else "non-zero return code",
    )


# ----------------- 会话型 instance（一 runner ⇔ 一容器） -----------------


def handle_instance_start(req: QueueRequest) -> QueueResponse:
    """Start this runner's apptainer instance.

    关键语义（为“runner 不必关、容器可复用”服务）：
      - runner 固定只有一个 instance 名（gp-00/gp-01/...）
      - lock 的“身份”是 (task_id, image) 而不是 (run_id)
      - 新的 run_id 只要指向同一 (task_id, image)，就允许复用并 rebind CURRENT_RUN_ID
      - 如果 instance 不存在但 lock 还在：自动清锁并允许重新 start
    """
    global CURRENT_RUN_ID, CURRENT_TASK_ID, CURRENT_IMAGE, CURRENT_SIF, CURRENT_BOUND_AT
    assert req.sif_path

    workdir_host, container_pwd = _resolve_cwd(req.cwd, req.run_id)
    env = os.environ.copy()
    if req.env:
        env.update(req.env)

    new_task_id, new_image = _job_sig_from_req(req)
    new_sif = str(req.sif_path or "")

    # auto-heal stale lock
    if CURRENT_RUN_ID and (not _instance_exists()):
        _clear_lock("lock_but_instance_missing")

    if CURRENT_RUN_ID:
        cur_task, cur_image = _job_sig_current()
        # 同一 job：允许复用 + rebind run_id
        if (cur_task == new_task_id) and (cur_image == new_image):
            old = CURRENT_RUN_ID
            CURRENT_RUN_ID = req.run_id
            CURRENT_BOUND_AT = _now()
            return QueueResponse(
                req_id=req.req_id,
                runner_id=req.runner_id,
                run_id=req.run_id,
                type=req.type,
                ok=True,
                returncode=0,
                stdout="",
                stderr=f"reused instance {INSTANCE_NAME}; rebound from {old}",
                runtime_sec=0.0,
                meta={"reused": "1", "instance": INSTANCE_NAME, "prev_run_id": old},
            )
        # 不同 job：busy
        msg = (
            f"runner busy: current_run_id={CURRENT_RUN_ID}, current_task_id={cur_task}, current_image={cur_image}, "
            f"new_run_id={req.run_id}, new_task_id={new_task_id}, new_image={new_image}"
        )
        return QueueResponse(
            req_id=req.req_id,
            runner_id=req.runner_id,
            run_id=req.run_id,
            type=req.type,
            ok=False,
            returncode=1,
            stdout="",
            stderr=msg,
            runtime_sec=0.0,
            error=msg,
            meta={"busy": "1", "current_run_id": str(CURRENT_RUN_ID)},
        )

    # CURRENT_RUN_ID is None: start fresh
    keepalive_cmd = "while true; do sleep 3600; done"
    cmd = [
        APPTAINER_BIN,
        "instance",
        "start",
        "--cleanenv",
        "--bind",
        f"{SHARE_ROOT}:/mnt/share",
    ]
    if container_pwd:
        cmd.extend(["--pwd", container_pwd])
    cmd.extend([req.sif_path, INSTANCE_NAME, "bash", "-lc", keepalive_cmd])

    t0 = _now()
    proc = subprocess.run(
        cmd,
        cwd=str(workdir_host),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=req.timeout_sec or None,
    )
    t1 = _now()

    if proc.returncode == 0:
        CURRENT_RUN_ID = req.run_id
        CURRENT_TASK_ID = new_task_id
        CURRENT_IMAGE = new_image
        CURRENT_SIF = new_sif
        CURRENT_BOUND_AT = _now()

    return QueueResponse(
        req_id=req.req_id,
        runner_id=req.runner_id,
        run_id=req.run_id,
        type=req.type,
        ok=(proc.returncode == 0),
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        runtime_sec=t1 - t0,
        error=None if proc.returncode == 0 else "non-zero return code",
        meta={"instance": INSTANCE_NAME, "task_id": new_task_id, "image": new_image},
    )


def handle_instance_exec(req: QueueRequest) -> QueueResponse:
    global CURRENT_RUN_ID, CURRENT_BOUND_AT
    assert req.cmd

    workdir_host, container_pwd = _resolve_cwd(req.cwd, req.run_id)

    env = os.environ.copy()
    if req.env:
        env.update(req.env)

    if CURRENT_RUN_ID is None:
        msg = "no active instance on this runner; call instance_start first"
        return QueueResponse(
            req_id=req.req_id,
            runner_id=req.runner_id,
            run_id=req.run_id,
            type=req.type,
            ok=False,
            returncode=1,
            stdout="",
            stderr=msg,
            runtime_sec=0.0,
            error=msg,
        )

    # allow rebind if job signature matches
    new_task_id, new_image = _job_sig_from_req(req)
    cur_task, cur_image = _job_sig_current()
    if (req.run_id != CURRENT_RUN_ID) and (new_task_id == cur_task) and (new_image == cur_image):
        old = CURRENT_RUN_ID
        CURRENT_RUN_ID = req.run_id
        CURRENT_BOUND_AT = _now()
        _write_heartbeat({"rebind_on_exec": "1", "prev_run_id": old})

    if CURRENT_RUN_ID != req.run_id:
        msg = f"runner bound to run_id={CURRENT_RUN_ID}, but got {req.run_id}"
        return QueueResponse(
            req_id=req.req_id,
            runner_id=req.runner_id,
            run_id=req.run_id,
            type=req.type,
            ok=False,
            returncode=1,
            stdout="",
            stderr=msg,
            runtime_sec=0.0,
            error=msg,
        )

    cmd = [APPTAINER_BIN, "exec", "--cleanenv", "--bind", f"{SHARE_ROOT}:/mnt/share"]
    if container_pwd:
        cmd.extend(["--pwd", container_pwd])
    cmd.extend([f"instance://{INSTANCE_NAME}", *req.cmd])

    t0 = _now()
    proc = subprocess.run(
        cmd,
        cwd=str(workdir_host),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=req.timeout_sec or None,
    )
    t1 = _now()

    return QueueResponse(
        req_id=req.req_id,
        runner_id=req.runner_id,
        run_id=req.run_id,
        type=req.type,
        ok=(proc.returncode == 0),
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        runtime_sec=t1 - t0,
        error=None if proc.returncode == 0 else "non-zero return code",
    )


def handle_instance_stop(req: QueueRequest) -> QueueResponse:
    workdir_host, _ = _resolve_cwd(req.cwd, req.run_id)
    env = os.environ.copy()
    if req.env:
        env.update(req.env)

    if CURRENT_RUN_ID is None and (not _instance_exists()):
        return QueueResponse(
            req_id=req.req_id,
            runner_id=req.runner_id,
            run_id=req.run_id,
            type=req.type,
            ok=True,
            returncode=0,
            stdout="",
            stderr="no active instance; stop is no-op",
            runtime_sec=0.0,
        )

    cmd = [APPTAINER_BIN, "instance", "stop", INSTANCE_NAME]
    t0 = _now()
    proc = subprocess.run(
        cmd,
        cwd=str(workdir_host),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=req.timeout_sec or None,
    )
    t1 = _now()

    # regardless of stop result, if instance is gone then clear lock
    if not _instance_exists():
        _clear_lock("instance_stop")

    return QueueResponse(
        req_id=req.req_id,
        runner_id=req.runner_id,
        run_id=req.run_id,
        type=req.type,
        ok=(proc.returncode == 0),
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        runtime_sec=t1 - t0,
        error=None if proc.returncode == 0 else "non-zero return code",
    )


# ----------------- put/get/cleanup -----------------


def handle_put(req: QueueRequest) -> QueueResponse:
    assert req.src and req.dst
    src = Path(req.src)
    dst = Path(req.dst)
    ok = True
    err = None

    t0 = _now()
    try:
        _ensure_dir(dst.parent)
        shutil.copy2(src, dst)
    except Exception as e:
        ok = False
        err = f"put failed: {e!r}"
    t1 = _now()

    return QueueResponse(
        req_id=req.req_id,
        runner_id=req.runner_id,
        run_id=req.run_id,
        type=req.type,
        ok=ok,
        returncode=0 if ok else 1,
        stdout="",
        stderr="" if ok else (err or ""),
        runtime_sec=t1 - t0,
        error=err,
    )


def handle_get(req: QueueRequest) -> QueueResponse:
    assert req.src and req.dst
    src = Path(req.src)
    dst = Path(req.dst)
    ok = True
    err = None

    t0 = _now()
    try:
        _ensure_dir(dst.parent)
        shutil.copy2(src, dst)
    except Exception as e:
        ok = False
        err = f"get failed: {e!r}"
    t1 = _now()

    return QueueResponse(
        req_id=req.req_id,
        runner_id=req.runner_id,
        run_id=req.run_id,
        type=req.type,
        ok=ok,
        returncode=0 if ok else 1,
        stdout="",
        stderr="" if ok else (err or ""),
        runtime_sec=t1 - t0,
        error=err,
    )


def handle_cleanup(req: QueueRequest) -> QueueResponse:
    return QueueResponse(
        req_id=req.req_id,
        runner_id=req.runner_id,
        run_id=req.run_id,
        type=req.type,
        ok=True,
        returncode=0,
        stdout="",
        stderr="cleanup noop",
        runtime_sec=0.0,
    )


if __name__ == "__main__":
    main()
