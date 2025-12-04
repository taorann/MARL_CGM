from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

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

# 一个 runner ⇔ 一个容器：名字固定为 gp-00/gp-01/...
INSTANCE_NAME = f"gp-{RUNNER_ID:02d}"

# 当前 runner 正在服务的 run_id；None 表示空闲
CURRENT_RUN_ID: Optional[str] = None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now() -> float:
    return time.time()


def main() -> None:
    inbox = runner_inbox(QUEUE_ROOT, RUNNER_ID)
    outbox = runner_outbox(QUEUE_ROOT, RUNNER_ID)
    _ensure_dir(inbox)
    _ensure_dir(outbox)

    while True:
        handled = False
        for req_path in sorted(inbox.glob("*.json")):
            handled = True
            try:
                handle_one_request(req_path, outbox)
            finally:
                try:
                    req_path.unlink()
                except FileNotFoundError:
                    pass
        if not handled:
            time.sleep(POLL_INTERVAL_SEC)


def handle_one_request(req_path: Path, outbox: Path) -> None:
    data: Dict[str, Any] = json.loads(req_path.read_text(encoding="utf-8"))
    req = QueueRequest(**data)

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
        )

    resp_path = outbox / f"{resp.req_id}.json"
    resp_path.write_text(json.dumps(resp.__dict__, ensure_ascii=False), encoding="utf-8")


# ----------------- 一次性 exec（直接对 SIF 调用） -----------------


def handle_exec(req: QueueRequest) -> QueueResponse:
    assert req.sif_path and req.cmd and req.cwd
    workdir_host = Path(req.cwd)
    _ensure_dir(workdir_host)

    env = os.environ.copy()
    if req.env:
        env.update(req.env)

    cmd = [
        APPTAINER_BIN,
        "exec",
        "--cleanenv",
        "--bind",
        f"{SHARE_ROOT}:/mnt/share",
        req.sif_path,
        *req.cmd,
    ]

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
    """
    启动本 runner 的唯一容器 instance: INSTANCE_NAME = gp-{RUNNER_ID:02d}
    规则：
      - CURRENT_RUN_ID 为 None：绑定到 req.run_id，启动 instance；
      - CURRENT_RUN_ID == req.run_id：视为幂等 start，直接 ok；
      - CURRENT_RUN_ID 其他：返回 busy 错误。
    """
    global CURRENT_RUN_ID

    assert req.sif_path and req.cwd
    workdir_host = Path(req.cwd)
    _ensure_dir(workdir_host)

    env = os.environ.copy()
    if req.env:
        env.update(req.env)

    # runner 空闲：启动 instance + 绑定 run_id
    if CURRENT_RUN_ID is None:
        cmd = [
            APPTAINER_BIN,
            "instance",
            "start",
            "--cleanenv",
            "--bind",
            f"{SHARE_ROOT}:/mnt/share",
            req.sif_path,
            INSTANCE_NAME,
        ]
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

    # 已绑定同一个 run_id：幂等
    if CURRENT_RUN_ID == req.run_id:
        return QueueResponse(
            req_id=req.req_id,
            runner_id=req.runner_id,
            run_id=req.run_id,
            type=req.type,
            ok=True,
            returncode=0,
            stdout="",
            stderr="instance already started for this run_id",
            runtime_sec=0.0,
            error=None,
        )

    # 被其他 run_id 占用
    msg = f"runner busy: current_run_id={CURRENT_RUN_ID}, new_run_id={req.run_id}"
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


def handle_instance_exec(req: QueueRequest) -> QueueResponse:
    """
    在本 runner 的 instance 中执行命令：
      apptainer exec ... instance://INSTANCE_NAME CMD...
    仅允许 CURRENT_RUN_ID == req.run_id。
    """
    assert req.cmd and req.cwd
    workdir_host = Path(req.cwd)
    _ensure_dir(workdir_host)

    env = os.environ.copy()
    if req.env:
        env.update(req.env)

    if CURRENT_RUN_ID is None:
        msg = "no active run_id on this runner; call instance_start first"
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

    cmd = [
        APPTAINER_BIN,
        "exec",
        "--cleanenv",
        "--bind",
        f"{SHARE_ROOT}:/mnt/share",
        f"instance://{INSTANCE_NAME}",
        *req.cmd,
    ]

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
    """
    停止本 runner 的 instance：
      apptainer instance stop INSTANCE_NAME
    仅允许 CURRENT_RUN_ID == req.run_id；成功后清空 CURRENT_RUN_ID。
    """
    global CURRENT_RUN_ID

    workdir_host = Path(req.cwd or ".")
    _ensure_dir(workdir_host)

    env = os.environ.copy()
    if req.env:
        env.update(req.env)

    if CURRENT_RUN_ID is None:
        return QueueResponse(
            req_id=req.req_id,
            runner_id=req.runner_id,
            run_id=req.run_id,
            type=req.type,
            ok=True,
            returncode=0,
            stdout="",
            stderr="no active run_id; stop is a no-op",
            runtime_sec=0.0,
            error=None,
        )

    if CURRENT_RUN_ID != req.run_id:
        msg = f"runner bound to run_id={CURRENT_RUN_ID}, cannot stop for {req.run_id}"
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

    cmd = [
        APPTAINER_BIN,
        "instance",
        "stop",
        INSTANCE_NAME,
    ]

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
        CURRENT_RUN_ID = None

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
    t0 = _now()
    t1 = _now()

    return QueueResponse(
        req_id=req.req_id,
        runner_id=req.runner_id,
        run_id=req.run_id,
        type=req.type,
        ok=True,
        returncode=0,
        stdout="",
        stderr="",
        runtime_sec=t1 - t0,
        error=None,
    )


if __name__ == "__main__":
    main()
