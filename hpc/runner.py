from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

from graph_planner.runtime.queue_protocol import (
    QueueRequest,
    QueueResponse,
    runner_inbox,
    runner_outbox,
)

APPTAINER_BIN = os.environ.get("APPTAINER_BIN", "singularity")
QUEUE_ROOT = Path(os.environ["QUEUE_ROOT"])
RUNNER_ID = int(os.environ["RUNNER_ID"])
SHARE_ROOT = Path(os.environ["SHARE_ROOT"])
CONTAINER_SHARE_ROOT = "/mnt/share"
POLL_INTERVAL_SEC = float(os.environ.get("RUNNER_POLL_INTERVAL", "0.5"))


def main() -> None:
    inbox = runner_inbox(QUEUE_ROOT, RUNNER_ID)
    outbox = runner_outbox(QUEUE_ROOT, RUNNER_ID)
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)

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

    # 根据 type 分派
    if req.type == "exec":
        resp = handle_exec(req)
    elif req.type == "instance_start":
        resp = handle_instance_start(req)
    elif req.type == "instance_exec":
        resp = handle_instance_exec(req)
    elif req.type == "instance_stop":
        resp = handle_instance_stop(req)
    else:
        # 其他类型目前只是简单 ack，一律 ok=True
        resp = QueueResponse(
            req_id=req.req_id,
            runner_id=req.runner_id,
            run_id=req.run_id,
            type=req.type,
            ok=True,
        )

    resp_path = outbox / f"{resp.req_id}.json"
    resp_path.write_text(
        json.dumps(resp.__dict__, ensure_ascii=False),
        encoding="utf-8",
    )


# 原有：一次性容器 exec
def handle_exec(req: QueueRequest) -> QueueResponse:
    assert req.sif_path and req.cmd and req.cwd
    workdir_host = Path(req.cwd)
    workdir_host.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if req.env:
        env.update(req.env)

    cmd = [
        APPTAINER_BIN,
        "exec",
        "--cleanenv",
        "--bind",
        f"{SHARE_ROOT}:{CONTAINER_SHARE_ROOT}",
        req.sif_path,
        *req.cmd,
    ]

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(workdir_host),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=req.timeout_sec or None,
    )
    t1 = time.time()

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


# 新增：启动一个 instance（轨迹开始）
def handle_instance_start(req: QueueRequest) -> QueueResponse:
    assert req.sif_path
    inst_name = f"gp-{req.run_id}"

    env = os.environ.copy()
    if req.env:
        env.update(req.env)

    # instance start 不太关心 cwd，用 SHARE_ROOT 保证目录存在即可
    cmd = [
        APPTAINER_BIN,
        "instance",
        "start",
        "--cleanenv",
        "--bind",
        f"{SHARE_ROOT}:{CONTAINER_SHARE_ROOT}",
        req.sif_path,
        inst_name,
    ]

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(SHARE_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=req.timeout_sec or None,
    )
    t1 = time.time()

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


# 新增：在已有 instance 里执行命令（轨迹中多步交互）
def handle_instance_exec(req: QueueRequest) -> QueueResponse:
    assert req.cmd
    inst_name = f"gp-{req.run_id}"

    workdir_host = Path(req.cwd or str(SHARE_ROOT))
    workdir_host.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if req.env:
        env.update(req.env)

    cmd = [
        APPTAINER_BIN,
        "exec",
        "--cleanenv",
        "--bind",
        f"{SHARE_ROOT}:{CONTAINER_SHARE_ROOT}",
        f"instance://{inst_name}",
        *req.cmd,
    ]

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(workdir_host),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=req.timeout_sec or None,
    )
    t1 = time.time()

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


# 新增：停止 instance（轨迹结束）
def handle_instance_stop(req: QueueRequest) -> QueueResponse:
    inst_name = f"gp-{req.run_id}"

    env = os.environ.copy()
    if req.env:
        env.update(req.env)

    cmd = [
        APPTAINER_BIN,
        "instance",
        "stop",
        inst_name,
    ]

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(SHARE_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=req.timeout_sec or None,
    )
    t1 = time.time()

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


if __name__ == "__main__":
    main()

