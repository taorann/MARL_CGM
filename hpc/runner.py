from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
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

# Runner liveness markers (used by ApptainerQueueRuntime to avoid routing to pending runners)
RUNNER_ROOT = QUEUE_ROOT / f"runner-{RUNNER_ID}"
READY_PATH = RUNNER_ROOT / "ready.json"
HEARTBEAT_PATH = RUNNER_ROOT / "heartbeat.json"

# 当前 runner 正在服务的 run_id；None 表示空闲
CURRENT_RUN_ID: Optional[str] = None


def _image_to_sif(image: str) -> Path:
    """Best-effort map a docker_image string to a local .sif path.

    This matches ApptainerQueueRuntime._image_to_sif and is used as a compatibility
    fallback if a legacy request omits `sif_path`.
    """
    normalized = image.replace("/", "-").replace(":", "-").replace("@", "-")
    sif_dir = Path(
        os.environ.get("GP_SIF_DIR")
        or os.environ.get("SIF_DIR")
        or str(Path.home() / "sif" / "sweb")
    ).expanduser().resolve()
    return sif_dir / f"{normalized}.sif"


def _coerce_legacy_to_queue_request(data: Dict[str, Any], *, req_id_fallback: str) -> QueueRequest:
    """Coerce older/looser request dicts into the QueueRequest schema.

    We have seen cases where the proxy writes an 'op' style payload directly
    into the runner inbox (missing req_id/runner_id/type). To avoid hard-failing
    on the runner side, we translate the minimal fields.
    """

    run_id = str(data.get("run_id") or "")
    if not run_id:
        run_id = "__default__"

    # Determine request type.
    typ = data.get("type")
    if not typ:
        op = str(data.get("op") or "").strip().lower()
        if op == "start":
            typ = "instance_start"
        elif op == "exec":
            typ = "instance_exec"
        elif op == "stop":
            typ = "instance_stop"
        elif op:
            # Unknown op -> noop
            typ = "noop"
        else:
            typ = "exec"  # legacy one-shot exec
    typ = str(typ)

    # Command normalization.
    cmd_val = data.get("cmd")
    cmd_list = None
    if isinstance(cmd_val, list):
        cmd_list = [str(x) for x in cmd_val]
    elif isinstance(cmd_val, str) and cmd_val:
        cmd_list = ["bash", "-lc", cmd_val]

    image = data.get("image")
    sif_path = data.get("sif_path")
    if not sif_path and isinstance(image, str) and image:
        sif_path = str(_image_to_sif(image))

    timeout_sec = data.get("timeout_sec")
    if timeout_sec is None:
        timeout_sec = data.get("timeout")
    try:
        timeout_sec = float(timeout_sec) if timeout_sec is not None else None
    except Exception:
        timeout_sec = None

    rid = data.get("runner_id")
    try:
        rid = int(rid) if rid is not None else RUNNER_ID
    except Exception:
        rid = RUNNER_ID

    req_id = str(data.get("req_id") or req_id_fallback or uuid.uuid4().hex)

    return QueueRequest(
        req_id=req_id,
        runner_id=rid,
        run_id=run_id,
        type=typ,  # type: ignore[arg-type]
        image=str(image) if image is not None else None,
        sif_path=str(sif_path) if sif_path is not None else None,
        cmd=cmd_list,
        cwd=str(data.get("cwd")) if data.get("cwd") is not None else None,
        env=dict(data.get("env") or {}),
        timeout_sec=timeout_sec,
        src=str(data.get("src")) if data.get("src") is not None else None,
        dst=str(data.get("dst")) if data.get("dst") is not None else None,
        meta=dict(data.get("meta") or {}),
    )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now() -> float:
    return time.time()


def _write_ready() -> None:
    meta = {
        "rid": RUNNER_ID,
        "pid": os.getpid(),
        "host": os.uname().nodename,
        "ts": _now(),
    }
    _ensure_dir(RUNNER_ROOT)
    READY_PATH.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def _write_heartbeat(extra: Optional[Dict[str, Any]] = None) -> None:
    meta: Dict[str, Any] = {
        "rid": RUNNER_ID,
        "pid": os.getpid(),
        "host": os.uname().nodename,
        "ts": _now(),
        "current_run_id": CURRENT_RUN_ID,
    }
    if extra:
        meta.update(extra)
    _ensure_dir(RUNNER_ROOT)
    HEARTBEAT_PATH.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")



def _resolve_cwd(req_cwd: str | None, run_id: str) -> tuple[Path, str | None]:
    """Split cwd into a safe host-side cwd and an optional container-side pwd.

    remote_swe often passes container paths like "/repo". The runner must NOT treat
    them as host paths (would attempt mkdir /repo and fail). We instead:
      - use a per-run directory under $SHARE_ROOT/gp_work as host cwd;
      - pass the container path via apptainer --pwd when it looks like an absolute POSIX path.
    """
    container_pwd: str | None = None
    if isinstance(req_cwd, str) and req_cwd.startswith('/'):
        container_pwd = req_cwd

    host_base = Path(os.environ.get('RUNNER_WORKDIR_HOST', str(SHARE_ROOT / 'gp_work')))
    host_cwd = host_base / run_id / INSTANCE_NAME
    _ensure_dir(host_cwd)
    return host_cwd, container_pwd

def main() -> None:
    inbox = runner_inbox(QUEUE_ROOT, RUNNER_ID)
    outbox = runner_outbox(QUEUE_ROOT, RUNNER_ID)
    _ensure_dir(inbox)
    _ensure_dir(outbox)

    # mark runner as ready/alive for proxy-side routing
    _write_ready()
    _write_heartbeat()

    last_hb = 0.0
    while True:
        now = _now()
        if now - last_hb >= 2.0:
            _write_heartbeat()
            last_hb = now

        handled = False
        for req_path in sorted(inbox.glob("*.json")):
            handled = True
            handle_one_request(req_path, outbox)  # must not raise
            try:
                req_path.unlink()
            except FileNotFoundError:
                pass

        if not handled:
            time.sleep(POLL_INTERVAL_SEC)


def handle_one_request(req_path: Path, outbox: Path) -> None:
    """Always write a response JSON, even if execution fails."""
    req_id = req_path.stem
    # If request parsing fails, we still want to include run_id (best effort) in the response.
    parsed_run_id: str = ""
    try:
        data: Dict[str, Any] = json.loads(req_path.read_text(encoding="utf-8"))
        parsed_run_id = str(data.get("run_id") or "")
        try:
            req = QueueRequest(**data)
        except TypeError:
            # Backward compatibility: accept legacy payloads (missing req_id/runner_id/type).
            req = _coerce_legacy_to_queue_request(data, req_id_fallback=req_id)
            parsed_run_id = req.run_id

        # NOTE: CURRENT_RUN_ID means "which run_id this runner instance is currently bound to".
        # It must NOT be overwritten on every incoming request, otherwise instance_start becomes
        # a no-op and later instance_exec will try to exec into a non-existent instance.
        _write_heartbeat({"last_req_id": req.req_id, "type": req.type, "req_run_id": req.run_id})

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
    except Exception as e:
        # If parsing failed, still return a response keyed by filename.
        resp = QueueResponse(
            req_id=req_id,
            runner_id=RUNNER_ID,
            run_id=parsed_run_id or (CURRENT_RUN_ID or ""),
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

    cmd = [
        APPTAINER_BIN,
        "exec",
        "--cleanenv",
        "--bind",
        f"{SHARE_ROOT}:/mnt/share",
    ]
    if container_pwd:
        cmd.extend(["--pwd", container_pwd])

    cmd.extend([
        req.sif_path,
        *req.cmd,
    ])

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
    """Start this runner's single apptainer instance.

    Rules:
      - If CURRENT_RUN_ID is None: bind to req.run_id and start.
      - If CURRENT_RUN_ID == req.run_id: idempotent start (ok=True).
      - Otherwise: return busy.
    """
    global CURRENT_RUN_ID
    assert req.sif_path

    workdir_host, _ = _resolve_cwd(req.cwd, req.run_id)

    env = os.environ.copy()
    if req.env:
        env.update(req.env)

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
    """Execute a command inside this runner's apptainer instance."""
    global CURRENT_RUN_ID
    assert req.cmd

    workdir_host, container_pwd = _resolve_cwd(req.cwd, req.run_id)

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
    ]
    if container_pwd:
        cmd.extend(["--pwd", container_pwd])
    cmd.extend([
        f"instance://{INSTANCE_NAME}",
        *req.cmd,
    ])

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
    """Stop this runner's apptainer instance."""
    global CURRENT_RUN_ID

    workdir_host, _ = _resolve_cwd(req.cwd, req.run_id)

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
