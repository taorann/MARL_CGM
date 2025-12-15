from __future__ import annotations

import os
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Mapping

from .queue_protocol import (
    QueueRequest,
    QueueResponse,
    ExecResult,
    runner_inbox,
    runner_outbox,
)


class ApptainerQueueRuntime:
    def __init__(
        self,
        queue_root: Path,
        sif_dir: Path,
        num_runners: int,
        *,
        default_timeout_sec: float = 900.0,
        poll_interval_sec: float = 0.5,
        max_stdout_bytes: int = int(os.environ.get("GP_MAX_STDOUT_BYTES", "20000000")),
    ) -> None:
        self.queue_root = Path(queue_root)
        self.sif_dir = Path(sif_dir)
        self.num_runners = int(num_runners)
        self.default_timeout_sec = float(default_timeout_sec)
        self.poll_interval_sec = float(poll_interval_sec)
        self.max_stdout_bytes = int(max_stdout_bytes)
        self._run_to_runner: Dict[str, int] = {}

        # Runner liveness / startup waiting (for remote_swe).
        self.heartbeat_ttl_sec = float(os.environ.get("GP_RUNNER_HEARTBEAT_TTL_SEC", "180"))
        self.wait_for_runners_sec = float(os.environ.get("GP_WAIT_FOR_RUNNERS_SEC", "60"))
    # ----------------------
    # 一次性容器（原有接口）
    # ----------------------
    def exec(
        self,
        *,
        run_id: str,
        docker_image: str,
        cmd: List[str],
        cwd: Path,
        env: Optional[Mapping[str, str]] = None,
        timeout_sec: Optional[float] = None,
        meta: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        runner_id = self._choose_runner(run_id)
        sif_path = self._image_to_sif(docker_image)
        req = QueueRequest(
            req_id=self._new_req_id(),
            runner_id=runner_id,
            run_id=run_id,
            type="exec",
            image=docker_image,
            sif_path=str(sif_path),
            cmd=list(cmd),
            cwd=str(cwd),
            env=dict(env or {}),
            timeout_sec=float(timeout_sec or self.default_timeout_sec),
            meta=meta or {},
        )
        resp = self._roundtrip(req)
        return self._resp_to_exec_result(resp)

    # ----------------------
    # 新增：instance 会话接口
    # ----------------------
    def start_instance(
        self,
        *,
        run_id: str,
        docker_image: str,
        cwd: Path,
        env: Optional[Mapping[str, str]] = None,
        timeout_sec: Optional[float] = None,
        meta: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        """
        启动一个 Apptainer instance（每条轨迹一个），run_id 用来标识 instance。
        对应 runner 里的 handle_instance_start。
        """
        runner_id = self._choose_runner(run_id)
        sif_path = self._image_to_sif(docker_image)
        req = QueueRequest(
            req_id=self._new_req_id(),
            runner_id=runner_id,
            run_id=run_id,
            type="instance_start",
            image=docker_image,
            sif_path=str(sif_path),
            cmd=[],  # start 不需要命令
            cwd=str(cwd),
            env=dict(env or {}),
            timeout_sec=float(timeout_sec or self.default_timeout_sec),
            meta=meta or {},
        )
        resp = self._roundtrip(req)
        return self._resp_to_exec_result(resp)

    def exec_in_instance(
        self,
        *,
        run_id: str,
        docker_image: str,
        cmd: List[str],
        cwd: Path,
        env: Optional[Mapping[str, str]] = None,
        timeout_sec: Optional[float] = None,
        meta: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        """
        在已启动的 instance 中执行一条命令（多步交互）。
        对应 runner 里的 handle_instance_exec。
        """
        runner_id = self._choose_runner(run_id)
        sif_path = self._image_to_sif(docker_image)
        req = QueueRequest(
            req_id=self._new_req_id(),
            runner_id=runner_id,
            run_id=run_id,
            type="instance_exec",
            image=docker_image,
            sif_path=str(sif_path),  # runner 现在不强依赖，但留着方便调试
            cmd=list(cmd),
            cwd=str(cwd),
            env=dict(env or {}),
            timeout_sec=float(timeout_sec or self.default_timeout_sec),
            meta=meta or {},
        )
        resp = self._roundtrip(req)
        return self._resp_to_exec_result(resp)

    def stop_instance(
        self,
        *,
        run_id: str,
        cwd: Path,
        env: Optional[Mapping[str, str]] = None,
        timeout_sec: Optional[float] = None,
        meta: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        """
        停止一个 instance（轨迹结束）。
        对应 runner 里的 handle_instance_stop。
        """
        runner_id = self._choose_runner(run_id)
        req = QueueRequest(
            req_id=self._new_req_id(),
            runner_id=runner_id,
            run_id=run_id,
            type="instance_stop",
            image="",      # runner 不需要
            sif_path="",   # runner 不需要
            cmd=[],
            cwd=str(cwd),
            env=dict(env or {}),
            timeout_sec=float(timeout_sec or self.default_timeout_sec),
            meta=meta or {},
        )
        resp = self._roundtrip(req)
        return self._resp_to_exec_result(resp)

    # ----------------------
    # 文件 put/get & cleanup（原有逻辑）
    # ----------------------
    def put_file(self, *, run_id: str, src: Path, dst: Path, meta: Optional[Dict[str, str]] = None) -> None:
        runner_id = self._choose_runner(run_id)
        req = QueueRequest(
            req_id=self._new_req_id(),
            runner_id=runner_id,
            run_id=run_id,
            type="put",
            src=str(src),
            dst=str(dst),
            meta=meta or {},
        )
        self._roundtrip(req)

    def get_file(self, *, run_id: str, src: Path, dst: Path, meta: Optional[Dict[str, str]] = None) -> None:
        runner_id = self._choose_runner(run_id)
        req = QueueRequest(
            req_id=self._new_req_id(),
            runner_id=runner_id,
            run_id=run_id,
            type="get",
            src=str(src),
            dst=str(dst),
            meta=meta or {},
        )
        self._roundtrip(req)

    def cleanup_run(self, run_id: str) -> None:
        runner_id = self._choose_runner(run_id)
        req = QueueRequest(
            req_id=self._new_req_id(),
            runner_id=runner_id,
            run_id=run_id,
            type="cleanup",
        )
        self._roundtrip(req)

    # ----------------------
    # 内部工具函数
    # ----------------------

    def _heartbeat_path(self, rid: int) -> Path:
        return self.queue_root / f"runner-{rid}" / "heartbeat.json"

    def _alive_runner_ids(self) -> List[int]:
        """Return runner ids considered 'alive' based on heartbeat.json timestamps."""
        now = time.time()
        alive: List[int] = []
        # only consider [0, num_runners) to avoid routing to ids we never created
        for rid in range(max(int(self.num_runners), 1)):
            hp = self._heartbeat_path(rid)
            if not hp.exists():
                continue
            try:
                data = json.loads(hp.read_text(encoding="utf-8"))
                ts = float(data.get("ts", 0.0) or 0.0)
            except Exception:
                ts = hp.stat().st_mtime
            if now - ts <= float(self.heartbeat_ttl_sec):
                alive.append(rid)
        return alive

    def _maybe_reroute_to_alive(self, req: QueueRequest) -> None:
        """If there are alive runners and req.runner_id is not one of them, reroute."""
        alive = self._alive_runner_ids()
        if not alive:
            # optionally wait a bit for runners to come up, to avoid 15min 'start' hangs
            if float(self.wait_for_runners_sec) <= 0:
                return
            deadline = time.time() + float(self.wait_for_runners_sec)
            while time.time() < deadline:
                time.sleep(min(float(self.poll_interval_sec), 1.0))
                alive = self._alive_runner_ids()
                if alive:
                    break
            if not alive:
                return

        if int(req.runner_id) in alive:
            return

        # deterministic mapping to currently alive runners
        new_rid = alive[hash(req.run_id) % len(alive)]
        if req.meta is None:
            req.meta = {}
        req.meta["reroute_from"] = str(req.runner_id)
        req.meta["reroute_to"] = str(new_rid)
        req.runner_id = int(new_rid)

    def _choose_runner(self, run_id: str) -> int:
        # Prefer alive runners to avoid routing to PENDING / dead runners (common on Slurm).
        alive = self._alive_runner_ids()
        rid = self._run_to_runner.get(run_id)

        if alive:
            if rid is not None and int(rid) in alive:
                return int(rid)
            new_rid = alive[hash(run_id) % len(alive)]
            self._run_to_runner[run_id] = int(new_rid)
            return int(new_rid)

        # Fallback: deterministic modulo (may hang if Slurm hasn't started runners yet).
        if rid is None:
            rid = hash(run_id) % max(int(self.num_runners), 1)
            self._run_to_runner[run_id] = int(rid)
        return int(rid)

    def _new_req_id(self) -> str:
        return uuid.uuid4().hex

    def _image_to_sif(self, docker_image: str) -> Path:
        normalized = (
            docker_image.replace("/", "-")
            .replace(":", "-")
            .replace("@", "-")
        )
        return self.sif_dir / f"{normalized}.sif"

    def _roundtrip(self, req: QueueRequest) -> QueueResponse:
        # Reroute to an alive runner if possible (prevents long timeouts when some runners are pending).
        self._maybe_reroute_to_alive(req)
        root = self.queue_root
        inbox = runner_inbox(root, req.runner_id)
        outbox = runner_outbox(root, req.runner_id)
        inbox.mkdir(parents=True, exist_ok=True)
        outbox.mkdir(parents=True, exist_ok=True)

        req_path = inbox / f"{req.req_id}.json"
        with req_path.open("w", encoding="utf-8") as f:
            json.dump(req.__dict__, f, ensure_ascii=False)

        resp_path = outbox / f"{req.req_id}.json"
        deadline = time.time() + (req.timeout_sec or self.default_timeout_sec) * 1.5
        while True:
            if resp_path.exists():
                with resp_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return QueueResponse(**data)
            if time.time() > deadline:
                raise TimeoutError(f"Timeout waiting for response for {req.req_id}")
            time.sleep(self.poll_interval_sec)

    def _resp_to_exec_result(self, resp: QueueResponse) -> ExecResult:
        stdout = (resp.stdout or "")[: self.max_stdout_bytes]
        stderr = (resp.stderr or "")[: self.max_stdout_bytes]
        if not resp.ok:
            return ExecResult(
                returncode=resp.returncode if resp.returncode is not None else -1,
                stdout=stdout,
                stderr=stderr,
                runtime_sec=resp.runtime_sec or 0.0,
                ok=False,
                error=resp.error or "ApptainerQueueRuntime request failed",
            )
        return ExecResult(
            returncode=resp.returncode or 0,
            stdout=stdout,
            stderr=stderr,
            runtime_sec=resp.runtime_sec or 0.0,
            ok=True,
            error=None,
        )

