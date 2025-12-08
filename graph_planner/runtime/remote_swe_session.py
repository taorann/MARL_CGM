#!/usr/bin/env python
from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RemoteSweSession:
    """
    Manage SSH calls to login24's swe_proxy plus runner bootstrap.

    * Optionally call hpc/ensure_runners.py to guarantee enough gp_runner_* jobs.
    * Forward start/exec/stop JSON requests to swe_proxy.py with GP_NUM_RUNNERS set.
    """

    ssh_target: str
    remote_repo: str
    image: str
    run_id: str

    remote_python: str = "python"
    swe_proxy_path: str = "hpc/swe_proxy.py"
    runner_manager_path: str = "hpc/ensure_runners.py"

    num_runners: int = 1
    ensure_runners: bool = True

    _runners_ensured: bool = False

    def _build_ssh_cmd(self) -> list[str]:
        repo = shlex.quote(self.remote_repo)
    
        cd_cmd = f"cd {repo}"
        env_prefix = (
            f"GP_NUM_RUNNERS={int(self.num_runners)} "
            f"PYTHONPATH=$PYTHONPATH:{repo}"
        )
    
        py = shlex.quote(self.remote_python or "python")
        proxy = shlex.quote(self.swe_proxy_path or "hpc/swe_proxy.py")
    
        remote_cmd = f"{cd_cmd} && {env_prefix} {py} {proxy}"
    
        return ["ssh", "-o", "BatchMode=yes", self.ssh_target, remote_cmd]


    def _call_proxy(self, payload: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        proc = subprocess.run(
            self._build_ssh_cmd(),
            input=json.dumps(payload).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Remote swe_proxy failed (rc={proc.returncode}). "
                f"stderr={proc.stderr.decode('utf-8', errors='ignore')}"
            )
        try:
            return json.loads(proc.stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to decode swe_proxy JSON response. stdout={proc.stdout!r}"
            ) from exc

    def _ensure_remote_runners(self) -> None:
        if not self.ensure_runners or self.num_runners <= 0 or self._runners_ensured:
            return

        cd_cmd = f"cd {shlex.quote(self.remote_repo)}"
        py = shlex.quote(self.remote_python or "python")
        manager = shlex.quote(self.runner_manager_path or "hpc/ensure_runners.py")
        remote_cmd = f"{cd_cmd} && {py} {manager} --target {int(self.num_runners)}"

        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", self.ssh_target, remote_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Remote ensure_runners failed (rc={proc.returncode}). "
                f"stderr={proc.stderr.decode('utf-8', errors='ignore')}"
            )
        self._runners_ensured = True

    def start(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        self._ensure_remote_runners()
        payload: Dict[str, Any] = {
            "op": "start",
            "run_id": self.run_id,
            "image": self.image,
            "timeout": timeout or 600.0,
        }
        return self._call_proxy(payload, timeout=timeout)

    def exec(
        self,
        cmd: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "op": "exec",
            "run_id": self.run_id,
            "image": self.image,
            "cmd": cmd,
            "timeout": timeout or 600.0,
        }
        if cwd is not None:
            payload["cwd"] = cwd
        if env:
            payload["env"] = env
        return self._call_proxy(payload, timeout=timeout)

    def stop(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "op": "stop",
            "run_id": self.run_id,
            "timeout": timeout or 600.0,
        }
        return self._call_proxy(payload, timeout=timeout)
