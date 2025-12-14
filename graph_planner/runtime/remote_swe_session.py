#!/usr/bin/env python
from __future__ import annotations

import json
import os
import sys
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional

def _dbg(msg: str) -> None:
    if os.environ.get("DEBUG") or os.environ.get("EBUG"):
        print(f"[remote_swe_session] {msg}", file=sys.stderr)




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
        _dbg(f"call op={payload.get('op')!r} run_id={payload.get('run_id')!r} image={payload.get('image')!r} cwd={payload.get('cwd')!r} timeout={timeout}")
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

        repo = shlex.quote(self.remote_repo)
        cd_cmd = f"cd {repo}"
        py = shlex.quote(self.remote_python or "python")
        manager = shlex.quote(self.runner_manager_path or "hpc/ensure_runners.py")
        env_prefix = (
            f"GP_NUM_RUNNERS={int(self.num_runners)} "
            f"PYTHONPATH=$PYTHONPATH:{repo}"
        )
        remote_cmd = f"{cd_cmd} && {env_prefix} {py} {manager} --target {int(self.num_runners)}"

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
        # Allow plenty of time for the initial container bootstrap even if callers
        # pass a smaller timeout (e.g., snippet-level defaults).
        effective_timeout = max(float(timeout or 0.0), 300.0)
        payload: Dict[str, Any] = {
            "op": "start",
            "run_id": self.run_id,
            "image": self.image,
            "timeout": effective_timeout,
        }
        return self._call_proxy(payload, timeout=effective_timeout)

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

    def build_repo_graph(
        self,
        repo_id: str = "",
        timeout: int = 3600,
        *,
        cwd: Optional[str] = None,
        repo: Optional[str] = None,
    ) -> str:
        """Build a repo-level graph in the remote SWE container and return base64(gzip(JSONL)).

        This calls swe_proxy with op='build_repo_graph'. The returned base64 string should be
        decoded client-side (SandboxRuntime) and can be cached on the host.

        Parameters
        ----------
        repo_id:
            Optional repo identifier (for logging/caching keys).
        timeout:
            Proxy-side timeout in seconds.
        cwd, repo:
            Optional remote-side working directory and repo mount path.
        """
        payload: Dict[str, Any] = {
            "op": "build_repo_graph",
            "run_id": self.run_id,
            "image": self.image,
            "repo_id": repo_id,
            "timeout": float(timeout),
        }
        if cwd is not None:
            payload["cwd"] = cwd
        if repo is not None:
            payload["repo"] = repo

        resp = self._call_proxy(payload, timeout=float(timeout))
        if not resp.get("ok", False):
            raise RuntimeError(
                f"remote build_repo_graph failed (rc={resp.get('returncode')}). "
                f"stderr={resp.get('stderr', '')}"
            )

        raw = (resp.get("stdout") or "").strip()
        if not raw:
            raise RuntimeError("remote build_repo_graph returned empty stdout")
        return raw

    def build_graph(
        self,
        issue_id: str,
        timeout: Optional[float] = None,
        *,
        cwd: Optional[str] = None,
        repo: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Construct an issue-specific code subgraph inside the remote SWE container."""

        payload: Dict[str, Any] = {
            "op": "build_graph",
            "run_id": self.run_id,
            "image": self.image,
            "issue_id": issue_id,
            "timeout": timeout or 600.0,
        }
        if cwd is not None:
            payload["cwd"] = cwd
        if repo is not None:
            payload["repo"] = repo

        resp = self._call_proxy(payload, timeout=timeout)
        if not resp.get("ok", False):
            raise RuntimeError(
                f"remote build_graph failed (rc={resp.get('returncode')}). "
                f"stderr={resp.get('stderr', '')}"
            )

        raw = (resp.get("stdout") or "").strip()
        if not raw:
            raise RuntimeError("remote build_graph returned empty stdout")
        try:
            return json.loads(raw)
        except Exception as exc:  # pragma: no cover - passthrough
            raise RuntimeError(
                f"failed to parse build_graph stdout as JSON: {raw[:200]!r}"
            ) from exc
