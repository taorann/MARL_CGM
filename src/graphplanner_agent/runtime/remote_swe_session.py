from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any

from graphplanner_agent.config.schema import default_remote_swe_ssh_args


class RemoteSweError(RuntimeError):
    pass


def normalize_sif_image_ref(image_ref: str) -> str:
    ref = str(image_ref or "").strip()
    if not ref:
        return ref
    if ref.endswith(".sif"):
        return Path(ref).stem
    return ref


def infer_sif_dir_from_ref(image_ref: str) -> str | None:
    ref = str(image_ref or "").strip()
    if not ref.endswith(".sif"):
        return None
    parent = Path(ref).parent
    if str(parent) in {"", "."}:
        return None
    return str(parent)


def _preview(value: str, limit: int = 2000) -> str:
    text = value or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"...<truncated {len(text) - limit} chars>"


def summarize_proxy_stdout(op: str, stdout: str) -> str:
    text = stdout or ""
    if op == "build_repo_graph" or (len(text) > 5000 and text.lstrip().startswith("H4sI")):
        head = text.lstrip()[:80].replace("\n", "")
        return f"<omitted base64+gzip payload len={len(text)} head={head!r}>"
    return _preview(text)


def _parse_proxy_response(stdout: str) -> dict[str, Any]:
    text = (stdout or "").strip()
    if not text:
        raise RemoteSweError("remote swe_proxy returned empty stdout")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if payload is None:
            raise RemoteSweError(f"remote swe_proxy returned non-JSON stdout: {_preview(text)}")
    if not isinstance(payload, dict):
        raise RemoteSweError("remote swe_proxy response must be a JSON object")
    return payload


def clean_remote_stderr(stderr: str) -> str:
    lines = []
    for line in (stderr or "").splitlines():
        text = line.strip()
        if not text:
            continue
        if "Permanently added" in text and "known hosts" in text:
            continue
        if text.startswith("Warning: Permanently added"):
            continue
        if text.startswith("Shared connection to ") and text.endswith(" closed."):
            continue
        if text.startswith("Connection to ") and text.endswith(" closed."):
            continue
        if text.startswith("debug1:") or text.startswith("debug2:") or text.startswith("debug3:"):
            continue
        lines.append(line)
    return "\n".join(lines)


@dataclass(slots=True)
class RemoteSweSession:
    ssh_target: str
    remote_repo: str
    image: str
    run_id: str
    remote_python: str = "python"
    swe_proxy_path: str = "hpc/swe_proxy.py"
    runner_manager_path: str = "hpc/ensure_runners.py"
    num_runners: int = 1
    ensure_runners: bool = True
    ssh_args: str | None = None
    sif_dir: str | None = None
    _runners_ensured: bool = False

    def __post_init__(self) -> None:
        original = self.image
        if not self.sif_dir:
            self.sif_dir = infer_sif_dir_from_ref(original)
        self.image = normalize_sif_image_ref(original)

    def _ssh_base_cmd(self) -> list[str]:
        cmd = ["ssh", "-o", "BatchMode=yes"]
        connect_timeout = (os.environ.get("GP_REMOTE_SWE_SSH_CONNECT_TIMEOUT") or "").strip()
        if connect_timeout:
            cmd.extend(["-o", f"ConnectTimeout={connect_timeout}"])
        insecure = os.environ.get("GP_REMOTE_SWE_INSECURE_SSH", "").strip().lower() in {"1", "true", "yes"}
        if insecure:
            cmd.extend(["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"])
        extra_args = (
            self.ssh_args
            or os.environ.get("GP_REMOTE_SWE_SSH_ARGS")
            or default_remote_swe_ssh_args("remote_swe")
            or ""
        ).strip()
        if extra_args:
            cmd.extend(shlex.split(extra_args))
        return cmd

    def _remote_env_prefix(self) -> str:
        repo = shlex.quote(self.remote_repo)
        parts = [
            f"GP_NUM_RUNNERS={int(self.num_runners)}",
            "PYTHONDONTWRITEBYTECODE=1",
            f"GRAPHPLANNER_REMOTE_REPO={repo}",
            f"GP_REMOTE_REPO={repo}",
        ]
        queue_root = (
            os.environ.get("GRAPHPLANNER_SANDBOX_QUEUE_ROOT")
            or os.environ.get("GP_QUEUE_ROOT")
            or os.environ.get("QUEUE_ROOT")
        )
        if queue_root:
            parts.append(f"QUEUE_ROOT={shlex.quote(queue_root)}")
        share_root = (
            os.environ.get("GRAPHPLANNER_SANDBOX_SHARE_ROOT")
            or os.environ.get("GP_SHARE_ROOT")
            or os.environ.get("SHARE_ROOT")
        )
        if share_root:
            parts.append(f"SHARE_ROOT={shlex.quote(share_root)}")
        if self.sif_dir:
            sif_dir = shlex.quote(self.sif_dir)
            parts.extend([f"GP_SIF_DIR={sif_dir}", f"SIF_DIR={sif_dir}"])
        for name in (
            "GRAPHPLANNER_REMOTE_HEARTBEAT_TTL_SEC",
            "GP_ENSURE_RUNNER_HEARTBEAT_TTL_SEC",
            "GP_ENSURE_RUNNER_GRACE_SEC",
            "RUNNER_PARTITION",
            "RUNNER_QOS",
        ):
            value = os.environ.get(name)
            if value:
                parts.append(f"{name}={shlex.quote(value)}")
        parts.append(f"PYTHONPATH=$PYTHONPATH:{repo}")
        return " ".join(parts)

    def proxy_ssh_cmd(self) -> list[str]:
        repo = shlex.quote(self.remote_repo)
        py = shlex.quote(self.remote_python or "python")
        proxy = shlex.quote(self.swe_proxy_path or "hpc/swe_proxy.py")
        remote_cmd = f"cd {repo} && {self._remote_env_prefix()} {py} {proxy}"
        return self._ssh_base_cmd() + [self.ssh_target, remote_cmd]

    def simple_ssh_cmd(self, command: str) -> list[str]:
        repo = shlex.quote(self.remote_repo)
        remote_cmd = f"cd {repo} && {command}"
        return self._ssh_base_cmd() + [self.ssh_target, remote_cmd]

    def ensure_remote_runners(self, timeout: float = 180.0) -> None:
        if not self.ensure_runners or self._runners_ensured:
            return
        repo = shlex.quote(self.remote_repo)
        py = shlex.quote(self.remote_python or "python")
        manager = shlex.quote(self.runner_manager_path or "hpc/ensure_runners.py")
        remote_cmd = f"cd {repo} && {self._remote_env_prefix()} {py} {manager} --target {int(self.num_runners)}"
        proc = subprocess.run(
            self._ssh_base_cmd() + [self.ssh_target, remote_cmd],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            raise RemoteSweError(
                "remote ensure_runners failed "
                f"rc={proc.returncode} stdout={_preview(proc.stdout)} stderr={_preview(clean_remote_stderr(proc.stderr))}"
            )
        self._runners_ensured = True

    def _call_proxy(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        if self.ensure_runners:
            self.ensure_remote_runners()
        payload_timeout = float(payload.get("timeout") or 0.0)
        ssh_timeout = float(timeout or 0.0) or payload_timeout
        ssh_timeout = max(ssh_timeout, payload_timeout * 2.0 + 30.0, 120.0)
        raw = json.dumps(payload, ensure_ascii=False)
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                self.proxy_ssh_cmd(),
                input=raw,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=ssh_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Remote swe_proxy timed out after {ssh_timeout:.1f}s") from exc
        runtime = time.perf_counter() - started
        if proc.returncode != 0:
            op = str(payload.get("op") or "exec")
            raise RemoteSweError(
                f"remote swe_proxy failed rc={proc.returncode} op={op!r} runtime={runtime:.1f}s "
                f"stdout={summarize_proxy_stdout(op, proc.stdout)} stderr={_preview(clean_remote_stderr(proc.stderr))}"
            )
        return _parse_proxy_response(proc.stdout)

    def start(self, timeout: float | None = None, cwd: str = "/testbed") -> dict[str, Any]:
        effective_timeout = max(float(timeout or 0.0), 300.0)
        return self._call_proxy(
            {
                "op": "start",
                "run_id": self.run_id,
                "image": self.image,
                "timeout": effective_timeout,
                "cwd": cwd or "/testbed",
            },
            timeout=effective_timeout,
        )

    def cleanup_pool(self, timeout: float | None = None, cwd: str = "/testbed") -> dict[str, Any]:
        effective_timeout = max(float(timeout or 0.0), 90.0)
        return self._call_proxy(
            {
                "op": "cleanup_pool",
                "run_id": self.run_id,
                "timeout": effective_timeout,
                "cwd": cwd or "/testbed",
            },
            timeout=effective_timeout,
        )

    def exec(self, cmd: str, *, cwd: str | None = None, env: dict[str, str] | None = None, timeout: float | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
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

    def stop(self, timeout: float | None = None) -> dict[str, Any]:
        return self._call_proxy({"op": "stop", "run_id": self.run_id, "timeout": timeout or 600.0}, timeout=timeout)

    def build_repo_graph(self, repo_id: str = "", timeout: int = 1200, *, cwd: str | None = None, repo: str | None = None) -> str:
        payload: dict[str, Any] = {
            "op": "build_repo_graph",
            "run_id": self.run_id,
            "image": self.image,
            "repo_id": repo_id,
            "timeout": float(timeout),
            "max_stdout_bytes": int(os.environ.get("GP_BUILD_REPO_GRAPH_STDOUT_BYTES", "200000000")),
        }
        if cwd is not None:
            payload["cwd"] = cwd
        if repo is not None:
            payload["repo"] = repo
        resp = self._call_proxy(payload, timeout=float(timeout))
        if not resp.get("ok", False):
            raise RemoteSweError(
                f"remote build_repo_graph failed rc={resp.get('returncode')} "
                f"error={resp.get('error')} stderr={_preview(str(resp.get('stderr') or ''))}"
            )
        raw = str(resp.get("stdout") or "").strip()
        if not raw:
            raise RemoteSweError("remote build_repo_graph returned empty stdout")
        return raw

    def check_remote_layout(self, timeout: float = 30.0) -> dict[str, Any]:
        py = shlex.quote(self.remote_python or "python")
        proxy = shlex.quote(self.swe_proxy_path or "hpc/swe_proxy.py")
        manager = shlex.quote(self.runner_manager_path or "hpc/ensure_runners.py")
        cmd = (
            "printf 'remote_repo=%s\\n' \"$PWD\"; "
            "whoami; hostname; "
            f"test -f {proxy} && echo swe_proxy_ok || echo swe_proxy_missing; "
            f"test -f {manager} && echo runner_manager_ok || echo runner_manager_missing; "
            f"{py} -V"
        )
        proc = subprocess.run(
            self.simple_ssh_cmd(cmd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {"ok": proc.returncode == 0, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": clean_remote_stderr(proc.stderr)}
