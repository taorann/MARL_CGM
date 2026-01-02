#!/usr/bin/env python
from __future__ import annotations

import json
import os
import sys
import shlex
import subprocess
import select
import time
import atexit
import signal
import weakref
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

def _dbg(msg: str) -> None:
    if os.environ.get("DEBUG") or os.environ.get("EBUG"):
        print(f"[remote_swe_session] {msg}", file=sys.stderr)


def _summarize_stdout(op: str, stdout: str, *, max_chars: int = 2000) -> str:
    """Safe preview for remote stdout.

    Some ops (notably build_repo_graph) intentionally return a large base64+gzip
    payload in stdout for host-side caching. Printing the raw payload looks like
    garbled text and can flood the console.
    """
    s = (stdout or "")
    if not s:
        return ""
    # Heuristic: build_repo_graph payload is base64 gzip and often starts with 'H4sI'.
    if op == "build_repo_graph" or (len(s) > 5000 and s.lstrip().startswith("H4sI")):
        head = s.lstrip()[:80].replace("\n", "")
        return f"<omitted base64+gzip payload len={len(s)} head={head!r}>"
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"...<truncated {len(s) - max_chars} chars>"



# -----------------------------------------------------------------------------
# Best-effort cleanup: if the local process exits (exception/KeyboardInterrupt),
# try to stop remote instances so runners are freed.
#
# Disable with: GP_DISABLE_REMOTE_ATEXIT_STOP=1
# Configure timeout with: GP_REMOTE_ATEXIT_STOP_TIMEOUT (seconds, default 60)
# -----------------------------------------------------------------------------

_ACTIVE_REMOTE_SWE_SESSIONS: "set[weakref.ReferenceType[Any]]" = set()
_CLEANUP_HOOK_INSTALLED: bool = False


def _cleanup_active_remote_swe_sessions(reason: str = "atexit") -> None:
    if os.environ.get("GP_DISABLE_REMOTE_ATEXIT_STOP", "").strip().lower() in {"1", "true", "yes"}:
        return
    try:
        timeout = float(os.environ.get("GP_REMOTE_ATEXIT_STOP_TIMEOUT", "60") or 60.0)
    except Exception:
        timeout = 60.0

    # Iterate over weakrefs to avoid keeping sessions alive.
    for ref in list(_ACTIVE_REMOTE_SWE_SESSIONS):
        sess = ref()
        if sess is None:
            continue
        try:
            # Stop should be idempotent on the proxy/runner side.
            sess.stop(timeout=timeout)
        except Exception:
            pass


def _install_cleanup_hooks_once() -> None:
    global _CLEANUP_HOOK_INSTALLED
    if _CLEANUP_HOOK_INSTALLED:
        return
    if os.environ.get("GP_DISABLE_REMOTE_ATEXIT_STOP", "").strip().lower() in {"1", "true", "yes"}:
        _CLEANUP_HOOK_INSTALLED = True
        return

    _CLEANUP_HOOK_INSTALLED = True
    try:
        atexit.register(_cleanup_active_remote_swe_sessions)
    except Exception:
        pass

    # Also handle SIGINT/SIGTERM (best-effort). SIGKILL cannot be handled.
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            prev = signal.getsignal(sig)

            def _handler(signum, frame, _prev=prev, _sig=sig):
                try:
                    _cleanup_active_remote_swe_sessions(reason=f"signal:{signum}")
                finally:
                    # Chain to previous handler.
                    if callable(_prev):
                        try:
                            _prev(signum, frame)
                            return
                        except Exception:
                            pass
                    if _prev == signal.SIG_IGN:
                        return
                    # Restore default then re-raise signal.
                    try:
                        signal.signal(_sig, signal.SIG_DFL)
                        os.kill(os.getpid(), signum)
                    except Exception:
                        return

            signal.signal(sig, _handler)
        except Exception:
            pass


def _register_active_session(sess: Any) -> None:
    try:
        _ACTIVE_REMOTE_SWE_SESSIONS.add(weakref.ref(sess))
    except Exception:
        return
    _install_cleanup_hooks_once()


def _deregister_active_session(sess: Any) -> None:
    # Remove matching weakrefs.
    try:
        dead = []
        for ref in list(_ACTIVE_REMOTE_SWE_SESSIONS):
            obj = ref()
            if obj is None or obj is sess:
                dead.append(ref)
        for ref in dead:
            _ACTIVE_REMOTE_SWE_SESSIONS.discard(ref)
    except Exception:
        pass



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
        # NOTE: login24 side uses ApptainerQueueRuntime._roundtrip(), whose own wait
        # budget is `payload.timeout * 1.5`. If we set the outer SSH timeout equal
        # to payload.timeout, the SSH layer may kill the request prematurely and
        # trigger a spurious retry (you observed duplicate `start`).
        payload_timeout = float(payload.get("timeout") or 0.0)
        ssh_timeout = float(timeout or 0.0)
        if ssh_timeout <= 0.0:
            ssh_timeout = payload_timeout
        # Give the proxy+runner time to finish: 2x payload timeout + small buffer.
        ssh_timeout = max(ssh_timeout, payload_timeout * 2.0 + 30.0, 120.0)

        _dbg(
            "call "
            f"op={payload.get('op')!r} "
            f"run_id={payload.get('run_id')!r} "
            f"image={payload.get('image')!r} "
            f"cwd={payload.get('cwd')!r} "
            f"payload_timeout={payload_timeout:.1f}s ssh_timeout={ssh_timeout:.1f}s"
        )

        t0 = time.perf_counter()
        payload_bytes = json.dumps(payload).encode("utf-8")

        proc = subprocess.Popen(
            self._build_ssh_cmd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if proc.stdin is None or proc.stdout is None or proc.stderr is None:
            try:
                proc.kill()
            except Exception:
                pass
            raise RuntimeError("Failed to spawn ssh process with stdio pipes")

        try:
            proc.stdin.write(payload_bytes)
            proc.stdin.close()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            raise

        # Stream remote stderr so long waits are visible (queue wait, Slurm pending, etc).
        stream_stdio = os.environ.get("GP_PRINT_REMOTE_STDIO", "").strip().lower() in {"1", "true", "yes"}
        hb_sec = float(os.environ.get("GP_SSH_HEARTBEAT_SEC", "0") or 0.0)
        last_hb = time.time()
        stderr_chunks: List[bytes] = []

        # We intentionally *don't* stream stdout; it must remain a single JSON object.
        while True:
            # Check for remote stderr output.
            try:
                r, _, _ = select.select([proc.stderr], [], [], 0.2)
            except Exception:
                r = []

            if r:
                try:
                    chunk = os.read(proc.stderr.fileno(), 4096)
                except Exception:
                    chunk = b""
                if chunk:
                    stderr_chunks.append(chunk)
                    if stream_stdio:
                        try:
                            sys.stderr.buffer.write(chunk)
                            sys.stderr.buffer.flush()
                        except Exception:
                            pass

            # Heartbeat while waiting (useful when stdout/stderr are quiet).
            if hb_sec > 0 and (time.time() - last_hb) >= hb_sec:
                print("[remote_swe_session] still waiting for remote_swe_proxy...", file=sys.stderr)
                last_hb = time.time()

            rc = proc.poll()
            if rc is not None:
                break

            # Enforce SSH timeout ourselves (Popen doesn't support it directly).
            if (time.perf_counter() - t0) > ssh_timeout:
                try:
                    proc.kill()
                except Exception:
                    pass
                raise TimeoutError(f"Remote swe_proxy timed out after {ssh_timeout:.1f}s")

        # Drain remaining streams.
        try:
            stdout_bytes = proc.stdout.read() or b""
        except Exception:
            stdout_bytes = b""
        try:
            rest = proc.stderr.read() or b""
            if rest:
                stderr_chunks.append(rest)
                if stream_stdio:
                    try:
                        sys.stderr.buffer.write(rest)
                        sys.stderr.buffer.flush()
                    except Exception:
                        pass
        except Exception:
            pass

        stderr_bytes = b"".join(stderr_chunks)
        dt = time.perf_counter() - t0
        rc = int(proc.returncode or 0)

        _dbg(
            f"done op={payload.get('op')!r} run_id={payload.get('run_id')!r} "
            f"rc={rc} dt={dt:.2f}s stdout_bytes={len(stdout_bytes)} stderr_bytes={len(stderr_bytes)}"
        )

        # NOTE: swe_proxy prints a JSON object to stdout even on failure and exits
        # non-zero. We should attempt to decode stdout to surface the real error.
        raw = stdout_bytes.decode("utf-8", errors="replace")
        parsed: Any = None
        try:
            parsed = json.loads(raw) if raw.strip() else None
        except Exception:
            parsed = None

        if rc != 0:
            stderr_txt = stderr_bytes.decode("utf-8", errors="ignore")
            if isinstance(parsed, dict):
                # Common keys used by swe_proxy.
                proxy_err = parsed.get("error") or parsed.get("stderr") or parsed.get("err")
                proxy_rc = parsed.get("returncode") or parsed.get("rc") or rc
                raise RuntimeError(
                    "Remote swe_proxy failed. "
                    f"ssh_rc={rc} proxy_rc={proxy_rc} proxy_error={proxy_err!r} "
                    f"proxy_resp={parsed!r} remote_stderr={stderr_txt!r}"
                )
            # No JSON => include stdout/stderr previews.
            stdout_preview = raw[:2000]
            raise RuntimeError(
                "Remote swe_proxy failed. "
                f"ssh_rc={rc} stdout={stdout_preview!r} remote_stderr={stderr_txt!r}"
            )

        # Forward remote stderr to local log when DEBUG is on (runner/proxy writes progress there).
        if (os.environ.get("DEBUG") or os.environ.get("EBUG")) and stderr_bytes:
            stderr_preview = stderr_bytes.decode("utf-8", errors="ignore")
            print("[remote_swe_session][remote_stderr]" + stderr_preview, file=sys.stderr)

        # Decode response (stdout must be a single JSON object).
        try:
            resp = parsed if parsed is not None else json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to decode swe_proxy JSON response. stdout={stdout_bytes!r}"
            ) from exc

        # Normalize keys across proxy versions.
        if isinstance(resp, dict):
            if "returncode" not in resp:
                if "rc" in resp:
                    resp["returncode"] = resp.get("rc")
                elif "code" in resp:
                    resp["returncode"] = resp.get("code")
            if "stdout" not in resp and "out" in resp:
                resp["stdout"] = resp.get("out")
            if "stderr" not in resp and "err" in resp:
                resp["stderr"] = resp.get("err")

            ok = resp.get("ok", True)
            try:
                resp_rc = int(resp.get("returncode") or 0)
            except Exception:
                resp_rc = 0
            if ok is False and resp_rc == 0:
                resp_rc = 1
                resp["returncode"] = resp_rc

            # Print a short preview on failures (or when explicitly enabled).
            if os.environ.get("GP_PRINT_REMOTE_STDIO") or (ok is False or resp_rc != 0):
                try:
                    op = str(payload.get("op") or "")
                    stdout_preview = (resp.get("stdout") or "")
                    stderr_preview = (resp.get("stderr") or "")
                    if isinstance(stdout_preview, list):
                        stdout_preview = "\n".join([str(x) for x in stdout_preview[:60]])
                    if isinstance(stderr_preview, list):
                        stderr_preview = "\n".join([str(x) for x in stderr_preview[:60]])
                    if stdout_preview:
                        print(
                            "[remote_swe_session][proxy_stdout]\n"
                            + _summarize_stdout(op, str(stdout_preview), max_chars=2000),
                            file=sys.stderr,
                        )
                    if stderr_preview:
                        print("[remote_swe_session][proxy_stderr]\n" + str(stderr_preview)[:2000], file=sys.stderr)
                except Exception:
                    pass

        return resp

    def _ensure_remote_runners(self) -> None:
        if not self.ensure_runners or self.num_runners <= 0:
            return
        # ensure_runners is idempotent; we run it before each start to recover from lost Slurm jobs.

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

    def start(self, timeout: Optional[float] = None, cwd: str = "/testbed") -> Dict[str, Any]:
        self._ensure_remote_runners()
        # Allow plenty of time for the initial container bootstrap even if callers
        # pass a smaller timeout (e.g., snippet-level defaults).
        effective_timeout = max(float(timeout or 0.0), 300.0)
        payload: Dict[str, Any] = {
            "op": "start",
            "run_id": self.run_id,
            "image": self.image,
            "timeout": effective_timeout,
            "cwd": cwd or "/repo",
        }
        resp = self._call_proxy(payload, timeout=effective_timeout)
        try:
            if isinstance(resp, dict) and bool(resp.get('ok', True)) and int(resp.get('returncode') or 0) == 0:
                _register_active_session(self)
        except Exception:
            pass
        return resp

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
        resp = self._call_proxy(payload, timeout=timeout)
        try:
            # Even if stop fails, we don't want to keep the session in registry forever.
            _deregister_active_session(self)
        except Exception:
            pass
        return resp

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
            "max_stdout_bytes": int(os.environ.get("GP_BUILD_REPO_GRAPH_STDOUT_BYTES", "200000000")),
        }
        if cwd is not None:
            payload["cwd"] = cwd
        if repo is not None:
            payload["repo"] = repo

        resp = self._call_proxy(payload, timeout=float(timeout))
        if not resp.get("ok", False):
            rc = resp.get("returncode", None)
            stderr = (resp.get("stderr") or "")
            err = resp.get("error", None)
            stdout_preview = (resp.get("stdout") or "")[:2000]
            raise RuntimeError(
                f"remote build_repo_graph failed (returncode={rc!r}). "
                f"error={err!r} stderr={stderr[:2000]!r} stdout_preview={stdout_preview!r}"
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
