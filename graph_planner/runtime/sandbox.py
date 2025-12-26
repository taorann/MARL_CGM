# graph_planner/runtime/sandbox.py
import os, json, random, string, time, shutil, tempfile, base64, shlex, re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import docker, base64, gzip

from ..agents.common.contracts import CGM_CONTRACT, CGMPatchErrorCode, ProtocolError, normalize_newlines

# 遥测
from ..infra import telemetry as telemetry_mod

from .apptainer_queue_runtime import ApptainerQueueRuntime
from .queue_protocol import ExecResult
from .remote_swe_session import RemoteSweSession

# R2E 组件（可选）
try:
    from r2egym.agenthub.runtime.docker import DockerRuntime as R2EDockerRuntime
    from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
    _HAS_R2E = True
except Exception:
    R2EDockerRuntime = None
    EnvArgs = None
    RepoEnv = None
    _HAS_R2E = False

def _rand_name(prefix="gp"):
    import string as _s, random as _r
    return f"{prefix}-" + "".join(_r.choices(_s.ascii_lowercase + _s.digits, k=8))

def _dbg(msg: str):
    if os.environ.get("DEBUG") or os.environ.get("EBUG"):
        print(f"[sandbox] {msg}")

@dataclass
class SandboxConfig:
    docker_image: str
    workdir: str
    mounts: Dict[str, str]
    env: Dict[str, str]
    pytest_cache_root: Optional[str] = None
    commit_hash: Optional[str] = None
    # 统一后端切换：
    backend: str = "auto"            # "auto" | "r2e" | "repoenv" | "docker" | "apptainer_queue" | "remote_swe"
    r2e_ds_json: Optional[str] = None  # 指向一个 JSON 文件，内容是 r2e 期望的 ds dict
    requires_build: bool = False       # SWE-bench 元数据：提示容器是否需要 build
    swebench_spec: Optional[Dict[str, Any]] = None  # 透传 swe-bench 的构建脚本信息
    force_docker_backend: bool = False
    port_forwards: Optional[List[Dict[str, Any]]] = None
    queue_root: Optional[str] = None
    sif_dir: Optional[str] = None
    num_runners: int = 1
    # remote_swe 后端所需配置（跨机 ssh 到 login24）
    ssh_target: Optional[str] = None
    remote_repo: Optional[str] = None
    remote_python: Optional[str] = None
    swe_proxy_path: Optional[str] = None
    runner_manager_path: Optional[str] = None
    # host-side repo root for graph scanning (multi-repo)
    repo_root_host: Optional[str] = None

class SandboxRuntime:
    """
    统一接口：
      run / apply_patch / get_patch / lint / test / reset_soft / close
    后端：
      - "repoenv"  : RepoEnv(EnvArgs(ds)) → 官方评测最友好
      - "r2e"      : R2E DockerRuntime(ds)（我们自己掌控容器，但仍用 R2E 底座）
      - "docker"   : 纯 docker-py（最自由）
      - "auto"     : 有 ds 用 "repoenv"，否则 "docker"
    """
    def __init__(self, cfg: SandboxConfig, force_backend: Optional[str] = None, run_id: Optional[str] = None):
        self.cfg = cfg
        self.run_id = run_id or "__default__"
        preferred_mode = force_backend or cfg.backend or "docker"
        if cfg.force_docker_backend:
            preferred_mode = "docker"
        if preferred_mode == "auto":
            preferred_mode = (
                "repoenv" if (_HAS_R2E and cfg.r2e_ds_json and os.path.exists(cfg.r2e_ds_json)) else "docker"
            )
        self._mode = preferred_mode

        self._env = None  # only populated when using RepoEnv as the backend
        self._aq: Optional[ApptainerQueueRuntime] = None
        self._remote: Optional[RemoteSweSession] = None
        # remote_swe: avoid duplicated expensive `start` calls within one SandboxRuntime.
        # (Multiple helpers call into _exec/build_repo_graph, which used to call start
        # every time and could race with SSH timeout.)
        self._remote_started: bool = False
        self._remote_start_lock = threading.Lock()
        self._last_read_file_error: Optional[dict] = None  # debug: last read_file_lines error payload

        if self._mode == "repoenv":
            try:
                self._init_repoenv_backend()
            except Exception as e:
                _dbg(f"repoenv init failed: {e!r}; falling back to docker")
                self._mode = "docker"
                self._init_docker_backend()
        elif self._mode == "r2e":
            self._init_r2e_backend()
        elif self._mode == "apptainer_queue":
            self._init_apptainer_backend()
        elif self._mode == "remote_swe":
            _dbg(f"init remote_swe: run_id={self.run_id} ssh_target={self.cfg.ssh_target} remote_repo={self.cfg.remote_repo} image={self.cfg.docker_image}")
            self._init_remote_swe_backend()
        else:
            self._init_docker_backend()
        self._exposed_ports: List[Dict[str, Any]] = getattr(self, "_exposed_ports", [])

    # ---------- backend: RepoEnv ----------
    def _init_repoenv_backend(self):
        if not _HAS_R2E:
            raise RuntimeError("r2egym is not available but backend='repoenv' was requested.")
        ds_path = self.cfg.r2e_ds_json
        if ds_path:
            ds_path = os.path.expanduser(ds_path)
            if not os.path.isabs(ds_path):
                ds_path = os.path.abspath(ds_path)
        if not (ds_path and os.path.exists(ds_path)):
            raise ValueError(f"r2e ds json not found: {self.cfg.r2e_ds_json}")

        with open(ds_path, "r") as f:
            ds = json.load(f)

        env_args = EnvArgs(ds=ds)
        env = RepoEnv(env_args)
        self._env = env
        self._rt = env.runtime  # r2e 的 DockerRuntime
        _dbg("repoenv initialized")

        # --- 关键保底：先用根目录作为 workdir 创建 repo_path，避免 chdir 失败 ---
        repo_path = getattr(self._rt, "repo_path", "/testbed")
        try:
            # 直接用 docker-py 在容器 root workdir 执行 mkdir，绕开 /testbed 不存在的问题
            self._rt.container.exec_run("bash -lc 'mkdir -p {}'".format(repo_path), workdir="/")
        except Exception as e:
            # Do not silently swallow repo-graph load failures; they are a common root cause of
            # unexpected fallback to legacy build_graph and missing candidates.
            if os.environ.get("DEBUG") or os.environ.get("EBUG"):
                try:
                    _dbg(f"repo_graph jsonl load failed: {e!r}")
                except Exception:
                    pass
            if os.environ.get("GP_DISABLE_BUILD_GRAPH") == "1":
                raise


        # 基本工具 + git 安全目录（现在 chdir 到 repo_path 已不会报 126）
        self._rt.run("python -m pip -q install --upgrade pip >/dev/null 2>&1 || true", timeout=180)
        self._rt.run("python -m pip -q install pytest >/dev/null 2>&1 || true", timeout=300)
        self._rt.run(f"git config --global --add safe.directory {repo_path} || true", timeout=30)

        self.repo = None

    # ---------- backend: R2E DockerRuntime（仍保留，训练期灵活） ----------
    def _init_r2e_backend(self):
        if not _HAS_R2E:
            raise RuntimeError("r2egym is not available but backend='r2e' was requested.")
        ds_path = self.cfg.r2e_ds_json
        if ds_path:
            ds_path = os.path.expanduser(ds_path)
            if not os.path.isabs(ds_path):
                ds_path = os.path.abspath(ds_path)
        if not (ds_path and os.path.exists(ds_path)):
            raise ValueError(f"r2e ds json not found: {self.cfg.r2e_ds_json}")

        with open(ds_path, "r") as f:
            ds = json.load(f)

        # 宿主挂载（只为把你的代码带进容器；真正工作目录在 /work）
        volumes = {}
        for host, container in (self.cfg.mounts or {}).items():
            if not os.path.isabs(host):
                raise ValueError(f"HOST mount path must be absolute: {host}")
            volumes[os.path.abspath(host)] = {"bind": container, "mode": "rw"}

        repo_src = next(iter((self.cfg.mounts or {}).values()), "/testbed")
        repo_path = "/work"

        self._rt = R2EDockerRuntime(
            ds=ds,
            repo_path=repo_path,
            command="/bin/bash",
            working_dir=repo_path,
            volumes=volumes,
            environment=self.cfg.env or {},
        )
        # 拷贝到 /work，避开 root_squash
        self._rt.run("mkdir -p /root/.local/bin /work", timeout=60)
        self._rt.run(f"rsync -a --delete {repo_src}/ {repo_path}/ || cp -a {repo_src}/. {repo_path}/", timeout=600)
        self._rt.run(f"git config --global --add safe.directory {repo_path} || true", timeout=30)
        self._rt.run("python -m pip -q install pytest >/dev/null 2>&1 || true", timeout=300)
        self.repo = None

    def _init_apptainer_backend(self) -> None:
        cfg = self.cfg
        if not cfg.queue_root or not cfg.sif_dir:
            raise ValueError("backend='apptainer_queue' requires queue_root and sif_dir.")
        queue_root = Path(os.path.expanduser(cfg.queue_root)).resolve()
        sif_dir = Path(os.path.expanduser(cfg.sif_dir)).resolve()
        self._aq = ApptainerQueueRuntime(
            queue_root=queue_root,
            sif_dir=sif_dir,
            num_runners=int(cfg.num_runners or 1),
        )
        self.workdir = cfg.workdir or "."
        _dbg(f"apptainer_queue backend initialized: workdir={self.workdir!r}")

    def _init_remote_swe_backend(self) -> None:
        cfg = self.cfg
        if not cfg.ssh_target or not cfg.remote_repo:
            raise ValueError("backend='remote_swe' requires ssh_target and remote_repo.")
        num_runners = int(cfg.num_runners or 1)
        self._remote = RemoteSweSession(
            ssh_target=cfg.ssh_target,
            remote_repo=os.path.expanduser(cfg.remote_repo),
            image=cfg.docker_image,
            run_id=self.run_id,
            remote_python=cfg.remote_python or "python",
            swe_proxy_path=cfg.swe_proxy_path or "hpc/swe_proxy.py",
            runner_manager_path=cfg.runner_manager_path or "hpc/ensure_runners.py",
            num_runners=num_runners,
            ensure_runners=True,
        )
        self.workdir = "/testbed"  # remote_swe: SWE-bench 容器内默认工作目录
        _dbg(
            f"remote_swe backend initialized: ssh={cfg.ssh_target!r}, "
            f"repo={cfg.remote_repo!r}, workdir={self.workdir!r}, num_runners={num_runners}"
        )

    # ---------- backend: docker-py（自管容器） ----------
    def _init_docker_backend(self):
        self.client = docker.from_env(timeout=120)
        volumes = {}
        for host, container in (self.cfg.mounts or {}).items():
            if not os.path.isabs(host):
                raise ValueError(f"HOST mount path must be absolute: {host}")
            volumes[os.path.abspath(host)] = {"bind": container, "mode": "rw"}
        if self.cfg.workdir:
            workdir = self.cfg.workdir
        elif "/testbed" in (self.cfg.mounts or {}).values():
            workdir = "/testbed"
        else:
            workdir = next(iter(self.cfg.mounts.values()), "/") if self.cfg.mounts else "/"
        self.workdir = workdir

        ports_arg = None
        if self.cfg.port_forwards:
            ports_arg = {}
            for spec in self.cfg.port_forwards:
                if not isinstance(spec, Mapping):
                    continue
                container_port = spec.get("container_port")
                if container_port is None:
                    continue
                try:
                    container_port = int(container_port)
                except Exception:
                    continue
                protocol = str(spec.get("protocol", "tcp")).lower()
                host_ip = spec.get("host_ip")
                host_port = spec.get("host_port")
                binding: Any
                if host_port is None or host_port == "":
                    binding = None
                else:
                    try:
                        host_port = int(host_port)
                    except Exception:
                        continue
                    if host_ip:
                        binding = (str(host_ip), host_port)
                    else:
                        binding = host_port
                key = f"{container_port}/{protocol}" if protocol else int(container_port)
                ports_arg[key] = binding
        self.container = self.client.containers.run(
            image=self.cfg.docker_image,
            command="/bin/bash",
            name=_rand_name("gp"),
            environment=self.cfg.env or {},
            working_dir=self.workdir,
            tty=True,
            stdin_open=True,
            detach=True,
            volumes=volumes,
            ports=ports_arg,
        )
        # git 安全目录兜底
        self._exec(f"git config --global --add safe.directory {self.workdir} || true")
        self.repo = None
        self._exposed_ports = []
        if ports_arg:
            try:
                self.container.reload()
                ports_info = (
                    self.container.attrs.get("NetworkSettings", {}).get("Ports", {})
                )
            except Exception:
                ports_info = {}
            for key, bindings in (ports_info or {}).items():
                if not bindings:
                    # None 表示 Docker 仍然会随机分配主机端口
                    self._exposed_ports.append(
                        {
                            "container_port": key,
                            "host_ip": None,
                            "host_port": None,
                        }
                    )
                    continue
                for binding in bindings:
                    self._exposed_ports.append(
                        {
                            "container_port": key,
                            "host_ip": binding.get("HostIp"),
                            "host_port": int(binding.get("HostPort"))
                            if binding.get("HostPort")
                            else None,
                        }
                    )

    # ---------- 通用执行 ----------
    def _ensure_remote_started(self, *, timeout: float) -> None:
        """Start remote SWE instance once per SandboxRuntime.

        Important: `swe_proxy` uses an internal wait budget that can exceed the
        nominal payload timeout. We therefore rely on RemoteSweSession._call_proxy
        to apply a larger outer SSH timeout, and we keep this call idempotent.
        """
        if self._mode != "remote_swe":
            return
        assert self._remote is not None, "remote_swe backend not initialized"
        if self._remote_started:
            return
        with self._remote_start_lock:
            if self._remote_started:
                return
            t0 = time.perf_counter()
            resp = self._remote.start(timeout=float(timeout), cwd=self.workdir or "/testbed")
            dt = time.perf_counter() - t0
            ok = bool(resp.get("ok", False))
            rc = resp.get("returncode", None)
            err = resp.get("error", None)
            stderr = resp.get("stderr", "") or ""
            _dbg(
                f"remote_swe started: run_id={self.run_id} dt={dt:.2f}s "
                f"ok={ok!r} rc={rc!r} error={err!r} stderr_bytes={len(stderr)}"
            )
            if not ok:
                # allow attaching to an already-running instance on this runner (runner-side rebind/reuse)
                msg = str(err or stderr or "")
                m = re.search(r"current_run_id=([A-Za-z0-9_\-]+)", msg)
                if m:
                    try:
                        cur = m.group(1)
                        _dbg(f"remote_swe start got busy; adopting current_run_id={cur}")
                        self._remote.run_id = cur
                        self._remote_started = True
                        return
                    except Exception:
                        pass
                raise RuntimeError(
                    f"remote_swe start failed: rc={rc!r} error={err!r} stderr={stderr[:2000]!r}"
                )
            self._remote_started = True

    def _exec(self, cmd: str, timeout: int = 900) -> Tuple[str, int]:
        if self._mode == "remote_swe":
            assert self._remote is not None, "remote_swe backend not initialized"
            start_timeout = max(float(timeout), 300.0)
            self._ensure_remote_started(timeout=start_timeout)
            resp = self._remote.exec(
                cmd,
                timeout=float(timeout),
                env=self.cfg.env,
                cwd=self.workdir,
            )
            out = (resp.get("stdout") or "") + (resp.get("stderr") or "")
            rc = int(resp.get("returncode", resp.get("rc", resp.get("code", 0)) or 0) or 0)
            return out, rc
        if self._mode == "apptainer_queue":
            q = "'" + cmd.replace("'", "'\"'\"'") + "'"
            exec_cmd = ["bash", "-lc", q]
            result: ExecResult = self._aq.exec(
                run_id=self.run_id,
                docker_image=self.cfg.docker_image,
                cmd=exec_cmd,
                cwd=Path(self.workdir),
                env=self.cfg.env,
                timeout_sec=float(timeout),
                meta={"src": "sandbox", "op": "exec"},
            )
            out = (result.stdout or "") + (result.stderr or "")
            return out, int(result.returncode)
        if self._mode in ("r2e", "repoenv"):
            out, rc = self._rt.run(cmd, timeout=timeout)
            try:
                rc_int = int(rc)
            except (TypeError, ValueError):
                rc_int = 0 if str(rc).strip() == "" else 1
            return out, rc_int
        # docker-py
        q = "'" + cmd.replace("'", "'\"'\"'") + "'"
        exec_cmd = f"bash -lc {q}"
        res = self.container.exec_run(exec_cmd, demux=True)
        rc = res.exit_code if hasattr(res, "exit_code") else res[0]
        out, err = res.output if hasattr(res, "output") else res[1]
        out = (out or b"") + (err or b"")
        return out.decode("utf-8", errors="ignore"), rc

    # ---------- ACI 接口 ----------
    def run(self, cmd: str, timeout: int = 900) -> Tuple[str, int]:
        return self._exec(cmd, timeout)

    def read_file_lines(self, path: str, start: int = 1, end: int = 1, timeout: int = 60) -> Tuple[List[str], int]:
        """Read a range of lines from a file inside the sandbox.

        Args:
          path: absolute path or path relative to current working directory in the sandbox
          start/end: 1-based inclusive line numbers (best-effort; clamped to file bounds)

        Returns (lines, rc). rc!=0 indicates a failure to read/parse.
        """
        try:
            s_i = int(start)
        except Exception:
            s_i = 1
        try:
            e_i = int(end)
        except Exception:
            e_i = s_i
        if s_i <= 0:
            s_i = 1
        if e_i < s_i:
            e_i = s_i

        req = {"path": str(path), "start": s_i, "end": e_i}
        try:
            payload = json.dumps(req, ensure_ascii=False)
        except Exception:
            payload = '{"path":%r,"start":%d,"end":%d}' % (str(path), s_i, e_i)
        b64 = base64.b64encode(payload.encode('utf-8')).decode('ascii')

        py = r'''
import base64, json, os, sys
req = json.loads(base64.b64decode(sys.argv[1]).decode('utf-8'))
path = req.get('path')
start = int(req.get('start', 1))
end = int(req.get('end', start))
if not isinstance(path, str) or not path:
    print(json.dumps({'lines': [], 'error': 'invalid_path', 'path': path}, ensure_ascii=False))
    raise SystemExit(2)
try:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.read().replace('\r\n','\n').replace('\r','\n').split('\n')
except Exception as e:
    print(json.dumps({'lines': [], 'error': 'open_failed', 'path': path, 'exc': repr(e)}, ensure_ascii=False))
    raise SystemExit(3)
n = len(lines)
s = max(1, min(start, n + 1))
e = max(s, min(end, n))
out = lines[s-1:e] if n > 0 else []
print(json.dumps({'lines': out}, ensure_ascii=False))
'''

        py_bin = 'python'
        try:
            if getattr(self, '_mode', None) == 'remote_swe':
                py_bin = getattr(self.cfg, 'remote_python', None) or 'python'
        except Exception:
            pass
        cmd = f"{py_bin} -c {shlex.quote(py)} {shlex.quote(b64)}"
        out, rc = self._exec(cmd, timeout=int(timeout or 60))
        # Parse JSON payload from stdout even when the command fails.
        data = None
        last_err = None
        try:
            for ln in reversed((out or "").splitlines()):
                ln_s = (ln or "").strip()
                if not ln_s:
                    continue
                try:
                    obj = json.loads(ln_s)
                except Exception:
                    continue
                if isinstance(obj, dict) and ("lines" in obj or "error" in obj):
                    data = obj
                    break
            if isinstance(data, dict) and data.get("error"):
                last_err = data
        except Exception:
            data = None
            last_err = None

        try:
            self._last_read_file_error = last_err
        except Exception:
            pass

        if not isinstance(data, dict):
            return [], int(rc) if rc != 0 else 4

        lines = data.get("lines") if isinstance(data.get("lines"), list) else []
        lines = [str(x) for x in lines]

        if (os.environ.get("DEBUG") or os.environ.get("EBUG")) and os.environ.get("GP_DEBUG_READ_FILE") == "1":
            try:
                print(f"[sandbox] read_file_lines path={path!r} start={s_i} end={e_i} rc={rc!r} error={data.get('error')!r}", file=sys.stderr)
            except Exception:
                pass

        return lines, (int(rc) if int(rc) != 0 else 0)

    def apply_patch_edits(self, edits: List[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Apply structured edits in-place within the sandbox.

        Each edit:
          { "path": str, "start": int, "end": int, "new_text": str }

        Conventions (best-effort):
        - If start >= 1: treat as 1-based line index, and end as 1-based inclusive.
        - If start <= 0: treat as 0-based slice indices, and end as 0-based exclusive.
        - If end < start: treat as insertion at start.

        Robustness:
        - Accept missing +x for scripts elsewhere (unrelated), but here handle CRLF, empty files, OOB indices,
          and missing directories; preserve trailing newline when possible.
        """
        try:
            payload = json.dumps(list(edits or []), ensure_ascii=False)
        except Exception:
            return {"success": False, "applied": 0, "paths": [], "error": "invalid_edits"}

        b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")

        py = r'''
import base64, json, os, sys
edits = json.loads(base64.b64decode(sys.argv[1]).decode('utf-8'))
ok = True
paths = []
applied = 0

def norm(s: str) -> str:
    return (s or '').replace('\r\n', '\n').replace('\r', '\n')

for e in edits:
    try:
        p = e.get('path')
        if not isinstance(p, str) or not p.strip():
            ok = False
            continue
        p = p.strip()
        start_raw = e.get('start', None)
        end_raw = e.get('end', None)
        try:
            s = int(start_raw) if start_raw is not None else 1
        except Exception:
            s = 1
        try:
            en = int(end_raw) if end_raw is not None else (s - 1 if s >= 1 else s)
        except Exception:
            en = (s - 1 if s >= 1 else s)

        nt = norm(e.get('new_text') or '')
        new_lines = nt.split('\n') if nt != '' else []

        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)

        content = ''
        had_trailing_nl = False
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            had_trailing_nl = content.endswith('\n') or content.endswith('\r')
        content = norm(content)
        lines = content.splitlines()

        # Index mapping
        if s >= 1:
            # 1-based inclusive [s, en]
            i0 = max(0, s - 1)
            i1 = max(0, en)  # exclusive end
            if en < s:
                i1 = i0
        else:
            # 0-based slice [s, en) (treat end<start as insertion)
            i0 = max(0, s)
            i1 = max(0, en)
            if en < s:
                i1 = i0

        # Clamp to file bounds
        n = len(lines)
        if i0 > n:
            i0 = n
        if i1 > n:
            i1 = n
        if i1 < i0:
            i1 = i0

        out = lines[:i0] + new_lines + lines[i1:]

        with open(p, 'w', encoding='utf-8') as f:
            if out:
                f.write('\n'.join(out))
                if had_trailing_nl:
                    f.write('\n')
            else:
                f.write('')

        paths.append(p)
        applied += 1
    except Exception:
        ok = False

print(json.dumps({'success': ok, 'applied': applied, 'paths': paths}, ensure_ascii=False))
'''

        cmd = "python -c " + shlex.quote(py) + " " + shlex.quote(b64)
        out, rc = self._exec(cmd, timeout=900)
        if rc != 0:
            return {"success": False, "applied": 0, "paths": [], "error": "exec_failed", "rc": rc, "stdout": out}
        try:
            return json.loads(out.strip().splitlines()[-1])
        except Exception:
            return {"success": True, "applied": 0, "paths": []}

    def apply_patch(self, unified_diff: str) -> bool:
        if self._mode in ("r2e", "repoenv"):
            return self._rt.apply_patch(unified_diff)
        # docker-py 路径
        heredoc = f"cat >/tmp/graph_planner.patch <<'EOF'\n{unified_diff}\nEOF"
        _, rc1 = self._exec(heredoc)
        if rc1 != 0: return False
        _, rc2 = self._exec("git apply --reject --whitespace=fix /tmp/graph_planner.patch")
        return rc2 == 0

    def get_patch(self) -> str:
        if self._mode in ("r2e", "repoenv"):
            return self._rt.get_patch()
        out, _ = self._exec("git diff")
        return out

    def lint(self) -> bool:
        _, rc = self._exec(
            "ruff --version >/dev/null 2>&1 || true; "
            "black --version >/dev/null 2>&1 || true; "
            "ruff check . || true; black --check . || true"
        )
        return rc == 0

    def test(self, selector: Optional[List[str]] = None, timeout: int = 1800) -> Dict:
        selector_tuple: Tuple[str, ...] = tuple(selector or ())
        sel = " ".join(selector_tuple)

        # remote_swe: prefer /repo (workdir) run_tests.sh if present, else fallback pytest
        if self._mode == "remote_swe":
            wd = (self.workdir or "/repo").rstrip("/")
            for script in (f"{wd}/run_tests.sh", "/repo/run_tests.sh"):
                # Some SWE-bench images ship run_tests.sh without +x; accept if file exists.
                if self._exec(f"test -f {script}")[1] == 0:
                    # SWE-bench run_tests.sh typically does not accept extra positional args; run full suite.
                    cmd = f"cd {os.path.dirname(script)} && bash {script}".strip()
                    start = time.time()
                    out, rc = self._exec(cmd, timeout=timeout)
                    duration = time.time() - start
                    result = {"mode": "run_tests.sh", "passed": rc == 0, "rc": rc, "stdout": out}
                    return self._finalize_test_result(
                        result,
                        command=cmd,
                        selector=selector_tuple,
                        duration=duration,
                    )

        # RepoEnv / R2E：尝试官方脚本 → 失败再回退 pytest
        if self._mode in ("repoenv", "r2e"):
            # 探测常见官方入口
            probes = [
                "test -f /testbed/run_tests.sh",
                "test -f /work/run_tests.sh",
                "test -d /r2e_tests",
            ]
            if any(self._exec(p)[1] == 0 for p in probes):
                # 先尝试 /testbed 下的脚本；没有就 /work；再没有就 /r2e_tests
                for candidate in ("/testbed/run_tests.sh", "/work/run_tests.sh"):
                    cmd = f"bash {candidate} --json /tmp/_r2e_eval.json"
                    start = time.time()
                    out, rc = self._exec(cmd, timeout=timeout)
                    duration = time.time() - start
                    if rc == 0 or "No such file" not in out:
                        dump, _ = self._exec("cat /tmp/_r2e_eval.json || true")
                        passed = False
                        try:
                            data = json.loads(dump) if dump.strip() else {}
                            passed = bool(data.get("passed", False))
                        except Exception:
                            # 退化关键词匹配
                            passed = ("PASSED" in out) and ("FAILED" not in out)
                        result = {"mode": "r2e", "passed": passed, "rc": 0 if passed else 1, "stdout": out}
                        return self._finalize_test_result(
                            result,
                            command=cmd,
                            selector=selector_tuple,
                            duration=duration,
                        )
                # r2e_tests 目录的自定义入口（按需定制）
                cmd = "bash /r2e_tests/run.sh || true"
                start = time.time()
                out, rc = self._exec(cmd, timeout=timeout)
                duration = time.time() - start
                passed = ("PASSED" in out) and ("FAILED" not in out)
                result = {"mode": "r2e", "passed": passed, "rc": 0 if passed else 1, "stdout": out}
                return self._finalize_test_result(
                    result,
                    command=cmd,
                    selector=selector_tuple,
                    duration=duration,
                )

        # 回退 pytest（禁用 --cache-dir，统一用 python -m pytest）
        cmd = f"python -m pytest -q {sel}".strip()
        _dbg(f"pytest cmd: {cmd}")
        start = time.time()
        out, rc = self._exec(cmd, timeout=timeout)
        duration = time.time() - start
        result = {"mode": "pytest", "passed": rc == 0, "rc": rc, "stdout": out}
        return self._finalize_test_result(
            result,
            command=cmd,
            selector=selector_tuple,
            duration=duration,
        )

    def _aci_root(self) -> Path:
        # Host-side cache root for GraphPlanner artifacts
        # Default: .aci (relative to current working directory)
        return Path(os.environ.get("GP_ACI_ROOT", "gp_artifacts"))

    def _repo_graph_cache_path(self, repo_id: str) -> Path:
        rid = (repo_id or "").strip() or "repo"
        return self._aci_root() / "subgraphs" / rid / "repo" / "repo_graph.jsonl"

    def _load_repo_graph_jsonl(self, path: Path) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                typ = obj.pop("type", None)
                if typ == "node":
                    nodes.append(obj)
                elif typ == "edge":
                    edges.append(obj)
                else:
                    # best-effort fallback
                    if "src" in obj and "dst" in obj:
                        edges.append(obj)
                    else:
                        nodes.append(obj)
        return {"nodes": nodes, "edges": edges}

    def load_repo_graph(self, repo_id: str = "", *, timeout: int = 3600) -> Dict[str, Any]:
        """Load (or build then load) the full repo_graph as a Python dict {nodes, edges}.

        This is intentionally NOT issue-conditioned: explore/find/expand should operate on the
        full repo graph for the current SWE container.
        """
        if self._mode != "remote_swe":
            raise RuntimeError(f"load_repo_graph is only supported for backend='remote_swe', got {self._mode!r}")
        rid = (repo_id or "").strip()
        if not rid:
            rid = str((self.cfg.env or {}).get("GP_REPO_ID") or "").strip() or "repo"
        p = Path(self.build_repo_graph(repo_id=rid, timeout=int(timeout), force=False))
        return self._load_repo_graph_jsonl(p)


    def build_repo_graph(self, repo_id: str = "", *, timeout: int = 3600, force: bool = False) -> str:
        """Build repo-level graph in remote_swe and cache JSONL on the host.

        Returns the host-side cache path to repo_graph.jsonl.
        """
        if self._mode != "remote_swe":
            raise RuntimeError(f"build_repo_graph is only supported for backend='remote_swe', got {self._mode!r}")
        if not self._remote:
            raise RuntimeError("remote_swe backend is not initialized")

        # Ensure remote instance started (idempotent)
        start_timeout = max(float(timeout), 300.0)
        self._ensure_remote_started(timeout=start_timeout)

        # Resolve repo id for caching
        rid = (repo_id or "").strip()
        if not rid:
            rid = str((self.cfg.env or {}).get("GP_REPO_ID") or "").strip()
        if not rid:
            rid = "repo"

        # Allow env-based forcing without changing call sites.
        # Useful when repo_graph schema changes (e.g., embedding snippets) and you
        # want to regenerate the cached JSONL.
        if not force:
            force = str(os.environ.get("GP_FORCE_REPO_GRAPH", "")).strip().lower() in {"1", "true", "yes", "y"}

        cache_path = self._repo_graph_cache_path(rid)
        if cache_path.exists() and not force:
            return str(cache_path)

        _dbg(f"remote_swe build_repo_graph: repo_id={rid} workdir={self.workdir} image={self.cfg.docker_image}")
        b64 = self._remote.build_repo_graph(repo_id=rid, timeout=int(timeout), cwd=self.workdir, repo=self.workdir)
        if not b64:
            raise RuntimeError("build_repo_graph returned empty base64 payload")

        try:
            blob = base64.b64decode(b64.encode("ascii"), validate=False)
            raw = gzip.decompress(blob)
        except Exception as exc:
            raise RuntimeError(
                f"failed to decode base64+gzip repo graph payload (len={len(b64)} chars)"
            ) from exc

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            tmp_path.write_bytes(raw)
            os.replace(str(tmp_path), str(cache_path))
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
        return str(cache_path)


    def build_issue_subgraph(
        self,
        issue_id: str,
        timeout: int = 900,
    ) -> Dict[str, Any]:
        """Build the working subgraph.

        For backend='remote_swe', we prefer a repo-level graph cached as JSONL on the host
        (via build_repo_graph). This avoids parsing large JSON over stdout and enables reuse
        across multiple issues from the same repo.

        For compatibility, if repo-level build fails, we fall back to remote build_graph (JSON).
        """
        if self._mode != "remote_swe":
            raise RuntimeError(
                f"build_issue_subgraph is only supported for backend='remote_swe', got {self._mode!r}"
            )
        if not self._remote:
            raise RuntimeError("remote_swe backend is not initialized")

        # Try repo-level cached JSONL first
        try:
            rid = str((self.cfg.env or {}).get("GP_REPO_ID") or "repo")
            jsonl_path = Path(self.build_repo_graph(repo_id=rid, timeout=int(timeout), force=False))
            return self._load_repo_graph_jsonl(jsonl_path)
        except Exception as e:
            # Do not silently swallow repo-graph load failures; they are a common root cause of
            # unexpected fallback to legacy build_graph and missing candidates.
            if os.environ.get("DEBUG") or os.environ.get("EBUG"):
                try:
                    _dbg(f"repo_graph jsonl load failed: {e!r}")
                except Exception:
                    pass
            if os.environ.get("GP_DISABLE_BUILD_GRAPH") == "1":
                raise


        # Fallback: legacy JSON stdout
        start_timeout = max(float(timeout), 300.0)
        self._ensure_remote_started(timeout=start_timeout)
        return self._remote.build_graph(
            issue_id=issue_id,
            timeout=float(timeout),
            cwd=self.workdir,
            repo=self.workdir,
        )

    def reset_soft(self) -> None:
        if self._mode in ("r2e", "repoenv"):
            self._rt.soft_git_reset()
        else:
            self._exec("git reset --hard HEAD && git clean -fd")

    def _finalize_test_result(
        self,
        result: Dict[str, Any],
        *,
        command: str,
        selector: Tuple[str, ...],
        duration: float,
    ) -> Dict[str, Any]:
        payload = {
            "kind": "test_run",
            "backend": self._mode,
            "command": command,
            "selector": list(selector),
            "duration_sec": round(duration, 3),
            "result": {
                "mode": result.get("mode"),
                "rc": result.get("rc"),
                "passed": bool(result.get("passed")),
            },
            "stdout": result.get("stdout", ""),
        }
        if self.cfg.workdir:
            payload["workdir"] = self.cfg.workdir
        if self._mode in ("repoenv", "r2e") and self.cfg.r2e_ds_json:
            payload["dataset_json"] = self.cfg.r2e_ds_json
        try:
            telemetry_mod.log_test_result(payload)
        except Exception as e:
            # Do not silently swallow repo-graph load failures; they are a common root cause of
            # unexpected fallback to legacy build_graph and missing candidates.
            if os.environ.get("DEBUG") or os.environ.get("EBUG"):
                try:
                    _dbg(f"repo_graph jsonl load failed: {e!r}")
                except Exception:
                    pass
            if os.environ.get("GP_DISABLE_BUILD_GRAPH") == "1":
                raise

        return result

    def close(self):
        if self._mode in ("r2e", "repoenv"):
            try:
                close = getattr(self._rt, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            if self._env is not None:
                try:
                    self._env.runtime = None
                except Exception:
                    pass
                self._env = None
            self._rt = None
        elif self._mode == "remote_swe":
            try:
                if self._remote:
                    self._remote.stop()
            except Exception:
                pass
            self._remote = None
        elif self._mode == "apptainer_queue":
            self._aq = None
        else:
            try: self.container.stop(timeout=5)
            except Exception: pass
            try: self.container.remove(force=True)
            except Exception: pass

    # ---------- 调试辅助 ----------
    @property
    def exposed_ports(self) -> List[Dict[str, Any]]:
        return list(self._exposed_ports)


class PatchApplier:
    """Apply unified diffs in a temporary workspace before committing changes."""

    def __init__(self) -> None:
        self._applied: Dict[str, set[str]] = {}

    def apply_in_temp_then_commit(
        self,
        repo_root: Path,
        patch_text: str,
        path: str,
        run_tests: Callable[[Path], Mapping[str, Any]],
        run_lint: Optional[Callable[[Path], Mapping[str, Any]]] = None,
        patch_id: Optional[str] = None,
        *,
        new_content: Optional[str] = None,
        stats: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, Any]:
        """Validate, trial, and commit a patch atomically.

        Parameters
        ----------
        repo_root:
            Root directory of the working repository on the host filesystem.
        patch_text:
            Unified diff text generated from the CGM candidate.
        path:
            Relative file path within ``repo_root`` touched by the patch.
        run_tests / run_lint:
            Callbacks executed inside the temporary copy. They must accept a
            ``Path`` argument pointing to the trial workspace and return a
            mapping containing status flags (``passed``/``ok``) and optional
            logs. ``run_lint`` is optional.
        patch_id:
            Optional deterministic identifier used to detect duplicate
            applications.
        new_content:
            Optional fully materialised file contents. When omitted the diff is
            applied to the current workspace to derive the new text.
        stats:
            Optional telemetry dictionary with ``n_hunks``/``added_lines``/
            ``removed_lines`` counters.
        """

        repo_root = Path(repo_root)
        if not repo_root.exists():
            raise ProtocolError(CGMPatchErrorCode.PATH_MISSING.value, f"repo root '{repo_root}' does not exist")

        normalized_path = path.strip()
        if not normalized_path:
            raise ProtocolError(CGMPatchErrorCode.PATH_MISSING.value, "patch path is empty")

        if patch_id:
            applied = self._applied.setdefault(normalized_path, set())
            if patch_id in applied:
                raise ProtocolError(CGMPatchErrorCode.DUPLICATE_PATCH.value, f"patch {patch_id} already applied to {normalized_path}")

        source_file = repo_root.joinpath(normalized_path)
        if not source_file.exists():
            raise ProtocolError(CGMPatchErrorCode.PATH_MISSING.value, f"target file '{normalized_path}' not found")

        try:
            original_text = normalize_newlines(source_file.read_text(encoding="utf-8"))
        except UnicodeDecodeError as exc:
            error = ProtocolError(
                CGMPatchErrorCode.ENCODING_UNSUPPORTED.value,
                f"file '{normalized_path}' is not UTF-8 encoded: {exc}",
            )
            error.__cause__ = exc
            raise error

        if new_content is None:
            analysis = _analyse_diff_fallback(original_text, patch_text, normalized_path)
            new_text = analysis.new_text
            computed_stats = {
                "n_hunks": analysis.n_hunks,
                "added_lines": analysis.added_lines,
                "removed_lines": analysis.removed_lines,
            }
        else:
            new_text = normalize_newlines(new_content)
            computed_stats = {
                "n_hunks": int((stats or {}).get("n_hunks", 0)),
                "added_lines": int((stats or {}).get("added_lines", 0)),
                "removed_lines": int((stats or {}).get("removed_lines", 0)),
            }

        if CGM_CONTRACT.constraints.get("newline_required") and not new_text.endswith("\n"):
            raise ProtocolError(CGMPatchErrorCode.NEWLINE_MISSING.value, f"resulting file '{normalized_path}' must end with newline")

        temp_dir = Path(tempfile.mkdtemp(prefix="gp-patch-", dir=str(repo_root.parent)))
        try:
            trial_root = temp_dir
            shutil.copytree(repo_root, trial_root, dirs_exist_ok=True)
            trial_file = trial_root.joinpath(normalized_path)
            trial_file.parent.mkdir(parents=True, exist_ok=True)
            trial_file.write_text(new_text, encoding="utf-8")

            lint_result = run_lint(trial_root) if run_lint else {"ok": True, "stdout": ""}
            lint_ok = bool(lint_result.get("ok"))
            if not lint_ok:
                error = ProtocolError("lint-failed", lint_result.get("stdout", "lint failed"))
                error.temp_path = temp_dir.name  # type: ignore[attr-defined]
                error.n_hunks = computed_stats["n_hunks"]  # type: ignore[attr-defined]
                error.added_lines = computed_stats["added_lines"]  # type: ignore[attr-defined]
                error.removed_lines = computed_stats["removed_lines"]  # type: ignore[attr-defined]
                raise error

            tests_result = run_tests(trial_root)
            tests_passed = bool(tests_result.get("passed"))
            if not tests_passed:
                error = ProtocolError("build-failed", tests_result.get("stdout", "tests failed"))
                error.temp_path = temp_dir.name  # type: ignore[attr-defined]
                error.n_hunks = computed_stats["n_hunks"]  # type: ignore[attr-defined]
                error.added_lines = computed_stats["added_lines"]  # type: ignore[attr-defined]
                error.removed_lines = computed_stats["removed_lines"]  # type: ignore[attr-defined]
                raise error

            temp_target = trial_file
            final_path = source_file
            final_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = final_path.parent.joinpath(f".{final_path.name}.gp_tmp")
            tmp_file.write_text(new_text, encoding="utf-8")
            with tmp_file.open("rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp_file, final_path)
            if patch_id:
                self._applied.setdefault(normalized_path, set()).add(patch_id)
            return {
                "ok": True,
                "applied": True,
                "path": normalized_path,
                "tests_passed": tests_passed,
                "lint_ok": lint_ok,
                "tests": tests_result,
                "lint": lint_result,
                "n_hunks": computed_stats["n_hunks"],
                "added_lines": computed_stats["added_lines"],
                "removed_lines": computed_stats["removed_lines"],
                "temp_path": temp_dir.name,
            }
        except ProtocolError as exc:
            if not hasattr(exc, "temp_path"):
                exc.temp_path = temp_dir.name  # type: ignore[attr-defined]
                exc.n_hunks = computed_stats["n_hunks"]  # type: ignore[attr-defined]
                exc.added_lines = computed_stats["added_lines"]  # type: ignore[attr-defined]
                exc.removed_lines = computed_stats["removed_lines"]  # type: ignore[attr-defined]
            raise
        except Exception as exc:  # pragma: no cover - defensive
            error = ProtocolError(CGMPatchErrorCode.DIRTY_WORKSPACE.value, str(exc))
            error.temp_path = temp_dir.name  # type: ignore[attr-defined]
            error.n_hunks = computed_stats["n_hunks"]  # type: ignore[attr-defined]
            error.added_lines = computed_stats["added_lines"]  # type: ignore[attr-defined]
            error.removed_lines = computed_stats["removed_lines"]  # type: ignore[attr-defined]
            raise error
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _analyse_diff_fallback(original_text: str, diff_text: str, path: str) -> "DiffAnalysis":
    """Fallback diff analysis shared with :class:`PatchApplier`."""

    diff_text = normalize_newlines(diff_text)
    lines = diff_text.splitlines()
    added = removed = n_hunks = 0
    new_lines: List[str] = []
    original_lines = original_text.splitlines()
    idx = 0
    current_orig = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("@@ "):
            n_hunks += 1
            idx += 1
            while idx < len(lines):
                segment = lines[idx]
                if segment.startswith("@@ ") or segment.startswith("diff --git") or segment.startswith("--- ") or segment.startswith("+++ "):
                    break
                if segment.startswith(" "):
                    if current_orig < len(original_lines):
                        new_lines.append(original_lines[current_orig])
                        current_orig += 1
                elif segment.startswith("-"):
                    removed += 1
                    current_orig += 1
                elif segment.startswith("+"):
                    added += 1
                    new_lines.append(segment[1:])
                idx += 1
            continue
        idx += 1
    new_lines.extend(original_lines[current_orig:])
    new_text = "\n".join(new_lines) + "\n"
    class DiffAnalysis:  # local alias to avoid importing from agents
        def __init__(self, new_text: str, n_hunks: int, added_lines: int, removed_lines: int) -> None:
            self.new_text = new_text
            self.n_hunks = n_hunks
            self.added_lines = added_lines
            self.removed_lines = removed_lines
    return DiffAnalysis(new_text, n_hunks, added, removed)
