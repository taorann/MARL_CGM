from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import shlex
import uuid
from pathlib import Path
from typing import Any

from graphplanner_agent.config import AgentConfig
from graphplanner_agent.datasets import TaskSpec
from graphplanner_agent.graph.schema import GraphNode, RepoGraph
from graphplanner_agent.runtime.remote_swe_session import RemoteSweError, RemoteSweSession, infer_sif_dir_from_ref, normalize_sif_image_ref
from graphplanner_agent.runtime.sandbox_base import CommandResult, TestResult
from graphplanner_agent.runtime.swebench_official import official_eval_command, result_from_official_run


WRONG_PYTHON_ENV_RETURNCODE = 97
WRONG_PYTHON_ENV_MARKER = "INFRA_WRONG_PYTHON_ENV"


def decode_repo_graph_payload(payload: str) -> RepoGraph:
    try:
        raw = gzip.decompress(base64.b64decode(payload.encode("ascii"), validate=False))
    except Exception as exc:
        raise RemoteSweError(f"failed to decode remote repo graph payload len={len(payload or '')}") from exc
    return repo_graph_from_jsonl(raw.decode("utf-8", "replace"))


def repo_graph_from_jsonl(text: str, root: str = "/testbed") -> RepoGraph:
    graph = RepoGraph(root=root)
    repo_id = "repo"
    graph.add_node(GraphNode(id=repo_id, kind="repository", name="repo", path="", start_line=1, end_line=1))
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        typ = obj.get("type")
        if typ == "edge" or ("src" in obj and "dst" in obj):
            src = str(obj.get("src") or obj.get("source") or "")
            dst = str(obj.get("dst") or obj.get("target") or "")
            kind = str(obj.get("kind") or obj.get("type") or "RELATED")
            if src and dst:
                graph.add_edge(src, dst, kind)
            continue
        node_id = str(obj.get("id") or "")
        if not node_id:
            continue
        span = obj.get("span")
        start, end = 1, 1
        if isinstance(span, list) and len(span) >= 2:
            start, end = int(span[0] or 1), int(span[1] or span[0] or 1)
        elif isinstance(span, dict):
            start = int(span.get("start") or span.get("start_line") or 1)
            end = int(span.get("end") or span.get("end_line") or start)
        else:
            start = int(obj.get("start_line") or obj.get("start") or 1)
            end = int(obj.get("end_line") or obj.get("end") or start)
        snippet = obj.get("text")
        if not isinstance(snippet, str):
            lines = obj.get("snippet_lines")
            if isinstance(lines, list):
                snippet = "\n".join(str(x) for x in lines) + ("\n" if lines else "")
            else:
                snippet = ""
        graph.add_node(
            GraphNode(
                id=node_id,
                kind=str(obj.get("kind") or obj.get("nodeType") or "node").lower(),
                name=str(obj.get("name") or obj.get("symbol") or node_id),
                path=str(obj.get("path") or ""),
                start_line=start,
                end_line=end,
                text=snippet or None,
                preview=str(obj.get("sig") or obj.get("doc") or obj.get("name") or ""),
                parent_id=str(obj.get("parent_id") or "") or None,
            )
        )
    for node in list(graph.nodes.values()):
        if node.id != repo_id and node.kind == "file":
            graph.add_edge(repo_id, node.id, "CONTAINS")
    return graph


class RemoteSweRuntime:
    def __init__(self, config: AgentConfig, session: RemoteSweSession | None = None):
        self.config = config
        self.session = session
        self.root = Path(config.sandbox_workdir)
        self.task: TaskSpec | None = None
        self.run_id = ""
        self.image_ref = ""
        self.image = ""
        self.sif_dir: str | None = None
        self.last_graph_cache_hit = False
        self.last_graph_cache_path: Path | None = None

    def start(self, task: TaskSpec) -> None:
        self.task = task
        self.root = Path(self.config.sandbox_workdir)
        image_ref = _image_ref_from_task(task)
        image = normalize_sif_image_ref(image_ref)
        if not image:
            raise RemoteSweError("remote_swe requires task.docker_image/image or sandbox.sif_path/sif_name")
        sif_dir = self.config.sandbox_sif_dir or infer_sif_dir_from_ref(image_ref)
        self.image_ref = image_ref
        self.image = image
        self.sif_dir = sif_dir
        # Remote runner.py derives its reusable job signature from the text
        # before the first "__" in run_id. SWE-bench task ids themselves contain
        # "__" (for example django__django-11740), so preserve per-issue
        # isolation by escaping that separator in the run id prefix.
        safe_task_id = str(task.task_id).replace("__", "--")
        self.run_id = f"gp-{safe_task_id}__{uuid.uuid4().hex[:8]}"
        if self.session is None:
            self.session = RemoteSweSession(
                ssh_target=self.config.sandbox_ssh_target,
                remote_repo=self.config.sandbox_remote_repo,
                image=image,
                run_id=self.run_id,
                remote_python=self.config.sandbox_remote_python,
                swe_proxy_path=self.config.sandbox_swe_proxy_path,
                runner_manager_path=self.config.sandbox_runner_manager_path,
                num_runners=self.config.sandbox_num_runners,
                ensure_runners=self.config.sandbox_ensure_runners_before_start,
                ssh_args=self.config.sandbox_ssh_args,
                sif_dir=sif_dir,
            )
        if self.config.sandbox_cleanup_pool_before_start:
            self.session.cleanup_pool(timeout=120.0, cwd=self.config.sandbox_workdir)
        resp = self.session.start(timeout=max(float(self.config.command_timeout), 300.0), cwd=self.config.sandbox_workdir)
        if not bool(resp.get("ok", True)) or int(resp.get("returncode") or 0) != 0:
            raise RemoteSweError(f"remote_swe start failed: {resp}")

    def stop(self) -> None:
        if self.session is None:
            return
        self.session.stop(timeout=60.0)

    def run(self, cmd: str, timeout: int = 120, cwd: str | None = None, env: dict[str, str] | None = None) -> CommandResult:
        if self.session is None:
            raise RemoteSweError("remote_swe runtime has not been started")
        effective_cwd = cwd or self.config.sandbox_workdir
        try:
            resp = self.session.exec(cmd, cwd=effective_cwd, env=env, timeout=float(timeout))
        except TimeoutError as exc:
            return CommandResult(cmd, 124, "", str(exc), timed_out=True)
        stdout = str(resp.get("stdout") or "")
        stderr = str(resp.get("stderr") or "")
        if resp.get("error"):
            stderr = (stderr + "\n" + str(resp.get("error"))).strip()
        rc = int(resp.get("returncode") if resp.get("returncode") is not None else (0 if resp.get("ok", False) else 1))
        return CommandResult(cmd, rc, stdout, stderr)

    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        py = r'''
import pathlib, sys
path = pathlib.Path(sys.argv[1])
start = int(sys.argv[2]) if sys.argv[2] != "" else None
end = int(sys.argv[3]) if sys.argv[3] != "" else None
lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
if start is None and end is None:
    print("\n".join(lines), end="\n" if lines else "")
else:
    s = max(1, start or 1)
    e = min(len(lines), end or len(lines))
    print("\n".join(lines[s-1:e]), end="\n" if e >= s else "")
'''
        result = self.run(
            "python -c "
            + shlex.quote(py)
            + " "
            + shlex.quote(path)
            + " "
            + shlex.quote("" if start is None else str(start))
            + " "
            + shlex.quote("" if end is None else str(end)),
            timeout=60,
        )
        if result.returncode != 0:
            raise RemoteSweError(f"remote read_file failed rc={result.returncode}: {result.stderr or result.stdout}")
        return result.stdout

    def write_file(self, path: str, content: str) -> None:
        py = r'''
import base64, pathlib, sys
path = pathlib.Path(sys.argv[1])
mode = sys.argv[2]
raw = base64.b64decode(sys.argv[3].encode("ascii"))
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("wb" if mode == "w" else "ab") as fh:
    fh.write(raw)
'''
        raw = content.encode("utf-8")
        chunk_size = int(os.environ.get("GP_REMOTE_WRITE_CHUNK_BYTES", "45000") or "45000")
        chunk_size = max(4096, min(chunk_size, 45000))
        chunks = [raw[i : i + chunk_size] for i in range(0, len(raw), chunk_size)] or [b""]
        for idx, chunk in enumerate(chunks):
            payload = base64.b64encode(chunk).decode("ascii")
            result = self.run(
                "python -c "
                + shlex.quote(py)
                + " "
                + shlex.quote(path)
                + " "
                + shlex.quote("w" if idx == 0 else "a")
                + " "
                + shlex.quote(payload),
                timeout=120,
            )
            if result.returncode != 0:
                raise RemoteSweError(f"remote write_file failed rc={result.returncode}: {result.stderr or result.stdout}")

    def snapshot(self, paths: list[str]) -> dict[str, str | None]:
        payload = base64.b64encode(json.dumps(paths or []).encode("utf-8")).decode("ascii")
        py = r'''
import base64, json, pathlib, sys
paths = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
snap = {}
for p in paths:
    path = pathlib.Path(p)
    snap[p] = path.read_text(encoding="utf-8", errors="replace") if path.exists() and path.is_file() else None
print(json.dumps(snap, ensure_ascii=False))
'''
        result = self.run("python -c " + shlex.quote(py) + " " + shlex.quote(payload), timeout=120)
        if result.returncode != 0:
            raise RemoteSweError(f"remote snapshot failed rc={result.returncode}: {result.stderr or result.stdout}")
        return json.loads(result.stdout.strip().splitlines()[-1] or "{}")

    def rollback(self, snapshot: dict[str, str | None]) -> None:
        payload = base64.b64encode(json.dumps(snapshot or {}).encode("utf-8")).decode("ascii")
        py = r'''
import base64, json, pathlib, sys
snap = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
for p, content in snap.items():
    path = pathlib.Path(p)
    if content is None:
        if path.exists() and path.is_file():
            path.unlink()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
print("ok")
'''
        result = self.run("python -c " + shlex.quote(py) + " " + shlex.quote(payload), timeout=180)
        if result.returncode != 0:
            raise RemoteSweError(f"remote rollback failed rc={result.returncode}: {result.stderr or result.stdout}")

    def build_graph(self) -> RepoGraph:
        if self.session is None:
            raise RemoteSweError("remote_swe runtime has not been started")
        repo_id = self.task.task_id if self.task else "repo"
        cached = self._load_graph_payload_cache()
        if cached is not None:
            self.last_graph_cache_hit = True
            return decode_repo_graph_payload(cached)
        self.last_graph_cache_hit = False
        payload = self.session.build_repo_graph(
            repo_id=repo_id,
            timeout=int(self.config.sandbox_remote_graph_timeout),
            cwd=self.config.sandbox_workdir,
            repo=self.config.sandbox_workdir,
        )
        self._store_graph_payload_cache(payload)
        return decode_repo_graph_payload(payload)

    def run_fail_to_pass(self, task: TaskSpec) -> TestResult:
        official_cmd = official_eval_command(task)
        if task.test_command:
            cmd = wrap_testbed_test_command(task.test_command)
        elif official_cmd:
            cmd = wrap_testbed_test_command(official_cmd)
            result = self.run(cmd, timeout=max(self.config.command_timeout, 1800), cwd=self.config.sandbox_workdir)
            if is_wrong_python_env(result):
                return TestResult(
                    "infra_bug",
                    cmd,
                    result.stdout,
                    result.stderr,
                    result.returncode,
                    parser_error="wrong_python_env",
                )
            return result_from_official_run(task, cmd, result)
        elif task.fail_to_pass:
            cmd = wrap_testbed_test_command(
                "python -m pytest -W ignore -q -o cache_dir=/tmp/pytest_cache "
                + " ".join(shlex.quote(s) for s in task.fail_to_pass)
            )
        else:
            cmd = wrap_testbed_test_command("python -m pytest -W ignore -q -o cache_dir=/tmp/pytest_cache")
        result = self.run(cmd, timeout=max(self.config.command_timeout, 1800), cwd=self.config.sandbox_workdir)
        status = "passed" if result.returncode == 0 else "failed"
        parser_error = None
        if result.timed_out:
            status = "timeout"
        elif is_wrong_python_env(result):
            status = "infra_bug"
            parser_error = "wrong_python_env"
        return TestResult(status, cmd, result.stdout, result.stderr, result.returncode, parser_error=parser_error)

    def _graph_cache_path(self) -> Path | None:
        if not self.config.sandbox_graph_cache:
            self.last_graph_cache_path = None
            return None
        material = {
            "base_commit": self.task.base_commit if self.task else None,
            "image": self.image,
            "image_ref": self.image_ref,
            "sif_dir": self.sif_dir,
            "remote_repo": self.config.sandbox_remote_repo,
            "workdir": self.config.sandbox_workdir,
            "frontend": os.environ.get("GP_REPO_GRAPH_FRONTEND", "treesitter"),
            "embed_snippets": os.environ.get("GP_REPO_GRAPH_EMBED_SNIPPETS", "0"),
        }
        digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()[:24]
        image_name = (self.image or "remote").replace("/", "-").replace(":", "-").replace("@", "-")
        cache_dir = Path(self.config.sandbox_graph_cache_dir)
        path = cache_dir / f"{image_name}-{digest}.b64gz"
        self.last_graph_cache_path = path
        return path

    def _load_graph_payload_cache(self) -> str | None:
        path = self._graph_cache_path()
        if path is None or not path.exists():
            return None
        text = path.read_text(encoding="utf-8").strip()
        return text or None

    def _store_graph_payload_cache(self, payload: str) -> None:
        path = self._graph_cache_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload.strip() + "\n", encoding="utf-8")
        tmp.replace(path)


def _image_ref_from_task(task: TaskSpec) -> str:
    candidates = [
        task.docker_image,
        task.sandbox.get("docker_image") if isinstance(task.sandbox, dict) else None,
        task.sandbox.get("image") if isinstance(task.sandbox, dict) else None,
        task.sandbox.get("container_image") if isinstance(task.sandbox, dict) else None,
        task.sandbox.get("sif_path") if isinstance(task.sandbox, dict) else None,
        task.sandbox.get("sif_name") if isinstance(task.sandbox, dict) else None,
        task.metadata.get("docker_image") if isinstance(task.metadata, dict) else None,
        task.metadata.get("image") if isinstance(task.metadata, dict) else None,
        task.metadata.get("sif_path") if isinstance(task.metadata, dict) else None,
        task.metadata.get("sif_name") if isinstance(task.metadata, dict) else None,
    ]
    for candidate in candidates:
        value = str(candidate or "").strip()
        if value:
            return value
    return ""


def wrap_testbed_test_command(cmd: str) -> str:
    """Run repro/pytest commands from the SWE-bench testbed Python, not host HOME."""
    prologue = r"""
cd /testbed
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PATH="/opt/miniconda3/envs/testbed/bin:/opt/conda/envs/testbed/bin:/opt/miniconda3/bin:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"
if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
  . /opt/miniconda3/etc/profile.d/conda.sh
  conda activate testbed >/dev/null 2>&1 || true
fi
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  . /opt/conda/etc/profile.d/conda.sh
  conda activate testbed >/dev/null 2>&1 || true
fi
hash -r
GP_PYTHON_EXE="$(python -c 'import os, sys; print(os.path.realpath(sys.executable))')"
case "$GP_PYTHON_EXE" in
  /home/*|/mnt/share/*)
    echo "INFRA_WRONG_PYTHON_ENV: python resolves to $GP_PYTHON_EXE" >&2
    exit 97
    ;;
esac
"""
    return prologue.strip() + "\n" + str(cmd or "").strip()


def is_wrong_python_env(result: CommandResult) -> bool:
    text = f"{result.stderr}\n{result.stdout}"
    return result.returncode == WRONG_PYTHON_ENV_RETURNCODE or WRONG_PYTHON_ENV_MARKER in text
