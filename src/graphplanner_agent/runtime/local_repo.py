from __future__ import annotations

import os
from pathlib import Path
import subprocess

from graphplanner_agent.datasets import TaskSpec
from graphplanner_agent.graph.build import build_python_graph
from graphplanner_agent.graph.schema import RepoGraph
from graphplanner_agent.runtime.sandbox_base import CommandResult, TestResult
from graphplanner_agent.runtime.swebench_official import official_eval_command, result_from_official_run


class LocalRepoRuntime:
    def __init__(self, root: Path):
        self.root = root

    def start(self, task: TaskSpec) -> None:
        self.root = task.repo_path

    def stop(self) -> None:
        return None

    def run(self, cmd: str, timeout: int = 120, cwd: str | None = None, env: dict[str, str] | None = None) -> CommandResult:
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        try:
            completed = subprocess.run(
                cmd,
                shell=True,
                cwd=str(self.root / cwd) if cwd else str(self.root),
                env=run_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            return CommandResult(cmd, completed.returncode, completed.stdout, completed.stderr)
        except subprocess.TimeoutExpired as exc:
            return CommandResult(cmd, 124, exc.stdout or "", exc.stderr or "", timed_out=True)

    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        lines = (self.root / path).read_text(encoding="utf-8").splitlines()
        if start is None and end is None:
            return "\n".join(lines) + ("\n" if lines else "")
        start = max(1, start or 1)
        end = min(len(lines), end or len(lines))
        return "\n".join(lines[start - 1 : end]) + ("\n" if end >= start else "")

    def write_file(self, path: str, content: str) -> None:
        full = self.root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")

    def snapshot(self, paths: list[str]) -> dict[str, str | None]:
        snap: dict[str, str | None] = {}
        for path in sorted(set(paths)):
            full = self.root / path
            snap[path] = full.read_text(encoding="utf-8") if full.exists() else None
        return snap

    def rollback(self, snapshot: dict[str, str | None]) -> None:
        for path, content in snapshot.items():
            full = self.root / path
            if content is None:
                if full.exists():
                    full.unlink()
            else:
                full.parent.mkdir(parents=True, exist_ok=True)
                full.write_text(content, encoding="utf-8")

    def build_graph(self) -> RepoGraph:
        return build_python_graph(self.root)

    def run_fail_to_pass(self, task: TaskSpec) -> TestResult:
        official_cmd = official_eval_command(task)
        if task.test_command:
            cmd = task.test_command
        elif official_cmd:
            cmd = official_cmd
            result = self.run(cmd)
            return result_from_official_run(task, cmd, result)
        elif task.fail_to_pass:
            cmd = "python -m pytest " + " ".join(task.fail_to_pass)
        else:
            cmd = "python -m pytest"
        result = self.run(cmd)
        status = "passed" if result.returncode == 0 else "failed"
        if result.timed_out:
            status = "timeout"
        return TestResult(status, cmd, result.stdout, result.stderr, result.returncode)
