from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from graphplanner_agent.datasets import TaskSpec
from graphplanner_agent.graph.schema import RepoGraph


@dataclass(slots=True)
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(slots=True)
class TestResult:
    status: str
    command: str
    stdout: str
    stderr: str
    returncode: int
    tests_status: dict[str, object] = field(default_factory=dict)
    resolved: bool | None = None
    parser_error: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def summary(self, limit: int = 2000) -> str:
        text = (self.stderr or self.stdout or "").strip()
        return text[-limit:] if len(text) > limit else text


class SandboxRuntime(Protocol):
    root: Path

    def start(self, task: TaskSpec) -> None: ...
    def stop(self) -> None: ...
    def run(self, cmd: str, timeout: int = 120, cwd: str | None = None, env: dict[str, str] | None = None) -> CommandResult: ...
    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> str: ...
    def write_file(self, path: str, content: str) -> None: ...
    def snapshot(self, paths: list[str]) -> dict[str, str | None]: ...
    def rollback(self, snapshot: dict[str, str | None]) -> None: ...
    def build_graph(self) -> RepoGraph: ...
    def run_fail_to_pass(self, task: TaskSpec) -> TestResult: ...
