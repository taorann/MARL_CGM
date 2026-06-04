from __future__ import annotations

from graphplanner_agent.config import AgentConfig
from graphplanner_agent.datasets import TaskSpec
from graphplanner_agent.runtime.local_repo import LocalRepoRuntime
from graphplanner_agent.runtime.remote_swe import RemoteSweRuntime
from graphplanner_agent.runtime.sandbox_base import SandboxRuntime


def make_runtime(task: TaskSpec, config: AgentConfig) -> SandboxRuntime:
    config.finalize()
    backend = config.sandbox_backend.lower()
    if backend == "local":
        return LocalRepoRuntime(task.repo_path)
    if backend == "remote_swe":
        return RemoteSweRuntime(config)
    raise ValueError(f"unsupported sandbox backend: {config.sandbox_backend}")
