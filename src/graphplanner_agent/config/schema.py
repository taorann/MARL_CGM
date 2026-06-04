from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(slots=True)
class AgentConfig:
    max_steps: int = 48
    command_timeout: int = 300
    planner_request_timeout: int = 300
    planner_model: str = "qwen3-coder-480b-a35b-instruct"
    planner_endpoint: str | None = None
    planner_api_key: str | None = None
    planner_temperature: float = 0.0
    planner_tool_calling: bool = False
    planner_enable_thinking: bool | None = True
    cgm_backend: str = "mock"
    cgm_endpoint: str | None = None
    cgm_http_max_attempts: int = 3
    max_patch_edits: int = 4
    allow_test_changes: bool = False
    require_failed_test_before_repair: bool = True
    max_repeated_action: int = 3
    planner_max_parse_retries: int = 2
    observation_mode: str = "json"
    cgm_mock_response: str | None = None
    sandbox_backend: str = "local"
    sandbox_ssh_target: str = "chongbin_cls@localhost"
    sandbox_remote_repo: str = "/appsnew/home/chongbin_pkuhpc/chongbin_cls/MARL_CGM"
    sandbox_remote_python: str = "python"
    sandbox_swe_proxy_path: str = "hpc/swe_proxy.py"
    sandbox_runner_manager_path: str = "hpc/ensure_runners.py"
    sandbox_num_runners: int = 1
    sandbox_workdir: str = "/testbed"
    sandbox_remote_graph_timeout: int = 1200
    sandbox_ssh_args: str | None = None
    sandbox_sif_dir: str | None = None
    sandbox_graph_cache: bool = True
    sandbox_graph_cache_dir: str = "runs/graph_cache"
    sandbox_ensure_runners_before_start: bool = True
    sandbox_cleanup_pool_before_start: bool = True
    console_verbose: bool = False

    @classmethod
    def from_env(cls) -> "AgentConfig":
        sandbox_backend = os.getenv("GRAPHPLANNER_SANDBOX_BACKEND") or os.getenv("GP_SANDBOX_BACKEND", "local")
        config = cls(
            max_steps=int(os.getenv("GRAPHPLANNER_MAX_STEPS", "48")),
            command_timeout=int(os.getenv("GRAPHPLANNER_COMMAND_TIMEOUT", "300")),
            planner_request_timeout=int(
                os.getenv("GRAPHPLANNER_PLANNER_TIMEOUT")
                or os.getenv("PLANNER_TIMEOUT")
                or os.getenv("GRAPHPLANNER_COMMAND_TIMEOUT", "300")
            ),
            planner_model=os.getenv("PLANNER_MODEL", "qwen3-coder-480b-a35b-instruct"),
            planner_endpoint=os.getenv("PLANNER_ENDPOINT"),
            planner_api_key=os.getenv("PLANNER_API_KEY"),
            planner_temperature=float(os.getenv("PLANNER_TEMPERATURE", "0")),
            planner_tool_calling=os.getenv("PLANNER_TOOL_CALLING", "0").lower() in {"1", "true", "yes", "on"},
            planner_enable_thinking=_parse_optional_bool(os.getenv("PLANNER_ENABLE_THINKING"), default=True),
            cgm_backend=os.getenv("CGM_BACKEND", "mock"),
            cgm_endpoint=os.getenv("CGM_ENDPOINT"),
            cgm_http_max_attempts=int(os.getenv("CGM_HTTP_MAX_ATTEMPTS", "3")),
            max_patch_edits=int(os.getenv("CGM_MAX_PATCH_EDITS", "4")),
            allow_test_changes=os.getenv("GRAPHPLANNER_ALLOW_TEST_CHANGES", "0") == "1",
            require_failed_test_before_repair=os.getenv("GRAPHPLANNER_REQUIRE_FAILED_TEST", "1") != "0",
            max_repeated_action=int(os.getenv("GRAPHPLANNER_MAX_REPEATED_ACTION", "3")),
            planner_max_parse_retries=int(os.getenv("GRAPHPLANNER_PARSE_RETRIES", "2")),
            observation_mode=os.getenv("GRAPHPLANNER_OBSERVATION_MODE", "json"),
            cgm_mock_response=os.getenv("CGM_MOCK_RESPONSE"),
            sandbox_backend=sandbox_backend,
            sandbox_ssh_target=os.getenv("GRAPHPLANNER_SANDBOX_SSH_TARGET")
            or os.getenv("GP_SANDBOX_SSH_TARGET", "chongbin_cls@localhost"),
            sandbox_remote_repo=os.getenv("GRAPHPLANNER_SANDBOX_REMOTE_REPO")
            or os.getenv("GP_SANDBOX_REMOTE_REPO", "/appsnew/home/chongbin_pkuhpc/chongbin_cls/MARL_CGM"),
            sandbox_remote_python=os.getenv("GRAPHPLANNER_SANDBOX_REMOTE_PYTHON")
            or os.getenv("GP_REMOTE_SWE_PYTHON", "python"),
            sandbox_swe_proxy_path=os.getenv("GRAPHPLANNER_SANDBOX_SWE_PROXY_PATH")
            or os.getenv("GP_REMOTE_SWE_PROXY_PATH", "hpc/swe_proxy.py"),
            sandbox_runner_manager_path=os.getenv("GRAPHPLANNER_SANDBOX_RUNNER_MANAGER_PATH")
            or os.getenv("GP_REMOTE_SWE_RUNNER_MANAGER_PATH", "hpc/ensure_runners.py"),
            sandbox_num_runners=int(os.getenv("GRAPHPLANNER_SANDBOX_NUM_RUNNERS") or os.getenv("GP_NUM_RUNNERS", "1")),
            sandbox_workdir=os.getenv("GRAPHPLANNER_SANDBOX_WORKDIR") or os.getenv("GP_SANDBOX_WORKDIR", "/testbed"),
            sandbox_remote_graph_timeout=int(os.getenv("GRAPHPLANNER_REMOTE_GRAPH_TIMEOUT") or os.getenv("GP_REPO_GRAPH_TIMEOUT_S", "1200")),
            sandbox_ssh_args=(os.getenv("GRAPHPLANNER_SANDBOX_SSH_ARGS") or os.getenv("GP_REMOTE_SWE_SSH_ARGS")),
            sandbox_sif_dir=os.getenv("GRAPHPLANNER_SANDBOX_SIF_DIR") or os.getenv("GP_SIF_DIR") or os.getenv("SIF_DIR"),
            sandbox_graph_cache=(
                os.getenv("GRAPHPLANNER_DISABLE_GRAPH_CACHE")
                or os.getenv("GP_DISABLE_GRAPH_CACHE")
                or "0"
            )
            not in {"1", "true", "yes", "on"},
            sandbox_graph_cache_dir=os.getenv("GRAPHPLANNER_GRAPH_CACHE_DIR")
            or os.getenv("GP_REPO_GRAPH_CACHE_DIR", "runs/graph_cache"),
            sandbox_ensure_runners_before_start=(
                os.getenv("GRAPHPLANNER_DISABLE_REMOTE_ENSURE")
                or os.getenv("GP_DISABLE_REMOTE_ENSURE")
                or "0"
            )
            not in {"1", "true", "yes", "on"},
            sandbox_cleanup_pool_before_start=(
                os.getenv("GRAPHPLANNER_DISABLE_REMOTE_CLEANUP")
                or os.getenv("GP_DISABLE_REMOTE_CLEANUP")
                or "0"
            )
            not in {"1", "true", "yes", "on"},
            console_verbose=os.getenv("GRAPHPLANNER_VERBOSE", "0") == "1",
        )
        config.finalize()
        return config

    def finalize(self) -> None:
        self.observation_mode = str(self.observation_mode or "json").strip().lower()
        if self.observation_mode not in {"json", "text"}:
            raise ValueError(f"invalid observation_mode: {self.observation_mode}; expected json or text")
        if not self.sandbox_ssh_args:
            self.sandbox_ssh_args = default_remote_swe_ssh_args(self.sandbox_backend)


def default_remote_swe_ssh_args(sandbox_backend: str) -> str | None:
    if str(sandbox_backend or "").lower() != "remote_swe":
        return None
    key = Path("/root/.ssh/id_ed25519_login24")
    if not key.exists():
        return None
    return (
        f"-i {key} -p 40022 -o StrictHostKeyChecking=no "
        "-o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=6"
    )


def _parse_optional_bool(value: str | None, *, default: bool | None = None) -> bool | None:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")
