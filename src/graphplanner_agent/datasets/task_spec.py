from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TaskSpec:
    task_id: str
    repo_path: Path
    issue_title: str
    issue_body: str
    base_commit: str | None = None
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    test_command: str | None = None
    docker_image: str | None = None
    sandbox: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def issue_text(self) -> str:
        return f"{self.issue_title}\n\n{self.issue_body}".strip()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskSpec":
        sandbox = dict(data.get("sandbox") or {})
        metadata = dict(data.get("metadata") or {})
        for key in ("repo", "version", "base_commit", "created_at"):
            if key in data and key not in metadata:
                metadata[key] = data[key]
        swebench_spec = data.get("swebench_spec") or metadata.get("swebench_spec") or {}
        if not isinstance(swebench_spec, dict):
            swebench_spec = {}
        eval_script_list = data.get("eval_script_list") or metadata.get("eval_script_list") or swebench_spec.get("eval_script_list")
        if eval_script_list and "eval_script_list" not in swebench_spec:
            swebench_spec["eval_script_list"] = _as_list(eval_script_list)
        if swebench_spec:
            metadata["swebench_spec"] = swebench_spec
        docker_image = _first(
            data,
            sandbox,
            metadata,
            [
                ("docker_image",),
                ("image",),
                ("image_name",),
                ("container_image",),
                ("sif_path",),
                ("sif_name",),
                ("environment", "docker_image"),
                ("environment", "image"),
                ("ds", "docker_image"),
            ],
        )
        if docker_image and not any(k in sandbox for k in ("docker_image", "image", "sif_path", "sif_name")):
            sandbox["sif_path" if str(docker_image).endswith(".sif") else "docker_image"] = docker_image
        repo_path = (
            data.get("repo_path")
            or sandbox.get("repo_path")
            or sandbox.get("workdir")
            or data.get("workdir")
            or data.get("testbed")
            or ("/testbed" if docker_image else ".")
        )
        return cls(
            task_id=str(data.get("task_id") or data.get("instance_id") or "task"),
            repo_path=Path(str(repo_path)),
            issue_title=str(data.get("issue_title") or data.get("title") or _nested(data, "issue", "title") or data.get("instance_id") or ""),
            issue_body=str(data.get("issue_body") or data.get("problem_statement") or _nested(data, "issue", "body") or ""),
            base_commit=data.get("base_commit") or data.get("commit") or metadata.get("base_commit"),
            fail_to_pass=_as_list(
                data.get("fail_to_pass")
                or data.get("FAIL_TO_PASS")
                or data.get("target_fail_to_pass_selectors")
                or _nested(data, "tests", "fail_to_pass")
                or _nested(data, "tests", "FAIL_TO_PASS")
                or _nested(data, "metadata", "target_fail_to_pass_selectors")
                or _nested(data, "metadata", "FAIL_TO_PASS")
            ),
            pass_to_pass=_as_list(
                data.get("pass_to_pass")
                or data.get("PASS_TO_PASS")
                or data.get("target_pass_to_pass_selectors")
                or _nested(data, "tests", "pass_to_pass")
                or _nested(data, "tests", "PASS_TO_PASS")
                or _nested(data, "metadata", "target_pass_to_pass_selectors")
                or _nested(data, "metadata", "PASS_TO_PASS")
            ),
            test_command=(
                data.get("test_command")
                or data.get("test_cmd")
                or data.get("run_tests")
                or _nested(data, "metadata", "test_command")
                or _nested(data, "metadata", "test_cmd")
            ),
            docker_image=str(docker_image) if docker_image else None,
            sandbox=sandbox,
            metadata=metadata,
        )


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except Exception:
                pass
        return [value]
    return [str(item) for item in value]


def _nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first(data: dict[str, Any], sandbox: dict[str, Any], metadata: dict[str, Any], paths: list[tuple[str, ...]]) -> Any:
    roots = (data, sandbox, metadata)
    for root in roots:
        for path in paths:
            value = _nested(root, *path)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if value:
                return value
    return None
