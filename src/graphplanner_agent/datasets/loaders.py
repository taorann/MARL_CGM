from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .task_spec import TaskSpec


def load_tasks(path: Path) -> list[TaskSpec]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [TaskSpec.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [TaskSpec.from_dict(item) for item in data]
    if isinstance(data, dict) and isinstance(data.get("tasks"), list):
        return [TaskSpec.from_dict(item) for item in data["tasks"]]
    if isinstance(data, dict):
        return [TaskSpec.from_dict(data)]
    raise ValueError(f"unsupported task file shape: {path}")


def dump_results(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
