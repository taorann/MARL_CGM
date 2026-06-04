from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TraceWriter:
    def __init__(self, jsonl_path: Path, markdown_path: Path | None = None):
        self.jsonl_path = jsonl_path
        self.markdown_path = markdown_path
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if self.markdown_path:
            self.markdown_path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        record = {"kind": kind, "payload": payload}
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        if self.markdown_path:
            with self.markdown_path.open("a", encoding="utf-8") as fh:
                fh.write(f"\n## {kind}\n\n```json\n{json.dumps(payload, indent=2, sort_keys=True)}\n```\n")
