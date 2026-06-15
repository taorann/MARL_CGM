from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any


@dataclass(slots=True)
class PatchEdit:
    path: str
    start: int
    end: int
    new_text: str


@dataclass(slots=True)
class Patch:
    edits: list[PatchEdit] = field(default_factory=list)
    summary: str = ""

    @property
    def touched_paths(self) -> list[str]:
        return sorted({edit.path for edit in self.edits})


def parse_cgm_output(raw: str | dict[str, Any]) -> Patch:
    data = raw
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("diff --git"):
            return parse_unified_diff(stripped)
        data = json.loads(_strip_json_wrapping(stripped))
    if isinstance(data, dict) and isinstance(data.get("diff"), str):
        return parse_unified_diff(data["diff"])
    if "patch" in data:
        data = data["patch"]
    edits = [
        PatchEdit(
            path=str(edit["path"]),
            start=int(edit["start"]),
            end=int(edit["end"]),
            new_text=_normalize_new_text(str(edit.get("new_text", "")), int(edit["start"]), int(edit["end"])),
        )
        for edit in data.get("edits", [])
    ]
    return Patch(edits=edits, summary=str(data.get("summary", "")))


def _normalize_new_text(text: str, start: int, end: int) -> str:
    if "\n" in text or "\\n" not in text:
        return text
    if end <= start and not re.search(r"\\n[ \t]*(?:def |class |if |for |while |try:|with |return |super\(|[A-Za-z_])", text):
        return text
    return text.replace("\\r\\n", "\n").replace("\\n", "\n")


def looks_like_diff_marker(text: str) -> bool:
    return bool(re.search(r"(^|\n)(<<<<<<<|=======|>>>>>>>|diff --git|@@ )", text))


def patch_text_artifact_reason(text: str) -> str | None:
    if looks_like_diff_marker(text):
        return "diff/conflict marker embedded in edit"
    if re.search(r"\{\s*['\"]patch['\"]\s*:", text) or re.search(r"\{\s*['\"]edits['\"]\s*:", text):
        return "CGM patch schema JSON embedded inside edit text"
    if re.search(r"\('',\)\s*\{", text):
        return "Python tuple/schema artifact embedded inside edit text"
    if re.search(r"(?m)^\s*\+\s*$", text):
        return "standalone diff plus marker embedded inside edit text"
    return None


def _strip_json_wrapping(text: str) -> str:
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
    return text


HUNK_RE = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_unified_diff(diff_text: str) -> Patch:
    edits: list[PatchEdit] = []
    current_path: str | None = None
    lines = diff_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            current_path = None
        elif line.startswith("+++ b/"):
            current_path = line[len("+++ b/") :]
        elif line.startswith("@@ "):
            if not current_path:
                raise ValueError("diff hunk appeared before target path")
            match = HUNK_RE.match(line)
            if not match:
                raise ValueError(f"invalid diff hunk header: {line}")
            old_start = int(match.group(1))
            old_len = int(match.group(2) or "1")
            edit_start = 1 if old_start == 0 and old_len == 0 else old_start
            edit_end = 0 if old_start == 0 and old_len == 0 else old_start + old_len - 1
            new_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("@@ ") and not lines[i].startswith("diff --git "):
                hunk_line = lines[i]
                if hunk_line.startswith("\\ No newline"):
                    i += 1
                    continue
                if hunk_line.startswith("+") and not hunk_line.startswith("+++ "):
                    new_lines.append(hunk_line[1:])
                elif hunk_line.startswith(" "):
                    new_lines.append(hunk_line[1:])
                elif hunk_line.startswith("-"):
                    pass
                i += 1
            edits.append(
                PatchEdit(
                    path=current_path,
                    start=edit_start,
                    end=edit_end,
                    new_text="\n".join(new_lines) + ("\n" if new_lines else ""),
                )
            )
            continue
        i += 1
    if not edits:
        raise ValueError("unified diff did not contain any hunks")
    return Patch(edits=edits, summary="unified diff patch")
