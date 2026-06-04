from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from graphplanner_agent.graph.guards import is_test_path
from graphplanner_agent.repair.patch_schema import Patch, PatchEdit, patch_text_artifact_reason
from graphplanner_agent.runtime.sandbox_base import SandboxRuntime


@dataclass(slots=True)
class PatchDecision:
    ok: bool
    reason: str


def validate_patch(root: Path, patch: Patch, max_edits: int = 4, allow_test_changes: bool = False) -> PatchDecision:
    def read_local(path: str) -> str | None:
        full = root / path
        if not root.exists():
            return None
        if not full.exists():
            raise FileNotFoundError(path)
        return full.read_text(encoding="utf-8")

    return _validate_patch(patch, max_edits, allow_test_changes, read_local)


def validate_patch_with_runtime(runtime: SandboxRuntime, patch: Patch, max_edits: int = 4, allow_test_changes: bool = False) -> PatchDecision:
    def read_runtime(path: str) -> str | None:
        return runtime.read_file(path)

    return _validate_patch(patch, max_edits, allow_test_changes, read_runtime)


def normalize_patch_with_runtime(runtime: SandboxRuntime, patch: Patch) -> tuple[Patch, list[str]]:
    def read_runtime(path: str) -> str | None:
        return runtime.read_file(path)

    return _normalize_patch(patch, read_runtime)


def normalize_patch(root: Path, patch: Patch) -> tuple[Patch, list[str]]:
    def read_local(path: str) -> str | None:
        full = root / path
        if not full.exists():
            return None
        return full.read_text(encoding="utf-8")

    return _normalize_patch(patch, read_local)


def _normalize_patch(patch: Patch, read_file) -> tuple[Patch, list[str]]:
    notes: list[str] = []
    edits: list[PatchEdit] = []
    for edit in patch.edits:
        normalized = edit
        if edit.path.endswith(".py"):
            try:
                content = read_file(edit.path)
            except Exception:
                content = None
            if content is not None:
                lines = content.splitlines()
                normalized, edit_notes = _normalize_python_edit(edit, lines)
                notes.extend(edit_notes)
        edits.append(normalized)
    return Patch(edits=edits, summary=patch.summary), notes


def _validate_patch(patch: Patch, max_edits: int, allow_test_changes: bool, read_file) -> PatchDecision:
    if not patch.edits:
        return PatchDecision(False, "patch has no edits")
    for edit in patch.edits:
        artifact = patch_text_artifact_reason(edit.new_text)
        if artifact:
            return PatchDecision(False, f"{artifact}: {edit.path}")
    if len(patch.edits) > max_edits:
        return PatchDecision(False, f"patch has too many edits: {len(patch.edits)} > {max_edits}")
    seen: set[tuple[str, int, int]] = set()
    for edit in patch.edits:
        key = (edit.path, edit.start, edit.end)
        if key in seen:
            return PatchDecision(False, f"duplicate edit range: {edit.path}:{edit.start}-{edit.end}")
        seen.add(key)
        if is_test_path(edit.path) and not allow_test_changes:
            return PatchDecision(False, f"test path edits are blocked: {edit.path}")
        if edit.path.startswith("/") or ".." in Path(edit.path).parts:
            return PatchDecision(False, f"unsafe patch path: {edit.path}")
        try:
            content = read_file(edit.path)
        except FileNotFoundError:
            if edit.start != 1 or edit.end != 0:
                return PatchDecision(False, f"new file edits must use start=1,end=0: {edit.path}")
            lines = []
        except Exception as exc:
            if edit.start == 1 and edit.end == 0:
                lines = []
            else:
                return PatchDecision(False, f"could not read patch target {edit.path}: {exc}")
        else:
            if content is None:
                continue
            lines = content.splitlines()
        if edit.start < 1 or edit.end < edit.start - 1 or edit.end > len(lines):
            return PatchDecision(False, f"invalid edit range: {edit.path}:{edit.start}-{edit.end}")
        collapsed = _validate_suspicious_range_collapse(edit.path, lines, edit.start, edit.end, edit.new_text)
        if collapsed:
            return PatchDecision(False, collapsed)
        structural = _validate_structural_replacement(edit.path, lines, edit.start, edit.end, edit.new_text)
        if structural:
            return PatchDecision(False, structural)
    return PatchDecision(True, "ok")


def apply_patch(runtime: SandboxRuntime, patch: Patch) -> None:
    by_path: dict[str, list] = {}
    for edit in patch.edits:
        by_path.setdefault(edit.path, []).append(edit)
    for path, edits in by_path.items():
        content = _read_existing_file(runtime, path, edits)
        lines = content.splitlines()
        for edit in sorted(edits, key=lambda e: e.start, reverse=True):
            replacement = edit.new_text.splitlines()
            lines[edit.start - 1 : edit.end] = replacement
        runtime.write_file(path, "\n".join(lines) + ("\n" if lines else ""))


def _read_existing_file(runtime: SandboxRuntime, path: str, edits: list) -> str:
    try:
        return runtime.read_file(path)
    except Exception:
        if all(edit.start == 1 and edit.end == 0 for edit in edits):
            return ""
        raise


CONTROL_HEADER_RE = re.compile(r"^\s*(if|elif|else|for|while|try|except|finally|with|match|case)\b.*:\s*(?:#.*)?$")
ASSIGNMENT_LHS_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*(?:\[[^\n=]*\]|\.[A-Za-z_][A-Za-z0-9_]*)?)\s*=")


def _validate_structural_replacement(path: str, lines: list[str], start: int, end: int, new_text: str) -> str | None:
    if start != end or not lines:
        return None
    original = lines[start - 1]
    if not CONTROL_HEADER_RE.match(original):
        return None
    replacement = new_text.splitlines()
    original_indent = len(original) - len(original.lstrip())
    keeps_header = any(
        CONTROL_HEADER_RE.match(line) and (len(line) - len(line.lstrip())) <= original_indent
        for line in replacement
    )
    if not keeps_header:
        return f"edit appears to remove Python control-flow header at {path}:{start}: {original.strip()}"
    return None


def _validate_suspicious_range_collapse(path: str, lines: list[str], start: int, end: int, new_text: str) -> str | None:
    if end <= start:
        return None
    replacement_lines = [line for line in new_text.splitlines() if line.strip()]
    original_lines = lines[start - 1 : end]
    original_nonempty = [line for line in original_lines if line.strip()]
    if len(replacement_lines) != 1 or len(original_nonempty) <= 1:
        return None
    replacement = replacement_lines[0].strip()
    for line in original_nonempty:
        if _same_assignment_lhs(line, replacement):
            return None
    if replacement.startswith(("return ", "raise ", "break", "continue", "pass")):
        return None
    return (
        f"edit collapses multi-line Python span to one unrelated line at {path}:{start}-{end}; "
        "use the exact single-line range or rewrite the complete block"
    )


def _normalize_python_edit(edit: PatchEdit, lines: list[str]) -> tuple[PatchEdit, list[str]]:
    notes: list[str] = []
    if edit.start < 1 or edit.end < edit.start - 1 or edit.end > len(lines):
        return edit, notes
    normalized = edit
    normalized, shrink_note = _shrink_single_line_replacement(normalized, lines)
    if shrink_note:
        notes.append(shrink_note)
    normalized, indent_note = _align_replacement_indent(normalized, lines)
    if indent_note:
        notes.append(indent_note)
    return normalized, notes


def _shrink_single_line_replacement(edit: PatchEdit, lines: list[str]) -> tuple[PatchEdit, str | None]:
    if edit.end <= edit.start:
        return edit, None
    replacement_lines = [line for line in edit.new_text.splitlines() if line.strip()]
    if len(replacement_lines) != 1:
        return edit, None
    replacement = replacement_lines[0]
    original_span = lines[edit.start - 1 : edit.end]
    match_offset = _find_single_line_intent(original_span, replacement)
    if match_offset is None:
        return edit, None
    line_no = edit.start + match_offset
    return (
        PatchEdit(path=edit.path, start=line_no, end=line_no, new_text=edit.new_text),
        f"normalized single-line edit range from {edit.path}:{edit.start}-{edit.end} to {line_no}-{line_no}",
    )


def _find_single_line_intent(original_span: list[str], replacement: str) -> int | None:
    stripped_replacement = replacement.strip()
    if not stripped_replacement:
        return None
    lhs_matches = [
        idx
        for idx, line in enumerate(original_span)
        if line.strip() and _same_assignment_lhs(line, stripped_replacement)
    ]
    if len(lhs_matches) == 1:
        return lhs_matches[0]
    exact_matches = [idx for idx, line in enumerate(original_span) if line.strip() == stripped_replacement]
    if len(exact_matches) == 1:
        return exact_matches[0]
    return None


def _same_assignment_lhs(original: str, replacement: str) -> bool:
    old = ASSIGNMENT_LHS_RE.match(original)
    new = ASSIGNMENT_LHS_RE.match(replacement)
    return bool(old and new and old.group(1).replace(" ", "") == new.group(1).replace(" ", ""))


def _align_replacement_indent(edit: PatchEdit, lines: list[str]) -> tuple[PatchEdit, str | None]:
    if not edit.path.endswith(".py") or not edit.new_text.strip():
        return edit, None
    if edit.start < 1 or edit.start > len(lines):
        return edit, None
    replacement = edit.new_text.splitlines()
    first_idx = next((idx for idx, line in enumerate(replacement) if line.strip()), None)
    if first_idx is None:
        return edit, None
    original_line = lines[edit.start - 1]
    if not original_line.strip():
        return edit, None
    original_indent = len(original_line) - len(original_line.lstrip(" "))
    current_indent = len(replacement[first_idx]) - len(replacement[first_idx].lstrip(" "))
    if current_indent == original_indent:
        return edit, None
    delta = original_indent - current_indent
    fixed: list[str] = []
    for line in replacement:
        if not line.strip():
            fixed.append(line)
        elif delta > 0:
            fixed.append(" " * delta + line)
        else:
            trim = min(len(line) - len(line.lstrip(" ")), -delta)
            fixed.append(line[trim:])
    new_text = "\n".join(fixed)
    if edit.new_text.endswith("\n") and not new_text.endswith("\n"):
        new_text += "\n"
    return (
        PatchEdit(path=edit.path, start=edit.start, end=edit.end, new_text=new_text),
        f"aligned edit indentation at {edit.path}:{edit.start} from {current_indent} to {original_indent} spaces",
    )
