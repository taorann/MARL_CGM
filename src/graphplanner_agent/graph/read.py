from __future__ import annotations

from pathlib import Path
import re

from .schema import GraphNode


def read_node(root: Path, node: GraphNode, view: str = "body") -> GraphNode:
    path = root / node.path
    lines = path.read_text(encoding="utf-8").splitlines()
    return read_node_from_lines(lines, node, view)


def read_node_from_runtime(runtime, node: GraphNode, view: str = "body") -> GraphNode:
    start, end = view_range(node, view, total_lines=None)
    if view.startswith("file_window:") or view.startswith("around_line:"):
        text = runtime.read_file(node.path, start, end)
    elif view == "header":
        text = runtime.read_file(node.path, node.start_line, min(node.end_line, node.start_line + 4))
        start, end = node.start_line, min(node.end_line, node.start_line + 4)
    else:
        text = runtime.read_file(node.path, node.start_line, node.end_line)
        start, end = node.start_line, node.end_line
    actual_lines = len(text.splitlines())
    if actual_lines:
        end = start + actual_lines - 1
    return GraphNode(
        id=node.id,
        kind=node.kind,
        name=node.name,
        path=node.path,
        start_line=start,
        end_line=end,
        text=text,
        preview=node.preview,
        parent_id=node.parent_id,
    )


def read_node_from_lines(lines: list[str], node: GraphNode, view: str = "body") -> GraphNode:
    start, end = view_range(node, view, total_lines=len(lines))
    text = "\n".join(lines[start - 1 : end]) + ("\n" if end >= start else "")
    return GraphNode(
        id=node.id,
        kind=node.kind,
        name=node.name,
        path=node.path,
        start_line=start,
        end_line=end,
        text=text,
        preview=node.preview,
        parent_id=node.parent_id,
    )


def view_range(node: GraphNode, view: str = "body", total_lines: int | None = None) -> tuple[int, int]:
    if view == "header":
        start, end = node.start_line, min(node.end_line, node.start_line + 4)
    elif view.startswith("around_line:"):
        line = int(view.split(":", 1)[1])
        start, end = max(1, line - 8), line + 8
    elif view.startswith("file_window:"):
        match = re.match(r"file_window:(\d+)-(\d+)", view)
        if not match:
            raise ValueError(f"invalid file_window view: {view}")
        start, end = int(match.group(1)), int(match.group(2))
    else:
        start, end = node.start_line, node.end_line
    start, end = max(1, start), max(start, end)
    if total_lines is not None:
        end = min(total_lines, end)
    return start, end


def line_numbered(node: GraphNode) -> str:
    if not node.text:
        return ""
    return "\n".join(f"{idx:>4}: {line}" for idx, line in enumerate(node.text.splitlines(), start=node.start_line))
