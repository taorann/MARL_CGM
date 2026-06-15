from __future__ import annotations

from graphplanner_agent.graph.guards import is_test_path
from graphplanner_agent.planner.protocol import PlannerAction


VALID_FIND_TYPES = {"file", "class", "function", "method", "assignment", "any"}
FIND_TYPE_ALIASES = {"module_assignment": "assignment"}
VALID_EXPAND_MODES = {"callers", "callees", "siblings", "imports", "contains", "uses", "related", "mechanism", "owner_flow"}


def validate_action(action: PlannerAction) -> str | None:
    params = action.params
    if action.tool == "explore_find":
        query = str(params.get("query", "")).strip()
        find_type = FIND_TYPE_ALIASES.get(str(params.get("find_type", "")).strip(), str(params.get("find_type", "")).strip())
        path_glob = str(params.get("path_glob", "")).strip()
        if not query and not path_glob:
            return "explore_find requires non-empty query or path_glob"
        if find_type not in VALID_FIND_TYPES:
            return f"explore_find find_type must be one of {sorted(VALID_FIND_TYPES)}"
        if path_glob and is_test_path(path_glob):
            return "Blocked explore_find path_glob targeting benchmark test paths."
    if action.tool == "grep_code":
        pattern = str(params.get("pattern", "")).strip()
        path_glob = str(params.get("path_glob", "")).strip()
        if not pattern:
            return "grep_code requires a non-empty pattern"
        if not path_glob:
            return "grep_code requires path_glob so text search is scoped to implementation files"
        if is_test_path(path_glob):
            return "Blocked grep_code path_glob targeting benchmark test paths."
        try:
            context_lines = int(params.get("context_lines", 2))
            limit = int(params.get("limit", 20))
        except Exception:
            return "grep_code context_lines and limit must be integers"
        if context_lines < 0 or context_lines > 20:
            return "grep_code context_lines must be between 0 and 20"
        if limit < 1 or limit > 50:
            return "grep_code limit must be between 1 and 50"
    if action.tool == "explore_expand":
        if not str(params.get("anchor", "")).strip():
            return "explore_expand requires an anchor node id"
        mode = str(params.get("expand_mode", "")).strip()
        if mode not in VALID_EXPAND_MODES:
            return f"explore_expand expand_mode must be one of {sorted(VALID_EXPAND_MODES)}"
        if mode == "owner_flow" and not str(params.get("symbol", "")).strip():
            return "explore_expand owner_flow requires symbol, e.g. the missing attribute or parameter name"
    if action.tool == "read":
        node_id = str(params.get("node_id", ""))
        view = str(params.get("view") or "body")
        if not node_id.strip():
            return "read requires node_id"
        if not _valid_view(view):
            return "read view must be body, header, around_line:N, or file_window:start-end"
        if is_test_path(node_id):
            return "Blocked read of benchmark test path."
    if action.tool == "memory_commit":
        if "select_ids" not in params or not params.get("select_ids"):
            return "memory_commit requires a non-empty select_ids list; M is curated by explicit model choice"
        for key in ("select_ids", "keep_ids"):
            if key in params and not _is_string_list(params.get(key)):
                return f"memory_commit {key} must be a list of node ids"
    if action.tool == "memory_delete":
        for key in ("delete_ids", "keep_ids"):
            if key in params and not _is_string_list(params.get(key)):
                return f"memory_delete {key} must be a list of node ids"
    if action.tool == "memory_commit_note" and not str(params.get("note", "")).strip():
        return "memory_commit_note requires a non-empty note"
    return None


def _valid_view(view: str) -> bool:
    if view in {"body", "header"}:
        return True
    if view.startswith("around_line:"):
        try:
            return int(view.split(":", 1)[1]) >= 1
        except Exception:
            return False
    if view.startswith("file_window:"):
        try:
            start, end = view.split(":", 1)[1].split("-", 1)
            return int(start) >= 1 and int(end) >= int(start)
        except Exception:
            return False
    return False


def _is_string_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)
