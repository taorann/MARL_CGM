"""Contracts and validators for planner and CGM interactions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from ...core.actions import (
    ActionUnion,
    ExploreAction,
    MemoryAction,
    NoopAction,
    RepairAction,
    SubmitAction,
)

__all__ = [
    "PlannerContract",
    "CGMContract",
    "ProtocolError",
    "PLANNER_CONTRACT",
    "CGM_CONTRACT",
    "PLANNER_SYSTEM_PROMPT",
    "CGM_SYSTEM_PROMPT",
    "CGM_PATCH_INSTRUCTION",
    "parse_action_block",
    "validate_planner_action",
    "validate_cgm_patch",
    "normalize_newlines",
]


class PlannerErrorCode(str, Enum):
    INVALID_MULTI_BLOCK = "invalid-multi-block"
    MISSING_FUNCTION_TAG = "missing-function-tag"
    UNKNOWN_ACTION = "unknown-action"
    DUPLICATE_PARAM = "duplicate-param"
    UNKNOWN_PARAM = "unknown-param"
    EXTRA_TEXT = "extra-text"
    INVALID_JSON_PARAM = "invalid-json-param"
    MISSING_REQUIRED_PARAM = "missing-required-param"


class CGMPatchErrorCode(str, Enum):
    INVALID_PATCH_SCHEMA = "invalid-patch-schema"
    MULTI_FILE_DIFF = "multi-file-diff"
    NEWLINE_MISSING = "newline-missing"
    RANGE_INVALID = "range-invalid"
    PATH_MISSING = "path-missing"
    INVALID_UNIFIED_DIFF = "invalid-unified-diff"
    HUNK_MISMATCH = "hunk-mismatch"
    ENCODING_UNSUPPORTED = "encoding-unsupported"
    DIRTY_WORKSPACE = "dirty-workspace"
    DUPLICATE_PATCH = "duplicate-patch"


class ProtocolError(ValueError):
    """Exception raised when planner or CGM output violates the contract."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class PlannerContract:
    """Single source of truth for planner prompts and schema."""

    SYSTEM_PROMPT: str
    ACTIONS: Tuple[str, ...]
    allowed_params: Mapping[str, Set[str]]
    required_params: Mapping[str, Set[str]] = field(default_factory=dict)
    errors: Tuple[str, ...] = (
        PlannerErrorCode.INVALID_MULTI_BLOCK.value,
        PlannerErrorCode.MISSING_FUNCTION_TAG.value,
        PlannerErrorCode.UNKNOWN_ACTION.value,
        PlannerErrorCode.DUPLICATE_PARAM.value,
        PlannerErrorCode.UNKNOWN_PARAM.value,
        PlannerErrorCode.EXTRA_TEXT.value,
        PlannerErrorCode.INVALID_JSON_PARAM.value,
        PlannerErrorCode.MISSING_REQUIRED_PARAM.value,
    )

    def normalise_action(self, name: str) -> str:
        action = (name or "").strip().lower()
        if action not in self.ACTIONS:
            raise ProtocolError(
                PlannerErrorCode.UNKNOWN_ACTION.value,
                f"action '{name}' is not supported; expected one of {sorted(self.ACTIONS)}",
            )
        return action


@dataclass(frozen=True)
class CGMContract:
    """Single source of truth for CGM prompts and patch schema."""

    SYSTEM_PROMPT: str
    schema: Mapping[str, Any]
    constraints: Mapping[str, Any]


@dataclass(frozen=True)
class CGMPatch:
    """Normalised CGM patch guaranteed to touch exactly one file."""

    path: str
    edits: List[Dict[str, Any]]
    summary: Optional[str] = None


PLANNER_SYSTEM_PROMPT = """You are GraphPlanner, a model-driven planning agent for code repair.

Output EXACTLY ONE JSON object inside a fenced ```json block. No extra text.

Concepts:
- repo_graph: full repository graph (read-only).
- working_subgraph (W): planner-facing, for exploration/reading/reasoning; may be noisy.
- memory_subgraph (M): patch-facing (for CGM); must be small and high-signal.

Actions:
1) explore/find
   - HARD RULE: provide at most ONE query term and at most ONE anchor.
   - Purpose: retrieve candidate code locations + snippets into working_subgraph.

2) explore/expand
   - Expand around ONE anchor to bring more code into working_subgraph.

3) memory (intent: commit | delete)
   - Select a small set of node_ids from working_subgraph and project them into memory_subgraph.
   - Prefer selector.nodes = [node_id, ...].
   - Do NOT describe this as "merge then clear working"; keep working readable by pruning unmemorized noise.

4) repair
   - HARD RULE: only call repair if memory_subgraph is NON-EMPTY.
   - Provide plan as a LIST of short steps (plan: ["...", "..."]).
   - Use memory_subgraph as the main context for patch generation.

5) submit
   - Run tests to validate the patch.

Always obey the hard rules above.
"""


PLANNER_CONTRACT = PlannerContract(
    SYSTEM_PROMPT=PLANNER_SYSTEM_PROMPT,
    ACTIONS=("explore", "memory", "repair", "submit", "noop"),
    allowed_params={
        # New: prefer <param name="action">{JSON}</param>
        # Compat: accept legacy params (op/anchors/...) and older "k" wrapper if emitted by older prompts.
        "explore": {"thought", "action", "k", "op", "anchors", "nodes", "query", "hop", "limit", "max_per_anchor", "total_limit", "dir_diversity_k"},
        "memory": {"thought", "action", "k", "target", "intent", "selector"},
        "repair": {"thought", "action", "k", "plan", "subplan", "focus_ids", "apply"},
        "submit": {"thought", "action", "k"},
        "noop": {"thought", "action", "k"},
    },
    required_params={
        "repair": set(),
        "memory": {"intent"},
        "explore": set(),
    },
)


CGM_SYSTEM_PROMPT = (
    "You are CodeFuse-CGM, a graph-aware assistant that generates precise code patches. "
    "Use the issue description, planner plan, graph context and snippets to derive the necessary edits. "
    "Reply with a JSON object containing a top-level \"patch\" field. The patch must include an \"edits\" array listing objects with \"path\", \"start\", \"end\" and \"new_text\" fields. Ensure new_text entries end with a newline."
)

CGM_PATCH_INSTRUCTION = CGM_SYSTEM_PROMPT

CGM_CONTRACT = CGMContract(
    SYSTEM_PROMPT=CGM_SYSTEM_PROMPT,
    schema={
        "patch": {
            "type": "object",
            "required": ["edits"],
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["path", "start", "end", "new_text"],
                    },
                    "minItems": 1,
                }
            },
        },
        "summary": {"type": "string"},
    },
    constraints={"one_file_per_patch": True, "newline_required": True},
)


_BLOCK_RE = re.compile(r"<function\s*=\s*([a-zA-Z0-9_.-]+)\s*>", re.IGNORECASE)
_END_RE = re.compile(r"</function>", re.IGNORECASE)
_PARAM_RE = re.compile(r"<param\s+name=\"([^\"]+)\">(.*?)</param>", re.DOTALL | re.IGNORECASE)
_CDATA_START = "<![CDATA["


def _looks_like_json(text: str) -> bool:
    if not text:
        return False
    first = text[0]
    if first in '{["' or first in '-0123456789':
        return True
    lowered = text.lower()
    return lowered in {"true", "false", "null"}


def normalize_newlines(text: str) -> str:
    """Return text with CRLF/CR normalised to LF."""

    if not isinstance(text, str):
        return str(text)
    return text.replace('\r\n', '\n').replace('\r', '\n')

def _normalise_json_value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    if not _looks_like_json(text):
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(
            PlannerErrorCode.INVALID_JSON_PARAM.value,
            f"unable to parse JSON value: {exc}"
        ) from exc


def parse_action_block(text: str) -> Dict[str, Any]:
    """Parse a planner action response.

    Preferred format:
        <function=ACTION>
          <param name="thought"><![CDATA[...]]></param>
          <param name="action">{JSON}</param>
        </function>

    Tolerances:
      - bare JSON (entire response is a single JSON object)
      - missing </function> terminator (best-effort parse)
      - legacy <param name="k">{JSON}</param>
    """

    if not isinstance(text, str):
        raise ProtocolError(PlannerErrorCode.MISSING_FUNCTION_TAG.value, "planner response must be a string")

    stripped = text.strip()

    # 0) Bare JSON fallback
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                action_name = PLANNER_CONTRACT.normalise_action(obj.get("type") or obj.get("name"))
                if not action_name:
                    raise ProtocolError(PlannerErrorCode.MISSING_FUNCTION_TAG.value, "json action missing 'type'")
                params = dict(obj)
                # ensure 'type' aligns
                params["type"] = action_name
                # keep optional thought if present
                return {"name": action_name, "params": params}
        except json.JSONDecodeError:
            pass
        except ProtocolError:
            raise
        except Exception:
            pass

    matches = list(_BLOCK_RE.finditer(text))
    if not matches:
        raise ProtocolError(PlannerErrorCode.MISSING_FUNCTION_TAG.value, "response does not contain <function=...>")
    if len(matches) > 1:
        raise ProtocolError(PlannerErrorCode.INVALID_MULTI_BLOCK.value, "response must contain exactly one function block")

    match = matches[0]
    action_name = PLANNER_CONTRACT.normalise_action(match.group(1))

    end_match = _END_RE.search(text, match.end())
    end_pos = end_match.start() if end_match else len(text)  # tolerate truncation
    inner = text[match.end():end_pos]

    params: Dict[str, Any] = {}
    last_end = 0
    for param_match in _PARAM_RE.finditer(inner):
        start, end = param_match.span()
        if inner[last_end:start].strip():
            raise ProtocolError(PlannerErrorCode.EXTRA_TEXT.value, "unexpected text between <param> elements")
        key = param_match.group(1).strip()
        allowed = PLANNER_CONTRACT.allowed_params.get(action_name, set())
        if key not in allowed:
            raise ProtocolError(PlannerErrorCode.UNKNOWN_PARAM.value, f"parameter '{key}' is not allowed for action '{action_name}'")
        if key in params:
            raise ProtocolError(PlannerErrorCode.DUPLICATE_PARAM.value, f"duplicate parameter '{key}'")
        raw_value = param_match.group(2).strip()
        if raw_value.startswith(_CDATA_START) and raw_value.endswith("]]>"):
            value = raw_value[len(_CDATA_START):-3]
        else:
            value = _normalise_json_value(raw_value)
        params[key] = value
        last_end = end

    # Best-effort recover for truncated <param name="action"> or <param name="k">
    if "action" not in params and "k" not in params:
        for key in ("action", "k"):
            m = re.search(rf'<param name="{key}">(.*)', inner, re.DOTALL)
            if not m:
                continue
            raw = m.group(1)
            raw = re.split(r"</param>|</function>", raw, maxsplit=1)[0].strip()
            if not raw:
                continue
            try:
                params[key] = _normalise_json_value(raw)
            except Exception:
                continue

    return {"name": action_name, "params": params}



def _coerce_bool(value: Any, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def _ensure_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v)]
    if isinstance(value, (str, int, float)):
        text = str(value)
        return [text] if text else []
    return []

def _extract_query_terms(query: str, max_terms: int = 16) -> List[str]:
    """Extract deterministic keyword terms from a free-form query string.

    We prefer identifiers and path-like tokens so the env can run robust repo search
    without having to parse natural-language sentences.
    """
    if not query:
        return []
    q = str(query)

    terms: List[str] = []
    seen: set[str] = set()

    def _push(t: str):
        t = (t or "").strip()
        if not t:
            return
        tl = t.lower()
        if tl in {"a","an","the","and","or","of","to","for","in","on","at","by","with","from","as","is","are","was","were","be","been","being"}:
            return
        if len(t) <= 1:
            return
        if tl in seen:
            return
        seen.add(tl)
        terms.append(t)

    # backtick code spans
    for m in re.finditer(r"`([^`]{1,80})`", q):
        frag = m.group(1).strip()
        if frag:
            # split by whitespace/punct inside code span
            for t in re.findall(r"[A-Za-z_][A-Za-z0-9_./-]*", frag):
                _push(t)

    # path-like tokens
    for t in re.findall(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+", q):
        _push(t)

    # identifiers / dotted names
    for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", q):
        _push(t)

    # also accept bare snake_case-ish words longer than 3
    for t in re.findall(r"[A-Za-z0-9_]{4,}", q):
        _push(t)

    if len(terms) > max_terms:
        terms = terms[:max_terms]
    return terms



def _ensure_dict_list(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [dict(value)]
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _attach_meta(action: ActionUnion, meta: Dict[str, Any]) -> ActionUnion:
    object.__setattr__(action, "_meta", dict(meta))
    return action


def validate_planner_action(result: Mapping[str, Any]) -> ActionUnion:
    """Validate planner parameters and convert to a typed action."""

    action_name = PLANNER_CONTRACT.normalise_action(result.get("name"))
    params = dict(result.get("params") or {})
    # Preferred: params["action"] is a full JSON action object. Compat: params["k"].
    action_payload: Dict[str, Any] | None = None
    if isinstance(params.get("action"), Mapping):
        action_payload = dict(params.get("action") or {})
    elif isinstance(params.get("k"), Mapping):
        action_payload = dict(params.get("k") or {})
    elif isinstance(params.get("action"), str):
        # Rare: action arrives as raw JSON string
        try:
            obj = json.loads(params.get("action") or "{}")
            if isinstance(obj, dict):
                action_payload = obj
        except Exception:
            action_payload = None

    if action_payload:
        # allow payload to carry thought
        payload_thought = action_payload.pop("thought", None)
        if payload_thought and not params.get("thought"):
            params["thought"] = payload_thought
        # ensure type matches
        if "type" not in action_payload and action_name:
            action_payload["type"] = action_name
        # merge into params so legacy validators can read fields like op/intent/plan
        params.update(action_payload)
    meta: Dict[str, Any] = {}

    required = PLANNER_CONTRACT.required_params.get(action_name, set())
    missing = [key for key in required if key not in params]
    if missing:
        raise ProtocolError(
            PlannerErrorCode.MISSING_REQUIRED_PARAM.value,
            f"action '{action_name}' missing required params: {', '.join(sorted(missing))}",
        )

    if action_name == "explore":
        op = str(params.get("op", "find")).lower()
        anchors = _ensure_dict_list(params.get("anchors"))
        nodes = _ensure_str_list(params.get("nodes"))
        query_raw = params.get("query")
        query: Any = query_raw
        trimmed: Dict[str, Any] = {}
        if isinstance(query, list) and len(query) > 1:
            trimmed["query"] = {"from": len(query), "to": 1}
            query = query[:1]
        if isinstance(anchors, list) and len(anchors) > 1:
            trimmed["anchors"] = {"from": len(anchors), "to": 1}
            anchors = anchors[:1]
        if trimmed:
            meta["trimmed"] = trimmed
        if isinstance(query_raw, list):

            # Prefer keyword-list form for deterministic parsing.

            q_terms = [str(v).strip() for v in query_raw if isinstance(v, (str, int, float)) and str(v).strip()]

            query = q_terms or None

        elif isinstance(query_raw, str):

            q_terms = _extract_query_terms(query_raw.strip())

            # Fallback: keep the original string as a single term if extraction fails.

            if not q_terms and query_raw.strip():

                q_terms = [query_raw.strip()]

            query = q_terms or None

        else:

            query = None


        # 兼容旧版：explore.read
        # 兼容旧版：explore.read 已弃用；将其规范化为 explore.expand (hop=0)
        # 语义：读取 nodes 的 snippet，但不扩展邻居。
        if op == "read":
            op = "expand"
            # read 通常给 nodes；若 anchors 为空则从 nodes 构造 anchors
            if not anchors and nodes:
                anchors = [{"id": nid} for nid in nodes if isinstance(nid, str) and nid.strip()]
            # hop=0 表示“只读不扩展”
            params["hop"] = 0

        # 兼容：如果 expand 给的是 nodes 而不是 anchors，把 nodes 转成 anchors
        if op == "expand" and not anchors and nodes:
            anchors = [{"id": nid} for nid in nodes if isinstance(nid, str) and nid.strip()]

        # op 分支校验（不要再强制所有 explore 都带 anchors）
        if op == "expand":
            if not anchors:
                raise ProtocolError(
                    PlannerErrorCode.MISSING_REQUIRED_PARAM.value,
                    "explore.expand requires anchors (or nodes convertible to anchors)",
                )
        elif op == "find":
            if not anchors and not query:
                raise ProtocolError(
                    PlannerErrorCode.MISSING_REQUIRED_PARAM.value,
                    "explore.find requires anchors or query",
                )
        else:
            raise ProtocolError(PlannerErrorCode.UNKNOWN_ACTION.value, f"unknown explore op: {op}")

        # 数值字段解析 + cap
        def _to_int(name: str, default: int) -> int:
            try:
                return int(params.get(name, default))
            except Exception:
                return default

        hop_raw = _to_int("hop", 1)
        limit_raw = _to_int("limit", 50)
        max_per_anchor_raw = params.get("max_per_anchor")
        total_limit_raw = params.get("total_limit")
        dir_diversity_k_raw = params.get("dir_diversity_k")

        hop = max(0, min(2, hop_raw))
        limit = max(1, min(100, limit_raw))

        if max_per_anchor_raw is None:
            max_per_anchor_raw = None
        else:
            try:
                max_per_anchor_raw = int(max_per_anchor_raw)
            except Exception:
                max_per_anchor_raw = None

        if total_limit_raw is None:
            total_limit_raw = None
        else:
            try:
                total_limit_raw = int(total_limit_raw)
            except Exception:
                total_limit_raw = None

        if dir_diversity_k_raw is None:
            dir_diversity_k_raw = None
        else:
            try:
                dir_diversity_k_raw = int(dir_diversity_k_raw)
            except Exception:
                dir_diversity_k_raw = None

        def _cap_optional(v: Optional[int], lo: int, hi: int) -> Optional[int]:
            if v is None:
                return None
            return max(lo, min(hi, v))

        max_per_anchor = _cap_optional(max_per_anchor_raw, 1, 200)
        total_limit = _cap_optional(total_limit_raw, 1, 300)
        dir_diversity_k = _cap_optional(dir_diversity_k_raw, 0, 50)

        capped_fields: Dict[str, Any] = {}
        if hop != hop_raw:
            capped_fields["hop"] = hop_raw
        if limit != limit_raw:
            capped_fields["limit"] = limit_raw
        if max_per_anchor_raw is not None and max_per_anchor != max_per_anchor_raw:
            capped_fields["max_per_anchor"] = max_per_anchor_raw
        if total_limit_raw is not None and total_limit != total_limit_raw:
            capped_fields["total_limit"] = total_limit_raw
        if dir_diversity_k_raw is not None and dir_diversity_k != dir_diversity_k_raw:
            capped_fields["dir_diversity_k"] = dir_diversity_k_raw
        if capped_fields:
            meta.setdefault("warnings", []).append("value-capped")
            meta["capped"] = True
            meta["capped_fields"] = capped_fields

        return _attach_meta(
            ExploreAction(
                op=op,
                anchors=anchors,
                nodes=nodes,
                query=query,
                hop=hop,
                limit=limit,
                max_per_anchor=max_per_anchor,
                total_limit=total_limit,
                dir_diversity_k=dir_diversity_k,
            ),
            meta,
        )
    if action_name == "memory":
        target = str(params.get("target", "explore"))
        intent_raw = str(params.get("intent", "commit") or "commit").strip().lower()
        # Normalise natural synonyms to strict intents.
        if intent_raw in {"commit", "save", "store", "remember", "keep", "add"}:
            intent = "commit"
        elif intent_raw in {"delete", "remove", "drop", "forget", "clear"}:
            intent = "delete"
        else:
            raise ProtocolError(
                PlannerErrorCode.UNKNOWN_PARAM.value,
                f"unknown memory intent: {intent_raw!r} (use 'commit' or 'delete')",
            )
        selector = params.get("selector")
        # 兼容旧版：scope 字段存在时忽略
        if isinstance(selector, str):
            selector = selector.strip() or None
        return _attach_meta(MemoryAction(target=target, intent=intent, selector=selector), meta)

    if action_name == "repair":
        apply_flag = bool(payload.get("apply", False))
        plan_raw = payload.get("plan", [])
        subplan_raw = payload.get("subplan", None)
        def _to_steps(x):
            if x is None:
                return []
            if isinstance(x, str):
                s = x.strip()
                return [s] if s else []
            if isinstance(x, list):
                out = []
                for v in x:
                    if isinstance(v, str):
                        s = v.strip()
                        if s:
                            out.append(s)
                return out
            return []
        plan_steps = _to_steps(plan_raw)
        for s in _to_steps(subplan_raw):
            if s not in plan_steps:
                plan_steps.append(s)
        if (not plan_steps) and apply_flag:
            meta.setdefault("warnings", []).append("missing-plan")
        plan_targets = payload.get("plan_targets", [])
        if not isinstance(plan_targets, list):
            plan_targets = []
        patch = payload.get("patch", None)
        if patch is not None and not isinstance(patch, dict):
            patch = None
        return _attach_meta(
            RepairAction(apply=apply_flag, issue=issue, plan=plan_steps, plan_targets=plan_targets, patch=patch),
            meta,
        )

    if action_name == "submit":
        return _attach_meta(SubmitAction(), meta)

    if action_name == "noop":
        return _attach_meta(NoopAction(), meta)

    raise ProtocolError(PlannerErrorCode.UNKNOWN_ACTION.value, f"unsupported action '{action_name}'")


def validate_cgm_patch(obj: Mapping[str, Any]) -> CGMPatch:
    """Validate CGM patch structure and ensure a single target file."""

    if not isinstance(obj, Mapping):
        raise ProtocolError(CGMPatchErrorCode.INVALID_PATCH_SCHEMA.value, "CGM output must be a mapping")
    patch = obj.get("patch")
    if not isinstance(patch, Mapping):
        raise ProtocolError(CGMPatchErrorCode.INVALID_PATCH_SCHEMA.value, "'patch' field missing or not an object")
    edits_raw = patch.get("edits")
    if not isinstance(edits_raw, Sequence) or not edits_raw:
        raise ProtocolError(CGMPatchErrorCode.INVALID_PATCH_SCHEMA.value, "'patch.edits' must be a non-empty list")

    edits: List[Dict[str, Any]] = []
    paths: Set[str] = set()
    for idx, item in enumerate(edits_raw):
        if not isinstance(item, Mapping):
            raise ProtocolError(CGMPatchErrorCode.INVALID_PATCH_SCHEMA.value, f"edit #{idx} is not an object")
        path = item.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ProtocolError(CGMPatchErrorCode.PATH_MISSING.value, f"edit #{idx} missing file path")
        try:
            start = int(item.get("start"))
            end = int(item.get("end"))
        except Exception as exc:
            raise ProtocolError(CGMPatchErrorCode.RANGE_INVALID.value, f"edit #{idx} has non-integer range") from exc
        if start <= 0 or end < start:
            raise ProtocolError(CGMPatchErrorCode.RANGE_INVALID.value, f"edit #{idx} has invalid span {start}->{end}")
        new_text = item.get("new_text")
        if not isinstance(new_text, str):
            raise ProtocolError(CGMPatchErrorCode.INVALID_PATCH_SCHEMA.value, f"edit #{idx} missing new_text string")
        if CGM_CONTRACT.constraints.get("newline_required") and not new_text.endswith("\n"):
            raise ProtocolError(CGMPatchErrorCode.NEWLINE_MISSING.value, f"edit #{idx} new_text must end with newline")
        normalized = {
            "path": path,
            "start": start,
            "end": end,
            "new_text": new_text,
        }
        edits.append(normalized)
        paths.add(path)

    if len(paths) != 1 and CGM_CONTRACT.constraints.get("one_file_per_patch"):
        raise ProtocolError(CGMPatchErrorCode.MULTI_FILE_DIFF.value, "patch must touch exactly one file")

    path = next(iter(paths))
    summary = obj.get("summary")
    if summary is not None and not isinstance(summary, str):
        summary = str(summary)
    return CGMPatch(path=path, edits=edits, summary=summary)
