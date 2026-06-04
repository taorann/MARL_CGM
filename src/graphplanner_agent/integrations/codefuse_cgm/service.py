"""HTTP service for graph-aware CodeFuse-CGM generation.

This module is intentionally self-contained under ``agent_rebuild``.  It
borrows the operational shape of the legacy service while fixing the output
contract boundary:

* prompt can use the official raw completion shape or chat-template shape;
* native unified-diff output is parsed into structured patch edits;
* parser tries complete unified diff before complete JSON patch;
* partial regex salvage is disabled by default and clearly marked when enabled.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger("graphplanner_agent.cgm.service")


Patch = Dict[str, Any]


DIFF_SYSTEM_PROMPT = (
    "You are CodeFuse-CGM, a graph-aware code repair model. "
    "Use the issue, code graph, and snippets to produce a minimal implementation patch. "
    "Prefer complete unified diff output."
)

JSON_SYSTEM_PROMPT = (
    "You are CodeFuse-CGM, a graph-aware code repair model. "
    "Use the issue, code graph, and snippets to produce a minimal implementation patch."
)

DIFF_INSTRUCTION = """Generate the minimal implementation patch.
Return exactly one complete unified diff and nothing else.
The diff must include file headers (`--- a/...`, `+++ b/...`) and at least one `@@` hunk.
Do not output JSON, markdown fences, analysis, or prose.
Do not edit tests.
Do not create or delete files; do not use `/dev/null`.
Do not output binary-file notices, property changes, SVN metadata, or reproduction scripts.
After the final implementation hunk, stop immediately.
Keep the patch minimal and syntactically valid."""

JSON_INSTRUCTION = """Generate the minimal implementation patch.
Return exactly one complete JSON object and nothing else:
{"patch":{"edits":[{"path":"...","start":1,"end":1,"new_text":"..."}],"summary":"..."}}
Each `new_text` is complete replacement source for the line span and must end with a newline.
Do not include unified diff markers inside `new_text`.
Do not edit tests."""


@dataclass(slots=True)
class PlanTarget:
    path: str
    start: int
    end: int
    id: str = ""
    confidence: float = 1.0
    why: str = ""


@dataclass(slots=True)
class Plan:
    targets: list[PlanTarget] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    priority_tests: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParseResult:
    patch: Patch
    parser: str
    raw_preview: str


@dataclass(slots=True)
class RuntimeProfile:
    runtime_mode: str
    encoder_path: str
    adapter_path: str
    use_adj: bool
    prompt_mode: str = "chat_template"
    last_graph_profile: dict[str, Any] | None = None
    last_attention_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_mode": self.runtime_mode,
            "encoder_path": self.encoder_path,
            "adapter_path": self.adapter_path,
            "use_adj": self.use_adj,
            "prompt_mode": self.prompt_mode,
            "last_graph_profile": self.last_graph_profile,
            "last_attention_mode": self.last_attention_mode,
        }


class GenerateRequest(BaseModel):
    """Request payload accepted by the new CGM service."""

    issue: Optional[Dict[str, Any]] = None
    plan: Optional[Dict[str, Any]] = None
    plan_text: Optional[str] = None
    subgraph: Optional[Sequence[Mapping[str, Any]]] = None
    graph: Optional[Mapping[str, Any]] = None
    prompt: Optional[str] = None
    answer: Optional[str] = None
    repo: Optional[str] = None
    language: Optional[str] = None
    task: Optional[str] = None
    snippets: Optional[Sequence[Mapping[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    model_overrides: Optional[Dict[str, Any]] = Field(default=None, alias="model_config")
    generation_config: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow", populate_by_name=True)


def _preview(text: str, limit: int = 1200) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit] + f"...<truncated {len(value) - limit} chars>"


def _safe_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _safe_int(value: Any, default: int = 1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _pick(mapping: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        if key in mapping:
            value = _safe_str(mapping.get(key))
            if value:
                return value
    return ""


def _pick_text(mapping: Mapping[str, Any]) -> str:
    lines = mapping.get("snippet_lines")
    if isinstance(lines, Sequence) and not isinstance(lines, str):
        text = "\n".join(str(x) for x in lines)
        if text.strip():
            start = _safe_int(mapping.get("start_line", mapping.get("start", mapping.get("line", 1))), 1)
            return _strip_display_line_numbers(text.strip(), start)
    text = _pick(mapping, "text", "code", "content", "body", "signature", "summary")
    if not text:
        return ""
    start = _safe_int(mapping.get("start_line", mapping.get("start", mapping.get("line", 1))), 1)
    return _strip_display_line_numbers(text, start)


def _strip_display_line_numbers(text: str, start_line: int) -> str:
    value = str(text or "")
    lines = value.splitlines()
    if not lines:
        return value
    expected = max(1, int(start_line or 1))
    stripped: list[str] = []
    numbered = 0
    for idx, line in enumerate(lines):
        match = re.match(r"^\s*(\d+): ?(.*)$", line)
        if match and int(match.group(1)) == expected + idx:
            numbered += 1
            stripped.append(match.group(2))
        else:
            stripped.append(line)
    nonblank = sum(1 for line in lines if line.strip())
    if numbered and numbered >= max(1, nonblank // 2):
        return "\n".join(stripped)
    return value


def _norm_path(path: str) -> str:
    p = (path or "").strip()
    if "\t" in p:
        p = p.split("\t", 1)[0].strip()
    if p.startswith("a/") or p.startswith("b/"):
        p = p[2:]
    return p


def _ensure_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def _diff_candidates(text: str) -> list[str]:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    candidates: list[str] = []
    for block in re.findall(r"```(?:diff|patch|git)?\s*([\s\S]*?)```", normalized, flags=re.IGNORECASE):
        if "@@ " in block and ("--- " in block or "diff --git " in block):
            candidates.append(block.strip())
    lines = normalized.splitlines()
    starts = [
        idx
        for idx, line in enumerate(lines)
        if line.startswith("diff --git ") or line.startswith("--- a/") or line.startswith("--- ")
    ]
    for start in starts:
        candidate = "\n".join(lines[start:]).strip()
        if "@@ " in candidate and ("--- " in candidate or "diff --git " in candidate):
            candidates.append(candidate)
    if "@@ " in normalized and ("--- " in normalized or "diff --git " in normalized):
        candidates.append(normalized)
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            out.append(candidate)
            seen.add(candidate)
    return out


def parse_unified_diff_patch(response: str) -> Patch | None:
    """Parse a complete unified diff into structured replacement edits.

    The parser supports raw diffs with or without ``diff --git`` headers and
    multiple hunks/files.  It returns ``None`` unless every hunk has a target
    path and a parseable header.
    """

    for candidate in _diff_candidates(response):
        patch = _parse_single_unified_diff(candidate)
        if patch is not None:
            return patch
    return None


def _parse_single_unified_diff(diff_text: str) -> Patch | None:
    lines = (diff_text or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    if not lines:
        return None

    edits: list[dict[str, Any]] = []
    pending_path = ""
    current_old = ""
    current_new = ""
    current_path = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            match = re.match(r"^diff --git\s+(.+?)\s+(.+?)$", line)
            pending_path = _norm_path(match.group(2)) if match else ""
            current_old = ""
            current_new = ""
            current_path = pending_path
            i += 1
            continue
        if line.startswith("--- "):
            current_old = _norm_path(line[4:])
            i += 1
            continue
        if line.startswith("+++ "):
            current_new = _norm_path(line[4:])
            if current_new and current_new != "/dev/null":
                current_path = current_new
            elif current_old and current_old != "/dev/null":
                current_path = current_old
            elif pending_path:
                current_path = pending_path
            i += 1
            continue
        if not line.startswith("@@ "):
            i += 1
            continue

        header = re.match(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@", line)
        if not header:
            return None
        path = current_path or current_new or pending_path
        if not path or path == "/dev/null":
            return None
        old_start = int(header.group(1))
        old_count = int(header.group(2) or "1")
        new_lines: list[str] = []
        i += 1
        while i < len(lines):
            seg = lines[i]
            if seg.startswith("@@ ") or seg.startswith("diff --git ") or seg.startswith("--- "):
                break
            if seg.startswith("\\ No newline at end of file"):
                i += 1
                continue
            if seg.startswith("+") and not seg.startswith("+++ "):
                new_lines.append(seg[1:])
            elif seg.startswith(" "):
                new_lines.append(seg[1:])
            elif seg.startswith("-") and not seg.startswith("--- "):
                pass
            elif seg == "":
                # A literal blank line in a hunk is represented by a leading
                # space.  A bare blank is usually the final split artifact.
                if i == len(lines) - 1:
                    i += 1
                    break
                return None
            else:
                return None
            i += 1
        start = old_start
        end = old_start + old_count - 1
        if old_count == 0:
            end = old_start - 1
        edits.append({"path": path, "start": start, "end": end, "new_text": _ensure_newline("\n".join(new_lines))})

    if not edits:
        return None
    edits.sort(key=lambda edit: (int(edit["start"]), int(edit["end"])), reverse=True)
    return {"edits": edits, "summary": "codefuse-cgm-diff"}


def _extract_json_object(text: str) -> Mapping[str, Any] | None:
    value = (text or "").strip()
    if not value:
        return None
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", value, flags=re.IGNORECASE)
    if fence:
        value = fence.group(1).strip()
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, Mapping) else None
    except Exception:
        pass
    start = value.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(value[start:])
    except Exception:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def parse_json_patch(response: str | Mapping[str, Any]) -> Patch | None:
    raw: Mapping[str, Any] | None
    if isinstance(response, Mapping):
        raw = response
    else:
        raw = _extract_json_object(str(response or ""))
    if raw is None:
        return None
    patch_obj: Mapping[str, Any] | None = None
    if isinstance(raw.get("patch"), Mapping):
        patch_obj = raw.get("patch")  # type: ignore[assignment]
    elif isinstance(raw.get("edits"), Sequence):
        patch_obj = raw
    elif all(key in raw for key in ("path", "start", "end")):
        patch_obj = {"edits": [dict(raw)], "summary": raw.get("summary")}
    elif isinstance(raw.get("changes"), Sequence):
        patch_obj = {"edits": list(raw.get("changes") or []), "summary": raw.get("summary")}
    if not isinstance(patch_obj, Mapping):
        return None
    edits_raw = patch_obj.get("edits")
    if not isinstance(edits_raw, Sequence) or isinstance(edits_raw, str):
        return None
    edits: list[dict[str, Any]] = []
    for entry in edits_raw:
        if not isinstance(entry, Mapping):
            continue
        path = _safe_str(entry.get("path"))
        if not path:
            continue
        start = _safe_int(entry.get("start"), 0)
        end = _safe_int(entry.get("end", start), start)
        if start <= 0 or end < start:
            continue
        raw_new_text = entry.get("new_text") if entry.get("new_text") is not None else entry.get("text")
        if not isinstance(raw_new_text, str) or not raw_new_text:
            continue
        new_text = raw_new_text
        if re.search(r"(?m)^\s*(diff --git |--- a/|\+\+\+ b/|@@ )", new_text):
            return None
        edits.append({"path": path, "start": start, "end": end, "new_text": _ensure_newline(new_text)})
    if not edits:
        return None
    return {"edits": edits, "summary": _safe_str(patch_obj.get("summary") or raw.get("summary") or "codefuse-cgm-json")}


def _extract_quoted_value(text: str, quote_index: int) -> str:
    out: list[str] = []
    escaped = False
    i = quote_index + 1
    while i < len(text):
        ch = text[i]
        if escaped:
            out.append("\\" + ch)
            escaped = False
            i += 1
            continue
        if ch == "\\":
            escaped = True
            i += 1
            continue
        if ch == '"':
            break
        out.append(ch)
        i += 1
    raw = "".join(out)
    try:
        return json.loads(f'"{raw}"')
    except Exception:
        return raw.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")


def parse_partial_patch(response: str) -> Patch | None:
    """Last-resort legacy salvage, intentionally disabled by default."""

    text = response or ""
    path_m = re.search(r'"path"\s*:\s*"([^"\n]+)"', text)
    start_m = re.search(r'"start"\s*:\s*(\d+)', text)
    end_m = re.search(r'"end"\s*:\s*(\d+)', text)
    key_m = re.search(r'"new_text"\s*:\s*', text)
    if not (path_m and start_m and key_m):
        return None
    path = path_m.group(1).strip()
    start = int(start_m.group(1))
    end = int(end_m.group(1)) if end_m else start
    idx = key_m.end()
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx < len(text) and text[idx] == '"':
        new_text = _extract_quoted_value(text, idx)
    else:
        new_text = text[idx:]
    marker = re.search(r"\n\s*(?:diff --git |--- a/|\+\+\+ b/|@@ )", new_text)
    if marker and marker.start() > 0:
        new_text = new_text[: marker.start()]
    if not path or start <= 0 or end < start or not new_text.strip():
        return None
    return {"edits": [{"path": path, "start": start, "end": end, "new_text": _ensure_newline(new_text.rstrip("\n"))}], "summary": "codefuse-cgm-partial"}


def parse_model_output(response: str, *, allow_partial: bool = False) -> ParseResult | None:
    raw = str(response or "")
    patch = parse_unified_diff_patch(raw)
    if patch is not None and _patch_artifact_reason(patch) is None:
        return ParseResult(patch=patch, parser="unified_diff", raw_preview=_preview(raw))
    patch = parse_json_patch(raw)
    if patch is not None and _patch_artifact_reason(patch) is None:
        return ParseResult(patch=patch, parser="json_patch", raw_preview=_preview(raw))
    if allow_partial:
        patch = parse_partial_patch(raw)
        if patch is not None and _patch_artifact_reason(patch) is None:
            return ParseResult(patch=patch, parser="partial_fallback", raw_preview=_preview(raw))
    return None


def _patch_artifact_reason(patch: Mapping[str, Any]) -> str | None:
    edits = patch.get("edits")
    if not isinstance(edits, Sequence) or isinstance(edits, str):
        return "patch edits missing"
    for edit in edits:
        if not isinstance(edit, Mapping):
            return "patch edit is not an object"
        text = str(edit.get("new_text") or "")
        reason = _text_artifact_reason(text)
        if reason:
            return reason
    return None


def _text_artifact_reason(text: str) -> str | None:
    if re.search(r"(?m)^\s*(diff --git |--- a/|\+\+\+ b/|@@ )", text):
        return "diff marker embedded in replacement text"
    if "('',)" in text or '("",)' in text:
        return "empty tuple generation artifact"
    if re.search(r"\b[Ee]ditted by\b|\b[Ee]dited by\b", text):
        return "model signature generation artifact"
    if re.search(r"\('',\)\s*\{", text):
        return "tuple/schema artifact embedded in replacement text"
    if re.search(r"\{\s*['\"]patch['\"]\s*:", text) or re.search(r"\{\s*['\"]edits['\"]\s*:", text):
        return "patch schema JSON embedded in replacement text"
    return None


def _canonical_node_type(node: Mapping[str, Any], path: str) -> str:
    kind = _pick(node, "kind", "type", "nodeType").lower()
    if kind in {"repo"}:
        return "Repo"
    if kind in {"package", "namespace"}:
        return "Package"
    if kind in {"file", "module"}:
        return "File"
    if kind in {"textfile", "text_file"}:
        return "TextFile"
    if kind in {"class", "type"}:
        return "Class"
    if kind in {"attribute", "attr", "field", "assignment", "module_assignment", "class_assignment"}:
        return "Attribute"
    if kind in {"lambda"}:
        return "Lambda"
    if kind in {"function", "func", "method", "chunk"}:
        return "Function"
    return "Function" if path else "File"


def _infer_repo_name(issue: Mapping[str, Any] | None, explicit_repo: str) -> str:
    if explicit_repo:
        return explicit_repo.split("/")[-1]
    issue_obj = issue if isinstance(issue, Mapping) else {}
    for key in ("repo", "repository", "project", "repo_name"):
        value = _safe_str(issue_obj.get(key))
        if value:
            return value.split("/")[-1]
    meta = issue_obj.get("metadata")
    if isinstance(meta, Mapping):
        for key in ("repo", "repository", "project", "repo_name"):
            value = _safe_str(meta.get(key))
            if value:
                return value.split("/")[-1]
    return "repo"


def _infer_language(issue: Mapping[str, Any] | None, explicit_language: str) -> str:
    if explicit_language:
        return explicit_language.lower()
    issue_obj = issue if isinstance(issue, Mapping) else {}
    value = _safe_str(issue_obj.get("language"))
    if value:
        return value.lower()
    meta = issue_obj.get("metadata")
    if isinstance(meta, Mapping):
        value = _safe_str(meta.get("language"))
        if value:
            return value.lower()
    return "python"


def _split_path(path: str) -> tuple[str, str]:
    normalized = (path or "").replace("\\", "/").lstrip("./")
    if "/" not in normalized:
        return "", normalized
    directory, filename = normalized.rsplit("/", 1)
    return directory, filename


def _dedupe_node_id(raw_id: str, seen: MutableMapping[str, int], fallback_idx: int) -> str:
    base = raw_id or f"node-{fallback_idx}"
    count = seen.get(base, 0)
    seen[base] = count + 1
    if count == 0:
        return base
    return f"{base}#{count}"


def _extract_edges_from_neighbors(
    canonical_nodes: Sequence[Mapping[str, Any]],
    id_alias: Mapping[str, str],
) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for node in canonical_nodes:
        source = _safe_str(node.get("id"))
        neighbors = node.get("neighbors")
        if not source or not isinstance(neighbors, Sequence) or isinstance(neighbors, str):
            continue
        for item in neighbors:
            if not isinstance(item, Mapping):
                continue
            raw_target = _pick(item, "to_id", "id", "nodeId", "to")
            if not raw_target:
                continue
            target = id_alias.get(raw_target, raw_target)
            edge_type = _pick(item, "type", "edgeType") or "RELATED"
            key = (source, target, edge_type)
            if key in seen:
                continue
            seen.add(key)
            edges.append({"source": source, "target": target, "type": edge_type, "edgeType": edge_type})
    return edges


def _extract_edges_raw(raw_edges: Sequence[Mapping[str, Any]], id_alias: Mapping[str, str]) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in raw_edges:
        raw_source = _pick(edge, "source", "src", "from")
        raw_target = _pick(edge, "target", "dst", "to")
        if not raw_source or not raw_target:
            continue
        source = id_alias.get(raw_source, raw_source)
        target = id_alias.get(raw_target, raw_target)
        edge_type = _pick(edge, "type", "edgeType", "kind") or "RELATED"
        key = (source, target, edge_type)
        if key in seen:
            continue
        seen.add(key)
        edges.append({"source": source, "target": target, "type": edge_type, "edgeType": edge_type})
    return edges


def _canonicalize_node(*, node: Mapping[str, Any], node_id: str, node_type: str, repo_name: str) -> dict[str, Any]:
    path = _pick(node, "path", "file", "abs_path")
    start_line = _safe_int(node.get("start_line", node.get("start", node.get("line", 1))), 1)
    end_line = _safe_int(node.get("end_line", node.get("end", start_line)), start_line)
    name = _pick(node, "name", "symbol", "label", "title")
    if not name and path:
        name = _split_path(path)[1] or path
    if not name:
        name = node_id
    signature = _pick(node, "signature", "header", "summary")
    comment = _pick(node, "comment", "docstring", "summary", "description")
    text = _pick_text(node) or signature
    file_path, file_name = _split_path(path)
    out: dict[str, Any] = {
        "id": node_id,
        "nodeId": node_id,
        "type": node_type,
        "kind": node_type.lower(),
        "nodeType": node_type,
        "name": name,
        "path": path,
        "start_line": max(1, start_line),
        "end_line": max(max(1, start_line), end_line),
        "text": text,
    }
    neighbors = node.get("neighbors")
    if isinstance(neighbors, Sequence) and not isinstance(neighbors, str):
        out["neighbors"] = [item for item in neighbors if isinstance(item, Mapping)]
    if text:
        out["snippet_lines"] = str(text).splitlines()
    if node_type == "Repo":
        out["name"] = name or repo_name
        return out
    if node_type == "Package":
        return out
    if node_type in {"File", "TextFile"}:
        out["fileName"] = file_name or name
        out["filePath"] = file_path
        return out
    if node_type == "Class":
        out["className"] = name
        out["classType"] = _pick(node, "classType", "class_type") or "class"
        out["comment"] = comment
        return out
    if node_type == "Attribute":
        out["attributeType"] = _pick(node, "attributeType", "fieldType", "kind", "type") or "attribute"
        out["comment"] = comment
        return out
    if node_type == "Lambda":
        return out
    out["header"] = signature or f"def {name}"
    out["comment"] = comment
    return out


def normalize_graph(
    raw: Any,
    *,
    issue: Mapping[str, Any] | None,
    language: str = "",
    repo_name: str = "",
) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        nodes_raw = raw.get("nodes")
        edges_raw = raw.get("edges") or raw.get("adjacency_edges")
        explicit_repo = _safe_str(raw.get("reponame") or raw.get("repo") or repo_name)
        explicit_lang = _safe_str(raw.get("language") or language)
    elif isinstance(raw, Sequence) and not isinstance(raw, str):
        nodes_raw = raw
        edges_raw = []
        explicit_repo = repo_name
        explicit_lang = language
    else:
        nodes_raw = []
        edges_raw = []
        explicit_repo = repo_name
        explicit_lang = language
    issue_obj = issue if isinstance(issue, Mapping) else {}
    repo = _infer_repo_name(issue_obj, explicit_repo or repo_name)
    lang = _infer_language(issue_obj, explicit_lang or language)
    nodes_list = [x for x in nodes_raw if isinstance(x, Mapping)] if isinstance(nodes_raw, Sequence) else []
    edges_list = [x for x in edges_raw if isinstance(x, Mapping)] if isinstance(edges_raw, Sequence) else []
    seen_ids: dict[str, int] = {}
    id_alias: dict[str, str] = {}
    canonical_nodes: list[dict[str, Any]] = []
    for idx, node in enumerate(nodes_list):
        raw_id = _pick(node, "id", "nodeId")
        node_id = _dedupe_node_id(raw_id, seen_ids, idx)
        if raw_id and raw_id not in id_alias:
            id_alias[raw_id] = node_id
        id_alias[node_id] = node_id
        path = _pick(node, "path", "file", "abs_path")
        node_type = _canonical_node_type(node, path)
        canonical_nodes.append(_canonicalize_node(node=node, node_id=node_id, node_type=node_type, repo_name=repo))

    repo_id = f"repo::{repo}"
    file_nodes: dict[str, str] = {}
    for node in list(canonical_nodes):
        path = _safe_str(node.get("path"))
        node_type = _safe_str(node.get("nodeType"))
        if not path or node_type in {"Repo", "Package", "File", "TextFile"}:
            continue
        if path in file_nodes:
            continue
        file_id = f"file::{path}"
        file_nodes[path] = file_id
        directory, filename = _split_path(path)
        canonical_nodes.append(
            {
                "id": file_id,
                "nodeId": file_id,
                "type": "File",
                "kind": "file",
                "nodeType": "File",
                "name": filename or path,
                "path": path,
                "fileName": filename or path,
                "filePath": directory,
                "text": "",
                "start_line": 1,
                "end_line": 1,
            }
        )
    canonical_nodes.append(
        {
            "id": repo_id,
            "nodeId": repo_id,
            "type": "Repo",
            "kind": "repo",
            "nodeType": "Repo",
            "name": repo,
            "path": repo,
            "text": repo,
            "start_line": 1,
            "end_line": 1,
            "snippet_lines": [repo],
        }
    )

    deduped_nodes: list[dict[str, Any]] = []
    used: set[str] = set()
    for idx, node in enumerate(canonical_nodes):
        node_obj = dict(node)
        raw_id = _safe_str(node_obj.get("id") or node_obj.get("nodeId")) or f"node-{idx}"
        if raw_id in used:
            raw_id = f"{raw_id}#{idx}"
        used.add(raw_id)
        node_obj["id"] = raw_id
        node_obj["nodeId"] = raw_id
        id_alias.setdefault(raw_id, raw_id)
        deduped_nodes.append(node_obj)
    canonical_nodes = deduped_nodes

    edges = _extract_edges_raw(edges_list, id_alias)
    edges.extend(_extract_edges_from_neighbors(canonical_nodes, id_alias))
    edge_seen: set[tuple[str, str, str]] = {(edge["source"], edge["target"], edge["type"]) for edge in edges}
    for node in canonical_nodes:
        node_id = _safe_str(node.get("id"))
        path = _safe_str(node.get("path"))
        node_type = _safe_str(node.get("nodeType"))
        if node_type in {"Repo", "Package", "File", "TextFile"} or not path or path not in file_nodes:
            continue
        file_id = file_nodes[path]
        key = (file_id, node_id, "CONTAINS")
        if key not in edge_seen:
            edge_seen.add(key)
            edges.append({"source": file_id, "target": node_id, "type": "CONTAINS", "edgeType": "CONTAINS"})
    for path, file_id in file_nodes.items():
        key = (repo_id, file_id, "CONTAINS")
        if key not in edge_seen:
            edge_seen.add(key)
            edges.append({"source": repo_id, "target": file_id, "type": "CONTAINS", "edgeType": "CONTAINS"})
    return {"nodes": canonical_nodes, "edges": edges, "reponame": repo, "language": lang}


@dataclass(slots=True)
class SnippetFormatter:
    max_snippets: int = 5
    max_lines_per_snippet: int = 40
    show_line_numbers: bool = True

    def format(self, snippets: Sequence[Mapping[str, Any]] | None) -> str:
        if not snippets:
            return ""
        blocks: list[str] = []
        for entry in list(snippets)[: self.max_snippets]:
            if not isinstance(entry, Mapping):
                continue
            path = _pick(entry, "path", "abs_path") or "unknown"
            start = entry.get("start") or entry.get("line")
            end = entry.get("end") or start
            header = f"{path}:{start}-{end}"
            lines = entry.get("snippet") or entry.get("lines") or entry.get("snippet_lines")
            if isinstance(lines, Sequence) and not isinstance(lines, str):
                base = _safe_int(start, 0) if start is not None else 0
                normalized_lines = _strip_display_line_numbers("\n".join(str(item) for item in lines), base or 1).splitlines()
                out_lines: list[str] = []
                for idx, raw in enumerate(normalized_lines[: self.max_lines_per_snippet]):
                    line = str(raw)
                    if self.show_line_numbers and base > 0:
                        out_lines.append(f"{base + idx:>5}: {line}")
                    else:
                        out_lines.append(line)
                blocks.append(header + "\n" + "\n".join(out_lines))
            else:
                body = _pick(entry, "text", "content")
                blocks.append(header + ("\n" + _preview(body, 1600) if body else ""))
        return "\n\n".join(blocks)


class ConversationEncoder:
    def __init__(self, tokenizer: Any, *, max_length: int, system_prompt: str, use_chat_template: bool = True) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.use_chat_template = bool(use_chat_template)

    def build_user_message(
        self,
        *,
        prompt: str,
        plan_text: str | None,
        snippets_text: str,
        issue_text: str | None,
        prompt_layout: str = "sectioned",
    ) -> str:
        if prompt_layout == "issue_only":
            return (issue_text or prompt or "").strip()
        sections: list[str] = []
        if issue_text:
            sections.append(f"[Issue]\n{issue_text.strip()}")
        sections.append(f"[Instruction]\n{prompt.strip()}")
        if plan_text:
            sections.append(f"[Plan]\n{plan_text.strip()}")
        if snippets_text:
            sections.append(f"[Snippets]\n{snippets_text}")
        return "\n\n".join(sections)

    def encode_prompt(
        self,
        *,
        prompt: str,
        plan_text: str | None,
        snippets_text: str,
        issue_text: str | None,
        prompt_layout: str = "sectioned",
    ) -> MutableMapping[str, Any]:
        import torch

        user = self.build_user_message(
            prompt=prompt,
            plan_text=plan_text,
            snippets_text=snippets_text,
            issue_text=issue_text,
            prompt_layout=prompt_layout,
        )
        if not self.use_chat_template:
            # Official CodeFuse-CGM training defaults to use_chat=false: the
            # model sees raw prompt tokens and learns to continue with answer
            # tokens directly.  Keep inference in that completion-style shape
            # when requested instead of adding chat role markers.
            raw_prompt = user if user.endswith("\n") else user + "\n"
            encoded = self.tokenizer(
                raw_prompt,
                add_special_tokens=False,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            if "attention_mask" not in encoded:
                encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
            return encoded
        messages: list[dict[str, str]] = [{"role": "user", "content": user}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        if hasattr(self.tokenizer, "apply_chat_template"):
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            if isinstance(encoded, torch.Tensor):
                return {"input_ids": encoded, "attention_mask": torch.ones_like(encoded)}
            return encoded
        text = "\n\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in messages) + "\n\nASSISTANT:"
        return self.tokenizer(text, max_length=self.max_length, truncation=True, return_tensors="pt")


def _node_sentence(node: Mapping[str, Any], *, repo_name: str, max_chars: int = 1024000) -> str:
    node_type = _pick(node, "nodeType", "type", "kind") or "Function"
    text = _pick_text(node)
    if node_type == "Repo":
        sentence = repo_name
    elif node_type == "Package":
        sentence = _pick(node, "name") or ""
    elif node_type == "File":
        file_path = _pick(node, "filePath")
        file_name = _pick(node, "fileName", "name")
        prefix = f"{file_path}/" if file_path else ""
        sentence = f"{prefix}{file_name}\n{text}".strip()
    elif node_type in {"TextFile", "Textfile"}:
        sentence = f"{_pick(node, 'name')}\n{text}".strip()
    elif node_type == "Class":
        sentence = f"{_pick(node, 'classType')} {_pick(node, 'className', 'name')}\n{_pick(node, 'comment')}\n{text}".strip(" ")
    elif node_type == "Attribute":
        sentence = f"{_pick(node, 'attributeType')} {_pick(node, 'name')}\n{_pick(node, 'comment')}\n{text}".strip(" ")
    elif node_type == "Function":
        comment = _pick(node, "comment")
        comment_prefix = f"{comment}\n" if comment else ""
        sentence = f"{_pick(node, 'header')} {_pick(node, 'name')}\n{comment_prefix}{text}".strip(" ")
    elif node_type == "Lambda":
        sentence = text.strip(" ")
    else:
        sentence = text.strip(" ")
    return sentence[:max_chars] if len(sentence) > max_chars else sentence


@dataclass(slots=True)
class GraphEncoding:
    embeddings: Any
    adjacency: Any
    node_count: int
    segment_count: int
    edge_count: int

    def profile(self) -> dict[str, Any]:
        import torch

        return {
            "node_count": self.node_count,
            "segment_count": self.segment_count,
            "edge_count": self.edge_count,
            "embedding_shape": list(self.embeddings.shape),
            "adjacency_shape": list(self.adjacency.shape) if self.adjacency is not None else None,
            "adjacency_nonzero": int(torch.count_nonzero(self.adjacency).item()) if self.adjacency is not None else 0,
        }


def encode_graph(*, graph: Mapping[str, Any], encoder: Any, tokenizer: Any, device: Any, embedding_dim: int = 256, max_segment_tokens: int = 512, save_adj: bool = True) -> GraphEncoding:
    import torch

    nodes = [x for x in graph.get("nodes", []) if isinstance(x, Mapping)]
    edges = [x for x in graph.get("edges", []) if isinstance(x, Mapping)]
    repo_name = _safe_str(graph.get("reponame") or graph.get("repo") or "repo")
    node_id_to_indices: dict[str, list[int]] = {}
    all_embeddings: list[Any] = []
    index_counter = 0
    for idx, node in enumerate(nodes):
        node_id = _pick(node, "nodeId", "id") or f"node-{idx}"
        sentence = _node_sentence(node, repo_name=repo_name)
        tokens = tokenizer.tokenize(sentence)
        if not tokens:
            node_id_to_indices[node_id] = [index_counter]
            all_embeddings.append(torch.zeros((embedding_dim,), dtype=torch.float32, device=device))
            index_counter += 1
            continue
        segments = (len(tokens) + max_segment_tokens - 1) // max_segment_tokens
        node_id_to_indices[node_id] = list(range(index_counter, index_counter + segments))
        for seg_idx in range(segments):
            start = seg_idx * max_segment_tokens
            end = min((seg_idx + 1) * max_segment_tokens, len(tokens))
            ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens[start:end]), device=device, dtype=torch.long).unsqueeze(0)
            emb = encoder(ids)
            if isinstance(emb, (tuple, list)):
                emb = emb[0]
            if emb.dim() == 3:
                emb = emb[:, 0, :]
            if emb.dim() == 2:
                emb = emb.squeeze(0)
            all_embeddings.append(emb.to(device))
            index_counter += 1
    if not all_embeddings:
        all_embeddings.append(torch.zeros((embedding_dim,), dtype=torch.float32, device=device))
    embeddings = torch.stack(all_embeddings, dim=0)
    segment_count = int(embeddings.shape[0])
    adjacency = None
    if save_adj:
        adjacency = torch.zeros((segment_count, segment_count), dtype=torch.float32, device=device)
        for edge in edges:
            source = _pick(edge, "source", "src")
            target = _pick(edge, "target", "dst")
            for src_idx in node_id_to_indices.get(source, []):
                for dst_idx in node_id_to_indices.get(target, []):
                    adjacency[src_idx, dst_idx] = 1.0
        for indices in node_id_to_indices.values():
            for i in indices:
                adjacency[i, i] = 1.0
            for left in range(len(indices)):
                for right in range(left + 1, len(indices)):
                    adjacency[indices[left], indices[right]] = 1.0
                    adjacency[indices[right], indices[left]] = 1.0
    return GraphEncoding(embeddings=embeddings, adjacency=adjacency, node_count=len(nodes), segment_count=segment_count, edge_count=len(edges))


def _resolve_dtype(name: str | None) -> Any:
    if not name:
        return None
    import torch

    try:
        return getattr(torch, name.strip().lower())
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {name}") from exc


def _primary_device(model: Any, fallback: str) -> Any:
    import torch

    if hasattr(model, "device") and model.device is not None:
        return torch.device(model.device)
    if hasattr(model, "hf_device_map"):
        device_map = getattr(model, "hf_device_map") or {}
        if isinstance(device_map, Mapping) and device_map:
            first = next(iter(device_map.values()))
            if isinstance(first, (list, tuple)) and first:
                first = first[0]
            return torch.device(first)
    try:
        return torch.device(next(model.parameters()).device)
    except StopIteration:
        return torch.device(fallback)


@dataclass(slots=True)
class GenerationConfig:
    model_name_or_path: str
    tokenizer_name_or_path: str | None = None
    max_length: int = 8192
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 0.9
    do_sample: bool = False
    num_return_sequences: int = 1
    device: str | None = None
    device_map: Any = None
    torch_dtype: str | None = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    attn_implementation: str | None = None


def build_graph_adapter(*, embedding_dim: int, hidden_dim: int, lm_hidden_dim: int) -> Any:
    """Build an adapter with legacy checkpoint-compatible parameter names."""

    from torch import nn

    class _GraphAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(embedding_dim, hidden_dim)
            self.gelu = nn.GELU()
            self.fc2 = nn.Linear(hidden_dim, lm_hidden_dim)

        def forward(self, x: Any) -> Any:
            return self.fc2(self.gelu(self.fc1(x)))

    return _GraphAdapter()


class GraphAwareGenerator:
    def __init__(
        self,
        config: GenerationConfig,
        *,
        encoder_path: str | None = None,
        adapter_path: str | None = None,
        adapter_hidden_dim: int = 4096,
        embedding_dim: int = 256,
        use_adj: bool = True,
        strict_adj: bool = False,
        system_prompt: str = DIFF_SYSTEM_PROMPT,
        use_chat_template: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.embedding_dim = embedding_dim
        self.use_adj = use_adj
        self.strict_adj = strict_adj
        model_root = Path(config.model_name_or_path)
        self.encoder_path = str(encoder_path or model_root / "codet5p-110m-embedding-cgm")
        self.adapter_path = str(adapter_path or model_root / "adapter-cgm" / "adapter.pth")
        if not Path(self.encoder_path).exists():
            raise FileNotFoundError(f"CGM graph encoder path not found: {self.encoder_path}")
        if not Path(self.adapter_path).exists():
            raise FileNotFoundError(f"CGM graph adapter weights not found: {self.adapter_path}")
        dtype = _resolve_dtype(config.torch_dtype)
        explicit_device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok_path = config.tokenizer_name_or_path or config.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=config.trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs: dict[str, Any] = {"trust_remote_code": bool(config.trust_remote_code)}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if config.device_map is not None:
            model_kwargs["device_map"] = config.device_map
        if config.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        if config.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        if config.attn_implementation:
            model_kwargs["attn_implementation"] = config.attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)
        if config.device_map is None:
            self.device = torch.device(explicit_device)
            self.model.to(self.device)
        else:
            self.device = _primary_device(self.model, explicit_device)
        self.model.eval()
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(self.encoder_path, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(self.encoder_path, torch_dtype=dtype or torch.float32, trust_remote_code=True)
        self.encoder.to(self.device)
        self.encoder.eval()
        lm_hidden = int(getattr(self.model.config, "hidden_size"))
        self.graph_adapter = build_graph_adapter(embedding_dim=embedding_dim, hidden_dim=adapter_hidden_dim, lm_hidden_dim=lm_hidden)
        state = torch.load(self.adapter_path, map_location="cpu")
        if isinstance(state, Mapping) and isinstance(state.get("state_dict"), Mapping):
            state = state["state_dict"]
        self.graph_adapter.load_state_dict(state)  # type: ignore[arg-type]
        self.graph_adapter.to(device=self.device, dtype=dtype or torch.float32)
        self.graph_adapter.eval()
        self.encoder_prompt = ConversationEncoder(
            self.tokenizer,
            max_length=config.max_length,
            system_prompt=system_prompt,
            use_chat_template=use_chat_template,
        )
        self.snippet_formatter = SnippetFormatter()
        prompt_mode = "chat_template" if use_chat_template else "raw_completion"
        self.profile = RuntimeProfile("graph", self.encoder_path, self.adapter_path, use_adj, prompt_mode)

    def _embed_tokens(self, input_ids: Any) -> Any:
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens(input_ids)
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()(input_ids)
        raise AttributeError("Loaded CGM language model does not expose input embeddings")

    def _build_graph_attention(self, *, graph_adj: Any, graph_len: int, qa_mask: Any) -> Any | None:
        import torch

        if graph_adj is None or not self.use_adj:
            return None
        if graph_adj.dim() == 2:
            graph_adj = graph_adj.unsqueeze(0)
        batch = int(qa_mask.shape[0])
        if graph_adj.shape[0] != batch:
            graph_adj = graph_adj.expand(batch, -1, -1)
        q_len = int(qa_mask.shape[-1])
        qa_causal = torch.tril(torch.ones((batch, q_len, q_len), device=qa_mask.device, dtype=graph_adj.dtype))
        graph_to_qa = qa_mask.unsqueeze(1).to(graph_adj.dtype).expand(batch, graph_len, q_len)
        qa_to_graph = torch.ones((batch, q_len, graph_len), device=qa_mask.device, dtype=graph_adj.dtype)
        full = torch.cat([torch.cat([graph_adj.to(qa_mask.device), graph_to_qa], dim=2), torch.cat([qa_to_graph, qa_causal], dim=2)], dim=1)
        return full[:, None, :, :]

    def generate(
        self,
        *,
        prompt: str,
        plan_text: str,
        issue_text: str | None,
        snippets: Sequence[Mapping[str, Any]],
        graph: Mapping[str, Any],
        prompt_layout: str = "sectioned",
    ) -> list[str]:
        import torch

        encoded = self.encoder_prompt.encode_prompt(
            prompt=prompt,
            plan_text=plan_text,
            snippets_text=self.snippet_formatter.format(snippets),
            issue_text=issue_text,
            prompt_layout=prompt_layout,
        )
        input_ids = encoded["input_ids"].to(self.device)
        qa_mask = encoded["attention_mask"].to(self.device)
        graph_encoded = encode_graph(
            graph=graph,
            encoder=self.encoder,
            tokenizer=self.encoder_tokenizer,
            device=self.device,
            embedding_dim=self.embedding_dim,
            save_adj=self.use_adj,
        )
        graph_embeddings = graph_encoded.embeddings.to(self.device)
        adapter_dtype = next(self.graph_adapter.parameters()).dtype
        if graph_embeddings.dtype != adapter_dtype:
            graph_embeddings = graph_embeddings.to(dtype=adapter_dtype)
        graph_prefix = self.graph_adapter(graph_embeddings).unsqueeze(0)
        token_embeddings = self._embed_tokens(input_ids)
        inputs_embeds = torch.cat([graph_prefix, token_embeddings], dim=1)
        graph_len = int(graph_prefix.shape[1])
        prefix_mask = torch.ones((qa_mask.shape[0], graph_len), dtype=qa_mask.dtype, device=qa_mask.device)
        two_d_mask = torch.cat([prefix_mask, qa_mask], dim=1)
        attention_mask = two_d_mask
        attention_mode = "graph_prefix_2d"
        if self.use_adj:
            graph_mask = self._build_graph_attention(graph_adj=graph_encoded.adjacency, graph_len=graph_len, qa_mask=qa_mask)
            if graph_mask is not None:
                attention_mask = graph_mask
                attention_mode = "graph_prefix_adj_4d"
        self.profile.last_graph_profile = graph_encoded.profile()
        self.profile.last_attention_mode = attention_mode
        kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": self.config.do_sample,
            "num_return_sequences": self.config.num_return_sequences,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        try:
            generated = self.model.generate(**kwargs)
        except Exception:
            if self.strict_adj or attention_mode != "graph_prefix_adj_4d":
                raise
            self.profile.last_attention_mode = "graph_prefix_2d_adj_fallback"
            kwargs["attention_mask"] = two_d_mask
            generated = self.model.generate(**kwargs)
        return [self.tokenizer.decode(seq, skip_special_tokens=True).strip() for seq in generated]


def _issue_text(issue: Mapping[str, Any] | None) -> str | None:
    if not isinstance(issue, Mapping):
        return None
    body = issue.get("body") or issue.get("description")
    if isinstance(body, str) and body.strip():
        return body.strip()
    title = issue.get("title")
    return title.strip() if isinstance(title, str) and title.strip() else None


def _coerce_plan(request: GenerateRequest) -> Plan:
    raw = request.plan or {}
    targets: list[PlanTarget] = []
    if isinstance(raw, Mapping):
        raw_targets = raw.get("targets")
        if isinstance(raw_targets, Sequence) and not isinstance(raw_targets, str):
            for idx, item in enumerate(raw_targets):
                if not isinstance(item, Mapping):
                    continue
                path = _safe_str(item.get("path"))
                start = _safe_int(item.get("start"), 0)
                end = _safe_int(item.get("end", start), start)
                if path and start > 0 and end >= start:
                    targets.append(PlanTarget(path=path, start=start, end=end, id=_safe_str(item.get("id") or f"target-{idx}"), confidence=float(item.get("confidence") or 1.0), why=_safe_str(item.get("why"))))
    if not targets and isinstance(request.snippets, Sequence):
        for idx, item in enumerate(request.snippets):
            if not isinstance(item, Mapping):
                continue
            path = _safe_str(item.get("path") or item.get("abs_path"))
            start = _safe_int(item.get("start") or item.get("line"), 0)
            end = _safe_int(item.get("end", start), start)
            if path and start > 0 and end >= start:
                targets.append(PlanTarget(path=path, start=start, end=end, id=f"snippet-{idx}", confidence=0.7, why="snippet-context"))
            if len(targets) >= 8:
                break
    return Plan(targets=targets, budget={}, priority_tests=[])


def _coerce_plan_text(request: GenerateRequest, plan: Plan) -> str:
    if isinstance(request.plan_text, str) and request.plan_text.strip():
        return request.plan_text.strip()
    lines: list[str] = []
    for target in plan.targets[:5]:
        item = f"{target.path}:{target.start}-{target.end}"
        if target.why:
            item += f" ({target.why})"
        lines.append(item)
    return "\n".join(lines)


def _extract_constraints(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    constraints = metadata.get("constraints")
    return constraints if isinstance(constraints, Mapping) else {}


def _requested_output_format(request: GenerateRequest, default_output_format: str, force_diff: bool) -> str:
    if force_diff:
        return "diff"
    constraints = _extract_constraints(request.metadata)
    raw = _safe_str((request.metadata or {}).get("output_format") if isinstance(request.metadata, Mapping) else "")
    raw = raw or _safe_str(constraints.get("output_format"))
    lowered = raw.lower()
    if lowered in {"json", "json_patch", "json_patch_object"}:
        return "json"
    if lowered in {"diff", "unified_diff", "patch"}:
        return "diff"
    return default_output_format


def _prompt_layout(request: GenerateRequest) -> str:
    metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
    raw = _safe_str(metadata.get("prompt_layout"))
    if raw.lower() == "issue_only":
        return "issue_only"
    return "sectioned"


def _instruction_for_request(request: GenerateRequest, *, default_output_format: str, force_diff: bool) -> str:
    fmt = _requested_output_format(request, default_output_format, force_diff)
    contract = DIFF_INSTRUCTION if fmt == "diff" else JSON_INSTRUCTION
    target_paths = _editable_target_paths(request)
    if target_paths:
        contract += "\nEditable target files: " + ", ".join(target_paths[:8]) + "."
    if force_diff:
        return contract
    if isinstance(request.prompt, str) and request.prompt.strip():
        return request.prompt.strip()
    return contract


def _editable_target_paths(request: GenerateRequest) -> list[str]:
    paths: list[str] = []
    if isinstance(request.plan, Mapping):
        targets = request.plan.get("targets")
        if isinstance(targets, Sequence) and not isinstance(targets, str):
            for target in targets:
                if not isinstance(target, Mapping):
                    continue
                path = _safe_str(target.get("path"))
                if path and path not in paths:
                    paths.append(path)
    if not paths and isinstance(request.snippets, Sequence):
        for snippet in request.snippets:
            if not isinstance(snippet, Mapping):
                continue
            path = _safe_str(snippet.get("path") or snippet.get("abs_path"))
            if path and path not in paths:
                paths.append(path)
    return paths


def _patch_outside_target_paths(patch: Mapping[str, Any], allowed_paths: Sequence[str]) -> list[str]:
    allowed = {str(path).strip() for path in allowed_paths if str(path).strip()}
    if not allowed:
        return []
    edits = patch.get("edits")
    if not isinstance(edits, Sequence) or isinstance(edits, str):
        return []
    outside: list[str] = []
    for edit in edits:
        if not isinstance(edit, Mapping):
            continue
        path = _safe_str(edit.get("path"))
        if path and path not in allowed and path not in outside:
            outside.append(path)
    return outside


@dataclass(slots=True)
class LocalCGMRuntime:
    generator: GraphAwareGenerator
    allow_partial: bool = False
    default_output_format: str = "diff"
    force_diff_prompt: bool = False
    last_parse: dict[str, Any] = field(default_factory=dict)

    def generate_patch(self, request: GenerateRequest) -> Patch:
        plan = _coerce_plan(request)
        graph = normalize_graph(
            request.graph if request.graph is not None else {"nodes": list(request.subgraph or []), "edges": []},
            issue=request.issue,
            language=str(request.language or ""),
            repo_name=str(request.repo or ""),
        )
        prompt = _instruction_for_request(request, default_output_format=self.default_output_format, force_diff=self.force_diff_prompt)
        prompt_layout = _prompt_layout(request)
        outputs = self.generator.generate(
            prompt=prompt,
            plan_text=_coerce_plan_text(request, plan),
            issue_text=_issue_text(request.issue),
            snippets=[x for x in list(request.snippets or []) if isinstance(x, Mapping)],
            graph=graph,
            prompt_layout=prompt_layout,
        )
        allowed_paths = _editable_target_paths(request)
        for idx, candidate in enumerate(outputs):
            parsed = parse_model_output(candidate, allow_partial=self.allow_partial)
            if parsed is None:
                continue
            outside = _patch_outside_target_paths(parsed.patch, allowed_paths)
            if outside:
                self.last_parse = {
                    "parser": parsed.parser,
                    "candidate_index": idx,
                    "rejected_reason": "patch touched non-target files",
                    "non_target_paths": outside,
                    "raw_preview": parsed.raw_preview,
                }
                continue
            self.last_parse = {"parser": parsed.parser, "candidate_index": idx, "raw_preview": parsed.raw_preview}
            return parsed.patch
        preview = _preview(outputs[0] if outputs else "")
        self.last_parse = {"parser": "none", "raw_preview": preview}
        raise RuntimeError(f"CGM output cannot be parsed as complete unified diff or JSON patch. first_output={preview}")


@dataclass(slots=True)
class RuntimeBundle:
    runtime: LocalCGMRuntime
    lock: asyncio.Lock
    runtime_mode: str = "graph"


def _apply_model_config(bundle: RuntimeBundle, model_config: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(model_config, Mapping):
        return {}
    cfg = bundle.runtime.generator.config
    original: dict[str, Any] = {}
    if "temperature" in model_config:
        original["temperature"] = cfg.temperature
        original["do_sample"] = cfg.do_sample
        cfg.temperature = float(model_config["temperature"])
        cfg.do_sample = cfg.temperature > 0
    if "top_p" in model_config:
        original["top_p"] = cfg.top_p
        cfg.top_p = float(model_config["top_p"])
    if "max_tokens" in model_config:
        original["max_new_tokens"] = cfg.max_new_tokens
        cfg.max_new_tokens = int(model_config["max_tokens"])
    if "num_return_sequences" in model_config:
        original["num_return_sequences"] = cfg.num_return_sequences
        cfg.num_return_sequences = int(model_config["num_return_sequences"])
    return original


def _restore_model_config(bundle: RuntimeBundle, snapshot: Mapping[str, Any]) -> None:
    cfg = bundle.runtime.generator.config
    for key, value in snapshot.items():
        setattr(cfg, key, value)


def create_app(bundle: RuntimeBundle, *, route: str = "/generate") -> FastAPI:
    app = FastAPI()

    @app.get("/healthz")
    async def healthcheck() -> dict[str, Any]:
        return {"ok": True, "runtime_mode": bundle.runtime_mode, "profile": bundle.runtime.generator.profile.to_dict()}

    @app.post(route)
    async def generate(request: GenerateRequest) -> JSONResponse:
        merged_overrides: dict[str, Any] = dict(request.model_overrides or {})
        if isinstance(request.generation_config, Mapping):
            merged_overrides.update(dict(request.generation_config))
        async with bundle.lock:
            overrides = _apply_model_config(bundle, merged_overrides)
            try:
                patch = bundle.runtime.generate_patch(request)
            except Exception as exc:
                LOGGER.exception("CGM /generate failure")
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            finally:
                _restore_model_config(bundle, overrides)
        response: dict[str, Any] = {
            "patch": patch,
            "summary": patch.get("summary"),
            "runtime_mode": bundle.runtime_mode,
            "metadata": {
                "runtime_profile": bundle.runtime.generator.profile.to_dict(),
                "parse": dict(bundle.runtime.last_parse),
                "partial_fallback_enabled": bundle.runtime.allow_partial,
            },
        }
        constraints = _extract_constraints(request.metadata)
        if constraints:
            response["metadata"]["constraints"] = dict(constraints)
        return JSONResponse(response)

    return app


def _build_runtime(args: argparse.Namespace) -> LocalCGMRuntime:
    default_output = "diff" if args.output_format in {"diff", "unified_diff"} else "json"
    use_chat_template = not bool(args.raw_prompt)
    system_prompt = "" if args.raw_prompt else (DIFF_SYSTEM_PROMPT if default_output == "diff" or args.zen else JSON_SYSTEM_PROMPT)
    max_new_tokens = int(args.max_new_tokens)
    if args.zen and not args.no_zen_token_cap:
        cap = int(args.zen_max_new_tokens_cap)
        if cap > 0 and max_new_tokens > cap:
            LOGGER.info("Capping --max-new-tokens from %s to %s for --zen diff mode", max_new_tokens, cap)
            max_new_tokens = cap
    config = GenerationConfig(
        model_name_or_path=args.model,
        tokenizer_name_or_path=args.tokenizer,
        max_length=int(args.max_input_tokens),
        max_new_tokens=max_new_tokens,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        do_sample=float(args.temperature) > 0,
        num_return_sequences=int(args.num_return_sequences),
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.dtype,
        load_in_8bit=bool(args.load_in_8bit),
        load_in_4bit=bool(args.load_in_4bit),
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=args.attn_implementation,
    )
    generator = GraphAwareGenerator(
        config,
        encoder_path=args.encoder_path,
        adapter_path=args.adapter_path,
        adapter_hidden_dim=int(args.adapter_hidden_dim),
        embedding_dim=int(args.embedding_dim),
        use_adj=not bool(args.disable_adj),
        strict_adj=bool(args.strict_adj),
        system_prompt=system_prompt,
        use_chat_template=use_chat_template,
    )
    return LocalCGMRuntime(
        generator=generator,
        allow_partial=bool(args.allow_partial),
        default_output_format=default_output,
        force_diff_prompt=bool(args.zen),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the graphplanner_agent CodeFuse-CGM service")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--route", default="/generate")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--encoder-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--adapter-hidden-dim", type=int, default=4096)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--disable-adj", action="store_true")
    parser.add_argument("--strict-adj", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="Enable legacy partial regex salvage. Off by default.")
    parser.add_argument("--output-format", choices=["diff", "unified_diff", "json"], default="diff")
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Use official use_chat=false style: raw prompt completion without tokenizer chat template.",
    )
    parser.add_argument("--zen", action="store_true", help="Force the service-level diff-native instruction contract.")
    parser.add_argument("--zen-max-new-tokens-cap", type=int, default=512)
    parser.add_argument("--no-zen-token-cap", action="store_true")
    parser.add_argument("--log-level", default="info")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    runtime = _build_runtime(args)
    app = create_app(RuntimeBundle(runtime=runtime, lock=asyncio.Lock()), route=args.route)
    uvicorn.run(app, host=args.host, port=args.port, log_level=str(args.log_level).lower())


if __name__ == "__main__":
    main()
