from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_TOOLS = {
    "run_failed_test",
    "explore_find",
    "grep_code",
    "explore_expand",
    "read",
    "memory_commit",
    "memory_delete",
    "memory_commit_note",
    "repair_review",
    "repair",
}


PLANNER_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_failed_test",
            "description": "Run trusted fail-to-pass tests and collect behavior evidence without exposing test source.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explore_find",
            "description": (
                "Find implementation graph nodes by query. Use public find_type values only; assignment covers module-level assignments. "
                "Use path_glob once a relevant file/package is known, so broad terms like formats do not drift to unrelated modules."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "find_type": {"type": "string", "enum": ["file", "class", "function", "method", "assignment", "any"]},
                    "class_name": {"type": "string"},
                    "path_glob": {
                        "type": "string",
                        "description": "Optional implementation path scope, e.g. astropy/io/ascii/*.py or astropy/io/ascii/core.py.",
                    },
                },
                "required": ["query", "find_type"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_code",
            "description": (
                "Search for exact implementation text within a scoped file or package. "
                "Use this after localization to find where a parameter/helper/string is actually used; then read the covering_node."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path_glob": {
                        "type": "string",
                        "description": "Required implementation path scope, e.g. astropy/io/ascii/core.py or astropy/io/ascii/*.py.",
                    },
                    "context_lines": {"type": "integer", "minimum": 0, "maximum": 20, "default": 2},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
                    "regex": {"type": "boolean", "default": False},
                },
                "required": ["pattern", "path_glob"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explore_expand",
            "description": (
                "Expand relations from an existing node. Use mechanism to lazily expose parent/base, override, "
                "composition, and pipeline candidates with code previews; use owner_flow with symbol after missing-attribute "
                "or wrong-owner patch failures."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "anchor": {"type": "string"},
                    "expand_mode": {
                        "type": "string",
                        "enum": ["callers", "callees", "siblings", "imports", "contains", "uses", "related", "mechanism", "owner_flow"],
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Required for owner_flow: missing attribute/parameter name such as formats.",
                    },
                },
                "required": ["anchor", "expand_mode"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read implementation code for a graph node by node_id. If view is omitted, the runtime uses body.",
            "parameters": {
                "type": "object",
                "properties": {"node_id": {"type": "string"}, "view": {"type": "string", "default": "body"}},
                "required": ["node_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_commit",
            "description": "Explicitly commit model-selected implementation evidence to curated CGM memory. Requires a prior read of each selected node; explore_find previews cannot be committed directly. Related nodes are not auto-added.",
            "parameters": {
                "type": "object",
                "properties": {
                    "select_ids": {"type": "array", "items": {"type": "string"}},
                    "keep_ids": {"type": "array", "items": {"type": "string"}},
                    "note": {"type": "string"},
                    "tag": {"type": "string"},
                },
                "required": ["select_ids"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_delete",
            "description": "Remove stale evidence from CGM memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "delete_ids": {"type": "array", "items": {"type": "string"}},
                    "keep_ids": {"type": "array", "items": {"type": "string"}},
                    "note": {"type": "string"},
                    "tag": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_commit_note",
            "description": "Write a planner-only note.",
            "parameters": {
                "type": "object",
                "properties": {"note": {"type": "string"}, "tag": {"type": "string"}},
                "required": ["note"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repair",
            "description": "Call CGM only after submitting a compact grounded evidence package from committed repair memory M.",
            "parameters": {
                "type": "object",
                "properties": {
                    "failure_seen": {
                        "type": "string",
                        "description": "Actual failure observed in the issue or runtime output. Do not include guessed causes.",
                    },
                    "evidence_chain": {
                        "type": "array",
                        "description": "Two to five read implementation nodes that support the repair intent: entry/state/decision/output/target.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "node_id": {"type": "string"},
                                "role": {"type": "string"},
                                "evidence": {"type": "string"},
                            },
                            "required": ["node_id", "role"],
                            "additionalProperties": False,
                        },
                    },
                    "target_nodes": {"type": "array", "items": {"type": "string"}},
                    "intent_analysis": {
                        "type": "string",
                        "description": (
                            "Intent analysis, not a patch instruction. Explain the deeper implementation mechanism "
                            "behind the issue behavior, the violated local invariant or issue-required behavior, and "
                            "why target_nodes are the patch locus. Do not prescribe exact replacement text, JSON patch, or diff. "
                            "If the latest repair_review returned ready advice for this same evidence package, "
                            "calling repair means you choose to adopt that critique."
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": (
                            "Planner self-assessed confidence in target_nodes plus intent_analysis, from 0 to 1. "
                            "Use high confidence only when localization and patch intent are both supported by read code; "
                            "use lower confidence when localization is plausible but exact behavior/message/API is uncertain."
                        ),
                    },
                },
                "required": [
                    "failure_seen",
                    "evidence_chain",
                    "target_nodes",
                    "intent_analysis",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repair_review",
            "description": (
                "Ask CGM to critique the proposed repair intent and evidence without generating or applying a patch. "
                "Use this after failed patches or when target confidence is uncertain."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "failure_seen": {
                        "type": "string",
                        "description": "Actual failure observed in the issue or runtime output. Do not include guessed causes.",
                    },
                    "evidence_chain": {
                        "type": "array",
                        "description": "Two to five read implementation nodes that support the repair intent: entry/state/decision/output/target.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "node_id": {"type": "string"},
                                "role": {"type": "string"},
                                "evidence": {"type": "string"},
                            },
                            "required": ["node_id", "role"],
                            "additionalProperties": False,
                        },
                    },
                    "target_nodes": {"type": "array", "items": {"type": "string"}},
                    "intent_analysis": {
                        "type": "string",
                        "description": (
                            "Intent to review, not exact patch text. Explain the mechanism, violated invariant, "
                            "and why target_nodes are the suspected patch locus."
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Planner confidence in this intent before CGM critique.",
                    },
                    "review_focus": {
                        "type": "string",
                        "description": (
                            "Optional. Use when a previous repair_review exists and you disagree with it, found counter-evidence, "
                            "or need CGM to review the same single plan more deeply."
                        ),
                    },
                },
                "required": [
                    "failure_seen",
                    "evidence_chain",
                    "target_nodes",
                    "intent_analysis",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass(slots=True)
class PlannerAction:
    tool: str
    params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.tool not in VALID_TOOLS:
            raise ValueError(f"unknown planner tool: {self.tool}")
        if not isinstance(self.params, dict):
            raise ValueError("planner action params must be an object")

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "PlannerAction":
        action = cls(tool=str(obj.get("tool", "")), params=dict(obj.get("params") or {}))
        action.validate()
        return action
