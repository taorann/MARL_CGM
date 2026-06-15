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
    "repair_propose",
    "repair_revise",
    "repair_submit",
    "discard_pending_patch",
    "repair_chunk",
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
                "Find implementation graph nodes or scoped implementation files. Use public find_type values only; assignment covers module-level assignments. "
                "Use path_glob once a relevant file/package is known, so broad terms do not drift. "
                "For file discovery or sibling context, query may be empty when path_glob is provided, e.g. path_glob=src/utils/*.ts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Implementation symbol/text query. May be empty only when path_glob scopes a file listing.",
                    },
                    "find_type": {"type": "string", "enum": ["file", "class", "function", "method", "assignment", "any"]},
                    "class_name": {"type": "string"},
                    "path_glob": {
                        "type": "string",
                        "description": "Optional implementation path scope, e.g. astropy/io/ascii/*.py or astropy/io/ascii/core.py.",
                    },
                },
                "required": ["find_type"],
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
            "name": "repair_propose",
            "description": (
                "Ask CGM to generate a candidate patch and store it as pending after patch validation/syntax checks. "
                "Does not run fail-to-pass/PASS_TO_PASS tests; planner must inspect and then submit, revise, or discard."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "failure_seen": {"type": "string"},
                    "evidence_chain": {
                        "type": "array",
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
                        "description": "Mechanism analysis for one candidate patch. Do not include exact patch text.",
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["failure_seen", "evidence_chain", "target_nodes", "intent_analysis", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repair_revise",
            "description": (
                "Ask CGM to revise the current pending patch using planner's patch review, history, and the same evidence package. "
                "Stores a new pending patch; does not run fail-to-pass/PASS_TO_PASS tests."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "failure_seen": {"type": "string"},
                    "evidence_chain": {
                        "type": "array",
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
                    "intent_analysis": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "revision_focus": {
                        "type": "string",
                        "description": "What CGM should fix in the pending patch; mention concrete risks or uncovered mechanism.",
                    },
                    "pending_patch_review": {
                        "type": "object",
                        "description": "Planner review of the pending patch: coverage, risks, and requested_change.",
                        "properties": {
                            "coverage": {"type": "string"},
                            "risks": {"type": "array", "items": {"type": "string"}},
                            "requested_change": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                },
                "required": [
                    "failure_seen",
                    "evidence_chain",
                    "target_nodes",
                    "intent_analysis",
                    "confidence",
                    "revision_focus",
                    "pending_patch_review",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repair_submit",
            "description": "Submit the current pending patch for official fail-to-pass and explicit PASS_TO_PASS verification.",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "description": "Why the pending patch is ready to test, including any accepted risks.",
                    }
                },
                "required": ["decision"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "discard_pending_patch",
            "description": "Discard the current pending patch when planner decides it is wrong, too risky, or stale.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repair_chunk",
            "description": (
                "Ask CGM for one small, coherent patch chunk and keep it applied if it passes patch validation and syntax checks. "
                "Use only for multi-site repairs where a single complete patch is too large; finish with repair for fail-to-pass verification."
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
                        "description": (
                            "One to five evidence nodes supporting this chunk. Use failure_seen for runtime/test behavior; "
                            "node_id must be a read implementation code node id or an explicit new_file:relative/path target. "
                            "Existing-file target_nodes entries must also appear here."
                        ),
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
                    "target_nodes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Committed code node ids to edit in this chunk, or new_file:relative/path.",
                    },
                    "intent_analysis": {
                        "type": "string",
                        "description": (
                            "Chunk intent, not exact patch text. Explain this one coherent sub-change and why it can be safely kept applied "
                            "before the final fail-to-pass verification."
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Planner self-assessed confidence in this chunk.",
                    },
                    "remaining_work": {
                        "type": "string",
                        "description": "Short description of what later chunks or final repair still need to verify or change.",
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
                        "description": (
                            "One to five evidence nodes supporting the repair intent. Use failure_seen for runtime/test behavior; "
                            "do not invent pseudo node ids like test_behavior. node_id must be a read implementation code node id "
                            "or an explicit new_file:relative/path target. Existing-file target_nodes entries must also appear here; "
                            "new_file targets are already explicit in target_nodes and may appear here when useful."
                        ),
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
                    "target_nodes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Committed code node ids to edit, or new_file:relative/path for an issue-required implementation file "
                            "that does not exist yet. New file paths must be implementation paths, never tests."
                        ),
                    },
                    "intent_analysis": {
                        "type": "string",
                        "description": (
                            "Intent analysis, not a patch instruction. Explain the deeper implementation mechanism "
                            "behind the issue behavior, the violated local invariant or issue-required behavior, and "
                            "why target_nodes are the patch locus. For new_file targets, explain why the file is absent but required by the issue/interface. "
                            "Do not prescribe exact replacement text, JSON patch, or diff. "
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
                        "description": (
                            "One to five evidence nodes supporting the repair intent. Use failure_seen for runtime/test behavior; "
                            "do not invent pseudo node ids like test_behavior. node_id must be a read implementation code node id "
                            "or an explicit new_file:relative/path target. Existing-file target_nodes entries must also appear here; "
                            "new_file targets are already explicit in target_nodes and may appear here when useful."
                        ),
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
                    "target_nodes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Committed code node ids to edit, or new_file:relative/path for an issue-required implementation file that does not exist yet.",
                    },
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
