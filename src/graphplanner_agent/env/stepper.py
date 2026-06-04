from __future__ import annotations

from dataclasses import dataclass, field
import json

from graphplanner_agent.config import AgentConfig
from graphplanner_agent.datasets import TaskSpec
from graphplanner_agent.env.action_handlers import handle_action
from graphplanner_agent.env.guards import validate_action
from graphplanner_agent.env.observations import build_observation
from graphplanner_agent.graph.schema import RepoGraph
from graphplanner_agent.memory import CgmMemory, TextNotes, WorkingMemory
from graphplanner_agent.planner.protocol import PLANNER_TOOL_SCHEMAS, PlannerAction
from graphplanner_agent.repair.cgm_client import CgmClient
from graphplanner_agent.repair.retry_policy import RepairHistory
from graphplanner_agent.runtime.sandbox_base import SandboxRuntime


@dataclass(slots=True)
class CodeRepairEnv:
    task: TaskSpec
    runtime: SandboxRuntime
    cgm: CgmClient
    config: AgentConfig
    graph: RepoGraph
    working: WorkingMemory = field(default_factory=WorkingMemory)
    memory: CgmMemory = field(default_factory=CgmMemory)
    notes: TextNotes = field(default_factory=TextNotes)
    repair_history: RepairHistory = field(default_factory=RepairHistory)
    failure_summary: dict[str, object] | None = None
    latest_result: dict[str, object] | None = None
    repair_feedback: str | None = None
    last_repair_attempt: dict[str, object] | None = None
    last_repair_review: dict[str, object] | None = None
    trajectory: list[dict[str, object]] = field(default_factory=list)
    planner_diagnostics: list[dict[str, object]] = field(default_factory=list)
    recent_actions: list[str] = field(default_factory=list)
    recent_action_signatures: list[str] = field(default_factory=list)
    action_counts: dict[str, int] = field(default_factory=dict)
    step_count: int = 0
    verified: bool = False
    done: bool = False
    status: str = "not_pass"

    @classmethod
    def create(cls, task: TaskSpec, runtime: SandboxRuntime, cgm: CgmClient, config: AgentConfig) -> "CodeRepairEnv":
        runtime.start(task)
        graph = runtime.build_graph()
        return cls(task=task, runtime=runtime, cgm=cgm, config=config, graph=graph)

    def observe(self) -> str:
        return build_observation(
            self.task,
            self.working,
            self.memory,
            self.notes,
            self.latest_result,
            self.failure_summary,
            self.repair_feedback,
            self.last_repair_attempt,
            self.last_repair_review,
            self.trajectory,
            self.planner_diagnostics,
            self.recent_actions,
            self.recent_action_signatures,
            len(self.graph.nodes),
            len(self.graph.edges),
            self.config.sandbox_backend,
            self.verified,
            repair_disabled_reason=self.repair_disabled_reason(),
            observation_mode=self.config.observation_mode,
        )

    def repair_disabled_reason(self) -> str | None:
        if self.config.require_failed_test_before_repair and self.failure_summary is None:
            return "repair is temporarily disabled until fail-to-pass behavior is collected with run_failed_test"
        if not self.memory.nodes:
            return "repair is temporarily disabled until hydrated implementation code is committed to repair_memory_M"
        missing = [node.id for node in self.memory.nodes.values() if not node.has_code]
        if missing:
            return f"repair is temporarily disabled until memory nodes have code bodies: {missing}"
        if self.repair_history.failed_with_same_memory(list(self.memory.nodes)) and not self._last_review_ready_for_current_memory():
            return "repair is temporarily disabled because the previous repair failed and repair_memory_M has not changed"
        return None

    def _last_review_ready_for_current_memory(self) -> bool:
        state = self.last_repair_review
        if not isinstance(state, dict):
            return False
        review = state.get("review")
        signature = state.get("package_signature")
        if not isinstance(review, dict) or not isinstance(signature, dict):
            return False
        if str(review.get("verdict") or "") != "ready":
            return False
        memory_ids = signature.get("memory_node_ids")
        return memory_ids == sorted(str(node_id) for node_id in self.memory.nodes)

    def action_disabled_reason(self, tool: str) -> str | None:
        if tool == "repair":
            return self.repair_disabled_reason()
        if tool == "repair_review":
            if self.config.require_failed_test_before_repair and self.failure_summary is None:
                return "repair_review is temporarily disabled until fail-to-pass behavior is collected with run_failed_test"
            if not self.memory.nodes:
                return "repair_review is temporarily disabled until hydrated implementation code is committed to repair_memory_M"
            missing = [node.id for node in self.memory.nodes.values() if not node.has_code]
            if missing:
                return f"repair_review is temporarily disabled until memory nodes have code bodies: {missing}"
        if tool == "explore_find" and self.repair_disabled_reason():
            latest_results = self.latest_result.get("results") if isinstance(self.latest_result, dict) else None
            if self.latest_result and self.latest_result.get("tool") == "explore_find" and latest_results:
                return (
                    "explore_find is temporarily disabled because the latest search returned candidates; "
                    "read a candidate, use grep_code with a scoped path_glob, commit a code-bearing read node, or expand from a candidate before searching again"
                )
        return None

    def planner_tool_schemas(self) -> list[dict]:
        disabled_tools = {
            schema.get("function", {}).get("name")
            for schema in PLANNER_TOOL_SCHEMAS
            if self.action_disabled_reason(str(schema.get("function", {}).get("name") or ""))
        }
        if not disabled_tools:
            return PLANNER_TOOL_SCHEMAS
        return [
            schema
            for schema in PLANNER_TOOL_SCHEMAS
            if schema.get("function", {}).get("name") not in disabled_tools
        ]

    def step(self, action: PlannerAction) -> dict[str, object]:
        self.step_count += 1
        self.recent_actions.append(action.tool)
        signature = json.dumps({"tool": action.tool, "params": action.params}, sort_keys=True)
        self.recent_action_signatures.append(signature)
        self.action_counts[signature] = self.action_counts.get(signature, 0) + 1
        if self.action_counts[signature] > self.config.max_repeated_action:
            reason = f"repeated identical action exceeded limit {self.config.max_repeated_action}; choose a different implementation node or commit/repair path"
            if _has_read_code_outside_memory(self):
                reason += "; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair"
            self.latest_result = {
                "tool": action.tool,
                "blocked": True,
                "reason": reason,
            }
            self._record_trajectory(action, self.latest_result)
            return self.latest_result
        blocked = validate_action(action)
        if blocked:
            self.latest_result = {"tool": action.tool, "blocked": True, "reason": blocked}
            self._record_trajectory(action, self.latest_result)
            return self.latest_result
        try:
            self.latest_result = handle_action(self, action)
        except Exception as exc:
            self.latest_result = {"tool": action.tool, "error": type(exc).__name__, "reason": str(exc)}
            if _is_remote_sandbox_lost_error(exc):
                self.latest_result.update(
                    {
                        "status": "infra_bug",
                        "done": True,
                        "error_origin": "remote_sandbox_lost",
                        "retryable": True,
                    }
                )
                self.done = True
                self.status = "bug"
        self._record_trajectory(action, self.latest_result)
        return self.latest_result

    def _record_trajectory(self, action: PlannerAction, result: dict[str, object]) -> None:
        record = {
            "step": self.step_count,
            "tool": action.tool,
            "params": _compact_action_params(action.tool, action.params),
            "status": _result_status(result),
            "summary": _result_summary(result),
        }
        reason = result.get("reason")
        if reason:
            record["reason"] = _compact_text(str(reason), 600)
        self.trajectory.append(record)

    def record_planner_diagnostic(self, diagnostic: dict[str, object]) -> None:
        self.planner_diagnostics.append(_compact_value(diagnostic, 1800))
        del self.planner_diagnostics[:-12]


def _result_status(result: dict[str, object]) -> str:
    if result.get("blocked"):
        return "blocked"
    if result.get("error"):
        return "error"
    status = result.get("status")
    if isinstance(status, str) and status:
        return status
    test = result.get("test")
    if isinstance(test, dict) and test.get("status"):
        return str(test["status"])
    return "ok"


def _is_remote_sandbox_lost_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "no active instance on this runner" in text


def _result_summary(result: dict[str, object]) -> dict[str, object]:
    tool = str(result.get("tool") or "")
    if tool == "run_failed_test":
        test = result.get("test") if isinstance(result.get("test"), dict) else {}
        return {
            "test_status": test.get("status"),
            "returncode": test.get("returncode"),
            "resolved": test.get("resolved"),
        }
    if tool == "explore_find":
        return {"result_count": len(result.get("results") or []), "warning": result.get("warning"), "path_glob": result.get("path_glob")}
    if tool == "grep_code":
        return {"hit_count": len(result.get("hits") or []), "path_glob": result.get("path_glob"), "pattern": result.get("pattern")}
    if tool == "explore_expand":
        return {"result_count": len(result.get("results") or []), "expand_mode": result.get("expand_mode"), "symbol": result.get("symbol")}
    if tool == "read":
        node = result.get("node") if isinstance(result.get("node"), dict) else {}
        return {"node": node, "blocked": result.get("blocked")}
    if tool == "memory_commit":
        return {
            "committed": result.get("committed"),
            "newly_added_ids": result.get("newly_added_ids"),
            "already_present_ids": result.get("already_present_ids"),
            "dropped_by_keep_ids": result.get("dropped_by_keep_ids"),
            "memory_changed": result.get("memory_changed"),
            "memory_count": len(result.get("memory") or []),
        }
    if tool == "memory_delete":
        return {
            "deleted_ids": result.get("deleted_ids"),
            "memory_changed": result.get("memory_changed"),
            "memory_count": len(result.get("memory") or []),
        }
    if tool == "repair":
        return {
            "status": result.get("status"),
            "blocked": result.get("blocked"),
            "rolled_back": result.get("rolled_back"),
            "touched_paths": result.get("touched_paths"),
            "error_origin": result.get("error_origin"),
        }
    if tool == "repair_review":
        review = result.get("review") if isinstance(result.get("review"), dict) else {}
        return {
            "status": result.get("status"),
            "blocked": result.get("blocked"),
            "verdict": review.get("verdict"),
            "confidence": review.get("confidence"),
            "suggested_next_action": review.get("suggested_next_action"),
            "error_origin": result.get("error_origin"),
        }
    return _compact_value(result, 800)


def _compact_value(value, limit: int):
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return _compact_text(str(value), limit)
    if len(text) <= limit:
        return value
    return _compact_text(text, limit)


def _compact_action_params(tool: str, params: dict[str, object]) -> object:
    if tool not in {"repair", "repair_review"}:
        return _compact_value(params, 700)
    compact: dict[str, object] = {}
    for key in ["failure_seen", "intent_analysis", "confidence"]:
        value = params.get(key)
        if value is not None:
            compact[key] = _compact_text(str(value), 220)
    for key in ["target_nodes"]:
        value = params.get(key)
        if isinstance(value, list):
            compact[key] = [str(item) for item in value[:8]]
    evidence_chain = params.get("evidence_chain")
    if isinstance(evidence_chain, list):
        compact["evidence_chain"] = [
            {
                "node_id": str(item.get("node_id") or ""),
                "role": str(item.get("role") or ""),
                "evidence": _compact_text(str(item.get("evidence") or ""), 180),
            }
            for item in evidence_chain[:6]
            if isinstance(item, dict)
        ]
    return compact


def _compact_text(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"...<truncated {len(text) - limit} chars>"


def _has_read_code_outside_memory(env: CodeRepairEnv) -> bool:
    committed = set(env.memory.nodes)
    return any(node_id not in committed and entry.node.has_code for node_id, entry in env.working.entries.items())
