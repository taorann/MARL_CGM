# graph_planner/env/planner_env.py
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from types import SimpleNamespace

DEBUG = bool(os.environ.get("DEBUG"))


def _dbg(msg: str) -> None:
    if DEBUG:
        print(f"[planner_env] {msg}")


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y"}
    try:
        return bool(x)
    except Exception:
        return default


def _safe_str(x: Any, default: str = "") -> str:
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return default


def _first_seq(seq: Iterable[Any]) -> Any:
    for x in seq:
        return x
    return None


def _norm_path(path: str) -> str:
    return path.replace("\\", "/") if isinstance(path, str) else ""


def _encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _decode_bytes(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


def _read_file_text(path: str, encoding: str = "utf-8") -> str:
    try:
        with open(path, "r", encoding=encoding) as f:
            return f.read()
    except Exception:
        return ""


def _write_file_text(path: str, text: str, encoding: str = "utf-8") -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding=encoding) as f:
            f.write(text)
    except Exception:
        pass


def _ensure_dir(path: str) -> None:
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Imports that may be heavy / optional
# ---------------------------------------------------------------------------

try:
    from copy import deepcopy
except Exception:
    def deepcopy(x):
        return x

try:
    from ..core.actions import (
        ActionUnion,
        ExploreAction,
        MemoryAction,
        NoopAction,
        RepairAction,
        SubmitAction,
    )
    from ..infra.config import Config, load as load_config
    try:
        from ..integrations.codefuse_cgm.formatting import GraphLinearizer, SnippetFormatter
    except Exception:
        # Lightweight dummies

        class GraphLinearizer:
            def linearize(self, payload):
                return "" if payload is None else str(payload)

        class SnippetFormatter:
            def format(self, snippets):
                return "" if not snippets else str(snippets)
    from ..memory import graph_adapter, mem_candidates, subgraph_store, text_memory
    from ..memory.subgraph_store import WorkingSubgraph
    from ..runtime.sandbox import SandboxConfig, SandboxRuntime
    from aci.schema import Plan, PlanTarget
    from aci.guard import GuardError, enforce_patch_guard
    from actor.collater import collate
    from actor import cgm_adapter
    from ..agents.rule_based.test_prioritizer import prioritize_tests
except Exception as _IMPORT_ERROR:  # pragma: no cover
    # The environment may not have all dependencies; this file is imported
    # in tooling contexts where only type information is needed.
    ActionUnion = object  # type: ignore[assignment]
    ExploreAction = MemoryAction = RepairAction = SubmitAction = NoopAction = object  # type: ignore[assignment]
    Config = object  # type: ignore[assignment]
    SandboxConfig = object  # type: ignore[assignment]
    SandboxRuntime = object  # type: ignore[assignment]
    Plan = PlanTarget = object  # type: ignore[assignment]
    subgraph_store = text_memory = graph_adapter = mem_candidates = object()  # type: ignore[assignment]


DEFAULT_MEMORY_CAPS = {
    "nodes": 200,
    "edges": 1000,
    "frontier": 50,
    "planner_tokens": 2000,
    "cgm_tokens": 16000,
}


def _get_io_config(cfg: Config) -> Mapping[str, Any]:
    try:
        cfg_dict = cfg.to_dict()
    except Exception:
        cfg_dict = {}
    io_cfg = cfg_dict.get("io") if isinstance(cfg_dict, Mapping) else {}
    if not isinstance(io_cfg, Mapping):
        io_cfg = {}
    return io_cfg


# ---------------------------------------------------------------------------
# PlannerEnv
# ---------------------------------------------------------------------------


class PlannerEnv:
    """Graph Planner 环境封装。

    负责：
      * 调用 SandboxRuntime 与容器交互；
      * 维护工作子图与记忆日志；
      * 将 Explore/Memory/Repair/Submit 动作映射为具体容器操作；
      * 将状态打包为 Observation 供上层 Agent 使用。
    """

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlannerEnv":
        cfg = SandboxConfig(**payload["sandbox"])
        return cls(issue=payload.get("issue", {}), sandbox_cfg=cfg)

    def __init__(self, issue: Dict[str, Any], sandbox_cfg: SandboxConfig):
        self.issue: Dict[str, Any] = issue or {}
        self.issue_id: str = str(self.issue.get("id") or "__default__")
        self.sandbox_cfg = sandbox_cfg
        effective_run_id = (
            os.environ.get("GRAPH_PLANNER_RUN_ID", "")
            or issue.get("run_id", "")
            or str(self.issue.get("id") or "__default__")
        )
        self.box = SandboxRuntime(sandbox_cfg, run_id=effective_run_id)
        self.config: Config = load_config()
        self.config_dict: Dict[str, Any] = self.config.to_dict()
        io_cfg = self.config_dict.get("io") if isinstance(self.config_dict, Mapping) else {}
        if not isinstance(io_cfg, Mapping):
            io_cfg = {}
        strict_env = os.environ.get("GRAPH_PLANNER_STRICT_IO") or os.environ.get("STRICT_PLANNER_IO")
        if strict_env is not None:
            self._strict_io = str(strict_env).strip().lower() in {"1", "true", "yes"}
        else:
            self._strict_io = bool(io_cfg.get("strict_planner_io", False))

        self.steps: int = 0
        self.last_info: Dict[str, Any] = {}
        self.repo_root_in_container: str = sandbox_cfg.workdir or "."
        # Optional host repo root for per-env graph scanning.
        self.repo_root_host: Optional[str] = getattr(sandbox_cfg, "repo_root_host", None)
        self.run_id: str = os.environ.get("GRAPH_PLANNER_RUN_ID", "") or self.issue.get("run_id", "")

        # 三图结构
        self.repo_graph: Optional[WorkingSubgraph] = None   # 完整仓库图（只读）
        self.memory_subgraph: WorkingSubgraph = subgraph_store.new()   # 长期记忆图
        self.working_subgraph: WorkingSubgraph = subgraph_store.new()  # 当前工作图

        # Repo 图索引
        self._repo_nodes_by_id: Dict[str, Dict[str, Any]] = {}
        self._repo_edges: List[Dict[str, Any]] = []

        # Text memory 相关
        self.memory_graph_store: Optional[text_memory.WorkingGraphStore] = None
        self.memory_text_store: Optional[text_memory.NoteTextStore] = None
        self.memory_state: Optional[text_memory.TurnState] = None

        # 最近一步 explore/read 的结果
        self.last_candidates: List[Dict[str, Any]] = []
        self.last_reads: List[Dict[str, Any]] = []

    # ------------------------------------------------------------
    # 兼容接口：env.subgraph → working_subgraph
    # ------------------------------------------------------------

    @property
    def subgraph(self) -> WorkingSubgraph:
        """Backward-compat alias for the current working_subgraph.

        Historically ``PlannerEnv.subgraph`` referred to the active code subgraph.
        After introducing the three-graph design (repo / working / memory),
        this alias keeps older integrations working by exposing the
        ``working_subgraph`` as the current observation graph.
        """
        return self.working_subgraph

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        self.steps = 0
        self.last_info = {"reset": True}
        backend_mode = getattr(self.box, "_mode", None)

        # === 1) 构建完整仓库图 repo_graph ===
        if backend_mode == "remote_swe" and hasattr(self.box, "build_issue_subgraph"):
            # remote_swe 模式：build_issue_subgraph 被视为完整仓库图
            try:
                repo_json = self.box.build_issue_subgraph(self.issue_id)
                self.repo_graph = subgraph_store.wrap(repo_json)
            except Exception:
                self.repo_graph = subgraph_store.new()
        else:
            # 本地 backend：从 ACI 子图缓存加载（如有需要可触发扫描构图）
            if hasattr(graph_adapter, "set_repo_root") and self.repo_root_host:
                graph_adapter.set_repo_root(self.repo_root_host)
            try:
                graph_adapter.connect()
            except Exception:
                pass
            try:
                self.repo_graph = subgraph_store.load(self.issue_id)
            except Exception:
                self.repo_graph = subgraph_store.new()

        # Repo 索引
        self._repo_nodes_by_id = {}
        self._repo_edges = []
        if self.repo_graph is not None:
            for n in getattr(self.repo_graph, "nodes", []):
                nid = n.get("id")
                if isinstance(nid, str):
                    self._repo_nodes_by_id[nid] = n
            self._repo_edges = list(getattr(self.repo_graph, "edges", []) or [])

        # === 2) 初始化 memory_subgraph（长期记忆） ===
        try:
            self.memory_subgraph = subgraph_store.load(self.issue_id)
        except Exception:
            self.memory_subgraph = subgraph_store.new()

        # === 3) 工作图 = 记忆图的拷贝（起点一致） ===
        self.working_subgraph = subgraph_store.wrap(self.memory_subgraph.to_json_obj())

        # === 4) 初始化 text_memory（基于 memory_subgraph） ===
        self.memory_graph_store = text_memory.WorkingGraphStore(self.memory_subgraph)
        self.memory_text_store = text_memory.NoteTextStore()
        self.memory_state = text_memory.TurnState(
            graph_store=self.memory_graph_store,
            text_store=self.memory_text_store,
        )
        self.memory_state.size = text_memory.Size(
            nodes=len(self.memory_subgraph.nodes),
            edges=len(self.memory_subgraph.edges),
            frontier=0,
            planner_tokens_est=0,
            cgm_tokens_est=0,
        )
        caps = self.config_dict.get("memory_caps") or DEFAULT_MEMORY_CAPS
        self.memory_state.caps = text_memory.Size(
            nodes=int(caps.get("nodes", DEFAULT_MEMORY_CAPS["nodes"])),
            edges=int(caps.get("edges", DEFAULT_MEMORY_CAPS["edges"])),
            frontier=int(caps.get("frontier", DEFAULT_MEMORY_CAPS["frontier"])),
            planner_tokens_est=int(caps.get("planner_tokens", DEFAULT_MEMORY_CAPS["planner_tokens"])),
            cgm_tokens_est=int(caps.get("cgm_tokens", DEFAULT_MEMORY_CAPS["cgm_tokens"])),
        )

        # 在 reset 时记录一次三图状态（如果配置了 GRAPH_PLANNER_TRACE_DIR）
        obs = self._obs()
        try:
            self._log_step_graphs()
        except Exception:
            pass

        return obs

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """单步：接受一个 action 字典，返回 (obs, reward, done, info)。"""
        self.steps += 1
        info: Dict[str, Any] = {}
        validation_meta: Dict[str, Any] = {}
        action_obj: ActionUnion

        # 解析 / 校验 action
        try:
            action_obj = self._parse_action(action)
        except ProtocolError as exc:
            info = {"error": exc.code, "detail": exc.detail}
            obs = self._obs()
            try:
                self._log_step_graphs()
            except Exception:
                pass
            return obs, -0.05, False, info

        # 按类型分发
        if isinstance(action_obj, ExploreAction):
            info = self._handle_explore(action_obj)
        elif isinstance(action_obj, MemoryAction):
            info = self._handle_memory(action_obj)
        elif isinstance(action_obj, RepairAction):
            info = self._handle_repair(action_obj)
        elif isinstance(action_obj, SubmitAction):
            info = self._handle_submit()
        elif isinstance(action_obj, NoopAction):
            info = {"kind": "noop"}
        else:
            info = {"kind": "noop"}

        kind = info.get("kind")
        if kind == "explore":
            self.memory_state.latest_explore = info
        elif kind and kind != "memory":
            self.memory_state.latest_non_explore = info

        self.last_info = info
        reward = self._compute_reward(info)
        done = bool(info.get("done"))

        obs = self._obs()
        # 每一步动作之后记录一次三图状态
        try:
            self._log_step_graphs()
        except Exception:
            pass

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Action 解析与 reward
    # ------------------------------------------------------------------
    def _parse_action(self, payload: Dict[str, Any]) -> ActionUnion:
        if not isinstance(payload, Mapping):
            raise ProtocolError("invalid-payload", "action payload must be a mapping")
        action_type = payload.get("type")
        if action_type == "explore":
            return ExploreAction(**payload)
        if action_type == "memory":
            return MemoryAction(**payload)
        if action_type == "repair":
            return RepairAction(**payload)
        if action_type == "submit":
            return SubmitAction(**payload)
        if action_type == "noop":
            return NoopAction(**payload)
        raise ProtocolError("unknown-action", f"unknown action type: {action_type}")

    def _compute_reward(self, info: Dict[str, Any]) -> float:
        # 简单 reward：成功提交 +1，失败提交 -1，其他略惩罚
        if info.get("kind") == "submit":
            if info.get("success"):
                return 1.0
            return -1.0
        if info.get("error"):
            return -0.1
        return -0.01

    # ------------------------------------------------------------------
    # Explore / Memory
    # ------------------------------------------------------------------
    def _handle_explore(self, act: ExploreAction) -> Dict[str, Any]:
        info: Dict[str, Any] = {"kind": "explore", "op": act.op}
        base_graph = self.repo_graph or self.working_subgraph

        if act.op == "expand":
            max_per_anchor = _safe_int(act.max_per_anchor or 8, 8)
            total_limit = _safe_int(act.total_limit or 32, 32)
            dir_k = _safe_int(act.dir_diversity_k or 4, 4)

            candidates = mem_candidates.build_mem_candidates(
                subgraph=base_graph,
                anchors=act.anchors,
                max_nodes_per_anchor=max_per_anchor,
                total_limit=total_limit,
                dir_diversity_k=dir_k,
            )
            self.last_candidates = candidates
            info["candidates"] = candidates
            candidate_ids = [
                c.get("id") for c in candidates
                if isinstance(c.get("id"), str)
            ]
            self._merge_repo_nodes_into_working(candidate_ids, status="explored")
            info["subgraph_stats"] = subgraph_store.stats(self.working_subgraph)
            return info

        if act.op == "read":
            resolved: List[Dict[str, Any]] = []
            for node_id in act.nodes:
                node = self._resolve_node(node_id)
                if node:
                    resolved.append(node)
            snippets: List[Dict[str, Any]] = []
            for node in resolved[: max(1, int(act.limit or 3))]:
                snippet = self._read_node_snippet(node)
                if snippet:
                    snippets.append(snippet)
            self.last_reads = snippets
            info["snippets"] = snippets
            # 把读取到的节点也 merge 进 working_subgraph
            node_ids = [
                n.get("id") for n in resolved
                if isinstance(n.get("id"), str)
            ]
            self._merge_repo_nodes_into_working(node_ids, status="explored")
            info["subgraph_stats"] = subgraph_store.stats(self.working_subgraph)
            return info

        info["error"] = f"unknown-explore-op:{act.op}"
        return info

    def _handle_memory(self, act: MemoryAction) -> Dict[str, Any]:
        info: Dict[str, Any] = {"kind": "memory"}

        if not self.memory_state:
            info["error"] = "memory-not-initialised"
            return info

        selector = act.selector or {}
        caps = self.memory_state.caps or text_memory.Size(**DEFAULT_MEMORY_CAPS)
        result = text_memory.memory_commit(
            self.memory_state,
            act.target,
            act.scope,
            selector,
            caps,
        )
        info.update(result or {})

        # 从 selector 中解析 node_ids，并写回 memory_subgraph
        node_ids: List[str] = []
        sel_nodes = selector.get("nodes") if isinstance(selector, Mapping) else None
        if isinstance(sel_nodes, str):
            node_ids = [sel_nodes]
        elif isinstance(sel_nodes, Sequence):
            node_ids = [nid for nid in sel_nodes if isinstance(nid, str)]

        if node_ids:
            self._merge_working_nodes_into_memory(node_ids, status="remembered")

        info["subgraph_stats"] = subgraph_store.stats(self.memory_subgraph)
        subgraph_store.save(self.issue_id, self.memory_subgraph)

        # 记忆动作之后，工作图重置为记忆图
        self._reset_working_to_memory()

        return info

    def _reset_working_to_memory(self) -> None:
        """Memory 动作之后，把工作图重置为记忆图。"""
        self.working_subgraph = subgraph_store.wrap(self.memory_subgraph.to_json_obj())

    # ------------------------------------------------------------------
    # Repair / Submit
    # ------------------------------------------------------------------
    def _handle_repair(self, act: RepairAction) -> Dict[str, Any]:
        info: Dict[str, Any] = {"kind": "repair", "apply": act.apply, "plan": act.plan}
        if not act.apply:
            return info

        if act.patch and isinstance(act.patch, dict) and act.patch.get("edits"):
            patch: Dict[str, Any] = dict(act.patch)
            if "summary" not in patch:
                patch["summary"] = act.plan or ""
            plan = self._build_plan(act.plan_targets or [])
            try:
                enforce_patch_guard(patch, plan, self.config)
            except GuardError as ge:
                info["guard_error"] = str(ge)
                info["applied"] = False
                return info

            apply_result = self._apply_patch_edits(patch.get("edits") or [])
            info.update(apply_result)
            info["plan_targets"] = act.plan_targets

            lint_report = self.box.lint()
            tests_report = self.box.test()
            info["lint"] = lint_report
            info["tests"] = tests_report
            info["applied"] = bool(apply_result.get("success"))
            info["priority_tests"] = prioritize_tests(
                self._observation_pack(),
                subgraph=self.working_subgraph,
            ).get("priority_tests", [])
            return info

        return self._repair_with_cgm(act, info)

    def _repair_with_cgm(self, act: RepairAction, info: Dict[str, Any]) -> Dict[str, Any]:
        plan_struct: Plan
        try:
            plan_struct = self._build_plan(act.plan_targets or [])
        except Exception as exc:
            info["applied"] = False
            info["no_repair"] = True
            info["error"] = f"plan-build-failed:{exc}"
            return info

        try:
            collate_cfg = deepcopy(self.config)
        except Exception:
            collate_cfg = SimpleNamespace(mode=getattr(self.config, "mode", "wsd"))

        if not hasattr(collate_cfg, "collate") or collate_cfg.collate is None:
            collate_cfg.collate = SimpleNamespace()  # type: ignore[attr-defined]
        else:
            collate_cfg.collate = deepcopy(collate_cfg.collate)

        collate_cfg.mode = getattr(collate_cfg, "mode", getattr(self.config, "mode", "wsd"))
        collate_cfg.prefer_test_files = getattr(self.config, "prefer_test_files", True)
        collate_cfg.collate.mode = getattr(collate_cfg.collate, "mode", collate_cfg.mode)
        collate_cfg.collate.budget_tokens = min(int(getattr(collate_cfg.collate, "budget_tokens", 40000)), 8000)
        collate_cfg.collate.max_chunks = getattr(collate_cfg.collate, "max_chunks", 64)
        collate_cfg.collate.per_file_max_chunks = getattr(collate_cfg.collate, "per_file_max_chunks", 8)
        base_collate_cfg = getattr(self.config, "collate", SimpleNamespace())
        collate_cfg.collate.enable_light_reorder = getattr(
            collate_cfg.collate,
            "enable_light_reorder",
            getattr(base_collate_cfg, "enable_light_reorder", False),
        )
        collate_cfg.collate.interleave_tests = getattr(
            collate_cfg.collate,
            "interleave_tests",
            getattr(base_collate_cfg, "interleave_tests", True),
        )

        try:
            # ✅ 使用 working_subgraph 作为 collate 的基础图
            chunks, meta = collate(self.working_subgraph, plan_struct, collate_cfg)
        except Exception as exc:
            info["applied"] = False
            info["no_repair"] = True
            info["error"] = f"collate-failed:{exc}"
            return info

        info["collate_meta"] = meta
        constraints = {"max_edits": 3}
        request_collated = {"chunks": chunks, "meta": meta}

        run_id = self.run_id or self.issue_id
        try:
            patch = cgm_adapter.generate(
                collated=request_collated,
                plan=act.plan or "",
                constraints=constraints,
                run_id=run_id,
                issue=self.issue,
            )
        except Exception as exc:
            info["applied"] = False
            info["no_repair"] = True
            info["error"] = f"cgm-error:{exc}"
            return info

        if not isinstance(patch, Mapping):
            info["applied"] = False
            info["no_repair"] = True
            info["error"] = "cgm-invalid-patch"
            return info

        patch_dict: Dict[str, Any] = dict(patch)
        patch_dict.setdefault("summary", act.plan or "")
        info["patch"] = patch_dict

        try:
            enforce_patch_guard(patch_dict, plan_struct, self.config)
        except GuardError as ge:
            info["guard_error"] = str(ge)
            info["applied"] = False
            info["no_repair"] = True
            return info

        apply_result = self._apply_patch_edits(patch_dict.get("edits") or [])
        info.update(apply_result)
        applied = bool(apply_result.get("success"))
        info["applied"] = applied
        info["plan_targets"] = act.plan_targets

        lint_ok = bool(self.box.lint())
        tests_report = self.box.test()
        info["lint_ok"] = lint_ok
        info["tests"] = tests_report
        info["priority_tests"] = prioritize_tests(
            self._observation_pack(),
            subgraph=self.working_subgraph,
        ).get("priority_tests", [])

        return info

    def _handle_submit(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"kind": "submit"}
        try:
            tests_report = self.box.test()
        except Exception as exc:
            info["success"] = False
            info["error"] = f"test-error:{exc}"
            return info

        info["tests"] = tests_report
        info["success"] = bool(tests_report)
        info["done"] = True
        return info

    def _apply_patch_edits(self, edits: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        try:
            result = self.box.apply_patch_edits(list(edits))
        except Exception as exc:
            return {"success": False, "error": f"apply-error:{exc}"}
        return dict(result or {})

    # ------------------------------------------------------------------
    # 辅助函数
    # ------------------------------------------------------------------
    def _obs(self) -> Dict[str, Any]:
        """构造给 LLM 的 observation：仅暴露 working_subgraph。

        - ``subgraph`` / ``subgraph_stats`` 指向当前工作图（working_subgraph）；
        - ``memory_stats`` 提供长期记忆图的规模信息；
        - ``observation_pack`` 为锚点规划 / 规则组件准备的摘要包，基于 memory_subgraph。
        """
        working_stats = subgraph_store.stats(self.working_subgraph)
        working_json = self.working_subgraph.to_json_obj()
        memory_stats = subgraph_store.stats(self.memory_subgraph)
        obs_pack = self._observation_pack(mem_stats=memory_stats, query_stats=None)

        return {
            "issue": self.issue,
            "steps": self.steps,
            "last_info": self.last_info,
            # LLM 看到的是“当前工作图”，里面节点已经带 status/tags（explored/remembered）
            "subgraph": working_json,
            "subgraph_stats": working_stats,
            # 记忆图的统计信息（可选）
            "memory_stats": memory_stats,
            "text_memory": text_memory.snapshot(self.memory_state) if self.memory_state else {},
            "reset": bool(self.last_info.get("reset")),
            "observation_pack": obs_pack,
        }

    def _log_step_graphs(self) -> None:
        """将当前 step 的工作图 & 记忆图落盘，便于回放 trajectory。"""
        try:
            issue_id = (
                self.issue_id
                or self.issue.get("issue_id")
                or self.issue.get("id")
                or "__default__"
            )
            out_dir = os.environ.get("GRAPH_PLANNER_TRACE_DIR")
            if not out_dir:
                return
            _ensure_dir(out_dir)
            step_prefix = f"{issue_id}.step_{self.steps:04d}"
            working_path = os.path.join(out_dir, f"{step_prefix}.working.json")
            memory_path = os.path.join(out_dir, f"{step_prefix}.memory.json")
            _write_file_text(working_path, json.dumps(self.working_subgraph.to_json_obj(), indent=2))
            _write_file_text(memory_path, json.dumps(self.memory_subgraph.to_json_obj(), indent=2))
        except Exception:
            pass

    def _observation_pack(
        self,
        mem_stats: Optional[Dict[str, Any]] = None,
        query_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # 记忆子图的统计信息（默认直接从 memory_subgraph 算）
        mem_stats = mem_stats or subgraph_store.stats(self.memory_subgraph)
        failure = self.issue.get("failure_frame") or {}
        issue_text = " ".join(
            str(x)
            for x in (
                self.issue.get("title"),
                self.issue.get("body"),
                self.issue.get("description"),
            )
            if x
        )
        pack = {
            "issue": issue_text,
            "top_assert": self.issue.get("top_assert"),
            "error_kind": self.issue.get("error_kind"),
            "failure_frame": failure,
            # 历史字段：subgraph_stats = memory_subgraph_stats
            "subgraph_stats": mem_stats,
            "memory_subgraph_stats": mem_stats,
            "query_subgraph_stats": query_stats,
            "cost": {
                "tokens": int(self.last_info.get("collate_meta", {}).get("est_tokens", 0)),
            },
            "cfg": self.config_dict,
        }
        return pack

    def _resolve_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        if not node_id:
            return None
        node = self.working_subgraph.get_node(node_id)
        if node:
            return dict(node)
        if self.repo_graph:
            repo_node = self._repo_nodes_by_id.get(node_id)
            if repo_node:
                return dict(repo_node)
        return None

    def _read_node_snippet(self, node: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        path = _norm_path(node.get("path") or "")
        if not path:
            return None
        abs_path = os.path.join(self.repo_root_in_container, path)
        text = _read_file_text(abs_path)
        if not text:
            return None
        start_line = int(node.get("start_line") or 1)
        end_line = int(node.get("end_line") or start_line)
        lines = text.splitlines()
        snippet_lines = lines[start_line - 1 : end_line]
        return {
            "id": node.get("id"),
            "path": path,
            "start": start_line,
            "end": end_line,
            "snippet": snippet_lines,
        }

    def _merge_repo_nodes_into_working(
        self,
        node_ids: Sequence[str],
        *,
        status: str = "explored",
    ) -> None:
        """把 repo_graph 中的若干节点及其边合并进 working_subgraph。"""
        if not node_ids:
            return

        working_nodes_by_id: Dict[str, Dict[str, Any]] = {}
        for n in getattr(self.working_subgraph, "nodes", []):
            nid = n.get("id")
            if isinstance(nid, str):
                working_nodes_by_id[nid] = n

        working_edges = list(getattr(self.working_subgraph, "edges", []) or [])
        working_ids = set(working_nodes_by_id.keys())

        for nid in node_ids:
            if not isinstance(nid, str):
                continue
            repo_node = self._repo_nodes_by_id.get(nid)
            if not repo_node:
                continue
            node_copy = dict(repo_node)
            node_copy["status"] = status
            tags = set(node_copy.get("tags") or [])
            tags.add(status)
            node_copy["tags"] = sorted(tags)
            working_nodes_by_id[nid] = node_copy
            working_ids.add(nid)

        for e in self._repo_edges:
            src_id = e.get("src")
            dst_id = e.get("dst")
            if not isinstance(src_id, str) or not isinstance(dst_id, str):
                continue
            if src_id in working_ids and dst_id in working_ids:
                working_edges.append(dict(e))

        self.working_subgraph = subgraph_store.wrap(
            {
                "nodes": list(working_nodes_by_id.values()),
                "edges": working_edges,
            }
        )

    def _merge_working_nodes_into_memory(
        self,
        node_ids: Sequence[str],
        *,
        status: str = "remembered",
    ) -> None:
        """把 working_subgraph 中的若干节点及其边合并进 memory_subgraph。"""
        if not node_ids:
            return

        mem_nodes_by_id: Dict[str, Dict[str, Any]] = {}
        for n in getattr(self.memory_subgraph, "nodes", []):
            nid = n.get("id")
            if isinstance(nid, str):
                mem_nodes_by_id[nid] = n

        mem_edges: List[Dict[str, Any]] = list(getattr(self.memory_subgraph, "edges", []) or [])
        mem_ids = set(mem_nodes_by_id.keys())

        working_nodes_by_id: Dict[str, Dict[str, Any]] = {}
        for n in getattr(self.working_subgraph, "nodes", []):
            nid = n.get("id")
            if isinstance(nid, str):
                working_nodes_by_id[nid] = n

        for nid in node_ids:
            if not isinstance(nid, str):
                continue
            w_node = working_nodes_by_id.get(nid)
            if not w_node:
                continue
            node_copy = dict(w_node)
            node_copy["status"] = status
            tags = set(node_copy.get("tags") or [])
            tags.add(status)
            node_copy["tags"] = sorted(tags)
            mem_nodes_by_id[nid] = node_copy
            mem_ids.add(nid)

        for e in getattr(self.working_subgraph, "edges", []) or []:
            src_id = e.get("src")
            dst_id = e.get("dst")
            if not isinstance(src_id, str) or not isinstance(dst_id, str):
                continue
            if src_id in mem_ids and dst_id in mem_ids:
                mem_edges.append(dict(e))

        self.memory_subgraph = subgraph_store.wrap(
            {
                "nodes": list(mem_nodes_by_id.values()),
                "edges": mem_edges,
            }
        )
