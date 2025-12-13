# graph_planner/env/planner_env.py
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
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
        """重置环境：重建 repo_graph，并清空 memory_subgraph / working_subgraph。"""
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

        # === 2) 初始化 memory_subgraph（现在：每次 reset 都清空） ===
        # 直接 new 一张空的长期记忆图，不再从磁盘加载历史记忆。
        self.memory_subgraph = subgraph_store.new()

        # === 3) 工作图 = 记忆图的拷贝（此时也是空图） ===
        self.working_subgraph = subgraph_store.wrap(self.memory_subgraph.to_json_obj())

        # === 4) 初始化 text_memory（基于 memory_subgraph） ===
        self.memory_graph_store = text_memory.WorkingGraphStore(self.memory_subgraph)
        self.memory_text_store = text_memory.NoteTextStore()
        self.memory_state = text_memory.TurnState(
            graph_store=self.memory_graph_store,
            text_store=self.memory_text_store,
        )
        # 此时 memory_subgraph 为空，size 从 0 开始计
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

        return self._obs()

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
        if self.memory_state:
            if kind == "explore":
                self.memory_state.latest_explore = info
            elif kind and kind != "memory":
                # 将非 explore 的工具输出也视作“observation”，便于 text memory commit
                self.memory_state.latest_non_explore = info
                self.memory_state.latest_observation = info

        self.last_info = info
        reward = self._compute_reward(info)
        done = bool(info.get("done"))

        obs = self._obs()
        self._log_step_graphs()
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
        """
        实现 ExploreAction 的三个子操作：
        - find:  根据 anchors 在 repo 图中做候选检索，不修改 working_subgraph；
        - expand: 围绕 anchors/nodes 在图上扩展，并 merge 进 working_subgraph；
        - read:  对指定 nodes 读取代码片段（snippets），并将这些节点标记为 explored。
        """

        info: Dict[str, Any] = {"kind": "explore", "op": act.op}

        if not self.repo_graph:
            info["error"] = "missing-repo-graph"
            return info

        # 数量预算：兼容旧字段 limit，同时支持 total_limit/max_per_anchor/dir_diversity_k
        total_limit = _safe_int(getattr(act, "total_limit", None) or act.limit or 32, 32)
        max_per_anchor = _safe_int(getattr(act, "max_per_anchor", None) or total_limit, total_limit)
        dir_diversity_k = _safe_int(getattr(act, "dir_diversity_k", None) or 4, 4)
        query = (getattr(act, "query", None) or "").strip()

        # -------- op = find --------
        if act.op == "find":
            # 1) 记忆召回：query 命中则返回 top-k note + node，并可选 merge 到 working
            if query:
                recall = self._recall_memory(query=query, k=min(total_limit, 20))
                if recall.get("memory_notes"):
                    info["memory_notes"] = recall["memory_notes"]
                if recall.get("memory_nodes"):
                    info["memory_nodes"] = recall["memory_nodes"]
                    # 将召回的记忆节点并入 working（只增量，不清空）
                    recall_ids = [n.get("id") for n in recall["memory_nodes"] if isinstance(n, dict)]
                    recall_ids = [nid for nid in recall_ids if isinstance(nid, str)]
                    if recall_ids:
                        self._merge_memory_nodes_into_working(recall_ids, status="recalled")

            # 2) repo 图检索：anchors 存在才做
            if act.anchors:
                candidates = mem_candidates.build_mem_candidates(
                    subgraph=self.repo_graph,
                    anchors=act.anchors,
                    max_nodes_per_anchor=max_per_anchor,
                    total_limit=total_limit,
                    dir_diversity_k=dir_diversity_k,
                )
            else:
                candidates = []

            self.last_candidates = candidates
            info["candidates"] = candidates
            info["subgraph_stats"] = subgraph_store.stats(self.working_subgraph)
            return info

        # -------- op = expand --------
        if act.op == "expand":
            candidates = mem_candidates.build_mem_candidates(
                subgraph=self.repo_graph,
                anchors=act.anchors,
                max_nodes_per_anchor=max_per_anchor,
                total_limit=total_limit,
                dir_diversity_k=dir_diversity_k,
            )
            self.last_candidates = candidates
            info["candidates"] = candidates

            node_ids: List[str] = []
            for c in candidates:
                nid = c.get("id")
                if isinstance(nid, str):
                    node_ids.append(nid)

            self._merge_repo_nodes_into_working(node_ids, status="explored")
            info["subgraph_stats"] = subgraph_store.stats(self.working_subgraph)
            return info

        # -------- op = read --------
        if act.op == "read":
            node_ids = list(act.nodes)
            if not node_ids and getattr(self, "last_candidates", None):
                for c in self.last_candidates:
                    nid = c.get("id")
                    if isinstance(nid, str):
                        node_ids.append(nid)

            # 去重并做 limit 截断
            uniq_ids: List[str] = []
            for nid in node_ids:
                if isinstance(nid, str) and nid not in uniq_ids:
                    uniq_ids.append(nid)
            if total_limit:
                uniq_ids = uniq_ids[: total_limit]

            snippets = []
            for nid in uniq_ids:
                snippet = self._read_node_snippet(nid)
                if snippet:
                    snippets.append(snippet)

            self.last_reads = snippets
            info["snippets"] = snippets

            self._merge_repo_nodes_into_working(uniq_ids, status="explored")
            info["subgraph_stats"] = subgraph_store.stats(self.working_subgraph)
            return info

        info["error"] = f"unknown-explore-op:{act.op}"
        return info

    def _handle_memory(self, act: MemoryAction) -> Dict[str, Any]:
        info: Dict[str, Any] = {"kind": "memory", "target": act.target, "intent": act.intent}

        if not self.memory_state:
            info["error"] = "memory-not-initialised"
            return info

        # 模型侧不再暴露 scope：env 内部固定 session
        scope = "session"

        selector = act.selector
        sel_map: Dict[str, Any] = selector if isinstance(selector, Mapping) else {}
        tag = sel_map.get("tag") if isinstance(sel_map.get("tag"), str) else (selector if isinstance(selector, str) else None)
        note_text = sel_map.get("note") if isinstance(sel_map.get("note"), str) else None

        # node_ids（可选）
        node_ids: List[str] = []
        sel_nodes = sel_map.get("nodes")
        if isinstance(sel_nodes, str):
            node_ids = [sel_nodes]
        elif isinstance(sel_nodes, Sequence):
            node_ids = [nid for nid in sel_nodes if isinstance(nid, str)]

        caps = getattr(self.memory_state, "caps", None) or text_memory.Size(**DEFAULT_MEMORY_CAPS)

        result: Dict[str, Any] = {}
        if act.intent == "commit":
            if act.target == "explore":
                # 选择性 commit：如果给了 nodes，就只把这些节点（及相关边）写入记忆
                if node_ids:
                    candidates: List[Dict[str, Any]] = []
                    for nid in node_ids:
                        w_node = self.working_subgraph.get_node(nid) if hasattr(self.working_subgraph, "get_node") else (self.working_subgraph.nodes or {}).get(nid)
                        if not isinstance(w_node, dict):
                            continue
                        cand = {
                            "id": nid,
                            "path": w_node.get("path"),
                            "span": w_node.get("span"),
                            "kind": w_node.get("kind"),
                            "summary": w_node.get("summary"),
                        }
                        candidates.append({k: v for k, v in cand.items() if v is not None})

                    mem_ids = set(getattr(self.memory_subgraph, "node_ids", set()) or set())
                    selected = set([c.get("id") for c in candidates if isinstance(c, dict) and c.get("id")])
                    allowed_ids = mem_ids | selected

                    edges: List[Dict[str, Any]] = []
                    for e in (getattr(self.working_subgraph, "edges", []) or []):
                        if not isinstance(e, dict):
                            continue
                        src = e.get("src")
                        dst = e.get("dst")
                        if not isinstance(src, str) or not isinstance(dst, str):
                            continue
                        if src in allowed_ids and dst in allowed_ids and (src in selected or dst in selected):
                            edges.append(dict(e))

                    fake_obs: Dict[str, Any] = {
                        "kind": "explore",
                        "candidates": candidates,
                        "edges": edges,
                        "frontier": len(candidates),
                    }
                    if note_text:
                        fake_obs["summary"] = note_text

                    old = self.memory_state.latest_explore
                    self.memory_state.latest_explore = fake_obs
                    result = text_memory.memory_commit(self.memory_state, "explore", scope, tag, caps) or {}
                    self.memory_state.latest_explore = old
                else:
                    # 未指定 nodes：直接 commit 最新 explore（全量候选）
                    result = text_memory.memory_commit(self.memory_state, "explore", scope, tag, caps) or {}

                # 标记 status/tags
                try:
                    ids_to_mark = node_ids
                    if not ids_to_mark:
                        # 如果未指定 nodes，则标记最新候选
                        le = self.memory_state.latest_explore or {}
                        cands = le.get("candidates") or []
                        ids_to_mark = [c.get("id") for c in cands if isinstance(c, dict) and isinstance(c.get("id"), str)]
                    for nid in ids_to_mark:
                        n = self.memory_subgraph.nodes.get(nid)
                        if isinstance(n, dict):
                            n["status"] = "remembered"
                            tags = set(n.get("tags") or [])
                            tags.add("remembered")
                            n["tags"] = sorted(tags)
                except Exception:
                    pass

            elif act.target == "observation":
                old_obs = self.memory_state.latest_observation
                if note_text:
                    obs = dict(old_obs or {})
                    obs.setdefault("kind", "observation")
                    obs["summary"] = note_text
                    self.memory_state.latest_observation = obs
                result = text_memory.memory_commit(self.memory_state, "observation", scope, tag, caps) or {}
                self.memory_state.latest_observation = old_obs
            else:
                result = {"ok": False, "error": "unknown-target", "msg": f"unknown memory target: {act.target}"}

        elif act.intent == "delete":
            if act.target == "explore":
                if node_ids:
                    # 手工删除指定节点及其 incident edges
                    removed = set(node_ids)
                    before_nodes = len(self.memory_subgraph.nodes or {})
                    before_edges = len(getattr(self.memory_subgraph, "edges", []) or [])
                    for nid in removed:
                        self.memory_subgraph.nodes.pop(nid, None)
                    self.memory_subgraph.node_ids = set((self.memory_subgraph.nodes or {}).keys())
                    new_edges = []
                    for e in (getattr(self.memory_subgraph, "edges", []) or []):
                        if not isinstance(e, dict):
                            continue
                        src = e.get("src")
                        dst = e.get("dst")
                        if src in removed or dst in removed:
                            continue
                        new_edges.append(e)
                    self.memory_subgraph.edges = new_edges
                    after_nodes = len(self.memory_subgraph.nodes or {})
                    after_edges = len(getattr(self.memory_subgraph, "edges", []) or [])
                    # 更新 size（token 预算不动，仅重算 nodes/edges）
                    try:
                        self.memory_state.size.nodes = after_nodes
                        self.memory_state.size.edges = after_edges
                        self.memory_state.next_version()
                    except Exception:
                        pass
                    result = {
                        "ok": True,
                        "deleted_nodes": sorted(list(removed)),
                        "deleted_nodes_count": before_nodes - after_nodes,
                        "deleted_edges_count": before_edges - after_edges,
                    }
                else:
                    sel = tag or "latest"
                    result = text_memory.memory_delete(self.memory_state, "explore", scope, sel) or {}
            elif act.target == "observation"
