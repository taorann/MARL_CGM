# graph_planner/env/planner_env.py
from __future__ import annotations

import json
import os
import time
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


def _plan_to_text(plan: Any) -> str:
    """Coerce repair.plan (List[str] | str | None) into a single string."""
    if plan is None:
        return ""
    if isinstance(plan, str):
        return plan.strip()
    if isinstance(plan, list):
        parts = [str(x).strip() for x in plan if str(x).strip()]
        return "\n".join(parts)
    return str(plan).strip()

def _safe_str(x: Any, default: str = "") -> str:
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return default


def _write_file_text(path: str, text: str, encoding: str = "utf-8") -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding=encoding) as f:
            f.write(text)
    except Exception:
        pass


from copy import deepcopy

from ..core.actions import (
    ActionUnion,
    ExploreAction,
    MemoryAction,
    NoopAction,
    RepairAction,
    SubmitAction,
)

# Shared protocol/contract error type used by the agent parser.
from ..agents.common.contracts import ProtocolError
from ..infra.config import Config, load as load_config
from ..infra import telemetry as telemetry_mod
from ..integrations.codefuse_cgm.formatting import GraphLinearizer, SnippetFormatter

from ..memory import graph_adapter, mem_candidates, subgraph_store, text_memory
from ..memory.subgraph_store import WorkingSubgraph, MemorySubgraph, Subgraph
from ..runtime.sandbox import SandboxConfig, SandboxRuntime
from .action_utils import normalize_explore_query_and_anchors

from aci.schema import Plan, PlanTarget
from aci.guard import GuardError, enforce_patch_guard

from actor.collater import collate
from actor import cgm_adapter

from ..infra.test_prioritizer import prioritize_tests


DEFAULT_MEMORY_CAPS = {
    "nodes": 200,
    "edges": 1000,
    "frontier": 50,
    "planner_tokens": 2000,
    "cgm_tokens": 16000,
}


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
        # Deterministic runner slot for concise logs (00/01/..). Should match apptainer_queue routing.
        try:
            self.runner_id: int = int(hash(effective_run_id) % max(int(getattr(sandbox_cfg, 'num_runners', 1) or 1), 1))
        except Exception:
            self.runner_id = 0

        self.box = SandboxRuntime(sandbox_cfg, run_id=effective_run_id)
        self.config: Config = load_config()
        self.config_dict: Dict[str, Any] = self.config.to_dict()
        # Episode stop condition: align with dataset/eval_engine max_steps when provided.
        issue_max_steps = self.issue.get("max_steps")
        env_max_steps = os.environ.get("GP_MAX_STEPS") or os.environ.get("GRAPH_PLANNER_MAX_STEPS")
        cfg_max_steps = getattr(self.config, "max_steps", None) or getattr(self.config, "episode_max_steps", None)
        try:
            self.max_steps = int(env_max_steps or issue_max_steps or cfg_max_steps or 40)
        except Exception:
            self.max_steps = 40
        # Unified telemetry (trajectory / events / metrics)
        self.telemetry = telemetry_mod.get_telemetry(run_id=effective_run_id)
        try:
            self.telemetry.start_run(meta={"component": "planner_env"})
        except Exception:
            pass
        io_cfg = self.config_dict.get("io") if isinstance(self.config_dict, Mapping) else {}
        if not isinstance(io_cfg, Mapping):
            io_cfg = {}
        strict_env = os.environ.get("GRAPH_PLANNER_STRICT_IO") or os.environ.get("STRICT_PLANNER_IO")
        if strict_env is not None:
            self._strict_io = str(strict_env).strip().lower() in {"1", "true", "yes"}
        else:
            self._strict_io = bool(io_cfg.get("strict_planner_io", False))
        # Step-level trace printing (useful for debugging without opening telemetry files)
        _print_env = os.environ.get("GP_PRINT_OPS") or os.environ.get("GRAPH_PLANNER_PRINT_OPS")
        if _print_env is None:
            self._print_ops = True
        else:
            self._print_ops = str(_print_env).strip().lower() not in {"0","false","no","off"}
        try:
            self._print_excerpt_chars = int(os.environ.get("GP_PRINT_EXCERPT_CHARS", "360"))
        except Exception:
            self._print_excerpt_chars = 360


        self.steps: int = 0
        self.last_info: Dict[str, Any] = {}
        self.repo_root_in_container: str = sandbox_cfg.workdir or "."
        if getattr(self.box, "_mode", None) == "remote_swe":
            self.repo_root_in_container = "/repo"
        # Optional host repo root for per-env graph scanning.
        self.repo_root_host: Optional[str] = getattr(sandbox_cfg, "repo_root_host", None)
        self.run_id: str = os.environ.get("GRAPH_PLANNER_RUN_ID", "") or self.issue.get("run_id", "")

        # 三图结构
        # - repo_graph: read-only truth graph
        # - working_subgraph: planner-facing cache (may be noisy)
        # - memory_subgraph: CGM-facing evidence graph (high-signal)
        self.repo_graph: Optional[Subgraph] = None
        self.working_subgraph: WorkingSubgraph = subgraph_store.new_working()
        self.memory_subgraph: MemorySubgraph = subgraph_store.new_memory()

        # Repo 图索引
        self._repo_nodes_by_id: Dict[str, Dict[str, Any]] = {}
        self._repo_edges: List[Dict[str, Any]] = []

        # Text memory (planner-only): a simple session-scoped note list.
        self.memory_text_store: Optional[text_memory.NoteTextStore] = None

        # 最近一步 explore/find|expand 的结果
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
        if backend_mode == "remote_swe":
            # remote_swe：先在远端容器里构图，再落盘/回传到 host 缓存，最后本地加载。
            # 注意：如果这里失败，后续 explore/find/expand 基本不可用，应 fail-fast。
            allow_empty = os.environ.get("GP_ALLOW_EMPTY_REPO_GRAPH", "0").strip().lower() in {"1", "true", "yes"}
            try:
                if hasattr(self.box, "load_repo_graph"):
                    repo_json = self.box.load_repo_graph(repo_id=str(getattr(self, "repo_id", "") or ""))
                elif hasattr(self.box, "build_issue_subgraph"):
                    # 兼容旧接口：将 build_issue_subgraph 视为 repo-level graph
                    repo_json = self.box.build_issue_subgraph(self.issue_id)
                else:
                    raise RuntimeError("remote_swe backend missing load_repo_graph/build_issue_subgraph")
                self.repo_graph = subgraph_store.wrap(repo_json)
            except Exception as e:
                _dbg(f"repo_graph build/load failed: {e!r}")
                if not allow_empty:
                    raise
                self.repo_graph = subgraph_store.new()

            # 强约束：remote_swe 默认必须有 repo_graph
            try:
                n_nodes = len(getattr(self.repo_graph, "nodes", {}) or {})
            except Exception:
                n_nodes = 0
            if (not allow_empty) and n_nodes <= 0:
                raise RuntimeError("remote_swe repo_graph is empty; set GP_ALLOW_EMPTY_REPO_GRAPH=1 to bypass")
        else:
            # 本地 backend：从 ACI 子图缓存加载（如有需要可触发扫描构图）
            if hasattr(graph_adapter, "set_repo_root") and self.repo_root_host:
                graph_adapter.set_repo_root(self.repo_root_host)
            try:
                graph_adapter.connect()
            except Exception:
                pass
            try:
                _req = getattr(graph_adapter, '_require_handle', None)
                if callable(_req):
                    gh = _req()
                    edges = []
                    for src, neighs in getattr(gh, 'adj', {}).items():
                        for dst, etype in (neighs or []):
                            edges.append({'src': str(src), 'dst': str(dst), 'etype': str(etype)})
                    self.repo_graph = subgraph_store.wrap({'nodes': getattr(gh, 'nodes', {}), 'edges': edges})
                else:
                    self.repo_graph = subgraph_store.new()
            except Exception:
                self.repo_graph = subgraph_store.new()


        # Keep graph_adapter aligned with repo_graph (mem_candidates / expand).
        try:
            # graph_adapter 的 root 用于路径拼接/显示；remote_swe 用真实容器 repo root
            if getattr(self.box, "_mode", None) == "remote_swe":
                root = getattr(self.box, "workdir", None) or os.environ.get("GP_REMOTE_REPO_ROOT", "/testbed")
            else:
                root = (getattr(self, "repo_root_host", None) or self.repo_root_in_container)
            if hasattr(graph_adapter, "set_repo_root"):
                graph_adapter.set_repo_root(root)
            if hasattr(graph_adapter, "connect_from_subgraph"):
                graph_adapter.connect_from_subgraph(self.repo_graph, root=root)
            elif hasattr(graph_adapter, "connect_from_nodes_edges"):
                graph_adapter.connect_from_nodes_edges(self.repo_graph.nodes, self.repo_graph.edges, root=root)  # type: ignore
        except Exception:
            pass

        # Repo 索引
        # NOTE: WorkingSubgraph.nodes 在不同实现里可能是:
        #   1) List[Dict]  (json obj 直接透传)
        #   2) Dict[str, Dict]  (wrap 后按 id 索引)
        # 旧代码直接 `for n in nodes: n.get(...)` 会在 Dict 情况下拿到 key(str) 而崩溃。
        self._repo_nodes_by_id = {}
        self._repo_edges = []
        if self.repo_graph is not None:
            nodes_store = getattr(self.repo_graph, "nodes", None)
            if isinstance(nodes_store, dict):
                # nodes_store: id -> node
                for nid, node in nodes_store.items():
                    if isinstance(nid, str) and isinstance(node, dict):
                        self._repo_nodes_by_id[nid] = node
                    elif isinstance(nid, str):
                        # 极端兜底：至少保留 id
                        self._repo_nodes_by_id[nid] = {"id": nid}
            else:
                for n in (nodes_store or []):
                    if isinstance(n, dict):
                        nid = n.get("id")
                        if isinstance(nid, str):
                            self._repo_nodes_by_id[nid] = n
                    elif isinstance(n, str) and isinstance(nodes_store, dict):
                        # 兼容：如果混入了 id 字符串
                        node = nodes_store.get(n)
                        if isinstance(node, dict):
                            self._repo_nodes_by_id[n] = node
            self._repo_edges = list(getattr(self.repo_graph, "edges", []) or [])

        # === 2) 初始化 memory_subgraph（每次 reset 都清空） ===
        # MemorySubgraph is CGM-facing and starts empty each episode.
        self.memory_subgraph = subgraph_store.new_memory()

        # === 3) 初始化 working_subgraph（独立对象，不拷贝 memory） ===
        self.working_subgraph = subgraph_store.new_working()

        # === 4) 初始化 text memory（planner-only） ===
        # Text memory is a simple list of human-readable notes for the planner.
        # It is NOT part of the graph memory and is NOT fed into CGM/collate.
        self.memory_text_store = text_memory.NoteTextStore()

# Telemetry: start a fresh episode for this issue/task
        try:
            self.telemetry.start_episode(
                task_id=self.issue_id,
                meta={
                    "issue_id": self.issue_id,
                    "repo_root_host": self.repo_root_host,
                    "backend_mode": backend_mode,
                },
                tags=(f"issue:{self.issue_id}",),
            )
            self.telemetry.set_step(0)
        except Exception:
            pass


        return self._obs()

    def close(self) -> None:
        """Close underlying sandbox resources.

        Called by rLLM wrapper when a trajectory terminates.
        """
        try:
            box = getattr(self, "box", None)
            if box is not None and hasattr(box, "close"):
                box.close()
        finally:
            # best-effort telemetry flush
            try:
                tel = getattr(self, "telemetry", None)
                flush = getattr(tel, "flush", None) or getattr(tel, "close", None)
                if callable(flush):
                    flush()
            except Exception:
                pass


    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """单步：接受一个 action 字典，返回 (obs, reward, done, info)。"""
        self.steps += 1
        _t0 = time.perf_counter()
        try:
            self.telemetry.set_step(self.steps)
        except Exception:
            pass
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
            # Telemetry: record invalid action as a step
            try:
                self.telemetry.set_step(self.steps)
                self.telemetry.log_step({
                    "action": action,
                    "error": {"code": exc.code, "detail": exc.detail},
                    "reward": -0.05,
                    "done": False,
                })
                self.telemetry.metric("env.invalid_action", 1.0, step_id=self.steps)
            except Exception:
                pass
            # Stop the trajectory if we have reached the max step budget.
            if getattr(self, "max_steps", None) is not None and self.steps >= int(self.max_steps):
                info["stop_reason"] = "max_steps"
                return obs, -0.05, True, info
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

        # NOTE: text memory is handled only via explicit memory actions (target="observation").
        self.last_info = info
        reward = self._compute_reward(info)
        done = bool(info.get("done"))
        # Global stop: enforce max_steps budget (used by eval_engine/datasets).
        if getattr(self, "max_steps", None) is not None and self.steps >= int(self.max_steps):
            if not done:
                info["stop_reason"] = "max_steps"
            done = True
        info["done"] = bool(done)

        obs = self._obs()
        self._log_step_graphs()
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Action 解析与 reward
    # ------------------------------------------------------------------

    def _parse_action(self, payload: Any) -> ActionUnion:
        """Parse an incoming action.

        The rLLM stack may pass:
        - a plain dict (our canonical representation)
        - a Pydantic model instance (e.g. ExploreAction / NoopAction)
        - a JSON string

        We normalise all of them to a mapping and then instantiate the corresponding
        action model.
        """

        # Fast path: already a typed action.
        if isinstance(payload, (ExploreAction, MemoryAction, RepairAction, SubmitAction, NoopAction)):
            return payload

        # Sometimes the upstream executor serialises actions as JSON strings.
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception as exc:
                raise ProtocolError("invalid-payload", f"action payload must be JSON: {exc}") from exc

        # Pydantic v2: BaseModel.model_dump(); v1: BaseModel.dict()
        if not isinstance(payload, Mapping):
            if hasattr(payload, "model_dump"):
                try:
                    payload = payload.model_dump()  # type: ignore[attr-defined]
                except Exception:
                    pass
            elif hasattr(payload, "dict"):
                try:
                    payload = payload.dict()  # type: ignore[attr-defined]
                except Exception:
                    pass

        if not isinstance(payload, Mapping):
            raise ProtocolError("invalid-payload", "action payload must be a mapping")

        action_type = payload.get("type")
        if action_type == "explore":
            # Backward compatibility: normalize legacy explore.read to explore.expand hop=0
            try:
                if payload.get("op") == "read":
                    payload = dict(payload)
                    payload["op"] = "expand"
                    payload["hop"] = 0
                    nodes = payload.get("nodes") or []
                    if nodes and not payload.get("anchors"):
                        payload["anchors"] = [{"id": nid} for nid in nodes if isinstance(nid, str)]
            except Exception:
                pass
            return ExploreAction(**payload)
        if action_type == "memory":
            return MemoryAction(**payload)
        if action_type == "repair":
            # v5 gate: repair requires non-empty memory_subgraph (high-signal evidence for CGM).
            try:
                mem_nodes = getattr(getattr(self, "memory_subgraph", None), "nodes", {}) or {}
                if isinstance(mem_nodes, list):
                    mem_n = len(mem_nodes)
                else:
                    mem_n = len(mem_nodes)
                if mem_n <= 0:
                    title = str((self.issue or {}).get("title") or "")
                    query = title.strip() or "locate relevant code for the issue"
                    return ExploreAction(type="explore", op="find", query=query, anchors=[], nodes=[])
            except Exception:
                pass
            # Guardrail: require at least one explore before repair to avoid blind patching.
            try:
                if int(getattr(self, "steps", 0)) <= 1:
                    stats = subgraph_store.stats(self.working_subgraph)
                    if not stats.get("n_nodes"):
                        title = str((self.issue or {}).get("title") or "")
                        query = title.strip() or "locate relevant code for the issue"
                        return ExploreAction(type="explore", op="find", query=query, anchors=[], nodes=[])
            except Exception:
                pass
            # v5 simplified protocol: the planner only provides a high-level plan.
            # The environment always applies the patch and supplies issue metadata.
            try:
                payload = dict(payload)
            except Exception:
                payload = {"type": "repair"}
            payload.setdefault("apply", True)
            payload.setdefault("issue", self.issue or {})
            # Do not accept planner-proposed patches/targets.
            payload.pop("patch", None)
            payload.pop("plan_targets", None)
            payload.pop("targets", None)
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
        实现 ExploreAction 的两个子操作：
        - find:  根据 anchors/query 在 repo 图中做候选检索，不修改 working_subgraph；
        - expand: 围绕 anchors 在图上扩展，并 merge 进 working_subgraph；
        """

        info: Dict[str, Any] = {"kind": "explore", "op": act.op}

        # v5 hard rules: use a single query and a single anchor for explore.
        query_value, anchors, trimmed = normalize_explore_query_and_anchors(
            op=act.op,
            query=getattr(act, "query", None),
            anchors=getattr(act, "anchors", None),
            nodes=getattr(act, "nodes", None),
            frontier_anchor_id=getattr(self, "frontier_anchor_id", None),
        )
        if trimmed:
            info["trimmed"] = trimmed

        if not self.repo_graph:
            info["error"] = "missing-repo-graph"
            return info

        # 数量预算：兼容旧字段 limit，同时支持 total_limit/max_per_anchor/dir_diversity_k
        total_limit = _safe_int(getattr(act, "total_limit", None) or act.limit or 32, 32)
        max_per_anchor = _safe_int(getattr(act, "max_per_anchor", None) or total_limit, total_limit)
        dir_diversity_k = _safe_int(getattr(act, "dir_diversity_k", None) or 4, 4)
        # Clamp expand size to keep prompts/snippets manageable.
        if act.op == "expand":
            _cap = _safe_int(os.environ.get("GP_EXPAND_TOTAL_LIMIT", "20"), 20)
            if _cap > 0:
                total_limit = min(total_limit, _cap)
                max_per_anchor = min(max_per_anchor, _cap)
        # Keep the raw query string intact for the query DSL (symbol:/path:/+/-/quotes).
        # We still compute tokenized terms only for debugging/telemetry.
        query_terms = mem_candidates.extract_query_terms(query_value)
        if query_terms:
            info["query_terms"] = list(query_terms)
        raw_query = (query_value or "").strip() if isinstance(query_value, str) else " ".join(query_terms).strip()
        query = raw_query


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
            if anchors:
                candidates = mem_candidates.build_mem_candidates(
                    subgraph=self.repo_graph,
                    anchors=anchors,
                    max_nodes_per_anchor=max_per_anchor,
                    total_limit=total_limit,
                    dir_diversity_k=dir_diversity_k,
                )
            else:
         
