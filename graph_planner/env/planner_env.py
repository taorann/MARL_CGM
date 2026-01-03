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
                if query:
                    candidates = mem_candidates.search_repo_candidates_by_query(
                        repo_graph=self.repo_graph,
                        query=query,
                        total_limit=total_limit,
                        dir_diversity_k=dir_diversity_k,
                    )
                else:
                    candidates = []

            # Expose candidates to the agent, but do NOT mutate working_subgraph.
            # The planner should explicitly select an anchor via explore_expand.
            self.last_candidates = candidates
            info["candidates"] = candidates

            # Mark which already-present working nodes are in the last candidate set.
            cand_ids: List[str] = []
            for c in candidates:
                nid = c.get("id") if isinstance(c, dict) else None
                if isinstance(nid, str):
                    cand_ids.append(nid)
            cand_set = set(cand_ids)
            if cand_set:
                try:
                    for n in (self.working_subgraph.nodes or []):
                        if isinstance(n, dict) and isinstance(n.get("id"), str):
                            n["in_last_candidates"] = n.get("id") in cand_set
                except Exception:
                    pass

            # Attach lightweight previews for the top-k candidates (for debugging/inspection).
            k_preview = _safe_int(os.environ.get("GP_CANDIDATE_PREVIEW_K", "6"), 6)
            max_lines = _safe_int(os.environ.get("GP_CANDIDATE_PREVIEW_LINES", "24"), 24)
            previews: List[Dict[str, Any]] = []
            for c in candidates[: max(0, k_preview)]:
                try:
                    nid = c.get("id") if isinstance(c, dict) else None
                    if not isinstance(nid, str):
                        continue
                    sn = self._read_node_snippet(nid)
                    if sn and isinstance(sn.get("snippet_lines"), list):
                        previews.append({
                            "id": nid,
                            "path": sn.get("path"),
                            "start": sn.get("start"),
                            "end": sn.get("end"),
                            "snippet_lines": list(sn.get("snippet_lines", [])[: max(0, max_lines)]),
                        })
                except Exception:
                    continue
            if previews:
                info["candidate_previews"] = previews

            info["subgraph_stats"] = subgraph_store.stats(self.working_subgraph)
            return info

        # -------- op = expand --------
        if act.op == "expand":
            candidates = mem_candidates.build_mem_candidates(
                subgraph=self.repo_graph,
                anchors=anchors,
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

            # Runtime guard: keep working subgraph bounded. This prevents rare
            # cases where repeated expands blow up local state (and later
            # summaries/prompt formatting).
            max_working = _safe_int(os.environ.get("GP_MAX_WORKING_NODES", "350"), default=350)
            if max_working > 0 and len(self.working_subgraph.nodes) > max_working:
                keep_recent = _safe_int(os.environ.get("GP_KEEP_RECENT_UNMEM", "120"), default=120)
                subgraph_store.prune_working(self.working_subgraph, keep_recent_unmemorized=keep_recent)
                info["pruned_working"] = True
            info["subgraph_stats"] = subgraph_store.stats(self.working_subgraph)
            return info

        info["error"] = f"unknown-explore-op:{act.op}"
        return info


    def _handle_memory(self, act: MemoryAction) -> Dict[str, Any]:
        """Handle memory action.

        We maintain two distinct memories:

        - **Graph memory** (memory_subgraph): high-signal induced subgraph used for CGM/collate.
          The planner commits nodes by id (select_ids) and we project W[S] -> M.

        - **Text memory** (memory_text_store): a simple session-scoped list of notes, only exposed
          to the planner via observation. CGM does *not* see it.

        Selector semantics (v7):
          - select_ids / commit_ids / nodes / node_ids / ids : the ids to *commit to graph memory*
          - keep_ids : ids to keep in working only (NOT committed)
          - note / note_text / text : free-form text to commit to text memory when target="observation"
          - note_id / note_ids / selector : which note(s) to delete when intent="delete" and target="observation"
        """

        raw_selector = act.selector
        if isinstance(raw_selector, dict):
            selector = raw_selector
        elif isinstance(raw_selector, (list, tuple)):
            selector = {"select_ids": list(raw_selector)}
        elif isinstance(raw_selector, str):
            # Allow shorthand: a single id or 'latest'
            selector = ({"selector": raw_selector} if (act.target or "").lower() == "observation" else {"select_ids": [raw_selector]})
        else:
            selector = {}


        tag = selector.get("tag") if isinstance(selector.get("tag"), str) else None

        # ---- text memory path (planner-only) ----
        if (act.target or "").lower() == "observation":
            if self.memory_text_store is None:
                self.memory_text_store = text_memory.NoteTextStore()

            info: Dict[str, Any] = {"kind": "memory", "target": "observation", "intent": act.intent}

            if act.intent == "commit":
                note_text = (
                    selector.get("note_text")
                    or selector.get("note")
                    or selector.get("text")
                    or selector.get("content")
                )
                note_text = _safe_str(note_text, "").strip()
                if not note_text:
                    info["skipped_reason"] = "empty_note_text"
                    return info

                note_id = self.memory_text_store.append("session", note_text)
                info["note_id"] = note_id
                info["notes_total"] = len(self.memory_text_store.get("session"))
                return info

            if act.intent == "delete":
                raw = (
                    selector.get("note_id")
                    or selector.get("note_ids")
                    or selector.get("ids")
                    or selector.get("selector")
                    or "latest"
                )
                deleted: List[Any] = []
                if isinstance(raw, (list, tuple)):
                    for x in raw:
                        ok = self.memory_text_store.remove("session", selector=str(x))
                        if ok:
                            deleted.append(x)
                else:
                    ok = self.memory_text_store.remove("session", selector=str(raw))
                    if ok:
                        deleted.append(raw)

                info["deleted"] = deleted
                info["notes_total"] = len(self.memory_text_store.get("session"))
                return info

            info["skipped_reason"] = f"unknown_intent:{act.intent}"
            return info

        # ---- graph memory path (CGM-facing) ----
        raw_sel = (
            selector.get("select_ids")
            or selector.get("commit_ids")
            or selector.get("nodes")
            or selector.get("node_ids")
            or selector.get("ids")
            or []
        )
        if isinstance(raw_sel, str):
            selected_ids = [raw_sel]
        else:
            selected_ids = [str(x) for x in (raw_sel or []) if x is not None]

        raw_keep = selector.get("keep_ids") or []
        if isinstance(raw_keep, str):
            keep_ids = [raw_keep]
        else:
            keep_ids = [str(x) for x in (raw_keep or []) if x is not None]
        keep_set = {x for x in keep_ids if isinstance(x, str) and x}

        top_k = selector.get("top_k")
        try:
            top_k = int(top_k) if top_k is not None else 8
        except Exception:
            top_k = 8

        # Runtime guard: keep memorized nodes + a fixed recent tail of unmemorized nodes.
        # Do not let the planner control pruning aggressiveness.
        keep_recent = _safe_int(os.environ.get("GP_KEEP_RECENT_UNMEM", "120"), default=120)

        w_before = subgraph_store.stats(self.working_subgraph)
        m_before = subgraph_store.stats(self.memory_subgraph)

        info = {
            "kind": "memory",
            "target": act.target or "explore",
            "intent": act.intent,
            "selected_for_memory": 0,
            "auto_selected": False,
            "pruned_unmemorized": 0,
            "working_nodes_before": w_before.get("n_nodes", 0),
            "working_memorized_before": w_before.get("n_memorized", 0),
            "working_unmemorized_before": w_before.get("n_unmemorized", 0),
            "memory_nodes_before": m_before.get("n_nodes", 0),
        }

        # ---------- delete from graph memory ----------
        if act.intent == "delete":
            selected_set = {nid for nid in selected_ids if isinstance(nid, str) and nid}
            if selected_set:
                mem_nodes = getattr(self.memory_subgraph, "nodes", {}) or {}
                if isinstance(mem_nodes, dict):
                    for nid in list(mem_nodes.keys()):
                        if nid in selected_set:
                            del mem_nodes[nid]
                mem_edges = getattr(self.memory_subgraph, "edges", []) or []
                if isinstance(mem_edges, list):
                    new_edges = []
                    for e in mem_edges:
                        if not isinstance(e, dict):
                            continue
                        u = str(e.get("u") or e.get("src") or e.get("from") or "")
                        v = str(e.get("v") or e.get("dst") or e.get("to") or "")
                        if u in selected_set or v in selected_set:
                            continue
                        new_edges.append(e)
                    self.memory_subgraph.edges = new_edges

            for nid in selected_ids:
                wn = self.working_subgraph.get_node(nid)
                if isinstance(wn, dict):
                    wn["memorized"] = False
                    wn["memorized_at_step"] = None

            pruned = subgraph_store.prune_working(
                self.working_subgraph,
                keep_ids=keep_set,
                keep_recent_unmemorized=keep_recent,
            )
            info["pruned_unmemorized"] = int(pruned or 0)

            w_after = subgraph_store.stats(self.working_subgraph)
            m_after = subgraph_store.stats(self.memory_subgraph)
            info.update(
                {
                    "working_nodes_after": w_after.get("n_nodes", 0),
                    "working_memorized_after": w_after.get("n_memorized", 0),
                    "working_unmemorized_after": w_after.get("n_unmemorized", 0),
                    "memory_nodes_after": m_after.get("n_nodes", 0),
                }
            )
            return info

        # ---------- commit to graph memory ----------
        if act.intent != "commit":
            info["skipped_reason"] = f"unknown_intent:{act.intent}"
            return info

        selected_set = {nid for nid in selected_ids if isinstance(nid, str) and nid}
        if not selected_set:
            info["auto_selected"] = True
            candidates = []
            for nid in reversed(list(getattr(self.working_subgraph, "node_ids", []) or [])):
                n = self.working_subgraph.get_node(nid)
                if not isinstance(n, dict) or n.get("memorized"):
                    continue
                has_snip = bool(n.get("snippet") or n.get("snippet_lines"))
                touched = n.get("gp_last_touched_step")
                try:
                    touched = int(touched) if touched is not None else -1
                except Exception:
                    touched = -1
                score = (1 if has_snip else 0, touched)
                candidates.append((score, nid))
            candidates.sort(reverse=True)
            selected_set = {nid for _, nid in candidates[:top_k]}

        info["selected_for_memory"] = len(selected_set)

        if not selected_set:
            pruned = subgraph_store.prune_working(
                self.working_subgraph,
                keep_ids=keep_set,
                keep_recent_unmemorized=keep_recent,
            )
            info["pruned_unmemorized"] = int(pruned or 0)
            w_after = subgraph_store.stats(self.working_subgraph)
            m_after = subgraph_store.stats(self.memory_subgraph)
            info.update(
                {
                    "working_nodes_after": w_after.get("n_nodes", 0),
                    "working_memorized_after": w_after.get("n_memorized", 0),
                    "working_unmemorized_after": w_after.get("n_unmemorized", 0),
                    "memory_nodes_after": m_after.get("n_nodes", 0),
                    "skipped_reason": "empty_selection",
                }
            )
            return info

        # Ensure selected memorized nodes carry embedded snippets when present in
        # the repo graph snapshot (older working nodes may lack snippet_lines).
        for nid in selected_set:
            n = self.working_subgraph.get_node(nid)
            if not isinstance(n, dict):
                continue
            if n.get("snippet_lines"):
                continue
            r = self.repo_graph.get(nid)
            if isinstance(r, dict) and r.get("snippet_lines"):
                n["snippet_lines"] = r.get("snippet_lines")
                if r.get("sig"):
                    n["sig"] = r.get("sig")
                if r.get("doc"):
                    n["doc"] = r.get("doc")

        proj_nodes, proj_edges = subgraph_store.project_to_memory(self.working_subgraph, list(selected_set))
        subgraph_store.add_nodes(self.memory_subgraph, proj_nodes)
        subgraph_store.add_edges(self.memory_subgraph, proj_edges)

        for nid in selected_set:
            n = self.working_subgraph.get_node(nid)
            if isinstance(n, dict):
                n["memorized"] = True
                n["memorized_at_step"] = int(getattr(self, "steps", 0))
                if tag:
                    n["tag"] = tag

        keep_union = set(selected_set) | set(keep_set)
        pruned = subgraph_store.prune_working(
            self.working_subgraph,
            keep_ids=keep_union,
            keep_recent_unmemorized=keep_recent,
        )
        info["pruned_unmemorized"] = int(pruned or 0)

        w_after = subgraph_store.stats(self.working_subgraph)
        m_after = subgraph_store.stats(self.memory_subgraph)
        info.update(
            {
                "working_nodes_after": w_after.get("n_nodes", 0),
                "working_memorized_after": w_after.get("n_memorized", 0),
                "working_unmemorized_after": w_after.get("n_unmemorized", 0),
                "memory_nodes_after": m_after.get("n_nodes", 0),
            }
        )
        return info

    def _handle_repair(self, act: RepairAction) -> Dict[str, Any]:
        info: Dict[str, Any] = {"kind": "repair", "apply": act.apply, "plan": _plan_to_text(act.plan)}
        mem_nodes = getattr(self.memory_subgraph, "nodes", {}) or {}
        mem_count = len(mem_nodes) if isinstance(mem_nodes, dict) else (len(mem_nodes) if mem_nodes is not None else 0)
        info["memory_nodes"] = mem_count
        if mem_count == 0:
            info["repair_allowed"] = False
            info["skipped_reason"] = "empty_memory_subgraph"
            return info
        info["repair_allowed"] = True
        if not act.apply:
            return info

        if act.patch and isinstance(act.patch, dict) and act.patch.get("edits"):
            patch: Dict[str, Any] = dict(act.patch)
            if "summary" not in patch:
                patch["summary"] = _plan_to_text(act.plan) or ""
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
                # v5 semantics: prioritize tests based on the evidence set used for patch generation.
                subgraph=getattr(self, "memory_subgraph", None) or self.working_subgraph,
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
            # ✅ 使用 memory_subgraph 作为 collate 的基础图
            chunks, meta = collate(self.memory_subgraph, plan_struct, collate_cfg)
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
                plan=_plan_to_text(act.plan) or "",
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
        patch_dict.setdefault("summary", _plan_to_text(act.plan) or "")
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
            # v5 semantics: prioritize tests based on the evidence set used for patch generation.
            subgraph=getattr(self, "memory_subgraph", None) or self.working_subgraph,
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
        passed = bool(tests_report.get("passed", False)) if isinstance(tests_report, dict) else bool(tests_report)
        info["success"] = passed
        # Only finish when tests passed. Otherwise allow more repair steps.
        info["done"] = bool(passed)
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
            "runner_id": getattr(self, "runner_id", 0),
            "issue": self.issue,
            "steps": self.steps,
            "last_info": self.last_info,
            # LLM 看到的是“当前工作图”，里面节点已经带 status/tags（explored/remembered）
            "subgraph": working_json,
            "subgraph_stats": working_stats,
            # 记忆图的统计信息（可选）
            "memory_stats": memory_stats,
            "text_memory": {"notes": (self.memory_text_store.get('session', limit=12) if self.memory_text_store else []), "count": (len(self.memory_text_store.get('session')) if self.memory_text_store else 0)},
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

    def _read_node_snippet(self, node: Union[Mapping[str, Any], str]) -> Optional[Dict[str, Any]]:
        """Read snippet lines for a node.

        - local backends: read from evaluator filesystem
        - remote_swe: prefer snippet embedded in repo_graph; otherwise read inside remote container
        """
        try:
            if isinstance(node, str):
                node = self._resolve_node(node) or {}
            if not isinstance(node, Mapping):
                return None

            raw_path = str(node.get("path") or "").strip()
            if not raw_path:
                return None

            # Nodes may carry either repo-relative paths ("a/b.py") or absolute
            # in-sandbox paths ("/testbed/a/b.py"). We'll resolve to an absolute
            # path for file reads, but keep a stable display path.
            repo_root = str(self.repo_root or self.box.workdir or ".").rstrip("/")

            def _resolve_abs(p: str) -> str:
                if p.startswith("/"):
                    # Already absolute? keep iff it is under repo_root.
                    if repo_root and p.startswith(repo_root + "/"):
                        return p
                    # Otherwise treat as repo-relative with a leading slash.
                    p = p.lstrip("/")
                return os.path.join(repo_root, p)

            abs_path = _resolve_abs(raw_path)
            # Prefer repo-relative for readability in prompts/logs.
            if raw_path.startswith(repo_root + "/"):
                path = raw_path[len(repo_root) + 1 :]
            else:
                path = raw_path.lstrip("/")

            span = node.get("span") if isinstance(node, Mapping) else {}
            span_start = span.get("start") if isinstance(span, Mapping) else None
            span_end = span.get("end") if isinstance(span, Mapping) else None
            start_line = int(node.get("start_line") or span_start or 1)
            end_line = int(node.get("end_line") or span_end or start_line)


            # 0) If repo_graph already carries snippet_lines/snippet, prefer it.
            embedded = node.get("snippet_lines")
            if not isinstance(embedded, list):
                embedded = node.get("snippet")
            if isinstance(embedded, list) and embedded:
                # Best-effort truncate to keep prompt under control.
                max_lines = _safe_int(os.environ.get("GP_MAX_SNIPPET_LINES", "120"), 120)
                snippet_lines = [str(x) for x in embedded[: max(0, max_lines)]]
                return {
                    "id": node.get("id"),
                    "path": path,
                    "start": start_line,
                    "end": end_line,
                    "snippet_lines": snippet_lines,
                    "snippet": snippet_lines,
                }

            # remote_swe: read inside container (repo root is the remote workdir, typically /testbed)
            if getattr(self.box, "_mode", None) == "remote_swe":
                repo_root = getattr(self.box, "workdir", None) or os.environ.get("GP_REMOTE_REPO_ROOT", "/testbed")
                abs_path = os.path.join(str(repo_root), path)
                reader = getattr(self.box, "read_file_lines", None)
                if not callable(reader):
                    return None
                snippet_lines, rc = reader(abs_path, start_line, end_line, timeout=60)
                if int(rc) != 0:
                    return None
                snippet_lines = list(snippet_lines or [])
                if not snippet_lines:
                    return None
                return {
                    "id": node.get("id"),
                    "path": path,
                    "start": start_line,
                    "end": end_line,
                    "snippet_lines": snippet_lines,
                    "snippet": snippet_lines,
                }


            # local (prefer sandbox runtime reader for consistency)
            abs_path = os.path.join(self.repo_root_in_container, path)
            reader = getattr(self.box, "read_file_lines", None)
            if callable(reader):
                snippet_lines, rc = reader(abs_path, start_line, end_line, timeout=60)
                if int(rc) == 0:
                    snippet_lines = list(snippet_lines or [])
                    return {
                        "id": node.get("id"),
                        "path": path,
                        "start": start_line,
                        "end": end_line,
                        "snippet_lines": snippet_lines,
                        "snippet": snippet_lines,
                    }
            # fallback: direct filesystem read
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    lines2 = f.read().splitlines()
                snippet_lines = lines2[start_line - 1 : end_line]
                return {
                    "id": node.get("id"),
                    "path": path,
                    "start": start_line,
                    "end": end_line,
                    "snippet_lines": snippet_lines,
                    "snippet": snippet_lines,
                }
            except Exception:
                return None

        except Exception:
            return None

    def _recall_memory(self, *, query: str, k: int = 8) -> Dict[str, Any]:
        q = (query or "").strip().lower()
        if not q:
            return {"memory_notes": [], "memory_nodes": []}

        # --- 1) text memory notes ---
        notes: List[Dict[str, Any]] = []
        store = getattr(self, "memory_text_store", None)
        raw_notes = store.get("session") if store is not None else []
        if isinstance(raw_notes, list):
            # recent-first scan
            for rec in reversed(raw_notes):
                text = getattr(rec, "text", None)
                note_id = getattr(rec, "note_id", None)
                if not isinstance(text, str):
                    continue
                if q in text.lower():
                    notes.append({"note_id": note_id, "text": text})
                    if len(notes) >= k:
                        break

        # --- 2) graph memory nodes ---
        nodes: List[Dict[str, Any]] = []
        sub = getattr(self, "memory_subgraph", None)
        if sub is not None and hasattr(sub, "nodes"):
            try:
                for nid, n in (sub.nodes or {}).items():
                    if not isinstance(nid, str) or not isinstance(n, dict):
                        continue
                    path = str(n.get("path") or "")
                    summary = str(n.get("summary") or "")
                    kind = str(n.get("kind") or "")
                    hay = f"{nid} {path} {summary} {kind}".lower()
                    if q not in hay:
                        continue
                    score = 0
                    if q in path.lower():
                        score += 2
                    if q in summary.lower():
                        score += 1
                    nodes.append({"id": nid, "path": n.get("path"), "kind": n.get("kind"), "summary": n.get("summary"), "score": score})
            except Exception:
                nodes = []

        nodes.sort(key=lambda x: (-int(x.get("score") or 0), str(x.get("path") or "")))
        return {"memory_notes": notes[:k], "memory_nodes": nodes[:k]}

    def _merge_memory_nodes_into_working(
        self,
        node_ids: Sequence[str],
        *,
        status: str = "recalled",
    ) -> None:
        if not node_ids:
            return
        if not self.memory_subgraph or not self.working_subgraph:
            return

        ids = [nid for nid in node_ids if isinstance(nid, str)]
        if not ids:
            return

        nodes_to_add: List[Dict[str, Any]] = []
        for nid in ids:
            n = self.memory_subgraph.get_node(nid) if hasattr(self.memory_subgraph, "get_node") else (self.memory_subgraph.nodes or {}).get(nid)
            if not isinstance(n, dict) or not n.get("id"):
                # subgraph_store stores nodes keyed by id, so ensure id
                n = dict(n or {})
                n["id"] = nid
            node_copy = dict(n)
            node_copy["status"] = status
            tags = set(node_copy.get("tags") or [])
            tags.add(status)
            node_copy["tags"] = sorted(tags)
            nodes_to_add.append(node_copy)

        if nodes_to_add:
            subgraph_store.add_nodes(self.working_subgraph, nodes_to_add)

        # edges: add memory edges where both endpoints exist in working and at least one endpoint is in ids
        try:
            existing = {
                (e.get("src"), e.get("dst"), e.get("etype") or e.get("type"))
                for e in (self.working_subgraph.edges or [])
                if isinstance(e, dict)
            }
        except Exception:
            existing = set()

        for e in (self.memory_subgraph.edges or []):
            if not isinstance(e, dict):
                continue
            src = e.get("src")
            dst = e.get("dst")
            if not isinstance(src, str) or not isinstance(dst, str):
                continue
            if src not in self.working_subgraph.node_ids or dst not in self.working_subgraph.node_ids:
                continue
            if src not in ids and dst not in ids:
                continue
            key = (src, dst, e.get("etype") or e.get("type"))
            if key in existing:
                continue
            existing.add(key)
            self.working_subgraph.edges.append(dict(e))


    def _merge_repo_nodes_into_working(
            self,
            node_ids: Sequence[str],
            *,
            status: str = "explored",
        ) -> None:
            """Merge selected repo_graph nodes (and induced edges) into the working_subgraph.

            Notes:
              - working_subgraph is the planner-facing cache. Nodes here must carry
                ``memorized: bool`` (default False).
              - This must handle WorkingSubgraph.nodes being either dict[id->node] or list[node].
            """
            if not node_ids:
                return

            # --- normalize working nodes store ---
            nodes_store = getattr(self.working_subgraph, "nodes", {}) or {}
            if isinstance(nodes_store, dict):
                working_nodes_by_id: Dict[str, Dict[str, Any]] = dict(nodes_store)
            else:
                working_nodes_by_id = {}
                for n in (nodes_store or []):
                    if isinstance(n, dict) and isinstance(n.get("id"), str):
                        working_nodes_by_id[n["id"]] = n

            working_edges: List[Dict[str, Any]] = list(getattr(self.working_subgraph, "edges", []) or [])
            working_ids = set(working_nodes_by_id.keys())

            step_now = int(getattr(self, "steps", 0) or 0)

            # --- add/merge nodes ---
            for nid in node_ids:
                if not isinstance(nid, str) or not nid:
                    continue
                repo_node = self._repo_nodes_by_id.get(nid)
                if not isinstance(repo_node, dict):
                    continue

                existing = working_nodes_by_id.get(nid) or {}
                node_copy = dict(repo_node)

                # Preserve extra fields already stored in working (e.g., snippet_lines, memorized flags).
                if isinstance(existing, dict):
                    for k, v in existing.items():
                        if k not in node_copy:
                            node_copy[k] = v

                # v5: memorized state lives on working nodes
                node_copy["memorized"] = bool(node_copy.get("memorized", False))
                if node_copy.get("memorized_at_step") is None and node_copy["memorized"]:
                    node_copy["memorized_at_step"] = existing.get("memorized_at_step") if isinstance(existing, dict) else None

                # Recency metadata (used by planner prompt selection)
                node_copy.setdefault("gp_added_step", existing.get("gp_added_step", step_now) if isinstance(existing, dict) else step_now)
                node_copy["gp_last_touched_step"] = step_now

                # Status/tags for debugging
                node_copy["status"] = status
                tags = set(node_copy.get("tags") or [])
                tags.add(status)
                node_copy["tags"] = sorted(tags)

                working_nodes_by_id[nid] = node_copy
                working_ids.add(nid)

            # --- induced edges among kept node ids ---
            # De-dup edges by (src,dst,type)
            try:
                existing_edges = {
                    (e.get("src"), e.get("dst"), e.get("etype") or e.get("type"))
                    for e in working_edges
                    if isinstance(e, dict)
                }
            except Exception:
                existing_edges = set()

            for e in (self._repo_edges or []):
                if not isinstance(e, dict):
                    continue
                src_id = e.get("src")
                dst_id = e.get("dst")
                if not isinstance(src_id, str) or not isinstance(dst_id, str):
                    continue
                if src_id in working_ids and dst_id in working_ids:
                    key = (src_id, dst_id, e.get("etype") or e.get("type"))
                    if key in existing_edges:
                        continue
                    existing_edges.add(key)
                    working_edges.append(dict(e))

            # Write back
            try:
                self.working_subgraph.nodes = working_nodes_by_id
                self.working_subgraph.edges = working_edges
                self.working_subgraph.node_ids = list(working_nodes_by_id.keys())
            except Exception:
                # Fallback: wrap to ensure schema
                self.working_subgraph = subgraph_store.wrap({"nodes": list(working_nodes_by_id.values()), "edges": working_edges})


    def _merge_working_nodes_into_memory(
        self,
        node_ids: Sequence[str],
        *,
        status: str = "remembered",
    ) -> None:
        """把 working_subgraph 中的若干节点及其边合并进 memory_subgraph。

        注意：subgraph_store.WorkingSubgraph.nodes 是 dict[id -> node]，不是 list。
        """
        ids = [nid for nid in (node_ids or []) if isinstance(nid, str) and nid.strip()]
        if not ids:
            return
        if not self.memory_subgraph or not self.working_subgraph:
            return

        mem_nodes: Dict[str, Dict[str, Any]] = dict(getattr(self.memory_subgraph, "nodes", {}) or {})
        work_nodes: Dict[str, Dict[str, Any]] = dict(getattr(self.working_subgraph, "nodes", {}) or {})

        # 1) nodes
        for nid in ids:
            w_node = work_nodes.get(nid)
            if not isinstance(w_node, dict):
                continue
            node_copy = dict(w_node)
            node_copy["id"] = nid
            node_copy["status"] = status
            tags = set(node_copy.get("tags") or [])
            tags.add(status)
            node_copy["tags"] = sorted(tags)
            mem_nodes[nid] = node_copy

        # 2) edges（仅保留两端都在 memory 的边）
        mem_ids = set(mem_nodes.keys())
        mem_edges: List[Dict[str, Any]] = []
        # existing edges from memory
        for e in (getattr(self.memory_subgraph, "edges", []) or []):
            if not isinstance(e, dict):
                continue
            src = e.get("src")
            dst = e.get("dst")
            if isinstance(src, str) and isinstance(dst, str) and src in mem_ids and dst in mem_ids:
                mem_edges.append(dict(e))

        existing = {
            (e.get("src"), e.get("dst"), e.get("etype") or e.get("type"))
            for e in mem_edges
            if isinstance(e, dict)
        }
        for e in (getattr(self.working_subgraph, "edges", []) or []):
            if not isinstance(e, dict):
                continue
            src = e.get("src")
            dst = e.get("dst")
            if not isinstance(src, str) or not isinstance(dst, str):
                continue
            if src in mem_ids and dst in mem_ids:
                key = (src, dst, e.get("etype") or e.get("type"))
                if key in existing:
                    continue
                existing.add(key)
                mem_edges.append(dict(e))

        # 3) write back in-place
        self.memory_subgraph.nodes = mem_nodes
        self.memory_subgraph.edges = mem_edges
        self.memory_subgraph.node_ids = list(mem_nodes.keys())


