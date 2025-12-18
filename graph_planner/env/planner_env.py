# graph_planner/env/planner_env.py
from __future__ import annotations

import base64
import json
import os
import re
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
    # Shared protocol/contract error type used by the agent parser.
    from ..agents.common.contracts import ProtocolError
    from ..infra.config import Config, load as load_config
    from ..infra import telemetry as telemetry_mod
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
    from ..infra.test_prioritizer import prioritize_tests
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
    ProtocolError = ValueError  # type: ignore[assignment]


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

        self.steps: int = 0
        self.last_info: Dict[str, Any] = {}
        self.repo_root_in_container: str = sandbox_cfg.workdir or "."
        if getattr(self.box, "_mode", None) == "remote_swe":
            self.repo_root_in_container = "/repo"
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
        if backend_mode == "remote_swe" and hasattr(self.box, "build_repo_graph"):
            # remote_swe 模式：先确保有一份可用的全仓库 repo_graph（通常只需要构建/加载一次）
            try:
                cache_path = self.box.build_repo_graph(self.issue_id)
                from pathlib import Path

                if hasattr(self.box, "_load_repo_graph_jsonl"):
                    repo_json = self.box._load_repo_graph_jsonl(Path(cache_path))
                else:
                    # Fallback: treat as empty graph if loader is unavailable
                    repo_json = {"nodes": [], "edges": []}

                self.repo_graph = subgraph_store.wrap(repo_json)
            except Exception:
                self.repo_graph = subgraph_store.new()

        elif backend_mode == "remote_swe" and hasattr(self.box, "build_issue_subgraph"):
            # 兼容旧实现：如果没有 build_repo_graph，则退化为 build_issue_subgraph（但仍按“全仓库图”对待）
            try:
                repo_json = self.box.build_issue_subgraph(self.issue_id)
                self.repo_graph = subgraph_store.wrap(repo_json)
            except Exception:
                self.repo_graph = subgraph_store.new()

            # NOTE:
            # repo_graph.nodes 的具体形态可能是：
            #   - list[dict]   (每个元素是 {id:..., ...})
            #   - dict[str,dict] (id -> node)
            #   - list[str]   (仅包含 node id)
            # 为了避免 reset 阶段因为 "str has no attribute get" 直接崩溃，这里做一次兼容。

            nodes_obj = getattr(self.repo_graph, "nodes", [])
            if isinstance(nodes_obj, dict):
                node_iter = nodes_obj.values()
            else:
                node_iter = nodes_obj or []

            for n in node_iter:
                # 1) 最常见：dict node
                if isinstance(n, dict):
                    nid = n.get("id")
                    if isinstance(nid, str) and nid:
                        self._repo_nodes_by_id[nid] = n
                    continue

                # 2) 退化：直接给了字符串 id
                if isinstance(n, str) and n:
                    self._repo_nodes_by_id[n] = {"id": n}
                    continue

                # 3) 其它类型：忽略
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
                    "obs_summary": self._telemetry_obs_summary(),
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
        # Global stop: enforce max_steps budget (used by eval_engine/datasets).
        if getattr(self, "max_steps", None) is not None and self.steps >= int(self.max_steps):
            if not done:
                info["stop_reason"] = "max_steps"
            done = True
        info["done"] = bool(done)

        obs = self._obs()
        try:
            self._telemetry_log_step_end(action=action, info=info, reward=reward, done=done, t0=_t0)
        except Exception:
            pass
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
                if query:
                    candidates = self._search_repo_candidates_by_query(
                        query=query, total_limit=total_limit, dir_diversity_k=dir_diversity_k
                    )
                else:
                    candidates = []

            self.last_candidates = candidates
            info["candidates"] = candidates
            # auto-attach snippets for candidates (explore has no separate read op)
            cand_snippets = []
            for c in candidates:
                try:
                    nid = c.get("id")
                    if isinstance(nid, str):
                        n = self._resolve_node(nid) or {}
                        sn = self._read_node_snippet(n) if n else None
                        if sn:
                            c["snippet_lines"] = sn.get("snippet_lines", [])
                            cand_snippets.append(sn)
                            try:
                                self.working_subgraph.update_node(nid, {"snippet_lines": sn.get("snippet_lines", [])})
                            except Exception:
                                pass
                except Exception:
                    continue
            info["candidate_snippets"] = cand_snippets
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

        info["error"] = f"unknown-explore-op:{act.op}"
        return info


    def _search_repo_candidates_by_query(
        self,
        *,
        query: str,
        total_limit: int,
        dir_diversity_k: int = 4,
    ) -> List[Dict[str, Any]]:
        """Search repo_graph nodes when explore.find is called without anchors.

        The repo_graph is a coarse code graph (files/symbols/spans). When the model provides
        a free-form query, we do a lightweight lexical match against node id/name/path.

        Query conventions (best-effort, optional):
        - "path:<substr>"   : prioritize path matches
        - "symbol:<name>"  : prioritize symbol/id/name matches
        """
        q = (query or "").strip()
        if not q or not self.repo_graph:
            return []

        q_lower = q.lower()

        mode = "free"
        payload = q
        if q_lower.startswith("path:"):
            mode, payload = "path", q[5:].strip()
        elif q_lower.startswith("symbol:"):
            mode, payload = "symbol", q[7:].strip()

        # Tokenize payload: keep identifiers and path-ish tokens
        toks = re.findall(r"[A-Za-z0-9_./:-]+", payload)
        toks = [t.lower() for t in toks if t and len(t) >= 2]

        def norm_dir(p: str) -> str:
            try:
                pp = Path(p)
                return str(pp.parent.as_posix())
            except Exception:
                return ""

        scored: List[Dict[str, Any]] = []
        nodes = getattr(self.repo_graph, "nodes", {}) or {}
        for nid, node in nodes.items():
            if not isinstance(node, dict):
                continue
            nid_s = str(nid or node.get("id") or "")
            name = str(node.get("name") or node.get("symbol") or "")
            path = str(node.get("path") or "")
            kind = str(node.get("kind") or "").lower()

            hay = (nid_s + " " + name + " " + path).lower()
            if not hay:
                continue

            score = 0.0
            reasons: List[str] = []

            # Strong exact/substring match on whole payload
            p_lower = payload.lower()
            if p_lower and p_lower in hay:
                score += 3.0
                reasons.append("payload")

            # Token matches
            for t in toks:
                if t in hay:
                    score += 1.0
                    reasons.append(t)

            # Mode-specific boosts
            if mode == "path" and payload and payload.lower() in path.lower():
                score += 3.0
                reasons.append("path")
            if mode == "symbol" and payload:
                pl = payload.lower()
                if pl == nid_s.lower() or pl == name.lower():
                    score += 4.0
                    reasons.append("symbol_exact")
                elif pl in nid_s.lower() or pl in name.lower():
                    score += 2.0
                    reasons.append("symbol")

            if score <= 0:
                continue

            scored.append(
                {
                    "id": nid_s,
                    "kind": kind,
                    "path": node.get("path"),
                    "span": node.get("span"),
                    "degree": int(node.get("degree") or 0),
                    "from_anchor": False,
                    "score": float(score),
                    "reasons": list(dict.fromkeys(reasons))[:8],
                    "name": name or None,
                }
            )

        scored.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

        # Directory diversity: round-robin by parent dir to avoid over-concentration.
        if dir_diversity_k and dir_diversity_k > 0:
            buckets: Dict[str, List[Dict[str, Any]]] = {}
            order: List[str] = []
            for c in scored:
                d = norm_dir(str(c.get("path") or ""))
                if d not in buckets:
                    buckets[d] = []
                    order.append(d)
                buckets[d].append(c)

            mixed: List[Dict[str, Any]] = []
            rounds = 0
            while len(mixed) < total_limit:
                progressed = False
                for d in list(order):
                    if not buckets.get(d):
                        continue
                    # Take up to k per dir overall via round-robin rounds
                    if rounds < dir_diversity_k:
                        mixed.append(buckets[d].pop(0))
                        progressed = True
                        if len(mixed) >= total_limit:
                            break
                    else:
                        # After k rounds, just fill greedily
                        while buckets[d] and len(mixed) < total_limit:
                            mixed.append(buckets[d].pop(0))
                        progressed = True
                if not progressed:
                    break
                rounds += 1
            scored = mixed

        return scored[:total_limit]

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
            elif act.target == "observation":
                sel = None
                if isinstance(selector, str):
                    sel = selector
                elif isinstance(sel_map.get("note_id"), (str, int)):
                    sel = str(sel_map.get("note_id"))
                sel = (sel or "latest").strip()
                result = text_memory.memory_delete(self.memory_state, "observation", scope, sel) or {}
            else:
                result = {"ok": False, "error": "unknown-target", "msg": f"unknown memory target: {act.target}"}
        else:
            result = {"ok": False, "error": "unknown-intent", "msg": f"unknown memory intent: {act.intent}"}

        info.update(result or {})

        # 保持 graph_store 与 memory_subgraph 同步
        if getattr(self, "memory_graph_store", None) is not None:
            try:
                self.memory_graph_store.subgraph = self.memory_subgraph
            except Exception:
                pass

        info["subgraph_stats"] = subgraph_store.stats(self.memory_subgraph)
        # Trajectory-only memory/working subgraphs: default is NOT to persist to disk.
        if os.environ.get("GP_PERSIST_SUBGRAPHS") in ("1", "true", "True"):
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

    def _read_node_snippet(self, node: Union[Mapping[str, Any], str]) -> Optional[Dict[str, Any]]:
        """Read snippet lines for a node.

        - local backends: read from evaluator filesystem
        - remote_swe: read inside remote container (repo root fixed at /repo)
        """
        try:
            if isinstance(node, str):
                node = self._resolve_node(node) or {}
            if not isinstance(node, Mapping):
                return None

            path = (node.get("path") or "").lstrip("/")
            if not path:
                return None

            span = node.get("span") if isinstance(node, Mapping) else {}
            span_start = span.get("start") if isinstance(span, Mapping) else None
            span_end = span.get("end") if isinstance(span, Mapping) else None
            start_line = int(node.get("start_line") or span_start or 1)
            end_line = int(node.get("end_line") or span_end or start_line)

            # remote_swe: run inside container
            if getattr(self.box, "_mode", None) == "remote_swe":
                abs_path = os.path.join("/repo", path)
                req = {"path": abs_path, "start": start_line, "end": end_line}
                b64 = base64.b64encode(json.dumps(req).encode("utf-8")).decode("ascii")
                py = r"""
import base64, json, sys
req = json.loads(base64.b64decode(sys.argv[1]).decode('utf-8'))
p = req['path']; s = int(req.get('start',1)); e = int(req.get('end',s))
with open(p,'r',encoding='utf-8',errors='replace') as f:
    lines = f.read().splitlines()
snippet = lines[max(0,s-1):max(0,e)]
print(json.dumps({'snippet_lines': snippet}))
"""
                cmd = "python -c " + shlex.quote(py) + " " + shlex.quote(b64)
                out, rc = self.box.run(cmd, timeout=60)
                if rc != 0:
                    return None
                try:
                    resp = json.loads(out.strip().splitlines()[-1])
                    snippet_lines = resp.get("snippet_lines") or []
                except Exception:
                    snippet_lines = []
                return {
                    "id": node.get("id"),
                    "path": path,
                    "start": start_line,
                    "end": end_line,
                    "snippet_lines": snippet_lines,
                    "snippet": snippet_lines,  # backward compat
                }

            # local
            abs_path = os.path.join(self.repo_root_in_container, path)
            text = _read_file_text(abs_path)
            if not text:
                return None
            lines2 = text.splitlines()
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
    def _recall_memory(self, *, query: str, k: int = 8) -> Dict[str, Any]:
        q = (query or "").strip().lower()
        if not q:
            return {"memory_notes": [], "memory_nodes": []}

        # --- 1) text memory notes ---
        notes: List[Dict[str, Any]] = []
        store = getattr(self, "memory_text_store", None)
        raw_notes = None
        if store is not None and hasattr(store, "_notes"):
            # NoteTextStore: _notes[scope] -> List[_NoteRecord]
            try:
                raw_notes = getattr(store, "_notes", {}).get("session") or []
            except Exception:
                raw_notes = []
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
            existing = working_nodes_by_id.get(nid)
            node_copy = dict(repo_node)
            # Preserve any extra fields already stored in working_subgraph (e.g., snippet_lines/text).
            if isinstance(existing, dict):
                for k, v in existing.items():
                    if k not in node_copy:
                        node_copy[k] = v
            # Recency metadata (used by planner prompt selection)
            if isinstance(existing, dict) and "gp_added_step" in existing:
                node_copy["gp_added_step"] = existing.get("gp_added_step")
            else:
                node_copy["gp_added_step"] = getattr(self, "steps", 0)
            node_copy["gp_last_touched_step"] = getattr(self, "steps", 0)
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
        self.memory_subgraph.node_ids = set(mem_nodes.keys())

        # keep WorkingGraphStore in sync (if present)
        if getattr(self, "memory_graph_store", None) is not None:
            try:
                self.memory_graph_store.subgraph = self.memory_subgraph
            except Exception:
                pass
