# graph_planner/env/planner_env.py
from __future__ import annotations

import base64
import json
import os
import re
import uuid
import time
import shlex
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


# ----------------------------
# Explore query normalization
# ----------------------------
_QUERY_STOPWORDS = {
    "a","an","the","and","or","of","to","for","in","on","at","by","with","from","as",
    "is","are","was","were","be","been","being","this","that","these","those","it","its",
    "does","do","did","can","could","should","would","may","might","feel","feels","like",
    "bug","missing","expected","suddenly","again","output","inputs","outputs","model","models",
}

def _extract_query_terms(query: Any, max_terms: int = 16) -> List[str]:
    """Normalize explore.find query into a small list of keyword terms.

    Accepts:
      - list[str]: already keywordized
      - str: free-form sentence; we extract code-ish tokens (identifiers / paths / dotted names)
    """
    if query is None:
        return []
    terms: List[str] = []
    seen: set[str] = set()

    def _push(t: str):
        t = (t or "").strip()
        if not t:
            return
        tl = t.lower()
        if tl in _QUERY_STOPWORDS:
            return
        if len(t) <= 1:
            return
        if tl in seen:
            return
        seen.add(tl)
        terms.append(t)

    if isinstance(query, list):
        for v in query:
            if isinstance(v, (str, int, float)):
                _push(str(v))
        return terms[:max_terms]

    if not isinstance(query, str):
        _push(str(query))
        return terms[:max_terms]

    q = query.strip()
    if not q:
        return []

    # backtick spans first
    for m in re.finditer(r"`([^`]{1,80})`", q):
        frag = (m.group(1) or "").strip()
        for t in re.findall(r"[A-Za-z_][A-Za-z0-9_./-]*", frag):
            _push(t)

    # path-like
    for t in re.findall(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+", q):
        _push(t)

    # identifiers / dotted names
    for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", q):
        _push(t)

    # fallback long-ish tokens
    for t in re.findall(r"[A-Za-z0-9_]{4,}", q):
        _push(t)

    return terms[:max_terms]




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


def _append_jsonl(path: str, obj: Mapping[str, Any]) -> None:
    """Append a single JSON object as one line to `path` (best-effort)."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(obj), ensure_ascii=False) + "\n")
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

        # Lightweight, always-on JSONL step log (independent of telemetry backend).
        # This is useful when telemetry_mod is configured as a no-op and you still
        # want per-step traces on disk.
        self._telemetry_root: str = (
            os.environ.get("GP_TELEMETRY_DIR")
            or os.environ.get("GRAPH_PLANNER_TELEMETRY_DIR")
            or "logs/telemetry"
        )
        self._episode_id: Optional[str] = None
        self._episode_dir: Optional[str] = None
        self._steps_jsonl_path: Optional[str] = None

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


        # Keep graph_adapter aligned with repo_graph (mem_candidates / expand).
        try:
            root = "/repo" if getattr(self.box, "_mode", None) == "remote_swe" else (getattr(self, "repo_root_host", None) or self.repo_root_in_container)
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

        # === 2) 初始化 memory_subgraph（现在：每次 reset 都清空） ===
        # 直接 new 一张空的长期记忆图，不再从磁盘加载历史记忆。
        self.memory_subgraph = subgraph_store.new()

        # === 3) 工作图 = 记忆图的拷贝（此时也是空图） ===
        self.working_subgraph = subgraph_store.wrap(self.memory_subgraph.to_json_obj())

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

        # Always-on JSONL step log: logs/telemetry/<run_id>/episodes/<episode_id>/steps.jsonl
        try:
            self._telemetry_jsonl_start_episode(backend_mode=backend_mode)
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

            # best-effort JSONL flush
            try:
                self._telemetry_jsonl_flush()
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
        trimmed: Dict[str, Any] = {}
        query_value = act.query
        if isinstance(query_value, list):
            if len(query_value) > 1:
                trimmed["query"] = {"from": len(query_value), "to": 1}
            query_value = query_value[:1]
        anchors = list(act.anchors or [])
        if len(anchors) > 1:
            trimmed["anchors"] = {"from": len(anchors), "to": 1}
            anchors = anchors[:1]
        # If model omitted anchors for expand/read, fall back to the last selected frontier anchor.
        if act.op in ("expand", "read") and not anchors:
            fa = getattr(self, "frontier_anchor_id", None)
            if fa:
                anchors = [{"id": fa}]
                trimmed.setdefault("anchors", {"from": 0, "to": 1, "source": "frontier_anchor_id"})
        if trimmed:
            info["trimmed"] = trimmed

        if not self.repo_graph:
            info["error"] = "missing-repo-graph"
            return info

        # 数量预算：兼容旧字段 limit，同时支持 total_limit/max_per_anchor/dir_diversity_k
        total_limit = _safe_int(getattr(act, "total_limit", None) or act.limit or 32, 32)
        max_per_anchor = _safe_int(getattr(act, "max_per_anchor", None) or total_limit, total_limit)
        dir_diversity_k = _safe_int(getattr(act, "dir_diversity_k", None) or 4, 4)
        query_terms = _extract_query_terms(getattr(act, "query", None))
        query = " ".join(query_terms).strip()
        if query_terms:
            info["query_terms"] = list(query_terms)


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
                    candidates = self._search_repo_candidates_by_query(
                        query=query, total_limit=total_limit, dir_diversity_k=dir_diversity_k
                    )
                else:
                    candidates = []

            self.last_candidates = candidates
            info["candidates"] = candidates
            # v5: choose exactly one anchor and immediately attach it to working_subgraph
            anchor_id: Optional[str] = None
            # If the model provided an anchor, respect it (after trimming in the parser)
            if anchors:
                a0 = anchors[0]
                if isinstance(a0, dict) and isinstance(a0.get("id"), str):
                    anchor_id = a0.get("id")
                elif isinstance(a0, str):
                    anchor_id = a0
            # Otherwise pick the top candidate as the frontier anchor
            if not anchor_id and candidates:
                c0 = candidates[0] if isinstance(candidates[0], dict) else {}
                if isinstance(c0, dict) and isinstance(c0.get("id"), str):
                    anchor_id = c0.get("id")

            if anchor_id:
                self.frontier_anchor_id = anchor_id
                info["anchor_selected"] = anchor_id
                # ensure anchor node is present in working for the planner to read
                self._merge_repo_nodes_into_working([anchor_id], status="frontier")
                # best-effort: attach snippet lines to the anchor node
                try:
                    n = self._resolve_node(anchor_id) or {}
                    sn = self._read_node_snippet(n) if n else None
                    if sn and isinstance(sn.get("snippet_lines"), list):
                        self.working_subgraph.update_node(anchor_id, {"snippet_lines": sn.get("snippet_lines", [])})
                except Exception:
                    pass
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
        nodes_store = getattr(self.repo_graph, "nodes", {}) or {}
        items: List[Tuple[str, Dict[str, Any]]] = []
        if isinstance(nodes_store, dict):
            for nid, node in nodes_store.items():
                if isinstance(nid, str) and isinstance(node, dict):
                    items.append((nid, node))
        else:
            for node in (nodes_store or []):
                if isinstance(node, dict):
                    nid = node.get("id") or node.get("node_id") or node.get("name")
                    if isinstance(nid, str):
                        items.append((nid, node))
        for nid, node in items:
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

    keep_recent = selector.get("keep_recent_unmemorized")
    try:
        keep_recent = int(keep_recent) if keep_recent is not None else 20
    except Exception:
        keep_recent = 20

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
                base = getattr(self.box, "workdir", None) or "/testbed"
                abs_path = os.path.join(str(base), path)
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
                self.working_subgraph.node_ids = set(working_nodes_by_id.keys())
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
        self.memory_subgraph.node_ids = set(mem_nodes.keys())


    # ------------------------------------------------------------
    # Telemetry helpers (step-level trajectory + terminal excerpts)
    # ------------------------------------------------------------

    def _telemetry_jsonl_start_episode(self) -> None:
        """Create a per-episode JSONL step log directory.

        This is independent of telemetry_mod; it makes debugging/replays possible even
        when telemetry_mod is configured as a no-op.
        """
        try:
            root = str(getattr(self, "_telemetry_root", "") or "").strip()
            if not root:
                return
            run_id = (self.run_id or self.issue.get("run_id") or self.issue_id or "__default__").strip()
            ep = f"{self.issue_id}.{uuid.uuid4().hex[:8]}"
            self._episode_id = ep
            self._episode_dir = os.path.join(root, run_id, "episodes", ep)
            _ensure_dir(self._episode_dir)
            self._steps_jsonl_path = os.path.join(self._episode_dir, "steps.jsonl")

            # Write a small meta marker (optional, best-effort).
            meta_path = os.path.join(self._episode_dir, "episode_meta.json")
            _write_file_text(
                meta_path,
                json.dumps(
                    {
                        "run_id": run_id,
                        "episode_id": ep,
                        "issue_id": self.issue_id,
                        "backend_mode": getattr(self.box, "_mode", None),
                        "created_at": int(time.time()),
                    },
                    indent=2,
                ),
            )

            _append_jsonl(
                self._steps_jsonl_path,
                {
                    "event": "episode.start",
                    "step": 0,
                    "issue_id": self.issue_id,
                    "run_id": run_id,
                    "created_at": int(time.time()),
                },
            )
        except Exception:
            return

    def _telemetry_jsonl_append_step(
        self,
        *,
        action: Mapping[str, Any],
        info: Mapping[str, Any],
        reward: float,
        done: bool,
        dt_ms: Optional[float],
        preview: str,
    ) -> None:
        """Append a compact per-step record to the JSONL log."""
        path = getattr(self, "_steps_jsonl_path", None)
        if not isinstance(path, str) or not path:
            return

        # Key signals for debugging.
        kind = (info.get("kind") if isinstance(info, Mapping) else None) or (action.get("type") if isinstance(action, Mapping) else None)
        op = (info.get("op") if isinstance(info, Mapping) else None) or (action.get("op") if isinstance(action, Mapping) else None)

        # Pull out query/anchors in a resilient way.
        query_used = info.get("query_used") or action.get("query") or ""
        if not query_used and isinstance(info.get("query_terms"), list):
            try:
                query_used = " ".join(str(x) for x in info.get("query_terms") if x)
            except Exception:
                query_used = ""
        anchor_selected = info.get("anchor_selected") or info.get("anchor_expanded")
        skipped_reason = info.get("skipped_reason")

        w_stats = {}
        m_stats = {}
        try:
            w_stats = subgraph_store.stats(getattr(self, "working_subgraph", None))
            m_stats = subgraph_store.stats(getattr(self, "memory_subgraph", None))
        except Exception:
            pass

        record: Dict[str, Any] = {
            "step": int(getattr(self, "steps", 0)),
            "kind": kind,
            "op": op,
            "query_used": query_used,
            "anchor": anchor_selected,
            "reward": float(reward),
            "done": bool(done),
            "dt_ms": dt_ms,
            "working_nodes_total": int(w_stats.get("n_nodes") or 0),
            "working_memorized_count": int(w_stats.get("memorized") or 0),
            "working_unmemorized_count": int(w_stats.get("unmemorized") or 0),
            "memory_nodes_total": int(m_stats.get("n_nodes") or 0),
            "selected_for_memory": info.get("selected_for_memory"),
            "pruned_unmemorized": info.get("pruned_unmemorized"),
            "skipped_reason": skipped_reason,
            "preview": preview,
        }

        _append_jsonl(path, record)

    def _telemetry_obs_summary(self) -> Dict[str, Any]:
        """A lightweight summary of the observation/state for debugging.

        Keep this stable and small to avoid bloating step logs.
        """
        def _safe_stats(sg: Any) -> Dict[str, Any]:
            try:
                return subgraph_store.stats(sg)
            except Exception:
                return {}

        return {
            "step": int(getattr(self, "steps", 0)),
            "max_steps": int(getattr(self, "max_steps", 0) or 0),
            "working": _safe_stats(getattr(self, "working_subgraph", None)),
            "memory": _safe_stats(getattr(self, "memory_subgraph", None)),
            "last_candidates": int(len(getattr(self, "last_candidates", []) or [])),
        }

    def _telemetry_preview_text(self, action: Mapping[str, Any], info: Mapping[str, Any]) -> str:
        """Extract a short, human-readable excerpt for terminal printing/log preview."""
        try:
            kind = str(info.get("kind") or action.get("type") or "").strip().lower()
        except Exception:
            kind = ""

        # Candidates preview (find/expand)
        if kind == "explore":
            op = str(info.get("op") or action.get("op") or "").strip().lower()
            cands = info.get("candidates")
            if isinstance(cands, list) and cands:
                first = cands[0] if isinstance(cands[0], dict) else {}
                nid = first.get("id") if isinstance(first, dict) else None
                path = first.get("path") if isinstance(first, dict) else None
                header = f"{op}: {len(cands)} candidates" + (f" | top={nid}" if nid else "") + (f" | {path}" if path else "")
                # Prefer snippet_lines attached during find
                snippet_lines = None
                if isinstance(first, dict):
                    snippet_lines = first.get("snippet_lines")
                if not snippet_lines and isinstance(info.get("candidate_snippets"), list) and info.get("candidate_snippets"):
                    sn0 = info.get("candidate_snippets")[0]
                    if isinstance(sn0, dict):
                        snippet_lines = sn0.get("snippet_lines")
                if isinstance(snippet_lines, list) and snippet_lines:
                    body = "\n".join(str(x) for x in snippet_lines[:12])
                    return (header + "\n" + body).strip()
                return header
            return f"{op}: 0 candidates"

        if kind == "repair":
            applied = info.get("applied")
            plan = info.get("plan")
            tests = info.get("tests")
            header = f"repair: applied={bool(applied)}"
            parts = [header]
            if isinstance(plan, str) and plan.strip():
                parts.append("plan: " + plan.strip())
            if isinstance(tests, dict):
                # try common keys
                status = tests.get("status") or tests.get("ok") or tests.get("success")
                summary = tests.get("summary") or tests.get("stderr") or tests.get("stdout")
                if status is not None:
                    parts.append(f"tests: {status}")
                if isinstance(summary, str) and summary.strip():
                    parts.append("tests_out: " + summary.strip())
            return "\n".join(parts)

        if kind == "memory":
            intent = info.get("intent") or action.get("intent")
            target = info.get("target") or action.get("target")
            return f"memory: intent={intent} target={target}"

        if kind == "submit":
            return "submit"

        return ""

    def _telemetry_log_step_end(
        self,
        *,
        action: Mapping[str, Any],
        info: Mapping[str, Any],
        reward: float,
        done: bool,
        t0: float,
    ) -> None:
        """Write a step record and optionally print a small excerpt to stdout."""
        dt_ms = None
        try:
            dt_ms = (time.perf_counter() - float(t0)) * 1000.0
        except Exception:
            dt_ms = None

        # Keep a compact preview inside telemetry to make debugging easier.
        preview = ""
        try:
            preview = self._telemetry_preview_text(action, info)
        except Exception:
            preview = ""

        # Reduce payload size: keep full info, but also provide a compact preview.
        payload: Dict[str, Any] = {
            "step": int(getattr(self, "steps", 0)),
            "action": dict(action) if isinstance(action, Mapping) else action,
            "reward": float(reward),
            "done": bool(done),
            "dt_ms": dt_ms,
            "info": dict(info) if isinstance(info, Mapping) else info,
            "preview": preview,
            "obs_summary": self._telemetry_obs_summary(),
        }

        try:
            self.telemetry.log_step(payload)
        except Exception:
            pass

        # Always-on JSONL step logging (stable, easy to parse).
        try:
            self._telemetry_jsonl_append_step(
                action=action,
                info=info,
                reward=reward,
                done=done,
                dt_ms=dt_ms,
                preview=preview,
            )
        except Exception:
            pass

        # Terminal trace (truncated)
        if getattr(self, "_print_ops", False):
            try:
                excerpt = preview or ""
                maxc = int(getattr(self, "_print_excerpt_chars", 360) or 360)
                if len(excerpt) > maxc:
                    excerpt = excerpt[:maxc] + "..."
                kind = (info.get("kind") if isinstance(info, Mapping) else None) or (action.get("type") if isinstance(action, Mapping) else None)
                op = (info.get("op") if isinstance(info, Mapping) else None) or (action.get("op") if isinstance(action, Mapping) else None)
                w = payload.get("obs_summary", {}).get("working", {}) if isinstance(payload.get("obs_summary"), dict) else {}
                m = payload.get("obs_summary", {}).get("memory", {}) if isinstance(payload.get("obs_summary"), dict) else {}
                w_n = w.get("n_nodes") if isinstance(w, dict) else None
                w_mem = w.get("n_memorized") if isinstance(w, dict) else None
                w_un = w.get("n_unmemorized") if isinstance(w, dict) else None
                m_n = m.get("n_nodes") if isinstance(m, dict) else None
                sel = (info.get("selected_for_memory") if isinstance(info, Mapping) else None)
                prune = (info.get("pruned_unmemorized") if isinstance(info, Mapping) else None)
                extra = f" W={w_n}(mem={w_mem},un={w_un}) M={m_n}" if w_n is not None or m_n is not None else ""
                if sel is not None or prune is not None:
                    extra += f" sel={sel or 0} prune={prune or 0}"
                line = f"[gp-env] step={payload['step']} kind={kind} op={op} reward={payload['reward']} done={payload['done']}" + extra
                print(line)
                if excerpt:
                    print(excerpt)
            except Exception:
                pass

        # Episode end marker
        if done:
            try:
                self.telemetry.event("episode.end", {"final_step": int(getattr(self, "steps", 0)), "final_reward": float(reward)})
            except Exception:
                pass
