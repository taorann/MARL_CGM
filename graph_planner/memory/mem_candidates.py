# 2025-10-22 memory hardening
# -*- coding: utf-8 -*-
"""
MemCandidates builder for Step 3.1

从现有子图与锚点出发做 1-hop 扩展，生成用于记忆操作决策的候选集：
- 去重、过滤已在子图中的节点
- 计算 explainable 的 score 与 reasons
- 施加目录多样性与配额约束
- 输出按 score 降序的 candidates 列表（TypedDict）

设计要点：
- 采用“无向邻接”的 1-hop（graph_adapter 内已封装）
- 同文件/同目录启发式加分，t-file 适度放大
- 目录多样性（round-robin）避免单一路径淹没
- 与 RepoGraph 的仓库级导航思想、Memory-R1 的结构化记忆操作兼容
"""
from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, List, Dict, Set, Tuple

from .types import Anchor, Candidate, Node, SubgraphLike
from . import graph_adapter

# ----------------------------
# Query term extraction (shared)
# ----------------------------
_QUERY_STOPWORDS = {
    "a","an","the","and","or","of","to","for","in","on","at","by","with","from","as",
    "is","are","was","were","be","been","being","this","that","these","those","it","its",
    "does","do","did","can","could","should","would","may","might","feel","feels","like",
    "bug","missing","expected","suddenly","again","output","inputs","outputs","model","models",
}


def extract_query_terms(query: Any, max_terms: int = 16) -> List[str]:
    """Normalize an explore query to a small list of keyword-ish terms.

    Accepts:
      - list[str]/list[Any]: will be stringified and de-duplicated
      - str: free-form sentence; we extract code-ish tokens (identifiers / paths / dotted names)
    """
    if query is None:
        return []

    terms: List[str] = []
    seen = set()

    def push(t: str) -> None:
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
                push(str(v))
        return terms[:max_terms]

    if not isinstance(query, str):
        push(str(query))
        return terms[:max_terms]

    q = query.strip()
    if not q:
        return []

    # backtick spans first
    for m in re.finditer(r"`([^`]{1,80})`", q):
        frag = (m.group(1) or "").strip()
        for t in re.findall(r"[A-Za-z_][A-Za-z0-9_./-]*", frag):
            push(t)

    # path-like
    for t in re.findall(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+", q):
        push(t)

    # identifiers / dotted names
    for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", q):
        push(t)

    # fallback long-ish tokens
    for t in re.findall(r"[A-Za-z0-9_]{4,}", q):
        push(t)

    return terms[:max_terms]






# ----------------------------
# Query processing (strong/weak)
# ----------------------------
_GENERIC_WEAK_TERMS: set[str] = {
    # very common programming / repo words that tend to explode recall
    "matrix","vector","array","tensor","list","dict","set","map",
    "class","method","function","func","module","file","path","line","code",
    "test","tests","testing","unit","example","examples","doc","docs","documentation",
    "fix","bug","issue","error","fail","failed","failing","failure",
    "check","checks","verify","validation","validate",
    "build","run","runs","running","start","stop",
    "add","remove","delete","update","change",
    "read","write","open","close","load","save",
    "use","using","used",
    "python","py","pip","conda",
}


def process_query(
    query_raw: str,
    *,
    max_strong: int = 8,
    max_weak: int = 8,
) -> Dict[str, Any]:
    """Best-effort query normalization that avoids recall explosions.

    - Keep a single *raw* query string for logging / model context.
    - Derive **strong_terms** for stage-1 recall (code-like / specific).
    - Derive **weak_terms** for stage-2 rerank/boost (generic).

    Notes:
      * This function is intentionally heuristic and conservative.
      * All outputs are bounded to keep downstream prompts small.
    """
    q = (query_raw or "").strip()
    if not q:
        return {"query_raw": "", "strong_terms": [], "weak_terms": []}

    # Prefer explicitly quoted / backticked segments as strong hints.
    quoted_terms: List[str] = []
    for g1, g2, g3 in re.findall(r"`([^`]+)`|\"([^\"]+)\"|'([^']+)'", q):
        seg = (g1 or g2 or g3 or "").strip()
        if seg:
            quoted_terms.append(seg)

    # Extract terms from the full query and quoted segments.
    terms: List[str] = []
    terms.extend(extract_query_terms(q))
    for seg in quoted_terms:
        terms.extend(extract_query_terms(seg))
        if seg not in terms:
            terms.append(seg)

    # De-dup while preserving order.
    seen: set[str] = set()
    deduped: List[str] = []
    for t in terms:
        t = (t or "").strip()
        if not t:
            continue
        tl = t.lower()
        if tl in seen:
            continue
        seen.add(tl)
        deduped.append(t)

    strong: List[str] = []
    weak: List[str] = []

    def _is_code_like(tok: str) -> bool:
        if any(s in tok for s in ("/", "\\", ".", "_", "::", ":", "#")):
            return True
        if re.search(r"\d", tok):
            return True
        if re.match(r"[A-Za-z]+[A-Z][A-Za-z0-9]*", tok):
            return True
        return False

    for t in deduped:
        tl = t.lower()
        if tl in _QUERY_STOPWORDS:
            continue
        if _is_code_like(t) or len(t) >= 12:
            strong.append(t)
            continue
        if tl in _GENERIC_WEAK_TERMS:
            weak.append(t)
            continue
        weak.append(t)

    # Also consider a few unquoted word tokens (weak-only) but keep them bounded.
    for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", q.lower()):
        if w in _QUERY_STOPWORDS or w in _GENERIC_WEAK_TERMS:
            continue
        if w not in {s.lower() for s in strong} and w not in {x.lower() for x in weak}:
            weak.append(w)

    strong = strong[: max(0, int(max_strong or 0))]
    weak = weak[: max(0, int(max_weak or 0))]

    return {
        "query_raw": q,
        "strong_terms": strong,
        "weak_terms": [w for w in weak if w.lower() not in {s.lower() for s in strong}],
    }


def search_repo_candidates_by_query(
    repo_graph: SubgraphLike,
    *,
    query: str,
    total_limit: int,
    dir_diversity_k: int = 4,
) -> List[Candidate]:
    """Search repo_graph nodes when explore.find is called without anchors.

    Compared to a naive bag-of-words, we:
      - derive strong/weak terms
      - stage-1 recall uses strong terms (or weak fallback)
      - weak terms only boost candidates that already matched stage-1
    """
    q = (query or "").strip()
    if not q or not repo_graph:
        return []

    q_lower = q.lower()
    mode = "free"
    payload = q
    if q_lower.startswith("path:"):
        mode, payload = "path", q[5:].strip()
    elif q_lower.startswith("symbol:"):
        mode, payload = "symbol", q[7:].strip()

    qp = process_query(payload, max_strong=8, max_weak=8)
    payload_norm = (qp.get("query_raw") or payload).strip()
    payload_lower = payload_norm.lower()

    strong_terms: List[str] = list(qp.get("strong_terms") or [])
    weak_terms: List[str] = list(qp.get("weak_terms") or [])

    # Stage-1 recall terms: strong preferred, otherwise a small weak fallback.
    recall_terms: List[str] = strong_terms if strong_terms else weak_terms[:4]
    recall_terms_l = [t.lower() for t in recall_terms if t]
    strong_l = {t.lower() for t in strong_terms if t}
    weak_l = {t.lower() for t in weak_terms if t}

    def norm_dir(p: str) -> str:
        try:
            return str(PurePosixPath(_norm_posix(p)).parent)
        except Exception:
            return ""

    scored: List[Candidate] = []
    nodes_store = getattr(repo_graph, "nodes", {}) or {}

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
        nid_s = str(nid or node.get("id") or "")
        name = str(node.get("name") or node.get("symbol") or "")
        path = str(node.get("path") or "")
        kind = str(node.get("kind") or "").lower()

        hay_all = (nid_s + " " + name + " " + path).lower()
        if not hay_all:
            continue

        hay = hay_all
        if mode == "path":
            hay = (path or "").lower()
        elif mode == "symbol":
            hay = (nid_s + " " + name).lower()

        score = 0.0
        reasons: List[str] = []
        match_strong = 0
        match_weak = 0

        # Strong substring match on the whole payload
        if payload_lower and payload_lower in hay:
            score += 4.0
            reasons.append("payload")

        # Stage-1 recall matches (must hit at least one)
        recall_hit = False
        for t in recall_terms_l:
            if t and t in hay:
                recall_hit = True
                if t in strong_l:
                    score += 3.0
                    match_strong += 1
                else:
                    score += 1.2
                    match_weak += 1
                reasons.append(t)

        if not recall_hit:
            continue

        # Stage-2 weak boosts (only after recall hit)
        for t in weak_l:
            if t and t not in reasons and t in hay_all:
                score += 0.35
                match_weak += 1
                if len(reasons) < 8:
                    reasons.append(t)

        # Mode-specific boosts
        if mode == "path" and payload_lower and payload_lower in (path or "").lower():
            score += 2.0
            reasons.append("path")
        if mode == "symbol" and payload_lower:
            pl = payload_lower
            if pl == nid_s.lower() or pl == name.lower():
                score += 3.0
                reasons.append("symbol_exact")
            elif pl in nid_s.lower() or pl in name.lower():
                score += 1.5
                reasons.append("symbol")

        if score <= 0:
            continue

        scored.append(
            {
                "id": nid_s,
                "kind": kind,
                "path": (node.get("path") or None),
                "span": node.get("span"),
                "degree": int(node.get("degree") or 0),
                "from_anchor": False,
                "score": float(score),
                "reasons": list(dict.fromkeys(reasons))[:8],
                "name": (name or None),
                "match_strong": int(match_strong),
                "match_weak": int(match_weak),
            }
        )

    scored.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

    # Directory diversity: round-robin by parent dir to avoid over-concentration.
    if dir_diversity_k and dir_diversity_k > 0 and scored:
        buckets: Dict[str, List[Candidate]] = {}
        order: List[str] = []
        for c in scored:
            d = norm_dir(str(c.get("path") or ""))
            if d not in buckets:
                buckets[d] = []
                order.append(d)
            buckets[d].append(c)

        mixed: List[Candidate] = []
        rounds = 0
        while len(mixed) < total_limit:
            progressed = False
            for d in list(order):
                if not buckets.get(d):
                    continue
                if rounds < dir_diversity_k:
                    mixed.append(buckets[d].pop(0))
                    progressed = True
                    if len(mixed) >= total_limit:
                        break
                else:
                    while buckets[d] and len(mixed) < total_limit:
                        mixed.append(buckets[d].pop(0))
                    progressed = True
            if not progressed:
                break
            rounds += 1
        scored = mixed

    return scored[:total_limit]


# --------------------------
# 可调权重（默认足够保守）
# --------------------------
@dataclass(frozen=True)
class CandidateScoringWeights:
    w_from_anchor: float = 1.0
    w_degree: float = 0.6
    w_same_file: float = 0.8
    w_same_dir: float = 0.3
    w_test_file: float = 0.25
    w_novelty: float = 0.5  # 不在子图中的适度加分


@dataclass(frozen=True)
class CandidateSelectionBudget:
    max_per_anchor: int = 50          # 每个锚点最多保留的 1-hop 节点
    total_limit: int = 200            # 全局候选上限（进入 3.2 决策的数量）
    dir_diversity_k: int = 3          # 每个目录至少保留的 top-k（round-robin 选取）
    prefer_test_files: bool = True    # 轻度偏好 t-file（单测上下文）


def _norm_posix(path: str | None) -> str:
    return (path or "").replace("\\", "/")


def _is_test_file(node: Node) -> bool:
    kind = (node.get("kind") or "").lower()
    path = _norm_posix(node.get("path") or "").lower()
    if kind == "t-file":
        return True
    return "test" in path or "/tests/" in path or path.endswith("_test.py") or path.endswith("test.py")


def _dirname(node: Node) -> str:
    path = _norm_posix(node.get("path") or "")
    try:
        return str(PurePosixPath(path).parent)
    except Exception:
        return ""


def _filename(node: Node) -> str:
    return _norm_posix(node.get("path") or "").split("/")[-1]


def _score_node(
    node: Node,
    anchor_paths: Set[str],
    in_subgraph_ids: Set[str],
    weights: CandidateScoringWeights,
    from_anchor_flag: bool,
) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 0.0

    # 来源标记
    if from_anchor_flag:
        score += weights.w_from_anchor
        reasons.append("from_anchor")

    # 度（若 adapter 已给出）
    degree = int(node.get("degree") or 0)
    if degree > 0:
        score += weights.w_degree * min(degree / 8.0, 1.0)  # 归一化到 [0,1]
        reasons.append(f"degree={degree}")

    # 同文件
    path = _norm_posix(node.get("path") or "")
    if path and path in anchor_paths:
        score += weights.w_same_file
        reasons.append("same_file")

    # 同目录
    if path:
        dirnames = {p.rsplit("/", 1)[0] for p in anchor_paths if "/" in p}
        my_dir = path.rsplit("/", 1)[0] if "/" in path else ""
        if my_dir and my_dir in dirnames:
            score += weights.w_same_dir
            reasons.append("same_dir")

    # t-file
    if _is_test_file(node):
        score += weights.w_test_file
        reasons.append("t_file")

    # 新颖性（不在现有子图）
    if node.get("id") not in in_subgraph_ids:
        score += weights.w_novelty
        reasons.append("novel")

    return score, reasons


def _to_candidate(node: Node, score: float, reasons: List[str], from_anchor: bool) -> Candidate:
    path = _norm_posix(node.get("path")) or None
    return {
        "id": node.get("id"),
        "kind": (node.get("kind") or "").lower(),
        "path": path,
        "span": node.get("span"),
        "degree": int(node.get("degree") or 0),
        "from_anchor": bool(from_anchor),
        "score": float(score),
        "reasons": reasons,
        "name": node.get("name"),
    }


def build_mem_candidates(
    subgraph: SubgraphLike,
    anchors: Iterable[Anchor | Node],
    *,
    max_nodes_per_anchor: int = 50,
    total_limit: int = 200,
    dir_diversity_k: int = 3,
    weights: CandidateScoringWeights | None = None,
) -> List[Candidate]:
    """
    主入口：生成用于 3.2 决策头的候选列表（按 score 降序）。

    参数
    ----
    subgraph: SubgraphLike
        当前工作子图（需支持 `iter_node_ids()` / `contains(node_id)` / `get_node(node_id)`）
    anchors: Iterable[Anchor | Node]
        锚点（可为 Anchor 或已解析的 Node）
    max_nodes_per_anchor: int
        每个锚点最多保留的 1-hop 候选
    total_limit: int
        返回的候选全局上限
    dir_diversity_k: int
        目录多样性：同一目录将以 round-robin 方式至少保留 top-k
    weights: CandidateScoringWeights
        打分权重；不传则使用默认

    返回
    ----
    List[Candidate]
        候选节点列表（含 explainable 的 reasons）
    """
    weights = weights or CandidateScoringWeights()

    # 1) 解析锚点对应的路径集合（用于 same_file/same_dir）
    anchor_nodes: List[Node] = []
    for a in anchors:
        if isinstance(a, dict) and "id" in a:  # Node
            anchor_nodes.append(a)  # type: ignore
        else:
            # Anchor ←→ Node 解析由 adapter 负责
            resolved = graph_adapter.find_nodes_by_anchor(a)  # type: ignore
            anchor_nodes.extend(resolved)
    anchor_paths: Set[str] = {
        _norm_posix(n.get("path"))
        for n in anchor_nodes
        if n.get("path")
    }

    # 2) 获取现有子图节点 id 集（避免重复）
    in_subgraph_ids: Set[str] = set(subgraph.iter_node_ids())  # type: ignore

    # 3) 针对每个锚点做 1-hop 扩展并初步打分
    raw_bucket: Dict[str, Candidate] = {}
    for an in anchor_nodes:
        # one_hop_expand: 使用“无向邻接”，内部包含正/反向边 + 同文件/同目录启发式
        neighbors: List[Node] = graph_adapter.one_hop_expand(
            subgraph=subgraph,
            anchors=[an],
            max_nodes=max_nodes_per_anchor,
        )

        for nb in neighbors:
            nid = nb.get("id")
            if not nid:
                continue
            # 过滤：保持唯一
            already = raw_bucket.get(nid)
            score, reasons = _score_node(
                nb,
                anchor_paths=anchor_paths,
                in_subgraph_ids=in_subgraph_ids,
                weights=weights,
                from_anchor_flag=True,
            )
            cand = _to_candidate(nb, score, reasons, from_anchor=True)
            if (already is None) or (cand["score"] > already["score"]):
                raw_bucket[nid] = cand

    # 4) 目录多样性与全局配额
    # 先按分数降序分桶，再用 round-robin 每个目录取前 k，之后补齐到 total_limit
    by_dir: Dict[str, List[Candidate]] = defaultdict(list)
    for c in sorted(raw_bucket.values(), key=lambda x: x["score"], reverse=True):
        by_dir[_dirname(c)].append(c)

    # 每目录保底 k 的 round-robin
    selected: List[Candidate] = []
    queues = {d: deque(lst[:dir_diversity_k]) for d, lst in by_dir.items()}
    while queues and len(selected) < total_limit:
        for d in list(queues.keys()):
            if queues[d]:
                selected.append(queues[d].popleft())
                if len(selected) >= total_limit:
                    break
            else:
                queues.pop(d, None)

    # 若仍未满额，按分数全局补齐
    if len(selected) < total_limit:
        picked_ids = {c["id"] for c in selected}
        rest = [c for c in sorted(raw_bucket.values(), key=lambda x: x["score"], reverse=True)
                if c["id"] not in picked_ids]
        need = total_limit - len(selected)
        selected.extend(rest[:need])

    # 5) 最终排序与输出
    selected.sort(key=lambda x: x["score"], reverse=True)
    return selected
