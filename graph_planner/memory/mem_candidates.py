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

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, List, Dict, Set, Tuple, Optional

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
# Query DSL parsing for explore.find
# ----------------------------


@dataclass(frozen=True)
class QuerySpec:
    raw: str
    symbol_terms: Tuple[str, ...] = ()
    path_terms: Tuple[str, ...] = ()
    must_terms: Tuple[str, ...] = ()
    must_not_terms: Tuple[str, ...] = ()
    phrase_terms: Tuple[str, ...] = ()
    free_terms: Tuple[str, ...] = ()


_SEARCH_CACHE: Dict[int, Dict[str, Any]] = {}


def _tokenize_free_text(s: str, *, max_terms: int = 24) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9_./:-]+", s or "")
    out: List[str] = []
    seen: Set[str] = set()
    for t in toks:
        tl = (t or "").strip().lower()
        if not tl or len(tl) < 2:
            continue
        if tl in _QUERY_STOPWORDS:
            continue
        if tl in seen:
            continue
        seen.add(tl)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


def parse_query_dsl(q: str) -> QuerySpec:
    """Parse a lightweight DSL for explore.find.

    Supported (order-insensitive):
      - symbol:<term>   bias toward node name/id matches
      - path:<term>     bias toward node path matches
      - +term           required term (must appear)
      - -term           forbidden term (must not appear)
      - "phrase" / 'phrase' required-to-score phrase match (high boost)

    The function is best-effort and intentionally permissive.
    """
    raw = (q or "").strip()
    if not raw:
        return QuerySpec(raw="")

    # Extract quoted phrases first (keep as phrases; also remove from the free text).
    phrase_terms: List[str] = []
    tmp = raw
    for pat in (r"\"([^\"]{1,120})\"", r"'([^']{1,120})'"):
        for m in re.finditer(pat, raw):
            ph = (m.group(1) or "").strip()
            if ph:
                phrase_terms.append(ph)
        tmp = re.sub(pat, " ", tmp)

    # Normalise separators.
    tmp = tmp.replace("\n", " ").replace("\t", " ")

    symbol_terms: List[str] = []
    path_terms: List[str] = []
    must_terms: List[str] = []
    must_not_terms: List[str] = []

    tokens = tmp.split()
    i = 0
    free_parts: List[str] = []
    while i < len(tokens):
        tok = tokens[i]
        tl = tok.lower()
        if tl.startswith("symbol:"):
            val = tok.split(":", 1)[1].strip()
            if not val and i + 1 < len(tokens):
                i += 1
                val = tokens[i].strip()
            if val:
                symbol_terms.append(val)
            i += 1
            continue
        if tl.startswith("path:"):
            val = tok.split(":", 1)[1].strip()
            if not val and i + 1 < len(tokens):
                i += 1
                val = tokens[i].strip()
            if val:
                path_terms.append(val)
            i += 1
            continue

        if tok.startswith("+") and len(tok) > 1:
            must_terms.append(tok[1:])
            i += 1
            continue
        if tok.startswith("-") and len(tok) > 1:
            must_not_terms.append(tok[1:])
            i += 1
            continue

        free_parts.append(tok)
        i += 1

    free_text = " ".join(free_parts)
    free_terms = _tokenize_free_text(free_text, max_terms=24)
    # Also tokenise the directive payloads: helps when user writes `symbol:foo.bar` etc.
    # But keep the original directive payloads separately for exact scoring.
    must_terms = [t for t in _tokenize_free_text(" ".join(must_terms), max_terms=12)]
    must_not_terms = [t for t in _tokenize_free_text(" ".join(must_not_terms), max_terms=12)]

    return QuerySpec(
        raw=raw,
        symbol_terms=tuple(symbol_terms[:3]),
        path_terms=tuple(path_terms[:3]),
        must_terms=tuple(must_terms[:6]),
        must_not_terms=tuple(must_not_terms[:6]),
        phrase_terms=tuple(phrase_terms[:6]),
        free_terms=tuple(free_terms[:24]),
    )


def _get_repo_items_cache(repo_graph: SubgraphLike) -> Dict[str, Any]:
    """Build/return a lightweight per-process cache for repo_graph search."""
    key = id(repo_graph)
    cached = _SEARCH_CACHE.get(key)
    if cached is not None:
        return cached

    nodes_store = getattr(repo_graph, "nodes", {}) or {}
    # items: (nid, node, name, path, kind, hay, dir)
    items: List[Tuple[str, Dict[str, Any], str, str, str, str, str]] = []

    def _add(nid: str, node: Dict[str, Any]) -> None:
        nid_s = str(nid or node.get("id") or "")
        name = str(node.get("name") or node.get("symbol") or "")
        path = str(node.get("path") or "")
        kind = str(node.get("kind") or "").lower()
        hay = (nid_s + " " + name + " " + path).lower()
        d = ""
        try:
            d = str(PurePosixPath(_norm_posix(path)).parent)
        except Exception:
            d = ""
        items.append((nid_s, node, name, path, kind, hay, d))

    if isinstance(nodes_store, dict):
        for nid, node in nodes_store.items():
            if isinstance(nid, str) and isinstance(node, dict):
                _add(nid, node)
    else:
        for node in (nodes_store or []):
            if isinstance(node, dict):
                nid = node.get("id") or node.get("node_id") or node.get("name")
                if isinstance(nid, str):
                    _add(nid, node)

    out = {"items": items, "N": len(items), "df": {}}
    _SEARCH_CACHE[key] = out
    return out


def _df_for_token(cache: Dict[str, Any], token: str) -> int:
    """Compute df(token) lazily and cache it."""
    df_cache: Dict[str, int] = cache.get("df") or {}
    tl = (token or "").lower()
    if not tl:
        return 0
    if tl in df_cache:
        return int(df_cache[tl])
    cnt = 0
    for (_nid, _node, _name, _path, _kind, hay, _d) in cache.get("items", []):
        if tl in hay:
            cnt += 1
    df_cache[tl] = cnt
    cache["df"] = df_cache
    return cnt


def _term_specificity_boost(t: str) -> float:
    tl = (t or "").strip().lower()
    if not tl:
        return 0.0
    b = 0.0
    if "/" in tl:
        b += 0.9
    if "." in tl:
        b += 0.6
    if "_" in tl:
        b += 0.4
    if ":" in tl:
        b += 0.3
    if len(tl) >= 12:
        b += 0.4
    elif len(tl) >= 8:
        b += 0.2
    return b


def _is_strong_term(t: str, *, cache: Dict[str, Any]) -> bool:
    """Best-effort strong/weak split.

    Strong terms are those that are structurally specific (path/dotted/identifier)
    OR have low document frequency in the repo graph.
    """
    tl = (t or "").strip().lower()
    if not tl or tl in _QUERY_STOPWORDS:
        return False
    if any(ch in tl for ch in ("/", ".", "_")) and len(tl) >= 5:
        return True
    N = int(cache.get("N") or 0) or 1
    df = _df_for_token(cache, tl)
    return df <= max(3, int(0.02 * N))


def search_repo_candidates_by_query(
    repo_graph: SubgraphLike,
    *,
    query: str,
    total_limit: int,
    dir_diversity_k: int = 4,
) -> List[Candidate]:
    """Search repo_graph nodes for explore.find (no anchors).

    This is intentionally *not* a full-text search; it's a small, explainable
    lexical scorer over node id/name/path.

    Query DSL (order-insensitive, best-effort):
      - symbol:<term>         strong bias toward node id/name (exact > substring)
      - path:<term>           strong bias toward node path
      - +term                 required term (must appear)
      - -term                 forbidden term (must not appear)
      - "phrase" / 'phrase'  high-boost phrase match

    Design goals:
      1) avoid weak-term explosion (e.g. "matrix")
      2) return a small top list with stable ordering
      3) keep reasons explainable
    """
    q = (query or "").strip()
    if not q or not repo_graph:
        return []

    spec = parse_query_dsl(q)
    cache = _get_repo_items_cache(repo_graph)
    N = int(cache.get("N") or 0) or 1

    # Determine which free terms are "strong"; we require at least one strong hit
    # unless the query provides explicit directives (+/symbol/path/phrase).
    free_terms = list(spec.free_terms)
    strong_terms = [t for t in free_terms if _is_strong_term(t, cache=cache)]
    if not strong_terms and free_terms:
        # fallback: take the longest term as a pseudo-strong anchor
        strong_terms = [max(free_terms, key=len)]

    has_directive = bool(spec.symbol_terms or spec.path_terms or spec.must_terms or spec.phrase_terms)

    # Precompute IDF-like weights for terms (weak terms are down-weighted, not excluded).
    term_weights: Dict[str, float] = {}
    for t in set([*(spec.must_terms or ()), *free_terms]):
        tl = t.lower()
        if not tl or tl in _QUERY_STOPWORDS:
            continue
        df = _df_for_token(cache, tl)
        idf = math.log((N + 1.0) / (df + 1.0))
        w = (1.0 + _term_specificity_boost(t)) * idf
        # If too common, treat as weak.
        if df >= int(0.10 * N):
            w *= 0.3
        term_weights[tl] = float(max(0.05, min(w, 6.0)))

    def _norm_dir(p: str) -> str:
        try:
            return str(PurePosixPath(_norm_posix(p)).parent)
        except Exception:
            return ""

    scored: List[Candidate] = []
    for nid_s, node, name, path, kind, hay, _d in cache.get("items", []):
        hay = (hay or "").lower()
        if not hay:
            continue

        # forbidden terms
        bad = False
        for t in spec.must_not_terms:
            if t.lower() in hay:
                bad = True
                break
        if bad:
            continue

        # required terms
        for t in spec.must_terms:
            if t.lower() not in hay:
                bad = True
                break
        if bad:
            continue

        score = 0.0
        reasons: List[str] = []

        # Directives: symbol / path
        for sym in spec.symbol_terms:
            sl = sym.lower()
            if not sl:
                continue
            if sl == (nid_s or "").lower() or sl == (name or "").lower():
                score += 8.0
                reasons.append("symbol_exact")
            elif sl in (nid_s or "").lower() or sl in (name or "").lower():
                score += 4.0
                reasons.append("symbol")

        for pt in spec.path_terms:
            pl = pt.lower()
            if not pl:
                continue
            if pl in (path or "").lower():
                score += 7.0
                reasons.append("path")
            elif pl in (nid_s or "").lower():
                score += 4.0
                reasons.append("path_hint")

        # Phrase matches (high boost, optional)
        for ph in spec.phrase_terms:
            phl = ph.lower()
            if phl and phl in hay:
                score += 4.0
                reasons.append("phrase")

        # Term matches (free + must)
        matched_terms: List[Tuple[float, str]] = []
        for t in [*spec.must_terms, *free_terms]:
            tl = t.lower()
            if not tl or tl in _QUERY_STOPWORDS:
                continue
            if tl in hay:
                w = float(term_weights.get(tl) or 0.5)
                score += w
                matched_terms.append((w, tl))

        # Weak-term explosion guard: require at least one strong term match
        # unless the query uses directives.
        if not has_directive:
            if strong_terms:
                if not any(st.lower() in hay for st in strong_terms):
                    continue

        if score <= 0.0:
            continue

        # Prefer function/class nodes very slightly over raw files for anchoring.
        if kind in {"func", "function", "class", "method"}:
            score += 0.25
        elif kind in {"file", "t-file"}:
            score += 0.05

        matched_terms.sort(key=lambda x: x[0], reverse=True)
        for (_w, tl) in matched_terms[:5]:
            reasons.append(tl)

        scored.append(
            {
                "id": nid_s,
                "kind": kind,
                "path": (node.get("path") or None),
                "span": node.get("span"),
                "degree": int(node.get("degree") or 0),
                "from_anchor": False,
                "score": float(score),
                "reasons": list(dict.fromkeys(reasons))[:10],
                "name": (name or None),
            }
        )

    scored.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

    # Directory diversity: round-robin by parent dir to avoid over-concentration.
    if dir_diversity_k and dir_diversity_k > 0 and scored:
        buckets: Dict[str, List[Candidate]] = {}
        order: List[str] = []
        for c in scored:
            d = _norm_dir(str(c.get("path") or ""))
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
