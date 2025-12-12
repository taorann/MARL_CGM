# graph_planner/integrations/codefuse_cgm/formatting.py
"""
CodeFuse-CGM I/O helpers.

统一封装：
- 图线性化（GraphLinearizer）
- 代码片段文本化（SnippetFormatter）
- 片段选择 / collate（collate）
- ChatPrompt 组装与编码（ConversationEncoder, CgmInputBuilder）

语义上对齐 CodeFuse-CGM 官方：
- collate 决定“哪些代码片段进入上下文”（类似官方的 context 构造）；
- GraphLinearizer/SnippetFormatter 把图与片段转成可读文本；
- ConversationEncoder 按 CGM 协议拼 Issue / Instruction / Plan / Subgraph / Snippets；
- CgmInputBuilder 在推理阶段一站式构造 tokenizer 输入（支持子步规划 plan_text）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import json
from collections import defaultdict

import torch
from transformers import PreTrainedTokenizerBase

from aci.schema import CollateMeta
from ...agents.common.contracts import CGM_SYSTEM_PROMPT, CGM_CONTRACT
from ...memory.types import DocChunk
from ...memory import subgraph_store

# ---------------------------------------------------------------------------
# Graph formatting
# ---------------------------------------------------------------------------

GraphDict = Mapping[str, object]


def load_graph_document(path: Path | str | None) -> Optional[GraphDict]:
    """读取子图 JSON 文档，兼容 ``None``/缺失文件。"""
    if path is None:
        return None
    graph_path = Path(path)
    if not graph_path.exists():
        return None
    with graph_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _pick(value: Mapping[str, object], *keys: str) -> Optional[str]:
    """按优先级返回首个非空字符串字段。"""
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _pick_list(value: Mapping[str, object], *keys: str) -> Sequence[str]:
    """提取字符串序列字段并进行类型转换。"""
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, Sequence) and not isinstance(candidate, str):
            return [str(item) for item in candidate]
    return []


def _ellipsis(text: str, max_chars: int) -> str:
    """在超长时追加省略号以控制字段长度。"""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


@dataclass
class GraphLinearizer:
    """Render graph nodes into a structured, readable text block.

    输入数据假定来自 subgraph_store / serialize_subgraph 生成的 JSON
    （节点、边列表），字段抽取与截断策略参考了 CodeFuse-CGM 内部的
    `get{Lang}Sentence` / `graph2embedding`：
        - name / label / nodeType
        - summary/docstring/comment/description
        - text/code/content/body
    """

    max_nodes: int = 32
    max_chars_per_field: int = 512

    def linearize(self, graph: Optional[GraphDict]) -> str:
        """将子图节点列表转换为多段文本，供 LLM 阅读。"""
        if not graph:
            return ""

        nodes = graph.get("nodes")
        if not isinstance(nodes, Sequence):
            return ""

        sections: List[str] = []
        for node in nodes[: self.max_nodes]:
            if not isinstance(node, Mapping):
                continue
            name = _pick(node, "name", "label", "title", "id", "nodeId") or "(anonymous)"
            node_type = _pick(node, "nodeType", "type")
            header = f"- {name}"
            if node_type:
                header += f" [{node_type}]"

            body_parts: List[str] = []
            summary = _pick(
                node,
                "summary",
                "docstring",
                "comment",
                "description",
                "signature",
            )
            if summary:
                body_parts.append(_ellipsis(summary, self.max_chars_per_field))

            text = _pick(node, "text", "code", "content", "body")
            if text:
                body_parts.append(_ellipsis(text, self.max_chars_per_field))

            anchors = _pick_list(node, "anchors", "anchor", "keywords")
            if anchors:
                body_parts.append("Anchors: " + ", ".join(anchors[:6]))

            if body_parts:
                sections.append(header + "\n" + "\n".join(f"    {line}" for line in body_parts))
            else:
                sections.append(header)

        return "\n".join(sections)


# ---------------------------------------------------------------------------
# Snippet formatting
# ---------------------------------------------------------------------------


@dataclass
class SnippetFormatter:
    """Serialise candidate code snippets in a deterministic order."""
    max_snippets: int = 5
    max_lines_per_snippet: int = 40

    def format(self, snippets: Optional[Sequence[Mapping[str, object]]]) -> str:
        """将候选片段序列转换为 ``path:start-end`` + 代码正文格式。"""
        if not snippets:
            return ""
        blocks: List[str] = []
        for entry in snippets[: self.max_snippets]:
            if not isinstance(entry, Mapping):
                continue
            path = _pick(entry, "path", "abs_path") or "unknown"
            start = entry.get("start") or entry.get("line")
            end = entry.get("end") or start
            header = f"{path}:{start}-{end}"
            lines = entry.get("snippet") or entry.get("lines")
            if isinstance(lines, Sequence) and not isinstance(lines, str):
                normalized = [str(raw) for raw in lines[: self.max_lines_per_snippet]]
                blocks.append(header + "\n" + "\n".join(normalized))
            else:
                body = _pick(entry, "text", "content")
                if body:
                    blocks.append(header + "\n" + _ellipsis(body, 1024))
                else:
                    blocks.append(header)
        return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Collate: 从 subgraph + PlanTarget 中选出 DocChunk[]
# ---------------------------------------------------------------------------

def _is_tfile_path(path: str) -> bool:
    p = (path or "").lower()
    return ("test" in p) or ("/tests/" in p) or p.endswith("_test.py") or p.endswith("test.py")


def _overlap(a1: int, a2: int, b1: int, b2: int) -> bool:
    return max(a1, b1) <= min(a2, b2)


def _lines(ck: DocChunk) -> int:
    try:
        return int(ck["end"]) - int(ck["start"]) + 1
    except Exception:
        return 0


def _est_tokens(ck: DocChunk) -> int:
    """估算片段 token 数：有 text 时按 char/4，无 text 时按行*60。"""
    if ck.get("text") is not None:
        return int(len(ck.get("text") or "") / 4)
    return max(1, _lines(ck) * 60)


def _summarize(chunks: List[DocChunk]) -> CollateMeta:
    n = len(chunks)
    total_lines = 0
    max_lines = 0
    tfile_chunks = 0
    tokens = 0
    for ck in chunks:
        ln = _lines(ck)
        total_lines += ln
        max_lines = max(max_lines, ln)
        tokens += _est_tokens(ck)
        if _is_tfile_path(ck["path"]):
            tfile_chunks += 1

    avg = total_lines / max(1, n)
    return {
        "chunks": n,
        "total_lines": total_lines,
        "avg_lines": round(avg, 2),
        "max_lines": max_lines,
        "tfile_chunk_ratio": round(tfile_chunks / max(1, n), 4),
        "est_tokens": tokens,
        "reordered": False,
        "warnings": [],
    }


@dataclass
class _ScoreCtx:
    """用于启发式打分的上下文。"""
    plan_targets: List[Dict[str, Any]]
    file_degree: Dict[str, float]
    prefer_tests: bool


def _build_file_degree_index(subgraph) -> Dict[str, float]:
    """
    基于子图节点的 degree，近似出每个文件的“重要度”。用于加权排序。
    """
    sg = subgraph_store.wrap(subgraph)
    sums: Dict[str, int] = defaultdict(int)
    cnts: Dict[str, int] = defaultdict(int)
    for n in sg.nodes.values():
        p = n.get("path")
        if not p:
            continue
        d = int(n.get("degree") or 0)
        sums[p] += d
        cnts[p] += 1
    out: Dict[str, float] = {}
    for p, s in sums.items():
        out[p] = float(s) / max(1, cnts[p])
    return out


def _score_chunk(ck: DocChunk, ctx: _ScoreCtx) -> float:
    """
    启发式权重：
      +1.6  片段与 plan target 文件相同且窗口重叠
      +1.0  片段与 plan target 文件相同（无重叠）
      +0.6  非测试文件（或者 prefer_tests=False 时反向为测试文件加权）
      +0~0.8  文件 degree 的归一化权重
      +0~0.5  短片段奖励（越短越高）
    """
    w = 0.0
    path = ck["path"]

    # 与 PlanTarget 关系
    overlap_bonus = 0.0
    samefile_bonus = 0.0
    for t in ctx.plan_targets:
        if t["path"] == path:
            samefile_bonus = max(samefile_bonus, 1.0)
            if _overlap(int(t["start"]), int(t["end"]), int(ck["start"]), int(ck["end"])):
                overlap_bonus = max(overlap_bonus, 1.6)
    w += max(overlap_bonus, samefile_bonus)

    # 测试文件权重
    is_test = _is_tfile_path(path)
    if ctx.prefer_tests:
        w += (0.6 if is_test else 0.3)
    else:
        w += (0.6 if not is_test else 0.3)

    # 文件 degree（0~0.8）
    deg = float(ctx.file_degree.get(path, 0.0))
    deg_norm = min(1.0, deg / 12.0)  # 粗略归一化，12度及以上视为 1.0
    w += 0.8 * deg_norm

    # 短片段奖励（0~0.5）
    ln = _lines(ck)
    short_bonus = max(0.0, 0.5 - 0.0005 * ln)  # 0 行 ~ 1000 行 -> 0.5 ~ 0.0
    w += short_bonus

    return w


def _light_reorder_select(
    chunks: List[DocChunk],
    scores: Dict[int, float],
    budget_tokens: int,
    max_chunks: int,
    per_file_max_chunks: int,
) -> List[DocChunk]:
    """稳定重排：按分数从高到低选择，遵守 token 预算、总片段上限、单文件片段上限。"""
    order = sorted(
        range(len(chunks)),
        key=lambda i: (scores.get(i, 0.0), -_lines(chunks[i])),
        reverse=True,
    )
    kept: List[DocChunk] = []
    token_sum = 0
    per_file_count: Dict[str, int] = defaultdict(int)

    for i in order:
        ck = chunks[i]
        p = ck["path"]
        if per_file_count[p] >= per_file_max_chunks:
            continue
        tk = _est_tokens(ck)
        if token_sum + tk > budget_tokens:
            continue
        kept.append(ck)
        token_sum += tk
        per_file_count[p] += 1
        if len(kept) >= max_chunks:
            break

    # 如果因为单个大块导致一个都装不下，兜底放一个最高分的
    if not kept and chunks:
        kept = [chunks[order[0]]]

    return kept


def _interleave_tests(primary: List[DocChunk], tests: List[DocChunk]) -> List[DocChunk]:
    """
    简单交错：每插入 3~5 个非测试片段，穿插 1 个测试片段；若测试片段不多则更稀疏。
    """
    if not tests:
        return list(primary)
    # 计算步长：尽量让测试片段比例不超过 ~25%
    step = max(3, min(5, len(primary) // max(1, len(tests))))
    out: List[DocChunk] = []
    ti = 0
    for i, ck in enumerate(primary):
        out.append(ck)
        if (i + 1) % step == 0 and ti < len(tests):
            out.append(tests[ti])
            ti += 1
    # 余下的测试片段（如果还剩）
    while ti < len(tests):
        out.append(tests[ti])
        ti += 1
    return out


def collate(
    subgraph: Mapping[str, Any] | Any,
    plan: Any,      # aci.schema.Plan
    cfg: Any,       # infra.config.Config
) -> Tuple[List[DocChunk], CollateMeta]:
    """
    组装 CGM 需要的上下文（DocChunk[]）与统计（CollateMeta）。

    - 线性化：subgraph_store.linearize(subgraph, mode=cfg.collate.mode or cfg.mode)
    - 打分：考虑 plan target / 文件度量 / 是否测试文件 / 片段长度
    - 选择：预算内 + per_file_max_chunks + max_chunks
    - 轻量重排：启用则按分数排序选择；否则保持原顺序裁剪
    - 交错：启用 interleave_tests 则将测试片段以固定步长交错
    """
    # 线性化成 DocChunk[]
    mode = getattr(getattr(cfg, "collate", cfg), "mode", getattr(cfg, "mode", "wsd"))
    chunks = subgraph_store.linearize(subgraph, mode=mode)  # List[DocChunk]
    meta = _summarize(chunks)

    # 配置
    coll_cfg = getattr(cfg, "collate", cfg)
    budget_tokens = getattr(coll_cfg, "budget_tokens", 40000)
    max_chunks = getattr(coll_cfg, "max_chunks", 64)
    per_file_max = getattr(coll_cfg, "per_file_max_chunks", 8)
    prefer_tests = getattr(cfg, "prefer_test_files", True)
    enable_reorder = getattr(coll_cfg, "enable_light_reorder", False)
    interleave_tests = getattr(coll_cfg, "interleave_tests", True)

    # 打分上下文
    file_degree = _build_file_degree_index(subgraph)
    plan_targets = [
        t.__dict__ if hasattr(t, "__dict__") else dict(t)
        for t in getattr(plan, "targets", [])
    ]
    ctx = _ScoreCtx(plan_targets=plan_targets, file_degree=file_degree, prefer_tests=prefer_tests)
    scores = {i: _score_chunk(ck, ctx) for i, ck in enumerate(chunks)}

    # 预算内：只做 per_file_max 与交错，不重排
    if meta["est_tokens"] <= budget_tokens and len(chunks) <= max_chunks:
        limited: List[DocChunk] = []
        per_file_count: Dict[str, int] = defaultdict(int)
        token_sum = 0
        for ck in chunks:
            p = ck["path"]
            if per_file_count[p] >= per_file_max:
                continue
            tk = _est_tokens(ck)
            if token_sum + tk > budget_tokens:
                continue
            limited.append(ck)
            token_sum += tk
            per_file_count[p] += 1
            if len(limited) >= max_chunks:
                break

        if interleave_tests:
            prim = [c for c in limited if not _is_tfile_path(c["path"])]
            tchs = [c for c in limited if _is_tfile_path(c["path"])]
            final = _interleave_tests(prim, tchs)
        else:
            final = limited

        meta = _summarize(final)
        return final, meta

    # 超预算：按配置决定是否重排
    if enable_reorder:
        selected = _light_reorder_select(
            chunks=chunks,
            scores=scores,
            budget_tokens=budget_tokens,
            max_chunks=max_chunks,
            per_file_max_chunks=per_file_max,
        )
        meta = _summarize(selected)
        meta["reordered"] = True
        meta.setdefault("warnings", []).append(
            f"light_reorder_applied: est>{budget_tokens}"
        )
    else:
        # 不重排：按线性化原顺序裁剪（但仍遵守 per_file_max）
        selected = []
        per_file_count = defaultdict(int)
        token_sum = 0
        for ck in chunks:
            p = ck["path"]
            if per_file_count[p] >= per_file_max:
                continue
            tk = _est_tokens(ck)
            if token_sum + tk > budget_tokens:
                continue
            selected.append(ck)
            token_sum += tk
            per_file_count[p] += 1
            if len(selected) >= max_chunks:
                break
        meta = _summarize(selected)
        meta.setdefault("warnings", []).append(
            f"budget_exceeded_without_reorder: est>{budget_tokens}"
        )

    # 交错（对 selected 生效）
    if interleave_tests and selected:
        prim = [c for c in selected if not _is_tfile_path(c["path"])]
        tchs = [c for c in selected if _is_tfile_path(c["path"])]
        selected = _interleave_tests(prim, tchs)
        meta = _summarize(selected)
        meta["reordered"] = meta.get("reordered", False)

    return selected, meta


# ---------------------------------------------------------------------------
# Conversation encoding & unified builder
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = CGM_SYSTEM_PROMPT
PROMPT_CONTRACT = CGM_CONTRACT  # 预留：需要的时候可以用来检查约定


@dataclass
class ConversationEncoder:
    """Compose chat prompts for CGM training and inference."""

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 8192
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def build_user_message(
        self,
        *,
        prompt: str,
        plan_text: Optional[str],
        graph_text: str,
        snippets_text: str,
        issue_text: Optional[str],
    ) -> str:
        """拼装包含 Issue/Instruction/Plan/Subgraph/Snippets 的用户消息文本。

        其中：
        - prompt: GraphPlanner 主 LLM 的“高层指令”（相当于官方的 prompt）
        - plan_text: 子步规划文本（由 GraphPlanner 先行产出）
        - issue_text: 原始 issue / failure 描述
        """
        sections: List[str] = []
        if issue_text:
            sections.append(f"[Issue]\n{issue_text.strip()}")
        sections.append(f"[Instruction]\n{prompt.strip()}")
        if plan_text:
            sections.append(f"[Plan]\n{plan_text.strip()}")
        if graph_text:
            sections.append(f"[Subgraph]\n{graph_text}")
        if snippets_text:
            sections.append(f"[Snippets]\n{snippets_text}")
        return "\n\n".join(sections)

    def _apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> MutableMapping[str, object]:
        """调用 tokenizer 聊天模板或退化为 ``role: content`` 拼接。"""
        if hasattr(self.tokenizer, "apply_chat_template"):
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            if isinstance(encoded, torch.Tensor):
                attention = torch.ones_like(encoded)
                return {"input_ids": encoded, "attention_mask": attention}
            return encoded

        # Fallback: join messages manually.
        text_blocks = []
        for msg in messages:
            text_blocks.append(f"{msg['role'].upper()}: {msg['content']}")
        if add_generation_prompt:
            text_blocks.append("ASSISTANT:")
        encoded = self.tokenizer(
            "\n\n".join(text_blocks),
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        return encoded

    def encode_prompt(
        self,
        *,
        prompt: str,
        plan_text: Optional[str],
        graph_text: str,
        snippets_text: str,
        issue_text: Optional[str],
    ) -> MutableMapping[str, object]:
        """仅编码提示部分，用于推理阶段生成输入张量。"""
        user_message = self.build_user_message(
            prompt=prompt,
            plan_text=plan_text,
            graph_text=graph_text,
            snippets_text=snippets_text,
            issue_text=issue_text,
        )
        messages: List[Mapping[str, str]] = [{"role": "user", "content": user_message}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return self._apply_chat_template(messages, add_generation_prompt=True)

    def encode_example(
        self,
        *,
        prompt: str,
        response: str,
        plan_text: Optional[str],
        graph_text: str,
        snippets_text: str,
        issue_text: Optional[str],
    ) -> MutableMapping[str, object]:
        """编码单条监督样本并对提示 token 打上 ``-100`` 标签。"""
        user_message = self.build_user_message(
            prompt=prompt,
            plan_text=plan_text,
            graph_text=graph_text,
            snippets_text=snippets_text,
            issue_text=issue_text,
        )
        messages: List[Mapping[str, str]] = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response.strip()},
        ]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        full = self._apply_chat_template(messages, add_generation_prompt=False)
        prompt_only = self._apply_chat_template(messages[:-1], add_generation_prompt=True)

        input_ids = full["input_ids"].squeeze(0)
        attention_mask = full["attention_mask"].squeeze(0)
        prompt_len = prompt_only["input_ids"].shape[-1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_length": prompt_len,
        }


@dataclass
class CgmInputBuilder:
    """一站式构造 CGM 推理输入。

    用法（伪代码）：
        builder = CgmInputBuilder(tokenizer)
        graph_json = env.working_subgraph.to_json_obj()
        chunks, meta = collate(graph_json, plan_struct, cfg)
        snippets_text = builder.snippet_formatter.format(chunks)
        inputs = builder.build_inference_inputs(
            issue_text=...,             # 原始 issue
            planner_prompt=...,         # 你给 GraphPlanner 的高层 prompt
            plan_text=...,              # 子步规划文本（可为 None）
            subgraph=graph_json,
            snippets=chunks,
        )
        # 然后把 inputs["input_ids"]/["attention_mask"] 喂给 CGM 模型 / HTTP Service
    """

    tokenizer: PreTrainedTokenizerBase
    graph_linearizer: GraphLinearizer = field(default_factory=GraphLinearizer)
    snippet_formatter: SnippetFormatter = field(default_factory=SnippetFormatter)
    encoder: ConversationEncoder = field(init=False)

    def __post_init__(self) -> None:
        self.encoder = ConversationEncoder(tokenizer=self.tokenizer)

    def build_inference_inputs(
        self,
        *,
        issue_text: Optional[str],
        planner_prompt: str,
        plan_text: Optional[str],
        subgraph: Mapping[str, Any],
        snippets: Sequence[Mapping[str, Any]],
    ) -> MutableMapping[str, object]:
        """子图 + 片段 + Issue + 子步规划 → tokenizer 输入张量。"""
        graph_text = self.graph_linearizer.linearize(subgraph)
        snippets_text = self.snippet_formatter.format(snippets)
        return self.encoder.encode_prompt(
            prompt=planner_prompt,
            plan_text=plan_text,
            graph_text=graph_text,
            snippets_text=snippets_text,
            issue_text=issue_text,
        )


__all__ = [
    "GraphLinearizer",
    "SnippetFormatter",
    "ConversationEncoder",
    "CgmInputBuilder",
    "collate",
    "load_graph_document",
    "DEFAULT_SYSTEM_PROMPT",
    "PROMPT_CONTRACT",
]
