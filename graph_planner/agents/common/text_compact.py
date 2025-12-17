"""
Utilities for compacting long free-form text (especially GitHub issues) into a
bounded-size representation suitable for LLM prompts.

This module is deliberately dependency-free and safe to use in offline / sandbox
contexts.

Design goals:
- Keep the issue title verbatim.
- Keep a bounded amount of prose.
- Keep a small number of fenced code blocks (often repro snippets / tracebacks).
- Provide deterministic truncation with visible markers so debugging is easier.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Tuple, Optional


_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)\n```", re.DOTALL)


@dataclass
class CompactIssueConfig:
    # Hard cap on output size (characters).
    max_chars: int = 8000
    # Maximum number of fenced code blocks to keep.
    max_code_blocks: int = 2
    # Maximum lines per kept code block (if longer, keep head+tail).
    max_code_lines: int = 160
    # Prose (non-code) soft cap before we start adding code blocks.
    prose_chars: int = 2400
    # If True, keep code blocks in original order. Otherwise prefer earlier ones.
    keep_order: bool = True


def _trim_lines(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    # keep head+tail for tracebacks and long repros
    head_n = max_lines // 2
    tail_n = max_lines - head_n
    head = lines[:head_n]
    tail = lines[-tail_n:]
    return "\n".join(
        head
        + [f"... [TRUNCATED {len(lines) - max_lines} lines] ..."]
        + tail
    )


def _hard_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    # keep head+tail for readability
    head_n = max_chars // 2
    tail_n = max_chars - head_n - 64  # reserve for marker
    if tail_n < 0:
        tail_n = 0
    head = text[:head_n].rstrip()
    tail = text[-tail_n:].lstrip() if tail_n else ""
    return f"{head}\n\n... [TRUNCATED {len(text) - max_chars} chars] ...\n\n{tail}"


def compact_issue(issue: Dict[str, Any], cfg: Optional[CompactIssueConfig] = None) -> Dict[str, str]:
    """
    Compact a GitHub issue-like dict into a bounded-length dict with keys:
      - title: str
      - body: str

    Expected input fields:
      - issue.get("title", "")
      - issue.get("body", "") or issue.get("description", "")

    This function is safe if fields are missing or non-strings.
    """
    cfg = cfg or CompactIssueConfig()
    title = str(issue.get("title") or "").strip()
    body_raw = issue.get("body")
    if body_raw is None:
        body_raw = issue.get("description")
    body = str(body_raw or "")

    # Extract fenced code blocks.
    fences: List[Tuple[str, str]] = []
    def _fence_sub(m: re.Match) -> str:
        lang = (m.group(1) or "").strip()
        code = (m.group(2) or "")
        fences.append((lang, code))
        # Remove from prose; keep a placeholder so headings still read okay.
        return "\n\n[CODE_BLOCK]\n\n"

    prose = _FENCE_RE.sub(_fence_sub, body)

    # Normalize whitespace in prose a bit (but keep headings, newlines).
    prose = re.sub(r"\n{3,}", "\n\n", prose).strip()

    prose_part = prose[: cfg.prose_chars].rstrip()
    if len(prose) > cfg.prose_chars:
        prose_part += "\n\n... [PROSE TRUNCATED] ..."

    kept: List[str] = []
    if title:
        kept.append(title)
        kept.append("")  # blank line

    kept.append(prose_part)

    # Keep up to N code blocks, trimmed by lines.
    num_keep = min(cfg.max_code_blocks, len(fences))
    for i in range(num_keep):
        lang, code = fences[i]
        code_t = _trim_lines(code.strip("\n"), cfg.max_code_lines)
        lang_tag = lang if lang else ""
        kept.append("")
        kept.append(f"```{lang_tag}".rstrip())
        kept.append(code_t)
        kept.append("```")

    out = "\n".join(kept).strip()
    out = _hard_truncate(out, cfg.max_chars)

    return {"title": title, "body": out}


def compact_issue_text(title: str, body: str, *, target_tokens: int = 320) -> str:
    """Return a compact single-string issue context.

    We approximate token budget using a conservative chars-per-token heuristic to avoid
    overlong prompts in environments without a tokenizer.
    """
    try:
        tt = int(target_tokens)
    except Exception:
        tt = 320
    if tt <= 0:
        tt = 320
    # Rough heuristic: 1 token ~= 3.5-4 chars in English; use 4 for safety.
    max_chars = min(20000, max(800, tt * 4))
    cfg = CompactIssueConfig(max_chars=max_chars)
    compacted = compact_issue({"title": title or "", "body": body or ""}, cfg=cfg).get("body", "")
    # Avoid duplicating the title if the caller already prints it.
    t = (title or "").strip()
    if t and compacted.startswith(t):
        compacted = compacted[len(t):].lstrip()
    return compacted.strip()
