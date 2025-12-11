"""
actor/cgm_adapter.py

统一对接 CodeFuse-CGM 的补丁生成入口。

语义（更新版）：
    - 仅调用远程 CGM Ray actor；
    - 不再 fallback 到本地 rule-based cgm_adapter.generate；
    - 如果没有可用 actor 或调用失败，直接 raise，让上层环境决定如何处理。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import os


try:
    import ray  # type: ignore[import]
except Exception:  # pragma: no cover
    ray = None  # type: ignore[assignment]

# 默认的 CGM actor 名称，可以通过环境变量覆盖
_DEFAULT_ACTOR_NAME = os.environ.get("CODEFUSE_CGM_ACTOR", "codefuse_cgm_actor")


@dataclass
class CGMRequest:
    """传给远程 CGM 的统一请求结构。"""

    collated: Dict[str, Any]
    plan: str
    constraints: Dict[str, Any]
    run_id: str
    issue: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collated": self.collated,
            "plan": self.plan,
            "constraints": self.constraints,
            "run_id": self.run_id,
            "issue": self.issue,
        }


def _resolve_actor(name: str) -> Any:
    """根据名称解析 Ray actor；解析失败返回 None。"""
    if ray is None:
        return None
    try:
        return ray.get_actor(name)
    except Exception:
        return None


def generate(
    *,
    collated: Dict[str, Any],
    plan: str,
    constraints: Optional[Dict[str, Any]] = None,
    run_id: str,
    issue: Optional[Dict[str, Any]] = None,
    actor_name: Optional[str] = None,
) -> Dict[str, Any]:
    """调用远程 CGM 生成补丁。

    Args
    ----
    collated:
        由 collater 输出的上下文，形如 {"chunks": [...], "meta": {...}}。
    plan:
        文本形式的修复计划，通常来自 RepairAction.plan。
    constraints:
        限制条件，例如 {"max_edits": 3} 等。
    run_id:
        当前 run 的标识，用于日志/跟踪。
    issue:
        原始 issue 信息（标题、描述、失败栈等），供 CGM 做上下文增强。
    actor_name:
        Ray actor 名称；若 None 则使用环境变量 CODEFUSE_CGM_ACTOR 或默认值。

    Returns
    -------
    patch: dict
        CodeFuse-CGM 风格的补丁结构，至少包含 "edits" 字段。

    Raises
    ------
    RuntimeError:
        - Ray 不可用；
        - 找不到指定 actor；
        - 远程调用失败或返回非法类型。
    """
    constraints = constraints or {}
    issue = issue or {}
    actor_name = actor_name or _DEFAULT_ACTOR_NAME

    # 1) 确认 Ray 环境可用
    if ray is None:
        raise RuntimeError(
            "Ray is not available; CGM remote generation cannot be used. "
            "Rule-based fallback has been disabled by design."
        )

    # 2) 解析 Ray actor
    actor = _resolve_actor(actor_name)
    if actor is None:
        raise RuntimeError(
            f"No CGM Ray actor named '{actor_name}' is available. "
            "Rule-based CGM fallback is disabled; please ensure the actor is started."
        )

    # 3) 构造请求 payload
    req = CGMRequest(
        collated=collated,
        plan=plan,
        constraints=constraints,
        run_id=run_id,
        issue=issue,
    )

    # 4) 调用远程 CGM
    try:
        # 约定远程 actor 暴露 generate_patch.remote(payload: dict) -> patch: dict
        future = actor.generate_patch.remote(req.to_dict())
        patch = ray.get(future)
    except Exception as exc:  # pragma: no cover - 远程错误按统一形式抛给上层
        raise RuntimeError(f"CGM remote failed: {exc}") from exc

    if not isinstance(patch, dict):
        raise RuntimeError(
            f"CGM remote returned non-dict patch: {type(patch)!r}. "
            "Rule-based local patch generation has been removed."
        )

    return patch
