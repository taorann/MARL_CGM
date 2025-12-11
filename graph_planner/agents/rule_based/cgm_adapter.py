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
    subgraph_linearized: Optional[List[Dict[str, Any]]],
    plan: Plan,
    constraints: Optional[Dict[str, Any]] = None,
    snippets: Optional[List[Dict[str, Any]]] = None,
    plan_text: Optional[str] = None,
    issue: Optional[Dict[str, Any]] = None,
) -> Patch:
    """生成补丁：优先调用 CodeFuse CGM 客户端或本地 CGM Runtime。

    更新语义：
        - 不再回退到本地“规则打标记”补丁（_generate_local_patch）。
        - 如果 HTTP 客户端和本地 Runtime 都不可用或均失败，则直接抛出 RuntimeError，
          由调用方（PlannerEnv / service 等）自己决定怎么处理。
    """
    client = _get_client()
    runtime = _get_local_runtime()
    last_exc: Optional[Exception] = None

    # 1) 优先远程 CodeFuse CGM HTTP 服务
    if client is not None:
        try:
            return client.generate_patch(
                issue=issue,
                plan=plan,
                plan_text=plan_text,
                subgraph_linearized=subgraph_linearized,
                snippets=snippets,
                metadata={"constraints": constraints or {}},
            )
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            telemetry.log_event(
                {
                    "kind": "cgm",
                    "ok": False,
                    "error": str(exc),
                    "endpoint": getattr(client, "endpoint", ""),
                }
            )

    # 2) 其次尝试本地 CGM Runtime（本地 Seq2Seq 模型）
    if runtime is not None:
        try:
            patch = runtime.generate_patch(
                issue=issue,
                plan=plan,
                plan_text=plan_text,
                subgraph_linearized=subgraph_linearized,
                snippets=snippets,
                constraints=constraints,
            )
            return patch
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            telemetry.log_event(
                {
                    "kind": "cgm-local",
                    "ok": False,
                    "error": str(exc),
                    "model": getattr(
                        getattr(runtime, "generator", None),
                        "model_name",
                        "",
                    ),
                }
            )

    # 3) 没有 client/runtime，或者都失败：不再本地规则 patch，直接抛错
    if last_exc is not None:
        raise RuntimeError(
            f"CGM generation failed (no rule-based fallback): {last_exc}"
        ) from last_exc

    raise RuntimeError(
        "No CGM client or local runtime available; "
        "rule-based local patch generation has been disabled."
    )
