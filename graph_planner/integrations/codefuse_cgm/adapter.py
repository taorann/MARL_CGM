"""
graph_planner/integrations/codefuse_cgm/adapter.py

统一对接 CodeFuse-CGM 的补丁生成入口（本进程内）。

语义（对齐 4.1）：
    - 优先调用远程 CodeFuse CGM HTTP 服务（CodeFuseCGMClient）；
    - 如果 HTTP 客户端不可用或失败，并且有本地 _LocalCGMRuntime，则尝试本地模型；
    - 不再 fallback 到任何 rule-based 补丁；
    - 不在这里处理 Ray / actor 逻辑（Ray 统一放在 actor/cgm_adapter.py）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import os

from aci.schema import Patch, Plan
from graph_planner.integrations.codefuse_cgm.client import CodeFuseCGMClient
from graph_planner.infra import telemetry


# ============================================================
# 本地 CGM Runtime（占位实现）
# ============================================================

@dataclass
class _LocalCGMRuntime:
    """
    本地 CGM 推理 Runtime。

    目前只存一个 generator 占位，真正的生成逻辑可以之后再补。
    service.py 会通过 _LocalCGMRuntime(generator=...) 来构造。
    """

    generator: Any

    def generate_patch(
        self,
        *,
        issue: Optional[Dict[str, Any]],
        plan: Plan,
        plan_text: Optional[str],
        subgraph_linearized: Optional[List[Dict[str, Any]]],
        snippets: Optional[List[Dict[str, Any]]],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Patch:
        """
        这里暂时不实现本地推理逻辑，直接提醒需要补完。

        如果你之后真的要用本地 CGM（不通过 HTTP service），
        可以在这里把 generator + collated payload 接上。
        """
        raise NotImplementedError(
            "Local CGM runtime is not wired yet; "
            "please implement _LocalCGMRuntime.generate_patch if needed."
        )


# ============================================================
# HTTP 客户端 & 本地 Runtime getter
# ============================================================

_CLIENT: Optional[CodeFuseCGMClient] = None
_LOCAL_RUNTIME: Optional[_LocalCGMRuntime] = None  # 如需本地 runtime，可在其他模块赋值


def _get_client() -> Optional[CodeFuseCGMClient]:
    """
    返回全局复用的 CodeFuse CGM HTTP 客户端。

    使用环境变量配置：
        CODEFUSE_CGM_ENDPOINT: 必选，HTTP 服务的 base url（例如 http://127.0.0.1:8000/generate）
        CODEFUSE_CGM_API_KEY: 可选，若服务需要鉴权
        CODEFUSE_CGM_MODEL_NAME: 可选，转发给远程服务的模型名
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    endpoint = os.getenv("CODEFUSE_CGM_ENDPOINT")
    if not endpoint:
        return None

    api_key = os.getenv("CODEFUSE_CGM_API_KEY") or None
    model_name = os.getenv("CODEFUSE_CGM_MODEL_NAME") or None

    _CLIENT = CodeFuseCGMClient(
        endpoint=endpoint,
        api_key=api_key,
        model=model_name,
    )
    return _CLIENT


def _get_local_runtime() -> Optional[_LocalCGMRuntime]:
    """
    返回全局本地 runtime。

    默认 None；如果你想启用本地 CGM，可以在别处：

        from graph_planner.integrations.codefuse_cgm import adapter
        adapter._LOCAL_RUNTIME = _LocalCGMRuntime(generator=...)

    然后这里就能拿到。
    """
    return _LOCAL_RUNTIME


# ============================================================
# 主入口：generate()
# ============================================================

def generate(
    subgraph_linearized: Optional[List[Dict[str, Any]]],
    plan: Plan,
    constraints: Optional[Dict[str, Any]] = None,
    snippets: Optional[List[Dict[str, Any]]] = None,
    plan_text: Optional[str] = None,
    issue: Optional[Dict[str, Any]] = None,
) -> Patch:
    """
    生成补丁：优先调用 CodeFuse CGM HTTP 客户端，然后尝试本地 Runtime。

    语义（4.1 对齐版）：
        - 不再回退到本地“规则打标记”补丁；
        - 如果 HTTP 客户端和本地 Runtime 都不可用或均失败，则直接抛出 RuntimeError，
          由调用方（PlannerEnv / actor / service 等）决定如何处理。
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
