"""Agent package aggregating available controllers."""

from typing import Any

# 只对外暴露 rLLM 版的 GraphPlannerRLLMAgent
__all__ = ["GraphPlannerRLLMAgent"]


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial lazy import
    if name == "GraphPlannerRLLMAgent":
        from ..integrations.rllm.agent import GraphPlannerRLLMAgent as _Agent
        return _Agent
    raise AttributeError(name)
