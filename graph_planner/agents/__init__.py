"""Agent package aggregating available controllers."""

from typing import Any

from .rule_based.planner import PlannerAgent

# 只对外暴露规则 Planner 和 rLLM 版的 GraphPlannerRLLMAgent
__all__ = ["PlannerAgent", "GraphPlannerRLLMAgent"]


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial lazy import
    if name == "GraphPlannerRLLMAgent":
        from ..integrations.rllm.agent import GraphPlannerRLLMAgent as _Agent
        return _Agent
    raise AttributeError(name)
