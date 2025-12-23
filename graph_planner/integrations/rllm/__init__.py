"""rLLM integration helpers.

GraphPlanner evaluation assumes rLLM and the corresponding environment classes
are available. We therefore import and register eagerly and allow ImportError to
surface instead of silently substituting ``None``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ...infra.vendor import ensure_rllm_importable
from .agent import GraphPlannerRLLMAgent
from .cgm_agent import CGMRLLMAgent
from .dataset import (
    GRAPH_PLANNER_CGM_DATASET_NAME,
    GRAPH_PLANNER_DATASET_NAME,
    RegisteredDataset,
    ensure_dataset_registered,
)
from .registry import register_rllm_components

if not ensure_rllm_importable():
    raise RuntimeError("Unable to import vendored rLLM modules")

from .env import GraphPlannerRLLMEnv

register_rllm_components(
    GraphPlannerRLLMAgent,
    GraphPlannerRLLMEnv,
    name="graph_planner_repoenv",
)

from .cgm_env import CGMRLLMEnv

register_rllm_components(
    CGMRLLMAgent,
    CGMRLLMEnv,
    name="graph_planner_cgm",
)

__all__ = [
    "GraphPlannerRLLMAgent",
    "GraphPlannerRLLMEnv",
    "CGMRLLMAgent",
    "CGMRLLMEnv",
    "GRAPH_PLANNER_DATASET_NAME",
    "GRAPH_PLANNER_CGM_DATASET_NAME",
    "RegisteredDataset",
    "resolve_task_file",
    "load_task_entries",
    "register_dataset_from_file",
    "ensure_dataset_registered",
]


def __getattr__(name: str) -> Any:  # pragma: no cover
    """Lazy-load non-critical symbols."""

    if name in {
        "GRAPH_PLANNER_DATASET_NAME",
        "GRAPH_PLANNER_CGM_DATASET_NAME",
        "RegisteredDataset",
        "resolve_task_file",
        "load_task_entries",
        "register_dataset_from_file",
        "ensure_dataset_registered",
    }:
        module = import_module("graph_planner.integrations.rllm.dataset")
        return getattr(module, name)
    raise AttributeError(name)
