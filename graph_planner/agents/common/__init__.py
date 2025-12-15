# graph_planner/agents/common/__init__.py
"""
Common agent utilities.

We keep imports lazy to avoid circular-import issues across agent protocol modules.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["chat", "contracts", "text_protocol"]


def __getattr__(name: str) -> Any:
    # 1) Allow importing submodules directly:
    #   from graph_planner.agents.common import text_protocol
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")

    # 2) Backward-compat: allow importing symbols that live in submodules:
    #   from graph_planner.agents.common import SomeClass
    for sub in __all__:
        mod = importlib.import_module(f"{__name__}.{sub}")
        if hasattr(mod, name):
            return getattr(mod, name)

    raise AttributeError(f"module {__name__} has no attribute {name}")
