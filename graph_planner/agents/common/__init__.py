"""Common helpers shared by planner and CGM agents.

We keep imports lightweight here to avoid circular-import issues.
"""

from __future__ import annotations

from . import chat, contracts

__all__ = ["chat", "contracts", "text_protocol"]


def __getattr__(name: str):
    """Lazily import heavier modules on demand."""
    if name == "text_protocol":
        from . import text_protocol as mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
