# graph_planner/infra/metrics.py
from __future__ import annotations
from typing import Any, Dict, Mapping, Optional

def init_wandb(*args: Any, **kwargs: Any) -> None:
    """Best-effort wandb init; safe no-op if wandb unavailable."""
    try:
        import wandb  # type: ignore
        if kwargs:
            wandb.init(*args, **kwargs)
    except Exception:
        return

def log_metrics(metrics: Mapping[str, Any], *, step: Optional[int] = None, **kwargs: Any) -> None:
    """Best-effort metrics logging; safe no-op if wandb unavailable."""
    try:
        import wandb  # type: ignore
        payload: Dict[str, Any] = dict(metrics or {})
        if step is not None:
            wandb.log(payload, step=step, **kwargs)
        else:
            wandb.log(payload, **kwargs)
    except Exception:
        return

def make_gpu_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            snap["cuda_device_count"] = int(torch.cuda.device_count())
            snap["cuda_current_device"] = int(torch.cuda.current_device())
            try:
                snap["cuda_memory_allocated"] = int(torch.cuda.memory_allocated())
                snap["cuda_memory_reserved"] = int(torch.cuda.memory_reserved())
            except Exception:
                pass
    except Exception:
        pass
    return snap

def make_ray_snapshot() -> Dict[str, Any]:
    try:
        import ray  # type: ignore
        ctx: Dict[str, Any] = {}
        try:
            ctx["is_initialized"] = bool(ray.is_initialized())
        except Exception:
            ctx["is_initialized"] = False
        return ctx
    except Exception:
        return {}
