# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Unified telemetry backend.

This module unifies:
- Trajectory logging (steps.jsonl): one record per env.step()
- Event logging (events.jsonl): sparse, high-signal events (errors / external calls)
- Metrics logging (metrics.jsonl): numeric series for monitoring/plots

It keeps backward compatible helpers:
- log_event(...)          -> event(...)
- log_test_result(...)    -> event("sandbox.test_result", ...) + metrics
- emit_metrics(...)       -> console metric line
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

from .config import load


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, obj: Mapping[str, Any]) -> None:
    _ensure_dir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _now_ts() -> float:
    return time.time()


def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)


def _sanitize_id(raw: str) -> str:
    raw = raw or "unknown"
    raw = raw.strip().replace(" ", "_")
    # Keep it filesystem friendly.
    out = []
    for ch in raw:
        if ch.isalnum() or ch in ("_", "-", ".", ":"):
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    return s[:120] if len(s) > 120 else s


@dataclass
class TelemetryContext:
    run_id: str = ""
    task_id: str = ""
    episode_id: str = ""
    step_id: int = 0
    tags: Tuple[str, ...] = ()


class Telemetry:
    """Trajectory + events + metrics logger with a small shared context."""

    def __init__(self) -> None:
        self.cfg = load().telemetry
        self.ctx = TelemetryContext()
        self._run_dir: Optional[Path] = None
        self._episode_dir: Optional[Path] = None
        self._run_started: bool = False
        self._episode_started: bool = False

    # --------------------------
    # Context management
    # --------------------------
    def set_run(self, run_id: str, tags: Optional[Tuple[str, ...]] = None) -> None:
        run_id = _sanitize_id(run_id or "")
        if not run_id:
            run_id = f"run_{int(_now_ts())}"
        self.ctx.run_id = run_id
        if tags is not None:
            self.ctx.tags = tuple(tags)
        self._run_dir = Path(self.cfg.base_dir) / self.ctx.run_id

    def start_run(self, meta: Optional[Mapping[str, Any]] = None) -> None:
        if not self.cfg.enabled:
            return
        if self._run_started:
            return
        if not self.ctx.run_id:
            self.set_run(os.environ.get("GRAPH_PLANNER_RUN_ID", "") or f"run_{int(_now_ts())}")
        self._run_started = True
        run_meta = {
            "ts": _now_ts(),
            "run_id": self.ctx.run_id,
            "type": "run",
            "name": "run.start",
            "payload": dict(meta or {}),
            "tags": list(self.ctx.tags),
        }
        if self._run_dir is not None:
            _append_jsonl(self._run_dir / "run_meta.jsonl", run_meta)

    def start_episode(
        self,
        *,
        task_id: str,
        episode_id: Optional[str] = None,
        meta: Optional[Mapping[str, Any]] = None,
        tags: Optional[Tuple[str, ...]] = None,
    ) -> str:
        if not self.cfg.enabled:
            return ""
        if not self._run_started:
            self.start_run()
        self.ctx.task_id = _safe_str(task_id)
        if tags is not None:
            self.ctx.tags = tuple(tags)
        ep = _sanitize_id(episode_id or self.ctx.task_id or f"ep_{int(_now_ts())}")
        self.ctx.episode_id = ep
        self.ctx.step_id = 0
        self._episode_dir = Path(self.cfg.base_dir) / self.ctx.run_id / "episodes" / ep
        self._episode_started = True
        payload = {
            "ts": _now_ts(),
            "run_id": self.ctx.run_id,
            "task_id": self.ctx.task_id,
            "episode_id": self.ctx.episode_id,
            "type": "episode",
            "name": "episode.start",
            "payload": dict(meta or {}),
            "tags": list(self.ctx.tags),
        }
        if self._episode_dir is not None:
            _append_jsonl(self._episode_dir / "meta.jsonl", payload)
        return ep

    def set_step(self, step_id: int) -> None:
        try:
            self.ctx.step_id = int(step_id)
        except Exception:
            self.ctx.step_id = 0

    # --------------------------
    # Record helpers
    # --------------------------
    def _base_record(self, *, type_: str, name: str, severity: str = "INFO") -> Dict[str, Any]:
        return {
            "ts": _now_ts(),
            "run_id": self.ctx.run_id,
            "task_id": self.ctx.task_id,
            "episode_id": self.ctx.episode_id,
            "step_id": self.ctx.step_id,
            "type": type_,
            "name": name,
            "severity": severity,
            "tags": list(self.ctx.tags),
        }

    def _truncate(self, obj: Any) -> Any:
        limit = int(getattr(self.cfg, "truncate_chars", 4000) or 4000)

        def trunc_str(s: str) -> str:
            if len(s) <= limit:
                return s
            return s[:limit] + f"...(truncated,{len(s)} chars)"

        if obj is None:
            return None
        if isinstance(obj, str):
            return trunc_str(obj)
        if isinstance(obj, (int, float, bool)):
            return obj
        if isinstance(obj, Mapping):
            return {k: self._truncate(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._truncate(v) for v in obj]
        return trunc_str(_safe_str(obj))

    # --------------------------
    # Public logging API
    # --------------------------
    def log_step(self, payload: Mapping[str, Any]) -> None:
        if not self.cfg.enabled or not self._episode_started or self._episode_dir is None:
            return
        rec = self._base_record(type_="step", name="env.step")
        rec["payload"] = self._truncate(dict(payload))
        _append_jsonl(self._episode_dir / "steps.jsonl", rec)

    def event(self, name: str, payload: Mapping[str, Any], *, severity: str = "INFO") -> None:
        if not self.cfg.enabled or not self._episode_started or self._episode_dir is None:
            # Still allow legacy flat file if configured.
            if getattr(self.cfg, "write_legacy", False):
                rec = {"ts": _now_ts(), "name": name, "severity": severity, **dict(payload)}
                _append_jsonl(Path(self.cfg.events_path), rec)
            return
        rec = self._base_record(type_="event", name=name, severity=severity)
        rec["payload"] = self._truncate(dict(payload))
        _append_jsonl(self._episode_dir / "events.jsonl", rec)
        if getattr(self.cfg, "write_legacy", False):
            legacy = {"ts": rec["ts"], "run_id": rec["run_id"], "task_id": rec["task_id"], "episode_id": rec["episode_id"], "step_id": rec["step_id"], "name": name, "severity": severity, **dict(payload)}
            _append_jsonl(Path(self.cfg.events_path), self._truncate(legacy))

    def metric(
        self,
        name: str,
        value: float,
        *,
        step_id: Optional[int] = None,
        labels: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not self.cfg.enabled or not self._episode_started or self._episode_dir is None:
            return
        rec = self._base_record(type_="metric", name=name)
        if step_id is not None:
            rec["step_id"] = int(step_id)
        rec["payload"] = {"value": value, "labels": dict(labels or {})}
        _append_jsonl(self._episode_dir / "metrics.jsonl", rec)
        if getattr(self.cfg, "console_metrics", False):
            emit_metrics({name: value})

    def artifact(self, name: str, path: str, meta: Optional[Mapping[str, Any]] = None) -> None:
        if not self.cfg.enabled or not self._episode_started or self._episode_dir is None:
            return
        rec = self._base_record(type_="artifact", name=name)
        rec["payload"] = {"path": _safe_str(path), "meta": dict(meta or {})}
        _append_jsonl(self._episode_dir / "artifacts.jsonl", rec)

    def end_episode(self, status: str, summary: Optional[Mapping[str, Any]] = None) -> None:
        if not self.cfg.enabled or not self._episode_started or self._episode_dir is None:
            return
        rec = self._base_record(type_="episode", name="episode.end")
        rec["payload"] = {"status": status, "summary": self._truncate(dict(summary or {}))}
        _append_jsonl(self._episode_dir / "meta.jsonl", rec)
        self._episode_started = False

    def end_run(self, status: str, summary: Optional[Mapping[str, Any]] = None) -> None:
        if not self.cfg.enabled or self._run_dir is None:
            return
        rec = {
            "ts": _now_ts(),
            "run_id": self.ctx.run_id,
            "type": "run",
            "name": "run.end",
            "payload": {"status": status, "summary": self._truncate(dict(summary or {}))},
            "tags": list(self.ctx.tags),
        }
        _append_jsonl(self._run_dir / "run_meta.jsonl", rec)
        self._run_started = False


# --------------------------
# Global singleton
# --------------------------

_TELEMETRY: Optional[Telemetry] = None


def get_telemetry(run_id: Optional[str] = None) -> Telemetry:
    global _TELEMETRY
    if _TELEMETRY is None:
        _TELEMETRY = Telemetry()
    if run_id:
        _TELEMETRY.set_run(run_id)
    return _TELEMETRY


# --------------------------
# Backward-compatible helpers
# --------------------------

def log_event(event: Dict[str, Any]) -> None:
    """
    Legacy: append an event dict.

    We map it to a named event and preserve the payload as-is.
    """
    tel = get_telemetry()
    name = _safe_str(event.get("name") or event.get("event") or "legacy.event")
    tel.event(name, event, severity=_safe_str(event.get("severity") or "INFO"))


def log_test_result(result: Dict[str, Any]) -> None:
    """
    Legacy: write test run details.

    In unified telemetry:
    - emit an event `sandbox.test_result`
    - emit a few common metrics if available
    """
    tel = get_telemetry()
    tel.event("sandbox.test_result", result, severity="INFO")

    # best-effort common metrics
    try:
        if "duration_ms" in result:
            tel.metric("sandbox.ms", float(result["duration_ms"]))
        elif "duration" in result:
            tel.metric("sandbox.s", float(result["duration"]))
    except Exception:
        pass
    try:
        if "failed" in result and isinstance(result["failed"], int):
            tel.metric("tests.failed", float(result["failed"]))
        if "passed" in result and isinstance(result["passed"], int):
            tel.metric("tests.passed", float(result["passed"]))
        if "ran" in result and isinstance(result["ran"], int):
            tel.metric("tests.ran", float(result["ran"]))
    except Exception:
        pass

    # Optional legacy flat JSONL output
    cfg = load().telemetry
    if getattr(cfg, "write_legacy", False):
        payload = dict(result)
        payload.setdefault("ts", _now_ts())
        _append_jsonl(Path(cfg.test_runs_path), payload)


def emit_metrics(metrics: Dict[str, Any]) -> None:
    """Console-friendly metric line."""
    line = " | ".join(f"{k}={v}" for k, v in metrics.items())
    print(f"[metrics] {line}")
