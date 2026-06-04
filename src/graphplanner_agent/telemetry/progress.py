from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass(slots=True)
class ProgressTracker:
    path: Path | None = None
    counts: dict[str, int] = field(default_factory=lambda: {"pass": 0, "not_pass": 0, "bug": 0})
    baseline_counts: dict[str, int] = field(default_factory=lambda: {"pass": 0, "not_pass": 0, "bug": 0})
    baseline_label: str | None = None
    tasks: dict[str, dict[str, object]] = field(default_factory=dict)

    def seed_counts(self, counts: dict[str, int], *, label: str | None = None) -> None:
        self.baseline_counts = {key: max(0, int(counts.get(key, 0))) for key in ("pass", "not_pass", "bug")}
        self.baseline_label = label
        self._write()

    def start_task(self, task_id: str, metadata: dict[str, object] | None = None) -> None:
        self.tasks[task_id] = {
            "task_id": task_id,
            "phase": "starting",
            "metadata": metadata or {},
            "steps": [],
            "final_status": None,
            "reason": None,
        }
        self._write()

    def update_task(self, task_id: str, phase: str, metadata: dict[str, object] | None = None) -> None:
        task = self.tasks.setdefault(task_id, {"task_id": task_id, "steps": []})
        task["phase"] = phase
        if metadata:
            current = task.get("metadata")
            if not isinstance(current, dict):
                current = {}
            current.update(metadata)
            task["metadata"] = current
        self._write()

    def record_step(self, task_id: str, step: int, tool: str, status: str, elapsed: float, summary: str) -> None:
        task = self.tasks.setdefault(task_id, {"task_id": task_id, "steps": []})
        task["phase"] = "running"
        steps = task.setdefault("steps", [])
        if isinstance(steps, list):
            steps.append(
                {
                    "step": step,
                    "tool": tool,
                    "status": status,
                    "elapsed": round(elapsed, 3),
                    "summary": summary,
                }
            )
            del steps[:-80]
        self._write()

    def record(self, status: str, task_id: str | None = None, reason: str | None = None) -> dict[str, object]:
        if status not in self.counts:
            status = "bug"
        self.counts[status] += 1
        if task_id:
            task = self.tasks.setdefault(task_id, {"task_id": task_id, "steps": []})
            task["phase"] = "finished"
            task["final_status"] = status
            task["reason"] = reason
        summary = self.summary()
        self._write()
        return summary

    def summary(self) -> dict[str, object]:
        combined = {key: self.baseline_counts.get(key, 0) + self.counts.get(key, 0) for key in ("pass", "not_pass", "bug")}
        total = sum(combined.values())
        bug_excluded = total - combined["bug"]
        return {
            "total": total,
            **combined,
            "current_run_total": sum(self.counts.values()),
            "current_run_counts": dict(self.counts),
            "baseline_total": sum(self.baseline_counts.values()),
            "baseline_counts": dict(self.baseline_counts),
            "accuracy": combined["pass"] / total if total else 0.0,
            "bug_excluded_accuracy": combined["pass"] / bug_excluded if bug_excluded else 0.0,
        }

    def render_markdown(self) -> str:
        summary = self.summary()
        lines = [
            "# GraphPlanner Progress",
            "",
            f"- total: {summary['total']}",
            f"- pass: {summary['pass']}",
            f"- not_pass: {summary['not_pass']}",
            f"- bug: {summary['bug']}",
            f"- accuracy: {summary['accuracy']:.3f}",
            f"- bug_excluded_accuracy: {summary['bug_excluded_accuracy']:.3f}",
            f"- current_run_total: {summary['current_run_total']}",
            f"- baseline_total: {summary['baseline_total']}",
            "",
        ]
        if summary["baseline_total"]:
            if self.baseline_label:
                lines.append(f"- baseline_label: {self.baseline_label}")
            lines.append(f"- baseline_counts: `{json.dumps(summary['baseline_counts'], sort_keys=True)}`")
            lines.append(f"- current_run_counts: `{json.dumps(summary['current_run_counts'], sort_keys=True)}`")
            lines.append("")
        for task in self.tasks.values():
            lines.extend(_render_task(task))
        return "\n".join(lines)

    def _write(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(self.render_markdown(), encoding="utf-8")


def _render_task(task: dict[str, object]) -> list[str]:
    lines = [
        f"## {task.get('task_id')}",
        "",
        f"- phase: {task.get('phase')}",
    ]
    if task.get("final_status"):
        lines.append(f"- final_status: {task.get('final_status')}")
    if task.get("reason"):
        lines.append(f"- reason: {task.get('reason')}")
    metadata = task.get("metadata")
    if isinstance(metadata, dict) and metadata:
        lines.append(f"- metadata: `{json.dumps(metadata, ensure_ascii=False, sort_keys=True)}`")
    steps = task.get("steps")
    if isinstance(steps, list) and steps:
        lines.extend(["", "| step | tool | status | elapsed | summary |", "| ---: | --- | --- | ---: | --- |"])
        for step in steps[-40:]:
            if not isinstance(step, dict):
                continue
            lines.append(
                "| {step} | {tool} | {status} | {elapsed:.1f}s | {summary} |".format(
                    step=int(step.get("step") or 0),
                    tool=_cell(step.get("tool")),
                    status=_cell(step.get("status")),
                    elapsed=float(step.get("elapsed") or 0.0),
                    summary=_cell(step.get("summary")),
                )
            )
    lines.append("")
    return lines


def _cell(value: object) -> str:
    text = str(value or "").replace("\n", " ")
    return text.replace("|", "\\|")
