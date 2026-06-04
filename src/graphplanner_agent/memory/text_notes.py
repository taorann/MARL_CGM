from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TextNotes:
    notes: list[dict[str, str | None]] = field(default_factory=list)

    def add(self, note: str, tag: str | None = None) -> None:
        self.notes.append({"tag": tag, "note": note})

    def summary(self, limit: int = 8) -> list[dict[str, str | None]]:
        return self.notes[-limit:]
