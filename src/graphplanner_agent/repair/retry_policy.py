from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RepairHistory:
    attempts: list[str] = field(default_factory=list)
    last_status: str | None = None
    last_error_origin: str | None = None
    last_memory_ids: tuple[str, ...] = ()

    def duplicate(self, signature: str) -> bool:
        return signature in self.attempts

    def record(self, signature: str) -> None:
        self.attempts.append(signature)

    def record_outcome(self, status: str, memory_ids: list[str], error_origin: str | None = None) -> None:
        self.last_status = status
        self.last_error_origin = error_origin
        self.last_memory_ids = tuple(sorted(memory_ids))

    def failed_with_same_memory(self, memory_ids: list[str]) -> bool:
        if self.last_status not in {"patch_rejected", "syntax_failed", "test_failed"}:
            return False
        if self.last_memory_ids != tuple(sorted(memory_ids)):
            return False
        if self.last_status == "test_failed":
            return True
        if self.last_error_origin in {"cgm_patch_schema", "patch_format_validation", "generated_patch"}:
            return False
        return True
