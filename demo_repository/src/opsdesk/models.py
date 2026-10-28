from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Priority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Incident:
    incident_id: str
    title: str
    description: str
    service: str
    customer_count: int = 0
    tags: set[str] = field(default_factory=set)
    priority: Priority = Priority.NORMAL

    @property
    def searchable_text(self) -> str:
        return f"{self.title} {self.description} {self.service}".lower()
