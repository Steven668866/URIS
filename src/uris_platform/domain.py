from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SceneObject:
    name: str
    zone: str
    state: str = "idle"
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneState:
    room: str
    objects: list[SceneObject]
    notes: str = ""

    def find_object(self, name: str) -> SceneObject | None:
        normalized = name.lower().strip()
        for obj in self.objects:
            if obj.name.lower() == normalized:
                return obj
        return None


@dataclass(frozen=True)
class ActionPlan:
    action: str
    target: str | None
    steps: list[str]
    explanation: str
    confidence: float
    adaptation_note: str | None = None
