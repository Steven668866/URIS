from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class LiveTriggerPolicy:
    min_interval_seconds: float = 1.5
    allow_auto_scene_trigger: bool = False


@dataclass(frozen=True)
class LiveTriggerDecision:
    trigger: bool
    reason: str


def should_trigger_qwen(
    *,
    policy: LiveTriggerPolicy,
    now_ts: float,
    last_qwen_ts: float | None,
    user_submitted: bool,
    scene_signature_changed: bool,
) -> LiveTriggerDecision:
    if user_submitted:
        return LiveTriggerDecision(True, "user_submit")

    if not scene_signature_changed:
        return LiveTriggerDecision(False, "no_scene_change")

    if not policy.allow_auto_scene_trigger:
        return LiveTriggerDecision(False, "auto_disabled")

    if last_qwen_ts is not None and (now_ts - last_qwen_ts) < policy.min_interval_seconds:
        return LiveTriggerDecision(False, "cooldown")

    return LiveTriggerDecision(True, "scene_change")


def scene_signature_from_detections(detections: Iterable[dict]) -> str:
    """Compact signature for scene-change checks."""
    parts: list[str] = []
    for det in sorted(
        detections,
        key=lambda d: (
            str(d.get("label", "")),
            -float(d.get("confidence", 0.0)),
        ),
    ):
        label = str(det.get("label", "object"))
        conf_bucket = int(float(det.get("confidence", 0.0)) * 10)
        parts.append(f"{label}:{conf_bucket}")
    return "|".join(parts)
