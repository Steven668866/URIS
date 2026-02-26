from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from uris_platform.domain import ActionPlan, SceneState


def select_video_sampling_fps(duration_seconds: float) -> float:
    if duration_seconds < 10:
        return 1.0
    if duration_seconds < 30:
        return 0.5
    return 0.33


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


@lru_cache(maxsize=512)
def _intent_from_command(command: str) -> str:
    normalized = _normalize_text(command)
    if any(word in normalized for word in ("move", "reposition", "shift")):
        return "move"
    if any(word in normalized for word in ("clean", "wipe", "wash", "tidy")):
        return "clean"
    if any(word in normalized for word in ("bring", "fetch", "get")):
        return "fetch"
    return "unknown"


def _find_target(scene: SceneState, command: str) -> str | None:
    normalized = _normalize_text(command)
    for obj in scene.objects:
        if obj.name.lower() in normalized:
            return obj.name
    # Fallback for cleaning: prefer objects explicitly marked dirty
    if any(word in normalized for word in ("clean", "wipe", "wash")):
        for obj in scene.objects:
            if "dirty" in obj.state.lower():
                return obj.name
    return None


def _preference_steps(preferences: Sequence[str] | None) -> list[str]:
    if not preferences:
        return []
    steps: list[str] = []
    for pref in preferences:
        lowered = pref.lower()
        if "confirm" in lowered:
            steps.append("Confirm the action with the user before execution.")
        elif "safety" in lowered:
            steps.append("Run a safety scan for obstacles and humans nearby.")
    return steps


def plan_robot_response(
    scene: SceneState, command: str, preferences: Sequence[str] | None = None
) -> ActionPlan:
    intent = _intent_from_command(command)
    target = _find_target(scene, command)
    pref_steps = _preference_steps(preferences)

    if intent == "move":
        if not target:
            return ActionPlan(
                action="clarify_request",
                target=None,
                steps=[
                    "Ask which object should be moved.",
                    "Confirm destination zone before generating the recommendation.",
                ],
                explanation="I can simulate a move recommendation, but I need you to clarify the target and destination.",
                confidence=0.32,
                adaptation_note="Prefer explicit object names for furniture-layout recommendations.",
            )
        steps = [
            *pref_steps,
            f"Identify the {target} in the {scene.room} scene context.",
            "Generate a safety-aware reposition suggestion.",
            f"Simulate the requested destination change for the {target}.",
            "Return the recommendation and ask for confirmation.",
        ]
        return ActionPlan(
            action="reposition_object",
            target=target,
            steps=steps,
            explanation=(
                f"I will simulate a reposition recommendation for the {target} and provide a safety-aware confirmation."
            ),
            confidence=0.84 if target else 0.55,
            adaptation_note="Furniture-layout preferences are logged for future recommendation refinement.",
        )

    if intent == "clean":
        if not target:
            return ActionPlan(
                action="clarify_request",
                target=None,
                steps=["Ask which item or area should be cleaned."],
                explanation="Please tell me what needs cleaning so I can plan the task.",
                confidence=0.35,
                adaptation_note="Track common cleaning targets per room.",
            )
        return ActionPlan(
            action="clean_object",
            target=target,
            steps=[
                *pref_steps,
                f"Inspect the {target} condition.",
                f"Clean the {target} with a suitable tool.",
                "Dispose residue and restore item position.",
                "Report the cleaning result.",
            ],
            explanation=f"I selected the {target} as the cleaning target based on your command and scene state.",
            confidence=0.79,
            adaptation_note="Dirty-object priority was applied during target selection.",
        )

    if intent == "fetch":
        return ActionPlan(
            action="fetch_item",
            target=target,
            steps=[
                *pref_steps,
                "Clarify destination person/location if missing.",
                "Simulate safe retrieval steps.",
                "Present a delivery suggestion and confirm intent.",
            ],
            explanation="I can simulate a fetch-task recommendation, but I may need a destination confirmation.",
            confidence=0.68,
            adaptation_note="Future versions should integrate user-position tracking.",
        )

    return ActionPlan(
        action="clarify_request",
        target=None,
        steps=[
            "Acknowledge the request.",
            "Ask a clarifying question about the intended task outcome.",
            "Offer examples of supported commands (move, clean, fetch).",
        ],
        explanation="I need to clarify the request before generating a safe interaction recommendation.",
        confidence=0.25,
        adaptation_note="Unsupported commands are captured for prompt/template expansion.",
    )
