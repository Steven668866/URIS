from uris_platform.domain import SceneObject, SceneState
from uris_platform.services.scenario_engine import plan_robot_response, select_video_sampling_fps


def _scene():
    return SceneState(
        room="living_room",
        objects=[
            SceneObject(name="chair", zone="left", state="idle"),
            SceneObject(name="table", zone="center", state="set"),
            SceneObject(name="cup", zone="table", state="dirty"),
        ],
    )


def test_select_video_sampling_fps_policy():
    assert select_video_sampling_fps(5) == 1.0
    assert select_video_sampling_fps(15) == 0.5
    assert select_video_sampling_fps(45) == 0.33


def test_plan_robot_response_move_command_targets_object():
    plan = plan_robot_response(_scene(), "Please move the chair to the right", preferences=["Add a safety confirmation before moving furniture"])
    assert plan.action == "reposition_object"
    assert plan.target == "chair"
    assert any("confirm" in step.lower() for step in plan.steps)
    assert plan.confidence >= 0.5


def test_plan_robot_response_clean_command_prefers_dirty_object():
    plan = plan_robot_response(_scene(), "Clean up the dirty cup")
    assert plan.action == "clean_object"
    assert plan.target == "cup"
    assert "cup" in plan.explanation.lower()


def test_plan_robot_response_fallback_for_unknown_command():
    plan = plan_robot_response(_scene(), "Do something thoughtful")
    assert plan.action == "clarify_request"
    assert plan.target is None
    assert "clarify" in plan.explanation.lower()
