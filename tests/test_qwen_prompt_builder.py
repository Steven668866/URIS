import json

from uris_platform.prompts.qwen_interaction_prompt import build_qwen_interaction_prompt


def test_prompt_builder_contains_ux_and_academic_constraints():
    prompt = build_qwen_interaction_prompt(
        user_query="请告诉我桌子上有什么，并给出整理建议。",
        scene_summary="客厅场景，桌子上有杯子和纸巾。",
        detections=[
            {"label": "cup", "confidence": 0.93, "bbox": [100, 120, 180, 220]},
            {"label": "tissue", "confidence": 0.88, "bbox": [200, 140, 260, 210]},
        ],
        preferences=["回答简洁但专业"],
        recent_turns=[{"role": "user", "content": "现在画面里有什么？"}],
        object_registry=[
            {"obj_id": "obj-0001", "label": "cup", "center_norm": [0.2, 0.4], "status": "visible"},
        ],
        recent_scene_events=[{"type": "appear", "message": "cup appeared"}],
        reference_resolution={"resolved": False, "clarification_needed": False},
    )

    assert prompt["schema_version"] == "uris-qwen-live-v1"
    system = prompt["system_prompt"]
    user = prompt["user_prompt"]

    assert "不要输出思维链" in system
    assert "先给出直接回答" in system
    assert "区分“观察事实”和“推断/建议”" in system
    assert "interaction simulation" in system.lower()
    assert "JSON" in system
    assert "scene_summary" in user
    assert "detections" in user
    assert "object_registry" in user
    assert "recent_scene_events" in user
    assert "reference_resolution" in user
    assert "请告诉我桌子上有什么" in user


def test_prompt_builder_requires_structured_fields():
    prompt = build_qwen_interaction_prompt(
        user_query="这个杯子是不是脏的？",
        scene_summary="厨房台面上有一个杯子。",
        detections=[{"label": "cup", "confidence": 0.91, "bbox": [1, 2, 3, 4]}],
        preferences=[],
        recent_turns=[],
        object_registry=[],
        recent_scene_events=[],
        reference_resolution={},
    )
    schema = prompt["required_json_fields"]
    assert "observed_objects" in schema
    assert "recommendation_steps" in schema
    assert "confidence" in schema
    assert "limitations" in schema


def test_prompt_builder_compacts_context_in_fast_mode():
    prompt = build_qwen_interaction_prompt(
        user_query="总结当前场景并指出我说的那个杯子。",
        scene_summary="客厅场景",
        detections=[
            {"label": f"obj{i}", "confidence": 0.9 - (i * 0.01), "bbox": [i, i, i + 1, i + 1]}
            for i in range(12)
        ],
        preferences=[],
        recent_turns=[],
        object_registry=[
            {"obj_id": f"obj-{i:04d}", "label": "cup", "center_norm": [0.1 * (i % 5), 0.5], "status": "visible"}
            for i in range(14)
        ],
        recent_scene_events=[{"type": "count_change", "message": f"event-{i}"} for i in range(9)],
        reference_resolution={"resolved": False},
        compact_context=True,
    )
    user = prompt["user_prompt"]
    tail = user.split("context_json:", 1)[1]
    json_blob = tail.split("请输出：", 1)[0].replace("\\n", "\n").strip()
    ctx = json.loads(json_blob)

    assert len(ctx["detections"]) <= 8
    assert len(ctx["object_registry"]) <= 10
    assert len(ctx["recent_scene_events"]) <= 5
