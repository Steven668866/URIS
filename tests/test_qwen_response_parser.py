from uris_platform.services.qwen_adapter import (
    QwenLiveAdapter,
    build_fallback_qwen_response,
    parse_qwen_structured_response,
)


def test_parse_qwen_structured_response_extracts_json_block():
    text = """
    这里是自然语言回答：我看到桌面上有一个杯子，建议先确认是否需要清理。

    ```json
    {
      "user_response": "我看到桌面上有一个杯子，建议先确认是否需要清理。",
      "analysis_json": {
        "intent": "scene_query",
        "user_goal": "identify objects",
        "observed_objects": ["cup"],
        "spatial_relations": [],
        "scene_summary": "桌面上检测到杯子",
        "recommendation_steps": ["确认杯子是否为目标物体"],
        "clarification_needed": false,
        "clarifying_question": "",
        "confidence": 0.82,
        "evidence_basis": ["camera_frame", "yolo_detection"],
        "limitations": ["单帧视角有限"]
      }
    }
    ```
    """
    parsed = parse_qwen_structured_response(text)
    assert parsed["json_valid"] is True
    assert parsed["user_response"].startswith("我看到桌面上有一个杯子")
    assert parsed["analysis_json"]["intent"] == "scene_query"
    assert parsed["analysis_json"]["confidence"] == 0.82


def test_parse_qwen_structured_response_falls_back_when_json_missing():
    parsed = parse_qwen_structured_response("我看到一个杯子，但无法稳定识别其他物体。")
    assert parsed["json_valid"] is False
    assert parsed["user_response"].startswith("我看到一个杯子")
    assert parsed["analysis_json"]["clarification_needed"] is False


def test_build_fallback_qwen_response_uses_detection_summary():
    response = build_fallback_qwen_response(
        user_query="桌上有什么？",
        detection_summary="3 detections: table x1, cup x2",
    )
    assert response["json_valid"] is False
    assert "cup x2" in response["user_response"]
    assert response["analysis_json"]["evidence_basis"] == ["yolo_detection_summary"]


def test_qwen_live_adapter_caches_same_query_and_scene():
    adapter = QwenLiveAdapter(adapter_path=None)

    kwargs = dict(
        user_query="桌上有什么？",
        scene_summary="2 detections: cup x1, table x1",
        detections=[{"label": "cup", "confidence": 0.9, "bbox": [1, 2, 3, 4]}],
        preferences=(),
        recent_turns=(),
        object_registry=(),
        recent_scene_events=(),
        reference_resolution={},
        enable_cache=True,
        include_prompt_bundle=False,
        compact_prompt_context=True,
    )

    first = adapter.generate_live_response(**kwargs)
    second = adapter.generate_live_response(**kwargs)

    assert first.get("cache_hit") is False
    assert second.get("cache_hit") is True
    assert second["user_response"] == first["user_response"]


def test_qwen_live_adapter_cache_ignores_recent_turns_drift_for_same_scene_query():
    adapter = QwenLiveAdapter(adapter_path=None)

    base_kwargs = dict(
        user_query="桌上有什么？",
        scene_summary="2 detections: cup x1, table x1",
        detections=[{"label": "cup", "confidence": 0.9, "bbox": [1, 2, 3, 4]}],
        preferences=(),
        object_registry=(),
        recent_scene_events=(),
        reference_resolution={},
        enable_cache=True,
        include_prompt_bundle=False,
        compact_prompt_context=True,
    )

    first = adapter.generate_live_response(recent_turns=(), **base_kwargs)
    second = adapter.generate_live_response(
        recent_turns=(
            {"role": "user", "content": "桌上有什么？"},
            {"role": "assistant", "content": "这里有一个杯子和一张桌子。"},
        ),
        **base_kwargs,
    )

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True


def test_qwen_live_adapter_short_circuits_on_ambiguous_reference_with_clarification():
    adapter = QwenLiveAdapter(adapter_path=None)
    response = adapter.generate_live_response(
        user_query="那个杯子怎么处理？",
        scene_summary="2 detections: cup x2, table x1",
        detections=[
            {"label": "cup", "confidence": 0.92, "bbox": [1, 1, 2, 2]},
            {"label": "cup", "confidence": 0.88, "bbox": [3, 1, 4, 2]},
        ],
        preferences=(),
        recent_turns=(),
        object_registry=(
            {"obj_id": "obj-0001", "label": "cup", "center_norm": [0.2, 0.5], "status": "visible"},
            {"obj_id": "obj-0002", "label": "cup", "center_norm": [0.7, 0.5], "status": "visible"},
        ),
        recent_scene_events=(),
        reference_resolution={
            "resolved": False,
            "clarification_needed": True,
            "candidate_count": 2,
            "candidates": ["obj-0001", "obj-0002"],
            "clarifying_question": "需要澄清：你指的是左边还是右边的杯子？",
        },
        enable_cache=True,
        include_prompt_bundle=False,
        compact_prompt_context=True,
    )

    assert response["json_valid"] is True
    assert response["analysis_json"]["clarification_needed"] is True
    assert "澄清" in response["analysis_json"]["clarifying_question"]
    assert response["parse_error"] is None
