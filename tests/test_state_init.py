from uris_platform.state import initialize_session_state


def test_initialize_session_state_sets_defaults():
    store = {}
    initialize_session_state(store)
    assert store["scene_name"] == "living_room"
    assert store["interaction_history"] == []
    assert store["perf_history"] == []
    assert store["evaluation_feedback"] == []
    assert store["live_camera_enabled"] is False
    assert store["live_camera_status"] == "idle"
    assert store["live_detections"] == []
    assert store["live_scene_summary"] == ""
    assert store["live_object_registry"] == []
    assert store["live_scene_event_log"] == []
    assert store["live_detection_history"] == []
    assert store["live_event_summary"] == ""
    assert store["live_latest_reference_resolution"] is None
    assert store["live_fast_mode"] is True
    assert store["live_enable_response_cache"] is True
    assert store["live_store_prompt_bundle_debug"] is False
    assert store["live_qwen_cache_stats"] == {"hits": 0, "misses": 0}
    assert store["live_tracker_mode"] == "simple"
    assert store["live_tracker_status"] == "idle"
    assert store["live_qwen_history"] == []
    assert store["prompt_version"] == "uris-qwen-live-v1"
    assert store["robot_state"]["mode"] == "idle"


def test_initialize_session_state_preserves_existing_values():
    store = {"scene_name": "kitchen", "interaction_history": [{"command": "hi"}]}
    initialize_session_state(store)
    assert store["scene_name"] == "kitchen"
    assert store["interaction_history"] == [{"command": "hi"}]
