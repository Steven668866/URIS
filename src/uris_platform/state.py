from __future__ import annotations

from typing import Any, MutableMapping


DEFAULT_STATE: dict[str, Any] = {
    "scene_name": "living_room",
    "interaction_history": [],
    "perf_history": [],
    "evaluation_feedback": [],
    "live_camera_enabled": False,
    "live_camera_status": "idle",
    "live_camera_mode": "snapshot",
    "live_detections": [],
    "live_scene_summary": "",
    "live_object_registry": [],
    "live_scene_event_log": [],
    "live_detection_history": [],
    "live_event_summary": "",
    "live_latest_reference_resolution": None,
    "live_fast_mode": True,
    "live_enable_response_cache": True,
    "live_store_prompt_bundle_debug": False,
    "live_qwen_cache_stats": {"hits": 0, "misses": 0},
    "live_tracker_mode": "simple",
    "live_tracker_status": "idle",
    "live_last_frame_ts": None,
    "live_last_snapshot_digest": "",
    "live_last_qwen_ts": None,
    "live_last_qwen_query": "",
    "live_last_detection_meta": {},
    "live_qwen_history": [],
    "live_errors": [],
    "live_scene_signature": "",
    "prompt_version": "uris-qwen-live-v1",
    "robot_state": {
        "mode": "idle",
        "location": "dock",
        "last_action": None,
    },
    "user_preferences": [],
    "scene_overrides": {},
    "latest_plan": None,
}


def initialize_session_state(session_state: MutableMapping[str, Any]) -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in session_state:
            # Avoid sharing mutable defaults across sessions.
            if isinstance(value, dict):
                session_state[key] = dict(value)
            elif isinstance(value, list):
                session_state[key] = list(value)
            else:
                session_state[key] = value
