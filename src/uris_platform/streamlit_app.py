from __future__ import annotations

import json
import os
import time
import hashlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import streamlit as st
from PIL import Image
import numpy as np

from uris_platform.config import load_platform_config
from uris_platform.domain import SceneObject, SceneState
from uris_platform.prompts.qwen_interaction_prompt import PROMPT_VERSION
from uris_platform.services.evaluation import compute_evaluation_summary
from uris_platform.services.live_camera import (
    LiveTriggerPolicy,
    scene_signature_from_detections,
    should_trigger_qwen,
)
from uris_platform.services.live_scene_memory import (
    ingest_live_detections,
    resolve_reference_query,
)
from uris_platform.services.object_tracking import (
    LiveObjectTracker,
    available_tracking_modes,
)
from uris_platform.services.perf import timed_stage
from uris_platform.services.qwen_adapter import QwenLiveAdapter
from uris_platform.services.scenario_engine import (
    plan_robot_response,
    select_video_sampling_fps,
)
from uris_platform.services.vision_yolo import (
    build_live_scene_summary,
    run_mock_or_passthrough_detection,
    run_ultralytics_detection_on_bgr,
    yolo_runtime_status,
)
from uris_platform.state import initialize_session_state
from uris_platform.ui.components import (
    action_plan_as_dict,
    render_action_plan,
    render_hero,
    render_interaction_history,
    render_metric_cards,
    render_response_card,
    render_perf_table,
    render_scene_objects,
    render_status_badges,
    render_surface,
)
from uris_platform.ui.theme import inject_theme


ROOT_DIR = Path(__file__).resolve().parents[2]
SCENE_DIR = ROOT_DIR / "configs" / "scenes"
DEFAULT_QWEN_LORA_PATH = str(ROOT_DIR / "Qwen2.5-VL-URIS-Predictive-LoRA")


def _safe_page_config() -> None:
    try:
        st.set_page_config(
            page_title="URIS Home Robot Interaction Platform",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        # Streamlit disallows duplicate set_page_config in same run.
        pass


@st.cache_data(show_spinner=False)
def load_scene_templates(scene_dir: str) -> dict[str, dict[str, Any]]:
    templates: dict[str, dict[str, Any]] = {}
    path = Path(scene_dir)
    if not path.exists():
        return templates
    for file in sorted(path.glob("*.json")):
        try:
            templates[file.stem] = json.loads(file.read_text(encoding="utf-8"))
        except Exception:
            continue
    return templates


def _scene_from_payload(payload: dict[str, Any]) -> SceneState:
    objects = [
        SceneObject(
            name=str(obj.get("name", "object")),
            zone=str(obj.get("zone", "unknown")),
            state=str(obj.get("state", "idle")),
            attributes=dict(obj.get("attributes", {})),
        )
        for obj in payload.get("objects", [])
    ]
    return SceneState(
        room=str(payload.get("room", "living_room")),
        objects=objects,
        notes=str(payload.get("notes", "")),
    )


def _current_scene_payload(templates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scene_name = st.session_state.scene_name
    override = st.session_state.scene_overrides.get(scene_name)
    if isinstance(override, dict):
        return override
    return templates.get(scene_name, templates.get("living_room", {"room": scene_name, "objects": []}))


def _append_bounded(store_key: str, item: Any, max_len: int) -> None:
    bucket = st.session_state[store_key]
    bucket.append(item)
    if len(bucket) > max_len:
        del bucket[:-max_len]


def _record_interaction(command: str, scene: SceneState, plan_dict: dict[str, Any], max_len: int) -> None:
    _append_bounded(
        "interaction_history",
        {
            "timestamp": time.time(),
            "command": command,
            "room": scene.room,
            "plan": plan_dict,
        },
        max_len,
    )


def _record_perf(command: str, stages: dict[str, float], max_len: int) -> None:
    total_ms = sum(stages.values())
    _append_bounded(
        "perf_history",
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "command": command,
            "total_ms": total_ms,
            "stages": stages,
        },
        max_len,
    )


def _upsert_evaluation_feedback(record: dict[str, Any]) -> None:
    feedback = st.session_state.evaluation_feedback
    idx = int(record["interaction_index"])
    for pos, existing in enumerate(feedback):
        try:
            if int(existing.get("interaction_index")) == idx:
                feedback[pos] = record
                return
        except (TypeError, ValueError):
            continue
    feedback.append(record)


@st.cache_resource(show_spinner=False)
def _get_live_qwen_adapter(adapter_path: str) -> QwenLiveAdapter:
    return QwenLiveAdapter(adapter_path=adapter_path)


@st.cache_resource(show_spinner=False)
def _get_live_tracker(mode: str) -> LiveObjectTracker:
    return LiveObjectTracker(mode=mode)


def _webrtc_runtime_status() -> tuple[bool, str | None]:
    try:
        import streamlit_webrtc  # noqa: F401

        return True, None
    except Exception as exc:
        return False, str(exc)


def _snapshot_to_bgr(image_file) -> tuple[np.ndarray, tuple[int, int]]:
    pil_image = Image.open(image_file).convert("RGB")
    rgb = np.array(pil_image)
    bgr = rgb[:, :, ::-1].copy()
    h, w = bgr.shape[:2]
    return bgr, (w, h)


def _snapshot_digest(image_file) -> str:
    if hasattr(image_file, "getvalue"):
        data = image_file.getvalue()
    else:
        data = image_file.read()
    return hashlib.sha1(data).hexdigest()


def _run_snapshot_detection(frame_bgr: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    yolo_status = yolo_runtime_status()
    if yolo_status.available:
        try:
            normalized = run_ultralytics_detection_on_bgr(frame_bgr)
            return normalized, {
                "available": True,
                "mode": "ultralytics",
                "error": None,
            }
        except Exception as exc:
            normalized = run_mock_or_passthrough_detection(
                raw_detections=[],
                frame_width=int(frame_bgr.shape[1]),
                frame_height=int(frame_bgr.shape[0]),
            )
            return normalized, {
                "available": False,
                "mode": "fallback",
                "error": str(exc),
            }
    normalized = run_mock_or_passthrough_detection(
        raw_detections=[],
        frame_width=int(frame_bgr.shape[1]),
        frame_height=int(frame_bgr.shape[0]),
    )
    return normalized, {
        "available": False,
        "mode": "fallback",
        "error": yolo_status.reason or "ultralytics not installed",
    }


def _record_live_qwen_interaction(
    *,
    query: str,
    parsed: dict[str, Any],
    source_mode: str,
    room_name: str,
    max_events: int,
    vision_ms: float,
    reference_resolution: dict[str, Any] | None = None,
    event_summary: str | None = None,
    object_registry_size: int | None = None,
    cache_hit: bool | None = None,
) -> None:
    analysis = parsed.get("analysis_json") or {}
    observed = analysis.get("observed_objects") or []
    target = observed[0] if observed else None
    plan_dict = {
        "action": str(analysis.get("intent") or "interactive_query"),
        "target": target,
        "steps": list(analysis.get("recommendation_steps") or []),
        "explanation": str(parsed.get("user_response") or ""),
        "confidence": float(analysis.get("confidence") or 0.0),
        "adaptation_note": None,
    }
    _append_bounded(
        "interaction_history",
        {
            "timestamp": time.time(),
            "command": query,
            "room": room_name,
            "source": source_mode,
            "plan": plan_dict,
            "json_valid": bool(parsed.get("json_valid")),
            "clarification_needed": bool(analysis.get("clarification_needed")),
            "reference_resolution": dict(reference_resolution or {}),
            "event_summary": event_summary,
            "object_registry_size": object_registry_size,
            "cache_hit": cache_hit,
        },
        max_events,
    )
    _record_perf(
        query,
        {
            "vision_ms": vision_ms,
            "qwen_ms": float(parsed.get("latency_ms") or 0.0),
            "vision_to_response_latency_ms": vision_ms + float(parsed.get("latency_ms") or 0.0),
        },
        max_events,
    )


def _render_live_detection_table(detections: list[dict[str, Any]]) -> None:
    if not detections:
        st.info("No detections yet. Capture a snapshot and run detection.")
        return
    rows = []
    for det in detections:
        rows.append(
            {
                "label": det.get("label"),
                "confidence": round(float(det.get("confidence", 0.0)), 3),
                "track_id": det.get("track_id"),
                "bbox": det.get("bbox"),
                "center_norm": det.get("center_norm"),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def _render_live_object_registry_table(registry: list[dict[str, Any]]) -> None:
    if not registry:
        st.caption("Object registry is empty. Run detection to build grounded object memory.")
        return
    rows = []
    for obj in registry:
        rows.append(
            {
                "obj_id": obj.get("obj_id"),
                "label": obj.get("label"),
                "track_id": obj.get("track_id"),
                "status": obj.get("status"),
                "conf": round(float(obj.get("confidence", 0.0)), 3),
                "center_norm": obj.get("center_norm"),
                "seen_count": int(obj.get("seen_count", 0)),
                "mentions": int(obj.get("mention_count", 0)),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def _render_live_scene_events(event_log: list[dict[str, Any]]) -> None:
    if not event_log:
        st.caption("No scene-memory events yet.")
        return
    rows = []
    for event in reversed(event_log[-8:]):
        rows.append(
            {
                "type": event.get("type"),
                "message": event.get("message"),
                "ts": event.get("ts"),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def _live_camera_tab(*, cfg_max_events: int) -> None:
    st.session_state.live_camera_enabled = True
    st.markdown("### Live Camera Interaction (YOLO + Qwen)")
    st.caption(
        "Phase 1: live preview (optional WebRTC) + snapshot detection + on-demand Qwen interaction. "
        "Outputs are interaction recommendations for simulation only."
    )

    yolo_status = yolo_runtime_status()
    webrtc_ok, webrtc_reason = _webrtc_runtime_status()
    qwen_adapter = _get_live_qwen_adapter(DEFAULT_QWEN_LORA_PATH)
    tracker_mode = str(st.session_state.get("live_tracker_mode", "simple"))
    tracker = _get_live_tracker(tracker_mode)
    tracker_status = tracker.status
    st.session_state.live_tracker_status = (
        tracker_status.active_mode if tracker_status.available else f"fallback:{tracker_status.active_mode}"
    )
    cache_stats = qwen_adapter.cache_stats
    last_ts = st.session_state.live_last_frame_ts
    last_age = None
    if isinstance(last_ts, (int, float)):
        last_age = max(0.0, time.time() - last_ts)

    render_status_badges(
        [
            {
                "label": "Camera",
                "value": f"{st.session_state.live_camera_mode}:{st.session_state.live_camera_status}",
                "tone": "ok" if st.session_state.live_camera_status == "running" else "warn",
            },
            {
                "label": "YOLO",
                "value": "ready" if yolo_status.available else "fallback",
                "tone": "ok" if yolo_status.available else "warn",
            },
            {
                "label": "Qwen",
                "value": qwen_adapter.status.mode,
                "tone": "ok" if qwen_adapter.status.available else "warn",
            },
            {
                "label": "Tracker",
                "value": (
                    tracker_status.active_mode
                    if tracker_status.requested_mode == tracker_status.active_mode
                    else f"{tracker_status.requested_mode}->{tracker_status.active_mode}"
                ),
                "tone": "ok" if tracker_status.available else "warn",
            },
            {
                "label": "Last Frame",
                "value": f"{last_age:.1f}s ago" if last_age is not None else "none",
                "tone": "ok" if last_age is not None else "warn",
            },
            {
                "label": "Prompt",
                "value": str(st.session_state.prompt_version),
            },
            {
                "label": "Fast Mode",
                "value": "on" if st.session_state.get("live_fast_mode", True) else "off",
                "tone": "ok" if st.session_state.get("live_fast_mode", True) else "warn",
            },
            {
                "label": "Qwen Cache",
                "value": f"H{cache_stats.get('hits', 0)}/M{cache_stats.get('misses', 0)}",
                "tone": "ok" if cache_stats.get("hits", 0) else "info",
            },
        ]
    )

    render_metric_cards(
        [
            {
                "label": "Camera Mode",
                "value": str(st.session_state.live_camera_mode),
                "sub": f"status: {st.session_state.live_camera_status}",
            },
            {
                "label": "YOLO Runtime",
                "value": "Available" if yolo_status.available else "Fallback",
                "sub": "object detection ready" if yolo_status.available else "install ultralytics for real detection",
            },
            {
                "label": "Tracking",
                "value": tracker_status.active_mode,
                "sub": (
                    "stable track_id for object registry"
                    if tracker_status.available and tracker_status.active_mode != "off"
                    else (tracker_status.reason or "tracking disabled")
                ),
            },
            {
                "label": "Qwen Live Adapter",
                "value": qwen_adapter.status.mode,
                "sub": "LoRA path configured" if qwen_adapter.status.available else "fallback response mode",
            },
            {
                "label": "Prompt Version",
                "value": str(st.session_state.prompt_version),
                "sub": "academic + UX oriented",
            },
            {
                "label": "Qwen Cache",
                "value": f"{cache_stats.get('hits', 0)} hits",
                "sub": f"misses: {cache_stats.get('misses', 0)}",
            },
        ]
    )

    source_col, status_col = st.columns([1.1, 0.9])
    with source_col:
        st.markdown("#### Camera Source")
        render_surface(
            "Interaction Mode Design",
            "YOLO runs on camera snapshots or live-preview-assisted captures, while Qwen is triggered on demand for better responsiveness and lower latency.",
        )
        preferred_mode = st.radio(
            "Input mode",
            options=["snapshot", "webrtc_preview"],
            index=0 if st.session_state.live_camera_mode != "webrtc_preview" else 1,
            format_func=lambda x: "Snapshot Camera (stable)" if x == "snapshot" else "WebRTC Live Preview (optional)",
            horizontal=True,
        )
        if preferred_mode != st.session_state.live_camera_mode:
            st.session_state.live_camera_mode = preferred_mode
            st.rerun()

        tracker_options = available_tracking_modes()
        selected_tracker_mode = st.selectbox(
            "Tracking mode",
            options=tracker_options,
            index=tracker_options.index(tracker_mode) if tracker_mode in tracker_options else 0,
            help="Use simple tracker now; ByteTrack/OC-SORT are adapter-ready placeholders with fallback.",
        )
        if selected_tracker_mode != tracker_mode:
            st.session_state.live_tracker_mode = selected_tracker_mode
            st.rerun()
        if tracker_status.reason:
            st.caption(f"Tracker note: {tracker_status.reason}")

        if st.session_state.live_camera_mode == "webrtc_preview":
            if webrtc_ok:
                try:
                    from streamlit_webrtc import webrtc_streamer

                    st.session_state.live_camera_status = "running"
                    webrtc_streamer(
                        key="uris-live-preview",
                        media_stream_constraints={"video": True, "audio": False},
                    )
                    st.info(
                        "Live preview is active. Use the snapshot capture below to run detection + Qwen in this phase."
                    )
                except Exception as exc:
                    st.session_state.live_camera_status = "error"
                    st.error(f"WebRTC preview failed: {exc}")
            else:
                st.session_state.live_camera_status = "error"
                st.warning("WebRTC preview unavailable in current environment.")
                st.caption(f"Reason: {webrtc_reason}")
                st.code("pip install streamlit-webrtc av", language="bash")
        else:
            st.session_state.live_camera_status = "running"
            st.info("Snapshot camera mode uses your browser camera and is the most stable path for detection + interaction.")

        snapshot = st.camera_input(
            "Capture from camera for detection and interaction",
            key="live_camera_snapshot",
            help="Capture a fresh frame. Then run detection and ask a question based on the current scene.",
        )

        auto_detect = st.checkbox("Auto-run detection after capture", value=True)
        run_detect_click = st.button("Run Object Detection on Snapshot", type="primary")

        if snapshot is not None:
            snapshot_digest = _snapshot_digest(snapshot)
        else:
            snapshot_digest = None

        should_run_detection = False
        if snapshot is not None:
            if run_detect_click:
                should_run_detection = True
            elif auto_detect and snapshot_digest and snapshot_digest != st.session_state.get("live_last_snapshot_digest"):
                should_run_detection = True

        if snapshot is not None and should_run_detection:
            try:
                frame_bgr, (w, h) = _snapshot_to_bgr(snapshot)
                with timed_stage("live_vision") as live_vision:
                    normalized, detector_info = _run_snapshot_detection(frame_bgr)
                    tracker_result = tracker.update(
                        detections=normalized.get("detections", []),
                        now_ts=time.time(),
                    )
                    normalized["detections"] = tracker_result.get("detections", normalized.get("detections", []))
                    scene_summary = build_live_scene_summary(normalized)
                now_ts = time.time()
                st.session_state.live_detections = normalized.get("detections", [])
                st.session_state.live_scene_summary = scene_summary
                st.session_state.live_last_frame_ts = now_ts
                st.session_state.live_scene_signature = scene_signature_from_detections(st.session_state.live_detections)
                st.session_state.live_last_snapshot_digest = snapshot_digest
                memory_update = ingest_live_detections(
                    registry=list(st.session_state.get("live_object_registry") or []),
                    detection_history=list(st.session_state.get("live_detection_history") or []),
                    event_log=list(st.session_state.get("live_scene_event_log") or []),
                    detections=list(st.session_state.live_detections),
                    scene_summary=scene_summary,
                    now_ts=now_ts,
                    max_history=max(12, min(cfg_max_events, 50)),
                    max_events=max(16, min(cfg_max_events * 2, 100)),
                )
                st.session_state.live_object_registry = memory_update["registry"]
                st.session_state.live_detection_history = memory_update["detection_history"]
                st.session_state.live_scene_event_log = memory_update["event_log"]
                st.session_state.live_event_summary = memory_update["event_summary"]
                st.session_state.live_last_detection_meta = {
                    "frame_width": w,
                    "frame_height": h,
                    "yolo_mode": detector_info["mode"],
                    "yolo_available": detector_info["available"],
                    "vision_latency_ms": round(live_vision.duration_ms, 3),
                    "yolo_error": detector_info.get("error"),
                    "tracker_requested_mode": tracker_status.requested_mode,
                    "tracker_active_mode": tracker_status.active_mode,
                    "tracker_available": tracker_status.available,
                    "tracker_reason": tracker_status.reason,
                    "tracker_track_count": (tracker_result.get("tracker_meta") or {}).get("track_count", 0),
                    "registry_size": len(st.session_state.live_object_registry),
                    "event_count": len(st.session_state.live_scene_event_log),
                }
                if detector_info.get("error"):
                    st.warning(f"YOLO unavailable or failed. Fallback summary used. Details: {detector_info['error']}")
                else:
                    st.success("Detection complete.")
            except Exception as exc:
                st.session_state.live_errors.append(
                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "stage": "snapshot_detection",
                        "error": str(exc),
                    }
                )
                st.error(f"Snapshot detection failed: {exc}")

    with status_col:
        st.markdown("#### Live Detection Status")
        if st.session_state.live_scene_summary:
            render_surface(
                "Current Scene Summary",
                str(st.session_state.live_scene_summary),
            )
        if st.session_state.get("live_event_summary"):
            render_surface(
                "Temporal Event Summary",
                str(st.session_state.live_event_summary),
            )
        st.json(
            {
                "live_camera_status": st.session_state.live_camera_status,
                "mode": st.session_state.live_camera_mode,
                "last_frame_ts": st.session_state.live_last_frame_ts,
                "scene_summary": st.session_state.live_scene_summary,
                "live_scene_signature": st.session_state.live_scene_signature,
                "live_event_summary": st.session_state.get("live_event_summary"),
                "live_tracker_mode": st.session_state.get("live_tracker_mode"),
                "live_tracker_status": st.session_state.get("live_tracker_status"),
                "live_object_registry_size": len(st.session_state.get("live_object_registry") or []),
                "live_event_log_size": len(st.session_state.get("live_scene_event_log") or []),
                "live_fast_mode": st.session_state.get("live_fast_mode"),
                "live_enable_response_cache": st.session_state.get("live_enable_response_cache"),
                "live_qwen_cache_stats": st.session_state.get("live_qwen_cache_stats"),
                "last_detection_meta": st.session_state.get("live_last_detection_meta"),
            },
            expanded=False,
        )
        _render_live_detection_table(st.session_state.live_detections)
        with st.expander("Object Registry (grounded scene memory)", expanded=False):
            _render_live_object_registry_table(st.session_state.get("live_object_registry") or [])
        with st.expander("Recent Scene Events", expanded=False):
            _render_live_scene_events(st.session_state.get("live_scene_event_log") or [])

    st.markdown("#### Qwen Interaction on Current Live Scene")
    left, right = st.columns([1.05, 0.95])
    with left:
        render_surface(
            "Response Style",
            "Answer-first, evidence-aware, academically rigorous. The system separates observation from inference and logs structured JSON for evaluation.",
        )
        with st.form("live_qwen_interaction_form", clear_on_submit=True):
            live_query = st.text_input(
                "Question / command",
                placeholder="例如：桌子上有哪些物体？请给我一个整理建议，并说明依据和不确定性。",
            )
            live_pref = st.text_input(
                "Optional response preference",
                placeholder="例如：请先给结论，再给证据和局限性",
            )
            allow_auto = st.checkbox("Enable auto scene-change trigger (experimental)", value=False)
            fast_mode = st.checkbox(
                "Fast mode (compact prompt context, recommended)",
                value=bool(st.session_state.get("live_fast_mode", True)),
                help="Reduces prompt payload size and UI debug overhead for faster responses.",
            )
            enable_response_cache = st.checkbox(
                "Enable Qwen response cache",
                value=bool(st.session_state.get("live_enable_response_cache", True)),
                help="Reuses answers for repeated query + scene state combinations.",
            )
            store_prompt_bundle_debug = st.checkbox(
                "Store prompt bundle in history (debug, slower)",
                value=bool(st.session_state.get("live_store_prompt_bundle_debug", False)),
                help="Keeps full prompt JSON in session history for ablation/debug; increases UI/state overhead.",
            )
            submit_live = st.form_submit_button("Analyze Current Scene with Qwen", type="primary")

        if submit_live and live_query.strip():
            st.session_state.live_fast_mode = bool(fast_mode)
            st.session_state.live_enable_response_cache = bool(enable_response_cache)
            st.session_state.live_store_prompt_bundle_debug = bool(store_prompt_bundle_debug)
            if live_pref.strip() and live_pref.strip() not in st.session_state.user_preferences:
                st.session_state.user_preferences.append(live_pref.strip())

            reference_resolution = resolve_reference_query(
                live_query.strip(),
                list(st.session_state.get("live_object_registry") or []),
            )
            st.session_state.live_latest_reference_resolution = reference_resolution
            if reference_resolution.get("resolved") and reference_resolution.get("selected_obj_id"):
                selected_id = str(reference_resolution.get("selected_obj_id"))
                updated_registry = []
                for obj in st.session_state.get("live_object_registry") or []:
                    obj_copy = dict(obj)
                    if str(obj_copy.get("obj_id")) == selected_id:
                        obj_copy["mention_count"] = int(obj_copy.get("mention_count", 0)) + 1
                    updated_registry.append(obj_copy)
                st.session_state.live_object_registry = updated_registry

            policy = LiveTriggerPolicy(min_interval_seconds=1.5, allow_auto_scene_trigger=allow_auto)
            decision = should_trigger_qwen(
                policy=policy,
                now_ts=time.time(),
                last_qwen_ts=st.session_state.live_last_qwen_ts,
                user_submitted=True,
                scene_signature_changed=True,
            )
            if not decision.trigger:
                st.warning(f"Qwen trigger blocked: {decision.reason}")
            else:
                parsed = qwen_adapter.generate_live_response(
                    user_query=live_query.strip(),
                    scene_summary=st.session_state.live_scene_summary or "No detection summary available",
                    detections=st.session_state.live_detections,
                    preferences=tuple(st.session_state.user_preferences),
                    recent_turns=tuple(
                        {"role": item.get("role", "user"), "content": item.get("content", item.get("command", ""))}
                        for item in st.session_state.live_qwen_history[-4:]
                    ),
                    object_registry=tuple(st.session_state.get("live_object_registry") or []),
                    recent_scene_events=tuple((st.session_state.get("live_scene_event_log") or [])[-5:]),
                    reference_resolution=dict(st.session_state.get("live_latest_reference_resolution") or {}),
                    enable_cache=bool(st.session_state.get("live_enable_response_cache", True)),
                    include_prompt_bundle=bool(st.session_state.get("live_store_prompt_bundle_debug", False)),
                    compact_prompt_context=bool(st.session_state.get("live_fast_mode", True)),
                )
                st.session_state.live_qwen_cache_stats = qwen_adapter.cache_stats
                st.session_state.live_last_qwen_ts = time.time()
                st.session_state.live_last_qwen_query = live_query.strip()
                parsed_record = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "query": live_query.strip(),
                    "response": parsed.get("user_response"),
                    "analysis_json": parsed.get("analysis_json"),
                    "json_valid": bool(parsed.get("json_valid")),
                    "latency_ms": round(float(parsed.get("latency_ms") or 0.0), 3),
                    "cache_hit": bool(parsed.get("cache_hit")),
                    "source_mode": st.session_state.live_camera_mode,
                    "prompt_version": PROMPT_VERSION,
                    "parse_error": parsed.get("parse_error"),
                    "prompt_bundle": parsed.get("prompt_bundle"),
                    "reference_resolution": st.session_state.get("live_latest_reference_resolution"),
                    "event_summary": st.session_state.get("live_event_summary"),
                    "object_registry_size": len(st.session_state.get("live_object_registry") or []),
                    "fast_mode": bool(st.session_state.get("live_fast_mode", True)),
                }
                _append_bounded("live_qwen_history", parsed_record, cfg_max_events)
                st.session_state.prompt_version = PROMPT_VERSION

                vision_ms = float(
                    (st.session_state.get("live_last_detection_meta") or {}).get("vision_latency_ms", 0.0)
                )
                _record_live_qwen_interaction(
                    query=live_query.strip(),
                    parsed=parsed,
                    source_mode=f"camera_{st.session_state.live_camera_mode}",
                    room_name=st.session_state.scene_name,
                    max_events=cfg_max_events,
                    vision_ms=vision_ms,
                    reference_resolution=dict(st.session_state.get("live_latest_reference_resolution") or {}),
                    event_summary=str(st.session_state.get("live_event_summary") or ""),
                    object_registry_size=len(st.session_state.get("live_object_registry") or []),
                    cache_hit=bool(parsed.get("cache_hit")),
                )
                st.success("Live interaction response generated and logged to Evaluation Lab.")

    with right:
        history = st.session_state.live_qwen_history
        if history:
            latest = history[-1]
            render_response_card(
                "Latest User-Facing Response",
                str(latest.get("response", "")),
                meta=[
                    f"source={latest.get('source_mode')}",
                    f"json_valid={latest.get('json_valid')}",
                    f"latency_ms={latest.get('latency_ms')}",
                    f"cache_hit={latest.get('cache_hit')}",
                    f"prompt={latest.get('prompt_version')}",
                    f"registry={latest.get('object_registry_size', 0)}",
                ],
            )
            render_status_badges(
                [
                    {
                        "label": "JSON Schema",
                        "value": "valid" if latest.get("json_valid") else "fallback",
                        "tone": "ok" if latest.get("json_valid") else "warn",
                    },
                    {
                        "label": "Parse",
                        "value": str(latest.get("parse_error") or "ok"),
                        "tone": "warn" if latest.get("parse_error") else "ok",
                    },
                    {
                        "label": "Reference",
                        "value": (
                            "resolved"
                            if (latest.get("reference_resolution") or {}).get("resolved")
                            else ("clarify" if (latest.get("reference_resolution") or {}).get("clarification_needed") else "none")
                        ),
                        "tone": (
                            "ok"
                            if (latest.get("reference_resolution") or {}).get("resolved")
                            else "warn"
                        ),
                    },
                    {
                        "label": "Cache",
                        "value": "hit" if latest.get("cache_hit") else "miss",
                        "tone": "ok" if latest.get("cache_hit") else "warn",
                    },
                ]
            )
            if latest.get("event_summary"):
                render_surface("Latest Temporal Event", str(latest.get("event_summary")))
            with st.expander("Structured JSON (research mode)", expanded=False):
                st.json(latest.get("analysis_json", {}), expanded=False)
                st.caption(
                    f"json_valid={latest.get('json_valid')} | latency_ms={latest.get('latency_ms')} | prompt={latest.get('prompt_version')}"
                )
                if latest.get("parse_error"):
                    st.caption(f"parse_error={latest.get('parse_error')}")
            with st.expander("Reference Resolution (grounding)", expanded=False):
                st.json(latest.get("reference_resolution") or {}, expanded=False)
            if latest.get("prompt_bundle"):
                with st.expander("Prompt Bundle (debug / ablation support)", expanded=False):
                    st.json(latest.get("prompt_bundle") or {}, expanded=False)
        else:
            st.info("No live Qwen responses yet. Capture a snapshot, run detection, and submit a question.")

    if st.session_state.live_errors:
        with st.expander("Live Camera Errors", expanded=False):
            st.json(st.session_state.live_errors[-10:], expanded=False)


def _repo_quick_stats() -> dict[str, Any]:
    entries = list(ROOT_DIR.iterdir())
    file_count = 0
    dir_count = 0
    for entry in entries:
        if entry.is_dir():
            dir_count += 1
        elif entry.is_file():
            file_count += 1
    large_files = []
    for entry in entries:
        if entry.is_file():
            size_mb = entry.stat().st_size / (1024 * 1024)
            if size_mb > 100:
                large_files.append((entry.name, round(size_mb, 1)))
    return {
        "root_items": len(entries),
        "root_dirs": dir_count,
        "root_files": file_count,
        "large_files": large_files[:5],
    }


def _scene_editor_panel(scene_payload: dict[str, Any]) -> None:
    with st.form("scene_editor_form"):
        editor_json = st.text_area(
            "Scene JSON",
            value=json.dumps(scene_payload, ensure_ascii=False, indent=2),
            height=300,
            help="Edit room objects, zones, and states to simulate a new environment.",
        )
        c1, c2, c3 = st.columns([1, 1, 2])
        apply_clicked = c1.form_submit_button("Apply Scene JSON", type="primary")
        reset_clicked = c2.form_submit_button("Reset Scene Override")
        if apply_clicked:
            try:
                payload = json.loads(editor_json)
                if not isinstance(payload, dict):
                    raise ValueError("Scene JSON must be an object.")
                st.session_state.scene_overrides[st.session_state.scene_name] = payload
                st.success("Scene override applied.")
            except Exception as exc:
                st.error(f"Invalid JSON: {exc}")
        if reset_clicked:
            st.session_state.scene_overrides.pop(st.session_state.scene_name, None)
            st.info("Scene override reset to template.")


def _interaction_console(scene: SceneState, cfg_max_events: int) -> None:
    st.markdown("### Command Console")
    with st.form("command_console_form", clear_on_submit=True):
        command = st.text_input(
            "User command",
            placeholder="e.g., Please move the chair to the right and clean the cup on the table (simulation only).",
        )
        pref = st.text_input(
            "Optional preference hint",
            placeholder="e.g., Add a safety confirmation before moving furniture",
        )
        submitted = st.form_submit_button("Simulate Robot Interaction", type="primary")

    if submitted and command.strip():
        if pref.strip() and pref.strip() not in st.session_state.user_preferences:
            st.session_state.user_preferences.append(pref.strip())

        with timed_stage("planning") as planning:
            plan = plan_robot_response(
                scene,
                command,
                preferences=tuple(st.session_state.user_preferences),
            )

        with timed_stage("render") as render_timing:
            # Simulate minimal render prep work for metrics consistency.
            plan_dict = action_plan_as_dict(plan)
            st.session_state.latest_plan = plan_dict
            st.session_state.robot_state["mode"] = "executing"
            st.session_state.robot_state["last_action"] = plan.action
            if plan.action == "clarify_request":
                st.session_state.robot_state["mode"] = "waiting_for_clarification"
            else:
                st.session_state.robot_state["mode"] = "idle"

        _record_interaction(command, scene, plan_dict, cfg_max_events)
        _record_perf(
            command,
            {
                "planning_ms": planning.duration_ms,
                "render_ms": render_timing.duration_ms,
            },
            cfg_max_events,
        )
        st.success("Interaction simulated. Review the recommendation panel and timing metrics.")

    st.markdown("### Preference Memory")
    if st.session_state.user_preferences:
        st.caption("Active adaptation hints used during planning:")
        for p in st.session_state.user_preferences[-6:]:
            st.markdown(f"- {p}")
    else:
        st.caption("No preferences stored yet. Add one when sending a command.")


def _ops_panel() -> None:
    perf_history = st.session_state.perf_history
    if perf_history:
        totals = [float(x.get("total_ms", 0.0)) for x in perf_history]
        st.line_chart({"total_ms": totals}, height=220)
    else:
        st.info("No performance samples yet. Run an interaction to populate metrics.")
    render_perf_table(perf_history)

    st.markdown("### Video Sampling Policy Preview")
    duration = st.slider("Duration (seconds)", min_value=1, max_value=120, value=20)
    st.metric("Recommended Sampling FPS", f"{select_video_sampling_fps(float(duration))}")

    st.markdown("### Runtime Diagnostics")
    st.json(
        {
            "cwd": os.getcwd(),
            "scene_dir_exists": SCENE_DIR.exists(),
            "scene_override_keys": list(st.session_state.scene_overrides.keys()),
            "interaction_history_size": len(st.session_state.interaction_history),
            "perf_history_size": len(st.session_state.perf_history),
            "evaluation_feedback_size": len(st.session_state.evaluation_feedback),
            "live_qwen_history_size": len(st.session_state.live_qwen_history),
            "live_camera_status": st.session_state.live_camera_status,
        },
        expanded=False,
    )


def _evaluation_panel() -> None:
    interactions = st.session_state.interaction_history
    perf_history = st.session_state.perf_history
    feedback_records = st.session_state.evaluation_feedback
    summary = compute_evaluation_summary(interactions, perf_history, feedback_records)

    def _fmt_pct(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{value * 100:.1f}%"

    def _fmt_ms(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{value:.2f} ms"

    render_metric_cards(
        [
            {
                "label": "Avg Response Latency",
                "value": _fmt_ms(summary["latency"]["avg_ms"]),
                "sub": f"p95: {_fmt_ms(summary['latency']['p95_ms'])}",
            },
            {
                "label": "Recommendation Consistency",
                "value": _fmt_pct(summary["consistency"]["score"]),
                "sub": f"repeated command groups: {summary['consistency']['repeated_command_groups']}",
            },
            {
                "label": "Simulated Task Completion",
                "value": _fmt_pct(summary["task_completion"]["simulated_rate"]),
                "sub": f"feedback overrides: {summary['task_completion']['feedback_override_count']}",
            },
            {
                "label": "User Satisfaction",
                "value": (
                    f"{summary['satisfaction']['avg_rating']:.2f}/5"
                    if summary["satisfaction"]["avg_rating"] is not None
                    else "N/A"
                ),
                "sub": f"ratings: {summary['satisfaction']['count']}",
            },
            {
                "label": "JSON Valid Rate",
                "value": _fmt_pct((summary.get("research_metrics") or {}).get("json_valid_rate")),
                "sub": f"samples: {(summary.get('research_metrics') or {}).get('json_valid_count', 0)}",
            },
            {
                "label": "Clarification Rate",
                "value": _fmt_pct((summary.get("research_metrics") or {}).get("clarification_rate")),
                "sub": f"samples: {(summary.get('research_metrics') or {}).get('clarification_count', 0)}",
            },
            {
                "label": "Reference Resolution",
                "value": _fmt_pct((summary.get("research_metrics") or {}).get("reference_resolution_rate")),
                "sub": f"attempts: {(summary.get('research_metrics') or {}).get('reference_attempt_count', 0)}",
            },
            {
                "label": "Cache Hit Rate",
                "value": _fmt_pct((summary.get("research_metrics") or {}).get("cache_hit_rate")),
                "sub": f"samples: {(summary.get('research_metrics') or {}).get('cache_observation_count', 0)}",
            },
        ]
    )
    render_status_badges(
        [
            {
                "label": "Evaluated Interactions",
                "value": str(len(feedback_records)),
                "tone": "ok" if feedback_records else "warn",
            },
            {
                "label": "Perf Samples",
                "value": str(len(perf_history)),
                "tone": "ok" if perf_history else "warn",
            },
            {
                "label": "Interaction Logs",
                "value": str(len(interactions)),
                "tone": "ok" if interactions else "warn",
            },
        ]
    )

    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.markdown("### Evaluation Feedback Recorder")
        if not interactions:
            st.info("No interactions yet. Run at least one simulation in Interaction Console.")
        else:
            options = []
            for idx, item in enumerate(interactions):
                cmd = str(item.get("command", "")).strip()
                plan = item.get("plan") or {}
                action = plan.get("action", "unknown")
                options.append((idx, f"#{idx} | {action} | {cmd[:80]}"))

            existing_by_idx = {}
            for rec in feedback_records:
                try:
                    existing_by_idx[int(rec.get("interaction_index"))] = rec
                except (TypeError, ValueError):
                    continue

            with st.form("evaluation_feedback_form"):
                selected_idx = st.selectbox(
                    "Interaction to evaluate",
                    options=[item[0] for item in options],
                    format_func=lambda i: dict(options).get(i, str(i)),
                )
                existing = existing_by_idx.get(selected_idx, {})
                rating = st.slider(
                    "Satisfaction rating (1-5)",
                    min_value=1,
                    max_value=5,
                    value=int(existing.get("satisfaction_rating", 4)),
                )
                completion_status = st.selectbox(
                    "Completion outcome (simulation judgement)",
                    options=["completed", "partial", "failed", "unknown"],
                    index=["completed", "partial", "failed", "unknown"].index(
                        str(existing.get("completion_status", "completed"))
                        if str(existing.get("completion_status", "completed")) in ["completed", "partial", "failed", "unknown"]
                        else "completed"
                    ),
                )
                note = st.text_area(
                    "Evaluator note (optional)",
                    value=str(existing.get("note", "")),
                    height=90,
                    placeholder="e.g., Recommendation was reasonable but lacked mention of object fragility.",
                )
                submit = st.form_submit_button("Save Evaluation Feedback", type="primary")
                if submit:
                    _upsert_evaluation_feedback(
                        {
                            "interaction_index": selected_idx,
                            "satisfaction_rating": rating,
                            "completion_status": completion_status,
                            "note": note.strip(),
                            "updated_at": datetime.now().isoformat(timespec="seconds"),
                        }
                    )
                    st.success("Evaluation feedback saved.")

        st.markdown("### Coverage")
        rated_count = len(
            {
                int(rec.get("interaction_index"))
                for rec in feedback_records
                if rec.get("interaction_index") is not None
            }
        )
        total_count = len(interactions)
        st.metric(
            "Feedback Coverage",
            f"{rated_count}/{total_count}" if total_count else "0/0",
            delta=(f"{(rated_count / total_count) * 100:.1f}%" if total_count else None),
        )
        if interactions and perf_history:
            st.line_chart(
                {"latency_ms": [float(item.get("total_ms", 0.0)) for item in perf_history]},
                height=200,
            )

    with c2:
        st.markdown("### Evaluation Summary")
        st.json(summary, expanded=False)

        st.markdown("### Feedback Records")
        if feedback_records:
            rows = []
            for rec in sorted(
                feedback_records,
                key=lambda x: int(x.get("interaction_index", -1)),
                reverse=True,
            ):
                idx = int(rec.get("interaction_index", -1))
                cmd = ""
                action = ""
                if 0 <= idx < len(interactions):
                    cmd = str(interactions[idx].get("command", ""))[:80]
                    action = str((interactions[idx].get("plan") or {}).get("action", ""))
                rows.append(
                    {
                        "interaction_index": idx,
                        "action": action,
                        "command": cmd,
                        "rating": rec.get("satisfaction_rating"),
                        "completion": rec.get("completion_status"),
                        "updated_at": rec.get("updated_at"),
                        "note": rec.get("note", ""),
                    }
                )
            st.dataframe(rows, width="stretch", hide_index=True)
            ratings = [
                int(rec.get("satisfaction_rating"))
                for rec in feedback_records
                if rec.get("satisfaction_rating") is not None
            ]
            if ratings:
                st.bar_chart({"rating": ratings}, height=180)
        else:
            st.info("No evaluation feedback recorded yet.")

        st.markdown("### Metric Definitions")
        st.caption(
            "Task completion is simulated from recommendation confidence and action type, then overridden by evaluator feedback when available."
        )
        st.caption(
            "Recommendation consistency is computed on repeated commands by checking whether the platform returns the same action/target pair."
        )


def _automation_panel(cfg_backend_mode: str) -> None:
    stats = _repo_quick_stats()
    st.markdown("### Automation Shortcuts")
    st.code(
        "\n".join(
            [
                "make run            # Launch new URIS platform",
                "make run-legacy     # Launch original VL demo",
                "make test           # Run platform unit tests",
                "make doctor         # Repo health / structure audit",
                "make benchmark      # Planner latency benchmark",
            ]
        ),
        language="bash",
    )

    st.markdown("### Project Health Snapshot")
    st.json(stats, expanded=False)

    st.markdown("### Migration Notes")
    st.info(
        "Legacy multimodal VL inference app is preserved under `legacy/legacy_video_reasoning_app.py`. "
        "This platform is system-first and model-agnostic, so future fine-tuned models can be plugged in via adapters. "
        "It focuses on interaction simulation and recommendation output, not physical robot motion control."
    )
    if cfg_backend_mode != "heuristic":
        st.warning(
            "Non-heuristic backend mode selected, but adapter is not implemented in this phase. "
            "Platform is currently running heuristic simulation."
        )


def main() -> None:
    _safe_page_config()
    inject_theme()

    cfg = load_platform_config(env=os.environ)
    initialize_session_state(st.session_state)

    templates = load_scene_templates(str(SCENE_DIR))
    if st.session_state.scene_name not in templates and templates:
        st.session_state.scene_name = cfg.default_scene if cfg.default_scene in templates else next(iter(templates))

    scene_payload = _current_scene_payload(templates)
    scene = _scene_from_payload(scene_payload)

    avg_latency = None
    if st.session_state.perf_history:
        avg_latency = mean(float(x.get("total_ms", 0.0)) for x in st.session_state.perf_history)

    render_hero(
        "URIS Home Robot Interaction Platform",
        "System-first simulation environment for multimodal home-robot interaction, task recommendation, and performance evaluation.",
        pills=[
            f"Backend: {cfg.backend_mode}",
            f"Scene: {st.session_state.scene_name}",
            "Phase 1 Platform Refactor",
            "Legacy VL App Preserved",
        ],
    )

    render_metric_cards(
        [
            {
                "label": "Interaction Rounds",
                "value": str(len(st.session_state.interaction_history)),
                "sub": "simulated user-to-robot task turns",
            },
            {
                "label": "Average Latency",
                "value": f"{avg_latency:.2f} ms" if avg_latency is not None else "N/A",
                "sub": "planning + render pipeline stages",
            },
            {
                "label": "Preferences",
                "value": str(len(st.session_state.user_preferences)),
                "sub": "adaptation hints currently active",
            },
            {
                "label": "Simulator State",
                "value": str(st.session_state.robot_state.get("mode", "idle")),
                "sub": f"last action: {st.session_state.robot_state.get('last_action') or 'none'}",
            },
        ]
    )

    with st.sidebar:
        st.header("Platform Controls")
        available_scenes = list(templates.keys()) or ["living_room"]
        selected_scene = st.selectbox(
            "Scene Template",
            options=available_scenes,
            index=available_scenes.index(st.session_state.scene_name)
            if st.session_state.scene_name in available_scenes
            else 0,
        )
        if selected_scene != st.session_state.scene_name:
            st.session_state.scene_name = selected_scene
            st.rerun()

        backend_mode = st.selectbox(
            "Backend Mode",
            options=["heuristic", "llm", "llm+yolo"],
            index=0 if cfg.backend_mode not in ["heuristic", "llm", "llm+yolo"] else ["heuristic", "llm", "llm+yolo"].index(cfg.backend_mode),
            help="Phase 1 implements heuristic mode; other modes are adapter placeholders.",
        )
        if backend_mode != cfg.backend_mode:
            st.caption("Set `URIS_PLATFORM_BACKEND_MODE` env var to persist backend mode.")

        if st.button("Reset Platform Session"):
            for key in [
                "interaction_history",
                "perf_history",
                "evaluation_feedback",
                "user_preferences",
                "scene_overrides",
                "latest_plan",
            ]:
                if key in st.session_state:
                    st.session_state[key] = [] if key in {"interaction_history", "perf_history", "evaluation_feedback", "user_preferences"} else {}
            for key in list(st.session_state.keys()):
                if str(key).startswith("live_"):
                    if key in {"live_detections", "live_qwen_history", "live_errors"}:
                        st.session_state[key] = []
                    elif key in {"live_camera_enabled"}:
                        st.session_state[key] = False
                    elif key in {"live_last_frame_ts", "live_last_qwen_ts"}:
                        st.session_state[key] = None
                    else:
                        st.session_state[key] = "" if isinstance(st.session_state.get(key), str) else None
            st.session_state.latest_plan = None
            st.session_state.robot_state = {"mode": "idle", "location": "dock", "last_action": None}
            st.session_state.live_camera_mode = "snapshot"
            st.session_state.live_camera_status = "idle"
            st.session_state.prompt_version = PROMPT_VERSION
            st.rerun()

        st.divider()
        st.caption("Legacy VL demo")
        st.code("streamlit run legacy/legacy_video_reasoning_app.py", language="bash")
        st.caption("This new platform focuses on interaction simulation, UI, and performance instrumentation (no robot motion control).")

    mission_tab, scene_tab, interaction_tab, live_tab, ops_tab, eval_tab, automation_tab = st.tabs(
        [
            "Mission Control",
            "Scenario Studio",
            "Interaction Console",
            "Live Camera",
            "Operations",
            "Evaluation Lab",
            "Automation",
        ]
    )

    with mission_tab:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.markdown("### Current Scene")
            st.caption(f"Room: `{scene.room}`")
            render_scene_objects([asdict(obj) for obj in scene.objects])
            if scene.notes:
                st.info(scene.notes)
        with c2:
            latest_plan = st.session_state.latest_plan
            from uris_platform.domain import ActionPlan

            render_action_plan(ActionPlan(**latest_plan) if latest_plan else None)

        st.markdown("### System Focus for This Phase")
        st.markdown(
            "- Upgrade UI and interaction flow to a research-demo-ready platform\n"
            "- Decouple model iteration from platform presentation and orchestration\n"
            "- Add performance instrumentation and automation for faster iteration\n"
            "- Preserve the original VL app under a legacy path\n"
            "- Output recommendations/simulations only, not physical robot control"
        )

    with scene_tab:
        st.markdown("### Scene Template Inspector")
        st.json(scene_payload, expanded=False)
        st.markdown("### Scene Editor")
        _scene_editor_panel(scene_payload)

    with interaction_tab:
        left, right = st.columns([1.1, 0.9])
        with left:
            _interaction_console(scene, cfg.max_timeline_events)
            render_interaction_history(st.session_state.interaction_history)
        with right:
            latest_plan = st.session_state.latest_plan
            from uris_platform.domain import ActionPlan

            render_action_plan(ActionPlan(**latest_plan) if latest_plan else None)

    with live_tab:
        _live_camera_tab(cfg_max_events=cfg.max_timeline_events)

    with ops_tab:
        _ops_panel()

    with eval_tab:
        _evaluation_panel()

    with automation_tab:
        _automation_panel(backend_mode)


if __name__ == "__main__":
    main()
