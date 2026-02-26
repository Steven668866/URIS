from uris_platform.services.live_scene_memory import ingest_live_detections, resolve_reference_query


def test_ingest_live_detections_reuses_stable_object_ids_for_nearby_same_label():
    registry = []
    detection_history = []
    event_log = []

    first = ingest_live_detections(
        registry=registry,
        detection_history=detection_history,
        event_log=event_log,
        detections=[
            {"label": "cup", "confidence": 0.93, "bbox": [10, 10, 40, 50], "center_norm": [0.2, 0.3]},
        ],
        scene_summary="1 detections: cup x1",
        now_ts=100.0,
    )
    second = ingest_live_detections(
        registry=first["registry"],
        detection_history=first["detection_history"],
        event_log=first["event_log"],
        detections=[
            {"label": "cup", "confidence": 0.91, "bbox": [12, 11, 42, 51], "center_norm": [0.21, 0.305]},
        ],
        scene_summary="1 detections: cup x1",
        now_ts=101.0,
    )

    assert len(second["registry"]) == 1
    assert second["registry"][0]["obj_id"] == first["registry"][0]["obj_id"]
    assert second["registry"][0]["seen_count"] == 2
    assert second["registry"][0]["status"] == "visible"


def test_ingest_live_detections_emits_count_change_events():
    first = ingest_live_detections(
        registry=[],
        detection_history=[],
        event_log=[],
        detections=[
            {"label": "cup", "confidence": 0.9, "bbox": [1, 1, 2, 2], "center_norm": [0.2, 0.2]},
            {"label": "table", "confidence": 0.95, "bbox": [0, 0, 10, 10], "center_norm": [0.5, 0.8]},
        ],
        scene_summary="2 detections: cup x1, table x1",
        now_ts=100.0,
    )
    second = ingest_live_detections(
        registry=first["registry"],
        detection_history=first["detection_history"],
        event_log=first["event_log"],
        detections=[
            {"label": "cup", "confidence": 0.91, "bbox": [1, 1, 2, 2], "center_norm": [0.2, 0.2]},
            {"label": "cup", "confidence": 0.87, "bbox": [3, 1, 4, 2], "center_norm": [0.3, 0.2]},
            {"label": "table", "confidence": 0.94, "bbox": [0, 0, 10, 10], "center_norm": [0.5, 0.8]},
        ],
        scene_summary="3 detections: cup x2, table x1",
        now_ts=101.0,
    )

    assert second["event_log"]
    latest = second["event_log"][-1]
    assert latest["type"] == "count_change"
    assert "cup" in latest["message"]
    assert second["event_summary"]


def test_resolve_reference_query_prefers_leftmost_object_for_left_reference():
    registry = [
        {
            "obj_id": "obj-0001",
            "label": "cup",
            "center_norm": [0.2, 0.4],
            "confidence": 0.92,
            "status": "visible",
            "mention_count": 0,
        },
        {
            "obj_id": "obj-0002",
            "label": "cup",
            "center_norm": [0.7, 0.4],
            "confidence": 0.89,
            "status": "visible",
            "mention_count": 0,
        },
    ]

    resolved = resolve_reference_query("左边那个杯子需要整理吗？", registry)

    assert resolved["resolved"] is True
    assert resolved["selected_obj_id"] == "obj-0001"
    assert resolved["method"] in {"directional", "label+directional"}
    assert resolved["clarification_needed"] is False


def test_resolve_reference_query_requests_clarification_for_ambiguous_deictic_reference():
    registry = [
        {
            "obj_id": "obj-0001",
            "label": "cup",
            "center_norm": [0.4, 0.4],
            "confidence": 0.92,
            "status": "visible",
            "mention_count": 0,
        },
        {
            "obj_id": "obj-0002",
            "label": "cup",
            "center_norm": [0.6, 0.4],
            "confidence": 0.89,
            "status": "visible",
            "mention_count": 0,
        },
    ]

    resolved = resolve_reference_query("那个杯子是什么情况？", registry)

    assert resolved["resolved"] is False
    assert resolved["clarification_needed"] is True
    assert resolved["candidate_count"] == 2
    assert "澄清" in resolved["clarifying_question"]


def test_ingest_live_detections_reuses_registry_entry_by_track_id_even_when_position_changes():
    first = ingest_live_detections(
        registry=[],
        detection_history=[],
        event_log=[],
        detections=[
            {
                "label": "cup",
                "confidence": 0.94,
                "bbox": [10, 10, 30, 40],
                "center_norm": [0.2, 0.3],
                "track_id": 7,
            }
        ],
        scene_summary="1 detections: cup x1",
        now_ts=10.0,
    )
    second = ingest_live_detections(
        registry=first["registry"],
        detection_history=first["detection_history"],
        event_log=first["event_log"],
        detections=[
            {
                "label": "cup",
                "confidence": 0.91,
                "bbox": [120, 120, 160, 180],
                "center_norm": [0.8, 0.8],  # far from previous center
                "track_id": 7,
            }
        ],
        scene_summary="1 detections: cup x1",
        now_ts=11.0,
        match_distance_threshold=0.05,  # would fail center-based matching if track_id not honored
    )

    assert len(second["registry"]) == 1
    assert second["registry"][0]["seen_count"] == 2
    assert second["registry"][0]["track_id"] == 7
