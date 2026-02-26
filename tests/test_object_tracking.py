from uris_platform.services.object_tracking import LiveObjectTracker, available_tracking_modes


def test_simple_tracker_assigns_stable_track_ids_across_nearby_frames():
    tracker = LiveObjectTracker(mode="simple")

    frame1 = tracker.update(
        detections=[
            {"label": "cup", "confidence": 0.93, "bbox": [10, 10, 30, 40], "center_norm": [0.2, 0.3]},
            {"label": "table", "confidence": 0.98, "bbox": [0, 60, 200, 200], "center_norm": [0.5, 0.8]},
        ],
        now_ts=100.0,
    )
    frame2 = tracker.update(
        detections=[
            {"label": "cup", "confidence": 0.9, "bbox": [12, 11, 32, 41], "center_norm": [0.21, 0.305]},
            {"label": "table", "confidence": 0.97, "bbox": [0, 60, 200, 200], "center_norm": [0.5, 0.8]},
        ],
        now_ts=101.0,
    )

    by_label_1 = {d["label"]: d["track_id"] for d in frame1["detections"]}
    by_label_2 = {d["label"]: d["track_id"] for d in frame2["detections"]}
    assert by_label_1["cup"] == by_label_2["cup"]
    assert by_label_1["table"] == by_label_2["table"]
    assert frame2["tracker_meta"]["mode"] == "simple"


def test_simple_tracker_creates_new_track_when_same_label_moves_far():
    tracker = LiveObjectTracker(mode="simple", distance_threshold=0.15)

    frame1 = tracker.update(
        detections=[
            {"label": "cup", "confidence": 0.93, "bbox": [10, 10, 30, 40], "center_norm": [0.2, 0.3]},
        ],
        now_ts=100.0,
    )
    frame2 = tracker.update(
        detections=[
            {"label": "cup", "confidence": 0.9, "bbox": [120, 120, 160, 180], "center_norm": [0.9, 0.9]},
        ],
        now_ts=101.0,
    )

    assert frame1["detections"][0]["track_id"] != frame2["detections"][0]["track_id"]


def test_available_tracking_modes_lists_research_ready_options():
    modes = available_tracking_modes()
    assert "simple" in modes
    assert "bytetrack" in modes
    assert "ocsort" in modes
