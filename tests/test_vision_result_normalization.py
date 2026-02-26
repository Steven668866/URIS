from uris_platform.services.vision_yolo import build_live_scene_summary, normalize_yolo_detections


def test_normalize_yolo_detections_sorts_and_counts_labels():
    raw = [
        {"label": "cup", "confidence": 0.72, "bbox": [10, 10, 40, 60]},
        {"label": "chair", "confidence": 0.95, "bbox": [0, 0, 100, 200]},
        {"label": "cup", "confidence": 0.88, "bbox": [60, 10, 90, 60]},
    ]
    result = normalize_yolo_detections(raw, frame_width=200, frame_height=200)

    assert [d["label"] for d in result["detections"]] == ["chair", "cup", "cup"]
    assert result["counts"] == {"chair": 1, "cup": 2}
    assert result["top_labels"] == ["chair", "cup"]
    assert result["detections"][0]["center_norm"] == [0.25, 0.5]


def test_build_live_scene_summary_mentions_counts_and_confidence():
    normalized = normalize_yolo_detections(
        [
            {"label": "cup", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
            {"label": "cup", "confidence": 0.8, "bbox": [10, 0, 20, 10]},
            {"label": "table", "confidence": 0.95, "bbox": [0, 10, 100, 80]},
        ],
        frame_width=100,
        frame_height=100,
    )
    summary = build_live_scene_summary(normalized)
    assert "table x1" in summary
    assert "cup x2" in summary
    assert "3 detections" in summary
