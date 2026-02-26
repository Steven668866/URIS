from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass
class YoloRuntimeStatus:
    available: bool
    reason: str | None = None


def normalize_yolo_detections(
    raw_detections: list[dict[str, Any]],
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for item in raw_detections:
        label = str(item.get("label", "object")).strip() or "object"
        confidence = float(item.get("confidence", 0.0))
        bbox = item.get("bbox") or [0, 0, 0, 0]
        if len(bbox) != 4:
            bbox = [0, 0, 0, 0]
        x1, y1, x2, y2 = [float(x) for x in bbox]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        cx = x1 + width / 2.0
        cy = y1 + height / 2.0
        center_norm = None
        if frame_width and frame_height and frame_width > 0 and frame_height > 0:
            center_norm = [
                round(cx / float(frame_width), 4),
                round(cy / float(frame_height), 4),
            ]
        normalized.append(
            {
                "label": label,
                "confidence": round(confidence, 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "center_norm": center_norm,
            }
        )

    normalized.sort(key=lambda d: (-float(d["confidence"]), d["label"]))
    counts = Counter(item["label"] for item in normalized)
    top_labels: list[str] = []
    for item in normalized:
        if item["label"] not in top_labels:
            top_labels.append(item["label"])
    return {
        "detections": normalized,
        "counts": dict(counts),
        "top_labels": top_labels,
        "total_detections": len(normalized),
        "frame_size": {"width": frame_width, "height": frame_height},
    }


def build_live_scene_summary(normalized: dict[str, Any]) -> str:
    counts = normalized.get("counts") or {}
    total = int(normalized.get("total_detections", 0))
    if total == 0:
        return "0 detections: no objects confidently detected"
    parts = [f"{label} x{count}" for label, count in sorted(counts.items())]
    top_conf = 0.0
    detections = normalized.get("detections") or []
    if detections:
        top_conf = max(float(d.get("confidence", 0.0)) for d in detections)
    return f"{total} detections: {', '.join(parts)} | top_conf={top_conf:.2f}"


def yolo_runtime_status() -> YoloRuntimeStatus:
    try:
        import ultralytics  # noqa: F401

        return YoloRuntimeStatus(available=True)
    except Exception as exc:  # pragma: no cover - runtime env dependent
        return YoloRuntimeStatus(available=False, reason=str(exc))


@lru_cache(maxsize=2)
def _load_ultralytics_model(model_name: str):  # pragma: no cover - runtime path
    from ultralytics import YOLO

    return YOLO(model_name)


def run_ultralytics_detection_on_bgr(
    frame_bgr,
    *,
    model_name: str = "yolov8n.pt",
    conf_threshold: float = 0.25,
    max_det: int = 20,
) -> dict[str, Any]:
    """Run ultralytics YOLO on a BGR frame and return normalized detections."""
    model = _load_ultralytics_model(model_name)
    results = model(frame_bgr, verbose=False, conf=conf_threshold, max_det=max_det)
    if not results:
        return normalize_yolo_detections([], frame_width=None, frame_height=None)

    result = results[0]
    names = result.names if hasattr(result, "names") else {}
    raw_detections: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        xyxy = boxes.xyxy.cpu().tolist() if hasattr(boxes, "xyxy") else []
        confs = boxes.conf.cpu().tolist() if hasattr(boxes, "conf") else []
        classes = boxes.cls.cpu().tolist() if hasattr(boxes, "cls") else []
        for bbox, conf, cls_id in zip(xyxy, confs, classes):
            label = names.get(int(cls_id), str(int(cls_id))) if isinstance(names, dict) else str(int(cls_id))
            raw_detections.append(
                {
                    "label": label,
                    "confidence": float(conf),
                    "bbox": bbox,
                }
            )

    height, width = frame_bgr.shape[:2]
    return normalize_yolo_detections(
        raw_detections,
        frame_width=int(width),
        frame_height=int(height),
    )


def run_mock_or_passthrough_detection(
    *,
    raw_detections: list[dict[str, Any]] | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> dict[str, Any]:
    """Utility used by the Streamlit fallback path and tests; real YOLO can be wired later."""
    return normalize_yolo_detections(
        raw_detections or [],
        frame_width=frame_width,
        frame_height=frame_height,
    )
