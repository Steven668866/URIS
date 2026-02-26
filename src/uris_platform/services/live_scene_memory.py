from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Iterable


def _safe_center_norm(item: dict[str, Any]) -> tuple[float, float] | None:
    center = item.get("center_norm")
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        return None
    try:
        return float(center[0]), float(center[1])
    except (TypeError, ValueError):
        return None


def _center_distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if a is None or b is None:
        return 999.0
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def _next_object_id(registry: Iterable[dict[str, Any]]) -> str:
    max_index = 0
    for item in registry:
        obj_id = str(item.get("obj_id", ""))
        if obj_id.startswith("obj-"):
            try:
                max_index = max(max_index, int(obj_id.split("-", 1)[1]))
            except ValueError:
                continue
    return f"obj-{max_index + 1:04d}"


def _counts_from_detections(detections: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(det.get("label", "object")) for det in detections))


def _bounded_append(items: list[dict[str, Any]], item: dict[str, Any], *, max_len: int) -> list[dict[str, Any]]:
    out = list(items)
    out.append(item)
    if len(out) > max_len:
        del out[:-max_len]
    return out


def _build_count_change_message(prev_counts: dict[str, int], curr_counts: dict[str, int]) -> str:
    labels = sorted(set(prev_counts) | set(curr_counts))
    changes = []
    for label in labels:
        prev_v = int(prev_counts.get(label, 0))
        curr_v = int(curr_counts.get(label, 0))
        if prev_v != curr_v:
            changes.append(f"{label}: {prev_v}->{curr_v}")
    if not changes:
        return "scene stable: no object count change"
    return "count change: " + ", ".join(changes)


def _match_registry_entry(
    *,
    registry: list[dict[str, Any]],
    detection: dict[str, Any],
    used_indices: set[int],
    match_distance_threshold: float,
) -> int | None:
    label = str(detection.get("label", "object"))
    det_track_id = detection.get("track_id")
    if det_track_id is not None:
        for idx, entry in enumerate(registry):
            if idx in used_indices:
                continue
            if str(entry.get("label", "object")) != label:
                continue
            if entry.get("track_id") == det_track_id:
                return idx

    det_center = _safe_center_norm(detection)
    best_idx = None
    best_dist = 999.0
    for idx, entry in enumerate(registry):
        if idx in used_indices:
            continue
        if str(entry.get("label", "object")) != label:
            continue
        dist = _center_distance(_safe_center_norm(entry), det_center)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    if best_idx is None:
        return None
    if best_dist > match_distance_threshold:
        return None
    return best_idx


def ingest_live_detections(
    *,
    registry: list[dict[str, Any]],
    detection_history: list[dict[str, Any]],
    event_log: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    scene_summary: str,
    now_ts: float,
    match_distance_threshold: float = 0.2,
    max_history: int = 20,
    max_events: int = 40,
) -> dict[str, Any]:
    """
    Update object registry and temporal memory using normalized detections.

    `detections` is expected to come from `normalize_yolo_detections`, but this function
    tolerates partial records.
    """
    next_registry = deepcopy(list(registry))
    used_indices: set[int] = set()

    for entry in next_registry:
        entry["status"] = "stale"

    for det in detections:
        match_idx = _match_registry_entry(
            registry=next_registry,
            detection=det,
            used_indices=used_indices,
            match_distance_threshold=match_distance_threshold,
        )
        if match_idx is None:
            entry = {
                "obj_id": _next_object_id(next_registry),
                "label": str(det.get("label", "object")),
                "bbox": list(det.get("bbox") or [0, 0, 0, 0]),
                "center_norm": list(det.get("center_norm") or []),
                "confidence": float(det.get("confidence", 0.0)),
                "track_id": det.get("track_id"),
                "first_seen_ts": float(now_ts),
                "last_seen_ts": float(now_ts),
                "seen_count": 1,
                "mention_count": 0,
                "status": "visible",
            }
            next_registry.append(entry)
            used_indices.add(len(next_registry) - 1)
            continue

        used_indices.add(match_idx)
        entry = next_registry[match_idx]
        entry["bbox"] = list(det.get("bbox") or entry.get("bbox") or [0, 0, 0, 0])
        entry["center_norm"] = list(det.get("center_norm") or entry.get("center_norm") or [])
        entry["confidence"] = float(det.get("confidence", entry.get("confidence", 0.0)))
        if det.get("track_id") is not None:
            entry["track_id"] = det.get("track_id")
        entry["last_seen_ts"] = float(now_ts)
        entry["seen_count"] = int(entry.get("seen_count", 0)) + 1
        entry["status"] = "visible"

    # Make registry deterministic and easier to display/debug.
    next_registry.sort(key=lambda item: str(item.get("obj_id", "")))

    curr_counts = _counts_from_detections(detections)
    prev_counts = {}
    if detection_history:
        prev_counts = dict((detection_history[-1] or {}).get("counts") or {})

    history_entry = {
        "ts": float(now_ts),
        "counts": curr_counts,
        "scene_summary": str(scene_summary or ""),
        "total_detections": len(detections),
    }
    next_history = _bounded_append(list(detection_history), history_entry, max_len=max_history)

    event_type = "scene_initialized" if not detection_history else "count_change"
    event_message = _build_count_change_message(prev_counts, curr_counts)
    if not detections:
        event_type = "no_detection"
        event_message = "no confident detections in latest frame"
    elif detection_history and prev_counts == curr_counts:
        event_type = "scene_stable"

    event_entry = {
        "ts": float(now_ts),
        "type": event_type,
        "message": event_message,
        "counts": curr_counts,
    }
    next_events = _bounded_append(list(event_log), event_entry, max_len=max_events)

    return {
        "registry": next_registry,
        "detection_history": next_history,
        "event_log": next_events,
        "event_summary": event_message,
        "visible_registry": [item for item in next_registry if item.get("status") == "visible"],
        "counts": curr_counts,
    }


def _query_has_any(query: str, tokens: list[str]) -> bool:
    return any(tok in query for tok in tokens)


def _label_aliases_for_registry(registry: list[dict[str, Any]]) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    base_aliases = {
        "cup": ["cup", "cups", "杯子", "杯"],
        "chair": ["chair", "chairs", "椅子"],
        "table": ["table", "desk", "桌子", "桌"],
        "bottle": ["bottle", "瓶子"],
        "tissue": ["tissue", "paper towel", "纸巾"],
    }
    for item in registry:
        label = str(item.get("label", "object")).lower()
        aliases[label] = base_aliases.get(label, [label])
    return aliases


def resolve_reference_query(user_query: str, registry: list[dict[str, Any]]) -> dict[str, Any]:
    query = (user_query or "").strip()
    query_lower = query.lower()
    visible = [item for item in registry if str(item.get("status", "visible")) == "visible"]
    alias_map = _label_aliases_for_registry(visible)

    direction_left = _query_has_any(query_lower + query, ["左边", "左侧", "left"])
    direction_right = _query_has_any(query_lower + query, ["右边", "右侧", "right"])
    deictic = _query_has_any(query_lower + query, ["那个", "这个", "that one", "this one", "that"])

    label_candidates = visible
    matched_label = None
    for label, aliases in alias_map.items():
        if any(alias.lower() in query_lower or alias in query for alias in aliases):
            label_candidates = [item for item in visible if str(item.get("label", "")).lower() == label]
            matched_label = label
            break

    candidates = list(label_candidates)
    method = "none"
    if direction_left and len(candidates) >= 1:
        with_center = [item for item in candidates if _safe_center_norm(item) is not None]
        if with_center:
            selected = min(with_center, key=lambda x: _safe_center_norm(x)[0])  # type: ignore[index]
            method = "label+directional" if matched_label else "directional"
            return {
                "query": query,
                "resolved": True,
                "selected_obj_id": selected.get("obj_id"),
                "selected_label": selected.get("label"),
                "selected_object": deepcopy(selected),
                "candidate_count": len(candidates),
                "candidates": [item.get("obj_id") for item in candidates],
                "method": method,
                "confidence": 0.8 if matched_label else 0.68,
                "clarification_needed": False,
                "clarifying_question": "",
            }

    if direction_right and len(candidates) >= 1:
        with_center = [item for item in candidates if _safe_center_norm(item) is not None]
        if with_center:
            selected = max(with_center, key=lambda x: _safe_center_norm(x)[0])  # type: ignore[index]
            method = "label+directional" if matched_label else "directional"
            return {
                "query": query,
                "resolved": True,
                "selected_obj_id": selected.get("obj_id"),
                "selected_label": selected.get("label"),
                "selected_object": deepcopy(selected),
                "candidate_count": len(candidates),
                "candidates": [item.get("obj_id") for item in candidates],
                "method": method,
                "confidence": 0.8 if matched_label else 0.68,
                "clarification_needed": False,
                "clarifying_question": "",
            }

    if len(candidates) == 1:
        selected = candidates[0]
        method = "label_match" if matched_label else "single_visible"
        return {
            "query": query,
            "resolved": True,
            "selected_obj_id": selected.get("obj_id"),
            "selected_label": selected.get("label"),
            "selected_object": deepcopy(selected),
            "candidate_count": len(candidates),
            "candidates": [selected.get("obj_id")],
            "method": method,
            "confidence": 0.76 if matched_label else 0.65,
            "clarification_needed": False,
            "clarifying_question": "",
        }

    if deictic and len(candidates) > 1:
        return {
            "query": query,
            "resolved": False,
            "selected_obj_id": None,
            "selected_label": matched_label,
            "selected_object": None,
            "candidate_count": len(candidates),
            "candidates": [item.get("obj_id") for item in candidates[:5]],
            "method": "ambiguous_deictic",
            "confidence": 0.32,
            "clarification_needed": True,
            "clarifying_question": "需要澄清：你指的是哪一个对象？请说明位置（如左边/右边）或类别。",
        }

    return {
        "query": query,
        "resolved": False,
        "selected_obj_id": None,
        "selected_label": matched_label,
        "selected_object": None,
        "candidate_count": len(candidates),
        "candidates": [item.get("obj_id") for item in candidates[:5]],
        "method": "no_resolution",
        "confidence": 0.3,
        "clarification_needed": False,
        "clarifying_question": "",
    }
