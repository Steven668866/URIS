from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence


def available_tracking_modes() -> list[str]:
    return ["simple", "bytetrack", "ocsort", "off"]


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


@dataclass(frozen=True)
class TrackerBackendStatus:
    requested_mode: str
    active_mode: str
    available: bool
    reason: str | None = None


class LiveObjectTracker:
    """
    Adapter-ready tracker wrapper.

    Phase 1/2 provides a deterministic lightweight tracker (`simple`) and keeps
    `bytetrack` / `ocsort` as interface-compatible modes that gracefully fall back.
    """

    def __init__(self, *, mode: str = "simple", distance_threshold: float = 0.18) -> None:
        requested = (mode or "simple").strip().lower()
        if requested not in set(available_tracking_modes()):
            requested = "simple"
        self.requested_mode = requested
        self.distance_threshold = float(distance_threshold)
        self._tracks: list[dict[str, Any]] = []
        self._next_track_id = 1
        self._status = self._detect_backend_status()

    def _detect_backend_status(self) -> TrackerBackendStatus:
        if self.requested_mode in {"simple", "off"}:
            return TrackerBackendStatus(
                requested_mode=self.requested_mode,
                active_mode=self.requested_mode,
                available=True,
                reason=None,
            )

        # Adapter placeholders for future real integration (ByteTrack / OC-SORT).
        # We intentionally avoid hard dependency imports here and degrade to simple.
        return TrackerBackendStatus(
            requested_mode=self.requested_mode,
            active_mode="simple",
            available=False,
            reason=f"{self.requested_mode} backend not installed yet; using simple tracker fallback",
        )

    @property
    def status(self) -> TrackerBackendStatus:
        return self._status

    def _match_track(self, detection: dict[str, Any], used_track_ids: set[int]) -> dict[str, Any] | None:
        label = str(detection.get("label", "object"))
        det_center = _safe_center_norm(detection)
        best: dict[str, Any] | None = None
        best_dist = 999.0
        for track in self._tracks:
            if int(track.get("track_id")) in used_track_ids:
                continue
            if str(track.get("label", "object")) != label:
                continue
            dist = _center_distance(_safe_center_norm(track), det_center)
            if dist < best_dist:
                best_dist = dist
                best = track
        if best is None or best_dist > self.distance_threshold:
            return None
        return best

    def update(self, *, detections: Sequence[dict[str, Any]], now_ts: float) -> dict[str, Any]:
        if self._status.active_mode == "off":
            return {
                "detections": [dict(det) for det in detections],
                "tracker_meta": {
                    "requested_mode": self._status.requested_mode,
                    "mode": self._status.active_mode,
                    "available": self._status.available,
                    "reason": self._status.reason,
                    "track_count": 0,
                },
            }

        updated: list[dict[str, Any]] = []
        used_track_ids: set[int] = set()
        for det in detections:
            det_copy = dict(det)
            matched = self._match_track(det_copy, used_track_ids)
            if matched is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                track_record = {
                    "track_id": track_id,
                    "label": str(det_copy.get("label", "object")),
                    "center_norm": list(det_copy.get("center_norm") or []),
                    "last_seen_ts": float(now_ts),
                }
                self._tracks.append(track_record)
            else:
                track_id = int(matched["track_id"])
                matched["center_norm"] = list(det_copy.get("center_norm") or matched.get("center_norm") or [])
                matched["last_seen_ts"] = float(now_ts)
            used_track_ids.add(int(track_id))
            det_copy["track_id"] = int(track_id)
            updated.append(det_copy)

        # Keep track list bounded and recent.
        stale_cutoff = float(now_ts) - 10.0
        self._tracks = [
            t for t in self._tracks if float(t.get("last_seen_ts", now_ts)) >= stale_cutoff
        ]

        return {
            "detections": deepcopy(updated),
            "tracker_meta": {
                "requested_mode": self._status.requested_mode,
                "mode": self._status.active_mode,
                "available": self._status.available,
                "reason": self._status.reason,
                "track_count": len(updated),
            },
        }
