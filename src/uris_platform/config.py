from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PlatformConfig:
    backend_mode: str = "heuristic"
    default_scene: str = "living_room"
    profiling_enabled: bool = True
    max_timeline_events: int = 30
    show_legacy_shortcuts: bool = True
    cache_scene_templates: bool = True


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_platform_config(env: Mapping[str, str] | None = None) -> PlatformConfig:
    source = env or {}
    return PlatformConfig(
        backend_mode=source.get("URIS_PLATFORM_BACKEND_MODE", "heuristic"),
        default_scene=source.get("URIS_PLATFORM_DEFAULT_SCENE", "living_room"),
        profiling_enabled=_parse_bool(
            source.get("URIS_PLATFORM_PROFILING_ENABLED"), True
        ),
        max_timeline_events=max(
            5, _parse_int(source.get("URIS_PLATFORM_MAX_TIMELINE_EVENTS"), 30)
        ),
        show_legacy_shortcuts=_parse_bool(
            source.get("URIS_PLATFORM_SHOW_LEGACY_SHORTCUTS"), True
        ),
        cache_scene_templates=_parse_bool(
            source.get("URIS_PLATFORM_CACHE_SCENE_TEMPLATES"), True
        ),
    )
