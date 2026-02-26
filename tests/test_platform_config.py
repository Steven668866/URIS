from uris_platform.config import load_platform_config


def test_load_platform_config_defaults():
    cfg = load_platform_config(env={})
    assert cfg.backend_mode == "heuristic"
    assert cfg.default_scene == "living_room"
    assert cfg.profiling_enabled is True
    assert cfg.max_timeline_events >= 20


def test_load_platform_config_env_overrides():
    cfg = load_platform_config(
        env={
            "URIS_PLATFORM_BACKEND_MODE": "llm",
            "URIS_PLATFORM_DEFAULT_SCENE": "kitchen",
            "URIS_PLATFORM_PROFILING_ENABLED": "false",
            "URIS_PLATFORM_MAX_TIMELINE_EVENTS": "12",
        }
    )
    assert cfg.backend_mode == "llm"
    assert cfg.default_scene == "kitchen"
    assert cfg.profiling_enabled is False
    assert cfg.max_timeline_events == 12
