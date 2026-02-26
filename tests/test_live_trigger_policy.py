from uris_platform.services.live_camera import LiveTriggerPolicy, should_trigger_qwen


def test_trigger_policy_user_submit_ignores_scene_change_flag():
    policy = LiveTriggerPolicy(min_interval_seconds=2.0, allow_auto_scene_trigger=False)
    decision = should_trigger_qwen(
        policy=policy,
        now_ts=10.0,
        last_qwen_ts=9.5,
        user_submitted=True,
        scene_signature_changed=False,
    )
    assert decision.trigger is True
    assert decision.reason == "user_submit"


def test_trigger_policy_blocks_auto_when_disabled():
    policy = LiveTriggerPolicy(min_interval_seconds=2.0, allow_auto_scene_trigger=False)
    decision = should_trigger_qwen(
        policy=policy,
        now_ts=10.0,
        last_qwen_ts=5.0,
        user_submitted=False,
        scene_signature_changed=True,
    )
    assert decision.trigger is False
    assert decision.reason == "auto_disabled"


def test_trigger_policy_respects_cooldown_for_auto():
    policy = LiveTriggerPolicy(min_interval_seconds=2.0, allow_auto_scene_trigger=True)
    blocked = should_trigger_qwen(
        policy=policy,
        now_ts=10.0,
        last_qwen_ts=9.2,
        user_submitted=False,
        scene_signature_changed=True,
    )
    allowed = should_trigger_qwen(
        policy=policy,
        now_ts=12.5,
        last_qwen_ts=9.2,
        user_submitted=False,
        scene_signature_changed=True,
    )
    assert blocked.trigger is False
    assert blocked.reason == "cooldown"
    assert allowed.trigger is True
    assert allowed.reason == "scene_change"
