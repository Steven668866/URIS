from uris_platform.services.evaluation import compute_evaluation_summary


def _interaction(command, action, target, confidence, room="living_room"):
    return {
        "timestamp": 1700000000.0,
        "command": command,
        "room": room,
        "plan": {
            "action": action,
            "target": target,
            "confidence": confidence,
            "steps": [],
            "explanation": "",
            "adaptation_note": None,
        },
    }


def test_compute_evaluation_summary_core_metrics():
    interactions = [
        _interaction("move chair right", "reposition_object", "chair", 0.84),
        _interaction("move chair right", "reposition_object", "chair", 0.83),
        _interaction("clean cup", "clean_object", "cup", 0.79),
        _interaction("do something thoughtful", "clarify_request", None, 0.25),
    ]
    perf_history = [
        {"total_ms": 12.0},
        {"total_ms": 24.0},
        {"total_ms": 18.0},
        {"total_ms": 30.0},
    ]
    summary = compute_evaluation_summary(interactions, perf_history, feedback_records=[])

    assert summary["interaction_count"] == 4
    assert summary["latency"]["avg_ms"] == 21.0
    assert summary["latency"]["p95_ms"] == 30.0
    assert summary["satisfaction"]["avg_rating"] is None
    assert summary["task_completion"]["simulated_rate"] == 0.75
    assert summary["consistency"]["score"] == 1.0


def test_compute_evaluation_summary_feedback_overrides_completion_and_tracks_satisfaction():
    interactions = [
        _interaction("move chair right", "reposition_object", "chair", 0.84),
        _interaction("clean cup", "clean_object", "cup", 0.79),
    ]
    perf_history = [{"total_ms": 10.0}, {"total_ms": 20.0}]
    feedback = [
        {"interaction_index": 0, "satisfaction_rating": 5, "completion_status": "failed"},
        {"interaction_index": 1, "satisfaction_rating": 3, "completion_status": "completed"},
    ]

    summary = compute_evaluation_summary(interactions, perf_history, feedback_records=feedback)
    assert summary["satisfaction"]["avg_rating"] == 4.0
    assert summary["satisfaction"]["count"] == 2
    assert summary["task_completion"]["simulated_rate"] == 0.5
    assert summary["task_completion"]["feedback_override_count"] == 2


def test_compute_evaluation_summary_includes_research_metrics_for_json_grounding_and_cache():
    interactions = [
        {
            **_interaction("桌上有什么", "scene_query", "cup", 0.8),
            "json_valid": True,
            "clarification_needed": False,
            "reference_resolution": {"resolved": True, "candidate_count": 1},
            "cache_hit": False,
        },
        {
            **_interaction("那个杯子是什么", "interactive_query", "cup", 0.7),
            "json_valid": True,
            "clarification_needed": True,
            "reference_resolution": {"resolved": False, "candidate_count": 2, "clarification_needed": True},
            "cache_hit": True,
        },
        {
            **_interaction("clean cup", "clean_object", "cup", 0.9),
            "json_valid": False,
            "clarification_needed": False,
            "reference_resolution": {},
            "cache_hit": True,
        },
    ]
    summary = compute_evaluation_summary(interactions, perf_history=[{"total_ms": 10.0}], feedback_records=[])

    rm = summary["research_metrics"]
    assert rm["json_valid_rate"] == 0.6667
    assert rm["json_valid_count"] == 3
    assert rm["clarification_rate"] == 0.3333
    assert rm["clarification_count"] == 3
    assert rm["reference_resolution_rate"] == 0.5
    assert rm["reference_attempt_count"] == 2
    assert rm["cache_hit_rate"] == 0.6667
    assert rm["cache_observation_count"] == 3
