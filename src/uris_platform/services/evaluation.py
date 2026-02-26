from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, median
from typing import Any, Iterable


def _nearest_rank_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil((percentile / 100.0) * len(ordered)))
    return float(ordered[rank - 1])


def _normalized_command(command: str) -> str:
    return " ".join((command or "").lower().split())


def _consistency(interaction_history: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[tuple[str | None, str | None]]] = defaultdict(list)
    for item in interaction_history:
        cmd_key = _normalized_command(str(item.get("command", "")))
        plan = item.get("plan") or {}
        grouped[cmd_key].append((plan.get("action"), plan.get("target")))

    repeated_groups = [vals for vals in grouped.values() if len(vals) >= 2]
    if not repeated_groups:
        return {"score": None, "repeated_command_groups": 0}

    weighted_numerator = 0
    weighted_denominator = 0
    for outputs in repeated_groups:
        counts: dict[tuple[str | None, str | None], int] = {}
        for output in outputs:
            counts[output] = counts.get(output, 0) + 1
        weighted_numerator += max(counts.values())
        weighted_denominator += len(outputs)

    score = weighted_numerator / weighted_denominator if weighted_denominator else None
    return {
        "score": round(float(score), 4) if score is not None else None,
        "repeated_command_groups": len(repeated_groups),
    }


def _feedback_index(feedback_records: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    latest_by_index: dict[int, dict[str, Any]] = {}
    for record in feedback_records:
        try:
            idx = int(record.get("interaction_index"))
        except (TypeError, ValueError):
            continue
        latest_by_index[idx] = record
    return latest_by_index


def _completion_score_from_plan(plan: dict[str, Any]) -> float:
    action = str(plan.get("action") or "")
    confidence = float(plan.get("confidence") or 0.0)
    return 1.0 if action != "clarify_request" and confidence >= 0.5 else 0.0


def _completion_score_from_feedback(record: dict[str, Any]) -> float | None:
    status = str(record.get("completion_status") or "").strip().lower()
    if status == "completed":
        return 1.0
    if status == "failed":
        return 0.0
    if status == "partial":
        return 0.5
    return None


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator / denominator), 4)


def compute_evaluation_summary(
    interaction_history: list[dict[str, Any]],
    perf_history: list[dict[str, Any]],
    feedback_records: list[dict[str, Any]],
) -> dict[str, Any]:
    latencies = [
        float(item.get("total_ms"))
        for item in perf_history
        if item.get("total_ms") is not None
    ]

    ratings = []
    for record in feedback_records:
        try:
            rating = int(record.get("satisfaction_rating"))
        except (TypeError, ValueError):
            continue
        if 1 <= rating <= 5:
            ratings.append(rating)

    feedback_by_idx = _feedback_index(feedback_records)
    completion_scores: list[float] = []
    override_count = 0
    for idx, interaction in enumerate(interaction_history):
        plan = interaction.get("plan") or {}
        completion = _completion_score_from_plan(plan)
        feedback = feedback_by_idx.get(idx)
        if feedback:
            override = _completion_score_from_feedback(feedback)
            if override is not None:
                completion = override
                override_count += 1
        completion_scores.append(completion)

    consistency = _consistency(interaction_history)

    json_valid_values: list[bool] = []
    clarification_values: list[bool] = []
    reference_attempts = 0
    reference_resolved = 0
    cache_values: list[bool] = []
    for item in interaction_history:
        if "json_valid" in item:
            json_valid_values.append(bool(item.get("json_valid")))

        if "clarification_needed" in item:
            clarification_values.append(bool(item.get("clarification_needed")))
        else:
            plan = item.get("plan") or {}
            if "action" in plan:
                clarification_values.append(str(plan.get("action")) == "clarify_request")

        if "cache_hit" in item and item.get("cache_hit") is not None:
            cache_values.append(bool(item.get("cache_hit")))

        rr = item.get("reference_resolution")
        if isinstance(rr, dict) and rr:
            candidate_count = int(rr.get("candidate_count") or 0)
            if candidate_count > 0 or "resolved" in rr or "clarification_needed" in rr:
                reference_attempts += 1
                if bool(rr.get("resolved")):
                    reference_resolved += 1

    return {
        "interaction_count": len(interaction_history),
        "latency": {
            "count": len(latencies),
            "avg_ms": round(float(mean(latencies)), 4) if latencies else None,
            "median_ms": round(float(median(latencies)), 4) if latencies else None,
            "p95_ms": round(float(_nearest_rank_percentile(latencies, 95)), 4)
            if latencies
            else None,
        },
        "satisfaction": {
            "count": len(ratings),
            "avg_rating": round(float(mean(ratings)), 4) if ratings else None,
        },
        "task_completion": {
            "simulated_rate": round(float(mean(completion_scores)), 4)
            if completion_scores
            else None,
            "count": len(completion_scores),
            "feedback_override_count": override_count,
        },
        "consistency": consistency,
        "research_metrics": {
            "json_valid_rate": _rate(sum(1 for x in json_valid_values if x), len(json_valid_values)),
            "json_valid_count": len(json_valid_values),
            "clarification_rate": _rate(sum(1 for x in clarification_values if x), len(clarification_values)),
            "clarification_count": len(clarification_values),
            "reference_resolution_rate": _rate(reference_resolved, reference_attempts),
            "reference_attempt_count": reference_attempts,
            "cache_hit_rate": _rate(sum(1 for x in cache_values if x), len(cache_values)),
            "cache_observation_count": len(cache_values),
        },
    }
