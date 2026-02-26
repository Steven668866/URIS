#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uris_platform.domain import SceneObject, SceneState
from uris_platform.services.scenario_engine import plan_robot_response


COMMANDS = [
    "Please move the chair to the right",
    "Clean the dirty cup on the table",
    "Bring the napkins to me",
    "Do something thoughtful",
    "Reposition the chair closer to the table",
]


def build_scene() -> SceneState:
    return SceneState(
        room="living_room",
        objects=[
            SceneObject(name="chair", zone="left", state="idle"),
            SceneObject(name="table", zone="center", state="set"),
            SceneObject(name="cup", zone="table", state="dirty"),
            SceneObject(name="napkins", zone="table", state="available"),
        ],
    )


def run_benchmark(iterations: int) -> dict:
    scene = build_scene()
    timings = []
    actions = {}
    for _ in range(iterations):
        cmd = random.choice(COMMANDS)
        start = time.perf_counter()
        plan = plan_robot_response(scene, cmd, preferences=("Add a safety confirmation",))
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings.append(elapsed_ms)
        actions[plan.action] = actions.get(plan.action, 0) + 1
    timings_sorted = sorted(timings)
    p95_index = max(0, min(len(timings_sorted) - 1, int(len(timings_sorted) * 0.95) - 1))
    return {
        "iterations": iterations,
        "mean_ms": round(statistics.mean(timings), 4),
        "median_ms": round(statistics.median(timings), 4),
        "p95_ms": round(timings_sorted[p95_index], 4),
        "min_ms": round(min(timings), 4),
        "max_ms": round(max(timings), 4),
        "actions": actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark URIS interaction planner latency")
    parser.add_argument("--iterations", type=int, default=100, help="Number of planner calls")
    args = parser.parse_args()
    report = run_benchmark(max(1, args.iterations))
    print("URIS Interaction Benchmark")
    print("=" * 60)
    for key, value in report.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
