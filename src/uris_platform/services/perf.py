from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class StageTiming:
    stage: str
    duration_ms: float


@contextmanager
def timed_stage(stage: str) -> Iterator[StageTiming]:
    started = time.perf_counter()
    record = StageTiming(stage=stage, duration_ms=0.0)
    try:
        yield record
    finally:
        record.duration_ms = (time.perf_counter() - started) * 1000
