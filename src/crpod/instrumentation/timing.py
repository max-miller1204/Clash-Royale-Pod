"""`Timer` — record wall-clock durations per pipeline stage.

Usage:
    timer = Timer()
    with timer.record("decode"):
        frames = list(VideoFrameIterator(...))
    with timer.record("yolo"):
        detections = detector.infer(frames)
    print(timer.report())
    # → {"decode": [0.123], "yolo": [4.567]}

The timer is process-local and lock-free — pipeline code is single-threaded
per video, so no synchronization is needed. Multiple recordings for the
same stage append to the list, which is what enables p50/p95/p99 reporting
in `scripts/latency_report.py`.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager


class Timer:
    """Per-stage wall-clock recorder.

    Each `record(stage)` call yields a context that times its block on
    entry/exit; the elapsed seconds get appended to the stage's list.
    Querying `durations` or `report()` returns the accumulated samples.
    """

    def __init__(self) -> None:
        self._durations: dict[str, list[float]] = {}

    @contextmanager
    def record(self, stage: str) -> Generator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._durations.setdefault(stage, []).append(elapsed)

    @property
    def durations(self) -> dict[str, list[float]]:
        """Snapshot of the per-stage duration lists (returned by reference;
        callers may mutate at their own risk)."""
        return self._durations

    def report(self) -> dict[str, dict[str, float]]:
        """Summary statistics per stage: count, total, mean, min, max.

        Percentiles (p50/p95/p99) are intentionally NOT computed here — a
        single `analyze-video` run produces one sample per stage, so the
        statistics carry no useful spread. `scripts/latency_report.py`
        aggregates across runs and computes percentiles there.
        """
        out: dict[str, dict[str, float]] = {}
        for stage, samples in self._durations.items():
            if not samples:
                continue
            out[stage] = {
                "count": float(len(samples)),
                "total": float(sum(samples)),
                "mean": float(sum(samples) / len(samples)),
                "min": float(min(samples)),
                "max": float(max(samples)),
            }
        return out
