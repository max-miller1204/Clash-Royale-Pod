"""Wave 4B — `Timer` shape contract.

The CLI / latency-audit script depends on the report shape, so it's
pinned here so a refactor can't break it silently.
"""

from __future__ import annotations

import time

from crpod.instrumentation import Timer


def test_record_appends_per_stage_durations() -> None:
    timer = Timer()
    with timer.record("decode"):
        pass
    with timer.record("yolo"):
        pass
    with timer.record("yolo"):
        pass

    assert set(timer.durations.keys()) == {"decode", "yolo"}
    assert len(timer.durations["decode"]) == 1
    assert len(timer.durations["yolo"]) == 2


def test_record_actually_measures_time() -> None:
    timer = Timer()
    with timer.record("sleep"):
        time.sleep(0.01)
    assert timer.durations["sleep"][0] >= 0.005  # generous lower bound


def test_report_summary_shape() -> None:
    timer = Timer()
    for _ in range(3):
        with timer.record("yolo"):
            pass

    report = timer.report()
    assert "yolo" in report
    assert report["yolo"].keys() == {"count", "total", "mean", "min", "max"}
    assert report["yolo"]["count"] == 3.0


def test_report_omits_empty_stages() -> None:
    """A stage with no samples (shouldn't normally happen, but defensive)
    is filtered out of `report()`."""
    timer = Timer()
    timer.durations["future_stage"] = []
    with timer.record("decode"):
        pass

    report = timer.report()
    assert "decode" in report
    assert "future_stage" not in report


def test_record_records_on_exception() -> None:
    """A `record` block that raises still records the duration so
    failed runs don't drop their partial timings."""
    timer = Timer()
    try:
        with timer.record("yolo"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert "yolo" in timer.durations
    assert len(timer.durations["yolo"]) == 1
