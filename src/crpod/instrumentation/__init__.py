"""Lightweight pipeline-stage timing — wave 4B.

Exports `Timer`, a process-local context manager + recorder used by
`crpod analyze-video` to attribute wall-clock time to each pipeline
stage (decode, YOLO, tracker, HUD, EV, blunders, report). The output
is consumed by `scripts/latency_report.py` and surfaced in
`docs/latency-budget.md`.
"""

from crpod.instrumentation.timing import Timer

__all__ = ["Timer"]
