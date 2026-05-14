"""Run `analyze_video` with per-stage instrumentation across N videos.

Emits a JSONL where each line is one run's `Timer.report()` payload, so
`scripts/latency_report.py` can compute percentiles across runs.

Usage:
    uv run python scripts/latency_audit.py video1.mp4 video2.mp4 video3.mp4 \\
        --weights output/models/crpod_v1_best.pt \\
        --jsonl docs/latency-runs.jsonl

A failing video logs the error and skips — one bad codec shouldn't
nuke the whole audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from crpod.instrumentation import Timer
from crpod.modeling.ev import EvModel
from crpod.pipeline import analyze_video


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="+", type=Path, help="paths to local match videos (.mp4)")
    ap.add_argument(
        "--weights",
        required=True,
        type=Path,
        help="trained YOLO weights (e.g. output/models/crpod_v1_best.pt)",
    )
    ap.add_argument("--model", type=Path, default=None, help="optional EV model .joblib")
    ap.add_argument("--target-fps", type=float, default=10.0)
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=Path("docs/latency-runs.jsonl"),
        help="append per-run Timer.report() lines to this path",
    )
    args = ap.parse_args(argv)

    if not args.weights.exists():
        print(f"error: weights not found: {args.weights}", file=sys.stderr)
        return 1

    model = EvModel.load(args.model) if args.model and args.model.exists() else None

    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl.open("a") as fh:
        for video in args.videos:
            if not video.exists():
                print(f"[skip] missing: {video}", file=sys.stderr)
                continue
            timer = Timer()
            print(f"== running: {video}", file=sys.stderr)
            try:
                analyze_video(
                    video,
                    yolo_weights=args.weights,
                    model=model,
                    target_fps=args.target_fps,
                    timer=timer,
                )
            except Exception as exc:
                print(f"[error] {video}: {exc}", file=sys.stderr)
                continue
            report = timer.report()
            report["_video"] = str(video)  # type: ignore[assignment]
            fh.write(json.dumps(report) + "\n")
            fh.flush()
            for stage, stats in report.items():
                if stage.startswith("_"):
                    continue
                print(f"  {stage}: total={stats['total']:.3f}s", file=sys.stderr)

    print(f"appended to {args.jsonl}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
