"""Aggregate per-stage timings across multiple `analyze-video` runs.

Reads a JSONL file where each line is a timer report from one run
(`Timer.report()` serialized). Computes p50/p95/p99 across runs per
stage and emits a Markdown table suitable for pasting into
`docs/latency-budget.md`.

Usage:
    # Capture per-run timings via the wave-4B audit harness:
    uv run python scripts/latency_audit.py video1.mp4 video2.mp4 video3.mp4 \\
        --weights output/models/crpod_v1_best.pt \\
        --jsonl docs/latency-runs.jsonl

    # Then aggregate to a Markdown table:
    uv run python scripts/latency_report.py docs/latency-runs.jsonl

The two-script split lets you replay aggregation after the fact without
re-running the (slow) pipeline.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _percentile(samples: list[float], q: float) -> float:
    """Inclusive percentile — q in [0, 1]. Linear interp between order
    statistics. For our 3-run smoke this just picks the right element;
    the function generalizes if the script gets pointed at 50+ runs.
    """
    if not samples:
        return float("nan")
    if len(samples) == 1:
        return float(samples[0])
    s = sorted(samples)
    rank = q * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", type=Path, help="path to a JSONL of per-run Timer.report() outputs")
    ap.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="optional path to append the aggregated table to (e.g. docs/latency-budget.md)",
    )
    args = ap.parse_args(argv)

    # Aggregate: per-stage list of total seconds per run.
    by_stage: dict[str, list[float]] = {}
    n_runs = 0
    for line in args.jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        n_runs += 1
        payload = json.loads(line)
        for stage, stats in payload.items():
            # The audit script tags each row with `_video` (string); skip
            # underscore-prefixed metadata keys.
            if stage.startswith("_") or not isinstance(stats, dict):
                continue
            by_stage.setdefault(stage, []).append(float(stats["total"]))

    if not by_stage:
        print(f"no rows in {args.jsonl}")
        return 1

    print(f"# Latency budget — aggregated across {n_runs} runs\n")
    print("| stage | p50 (s) | p95 (s) | p99 (s) | mean (s) | min | max |")
    print("|---|---|---|---|---|---|---|")
    rows = []
    for stage, samples in sorted(by_stage.items(), key=lambda kv: -sum(kv[1])):
        p50 = _percentile(samples, 0.5)
        p95 = _percentile(samples, 0.95)
        p99 = _percentile(samples, 0.99)
        mean = statistics.fmean(samples)
        row = (
            f"| {stage} | {p50:.3f} | {p95:.3f} | {p99:.3f} | "
            f"{mean:.3f} | {min(samples):.3f} | {max(samples):.3f} |"
        )
        print(row)
        rows.append(row)

    if args.md_out is not None:
        out = ["", f"## Aggregated across {n_runs} runs", ""]
        out.append("| stage | p50 (s) | p95 (s) | p99 (s) | mean (s) | min | max |")
        out.append("|---|---|---|---|---|---|---|")
        out.extend(rows)
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        with args.md_out.open("a") as fh:
            fh.write("\n".join(out) + "\n")
        print(f"\nAppended to {args.md_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
