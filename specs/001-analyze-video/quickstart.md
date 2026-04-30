# Quickstart: `crpod analyze-video`

A pod member's smoke-test recipe for the new subcommand. Use this to
verify the feature works end-to-end after `/speckit-implement` lands the
code, and to reproduce the success criteria from `spec.md`.

## Prerequisites

1. Repo cloned and the dev shell active (Nix flake or `uv sync` per
   `README.md`).
2. Trained detection weights on disk. The current canonical artifact is
   `output/models/crpod_v1_best.pt` (KataCR-public-dataset training run,
   mAP@0.5 = 0.885). Re-train via `scripts/brev_train.sh` if missing.
3. A short Clash Royale match video at hand. Suggested smoke source:
   ~30 s of any single match recorded from the player's phone (scrcpy or
   the in-game replay export). Keep it under
   `~/clash-videos/<match>.mp4`; do *not* commit it.

## Smoke run (US1: P1)

```bash
uv run crpod analyze-video \
    ~/clash-videos/sample.mp4 \
    --weights output/models/crpod_v1_best.pt \
    --out output/analysis/sample/
```

Expected behavior:

- stderr: a handful of `[crpod] frame=…/… plays=… ocr_fail=…%` progress
  lines spaced over the run.
- stdout: a pretty-printed `summary.json` content.
- exit code: `0`.
- `output/analysis/sample/summary.json` exists with the schema in
  `contracts/cli.md`.

What to verify by eye:

- `n_plays` is roughly the number of cards the human saw deployed (within
  ±10% — SC-002).
- `friendly_leak` is small (player rarely leaks if the recording is
  competitive).
- The `replay_id` matches `Path(video).stem`.

## Failure-mode smoke (US1: P1, error paths)

Verify each of the four FR-011 / SC-006 fail-fast paths surfaces in under
2 seconds:

```bash
# (a) missing video
uv run crpod analyze-video /tmp/does-not-exist.mp4 \
    --weights output/models/crpod_v1_best.pt
# expect: exit 1, stderr "error: video file not found: …"

# (b) missing weights
uv run crpod analyze-video ~/clash-videos/sample.mp4 \
    --weights /tmp/missing.pt
# expect: exit 1, stderr "error: weights file not found: …"

# (c) corrupt video — make a 1-byte file
echo > /tmp/bad.mp4
uv run crpod analyze-video /tmp/bad.mp4 \
    --weights output/models/crpod_v1_best.pt
# expect: exit 1, stderr "error: cannot decode video …"
```

The empty-detection path (FR-011 (d)) requires running the full pipeline
on a video the detector cannot resolve. Easiest reproducer: a video of a
non-Clash-Royale clip.

## Visualization parity smoke (US2: P2)

Run the same successful command from the smoke run; after it finishes:

```bash
ls output/analysis/sample/
# expect: summary.json, placements.png, tempo.png
```

Compare `placements.png` and `tempo.png` against the equivalents from a
recent `crpod analyze` run on the HF dataset — they should be visually
indistinguishable in style (palette, axis labels, layout).

## EV model smoke (US3: P3)

You need an EV model artifact. Train one against the HF dataset first if
you don't have one:

```bash
uv run crpod train \
    --weights output/models/crpod_v1_best.pt \
    --out output/models/ev.joblib \
    --max-replays 50
```

Then re-run the analyzer with `--model`:

```bash
uv run crpod analyze-video \
    ~/clash-videos/sample.mp4 \
    --weights output/models/crpod_v1_best.pt \
    --model output/models/ev.joblib \
    --out output/analysis/sample-with-ev/
```

The presence of EV predictions in the produced `AnalysisResult` is the
test signal. Whether the predictions land in `summary.json` (vs. a
sibling artifact) is left to the implementer per the data-model note,
provided BG-7 in the CLI contract holds.

## Pre-merge checks (CI parity)

Before opening the PR for this feature:

```bash
make lint format type-check test
```

All four must pass. CI runs the same four commands under `nix develop`.

## What "done" looks like

US1 done = smoke run produces `summary.json` + the four error paths fail
fast. US2 done = smoke run also produces both PNGs. US3 done = EV
predictions appear when `--model` is supplied and not when it isn't.

The full success bar is the SC-* set in `spec.md` — those require a
hand-annotated reference video, which the pod will produce against one
or two of the cleaner phone recordings the team already has.
