# CLI Contract: `crpod analyze-video`

This file defines the user-facing CLI contract for the new subcommand. It
is the surface that the rest of the project (notebooks, smoke tests, the
eventual final report) is allowed to depend on.

## Invocation

```text
crpod analyze-video VIDEO_PATH
    --weights PATH
    [--out DIR]
    [--model PATH]
    [--target-fps N]
```

| Argument | Required | Default | Meaning |
| -------- | -------- | ------- | ------- |
| `VIDEO_PATH` | yes | — | Positional. Path to a local video file. |
| `--weights` | yes | — | Path to trained YOLO weights (`.pt`). |
| `--out` | no | `output/analysis/<video-stem>/` | Directory for `summary.json` + visualizations. Created if missing. |
| `--model` | no | — | Path to a saved EV model (`.joblib`). When supplied, attaches `ev_predictions` to the in-memory result; presence in `summary.json` is left for a later spec, since US3 is P3 and the schema delta is small. |
| `--target-fps` | no | `10` | Frame-rate at which to decimate the video. Must match the rate the detection model was trained against. |

## Exit codes

| Code | Meaning |
| ---- | ------- |
| `0` | Pipeline completed; `summary.json` written. (`n_plays = 0` is allowed if the video genuinely contains no plays — but see code 1 (d).) |
| `1` | Validation or pipeline error. The stderr message identifies which: (a) video file missing, (b) weights file missing, (c) cannot decode video, (d) detection stream empty, (e) EV model file missing. |
| `2` | argparse usage error (unknown flag, missing required arg, non-positive `--target-fps`). |

The two-second fail-fast budget (SC-006) applies to codes 1(a), 1(b), and
1(e) — these MUST be raised before the YOLO model is loaded.

## stdout

On success, the same JSON written to `summary.json` is also printed to
stdout, pretty-printed (2-space indent). This mirrors the existing
`crpod analyze` behavior.

```json
{
  "replay_id": "match_2026_04_30",
  "arena": "video",
  "source_video": "/path/to/match_2026_04_30.mp4",
  "n_plays": 28,
  "n_interactions": 11,
  "friendly_leak": 1.5,
  "enemy_leak": 0.0
}
```

## stderr

Used for two things, both line-buffered:

1. **Progress lines** (FR-015): emitted ~every 5% of expected frames or
   every 15 s wall-clock, whichever fires first.

   ```text
   [crpod] frame=180/1800 plays=4 ocr_fail=12%
   ```

2. **Warnings**: visualization failures, OCR failure-rate alerts. Format:

   ```text
   [warn] viz skipped: <exception text>
   ```

3. **Errors** (always paired with a non-zero exit code):

   ```text
   error: video file not found: /path/to/missing.mp4
   error: weights file not found: /path/to/missing.pt
   error: cannot decode video (codec or truncated file): /path/to/bad.mp4
   error: detection stream empty — check weights match the game version
   error: EV model file not found: /path/to/missing.joblib
   ```

## Output directory contents

```text
<out>/
├── summary.json             # Always written on success.
├── placements.png           # Written when viz stage succeeds (US2).
└── tempo.png                # Written when viz stage succeeds (US2).
```

`summary.json` is the authoritative artifact; the PNGs are
visualizations of the same underlying `AnalysisResult`. Downstream
consumers should depend only on `summary.json`'s schema.

## Behavioral guarantees (testable)

These map directly onto the spec's user-story acceptance scenarios.

| ID | Guarantee | Maps to spec |
| -- | --------- | ------------ |
| BG-1 | A successful run writes `summary.json` and exits 0. | US1 scenario 1 |
| BG-2 | A missing video path exits 1 within 2 s with a clear message. | US1 scenario 2; SC-006 |
| BG-3 | A missing weights path exits 1 within 2 s with a clear message. | US1 scenario 3; SC-006 |
| BG-4 | An empty detection stream exits 1, identifying the empty stage. | US1 scenario 4 |
| BG-5 | When viz succeeds, `placements.png` and `tempo.png` are written. | US2 scenario 1 |
| BG-6 | When viz fails, `summary.json` is still written and a warning hits stderr. | US2 scenario 2 |
| BG-7 | When `--model` is supplied, EV predictions are computed for every interaction. | US3 scenario 1 |
| BG-8 | When `--model` is omitted, the run still succeeds. | US3 scenario 2 |
| BG-9 | The existing `crpod analyze` and `crpod train` subcommands' help text and exit codes are unchanged. | FR-014 |

## Non-goals (deliberately not in this contract)

- Reading from URLs, S3, GCS, or any non-filesystem source.
- Reading from a webcam or capture card (covered by future `crpod live`).
- Streaming output (e.g., emitting plays as JSON-lines while the video
  decodes).
- Configuring detection confidence threshold (left at `YoloDetector`'s
  default for v1).
