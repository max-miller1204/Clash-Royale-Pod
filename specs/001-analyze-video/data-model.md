# Phase 1 Data Model: `crpod analyze-video`

This feature **does not introduce new persisted entities**. It plumbs
through a video and reuses the existing pipeline contract dataclasses in
`src/crpod/types.py`. Per constitution Principle II ("Pipeline
Modularity"), all inter-stage data flows through those types — adding new
ones would fracture the contract that the HF-replay path also depends on.

This document records, for each entity, **what role it plays in the video
pipeline** and **which fields are populated by which stage**, so the
implementation has a single reference for who-writes-what.

## Existing entities re-used as-is

### `Detection` — `src/crpod/detection/yolo.py`

Per-frame, per-bbox detector output. Already produced by `YoloDetector.infer`.

| Field | Set by | Notes |
| ----- | ------ | ----- |
| `frame` | `YoloDetector.infer` | Iterator's `out_idx` (the post-decimation frame index). |
| `cls` | `YoloDetector.infer` | KataCR class name; the card-name mapping table is a separate concern (TODO.md). |
| `confidence` | `YoloDetector.infer` | Float in [0, 1]. |
| `xyxy` | `YoloDetector.infer` | Frame-pixel coordinates. |

### `Track` — `src/crpod/tracking/bytetrack.py`

A grouped sequence of `Detection`s sharing a stable identity over time.
Currently a stub; this feature fills in `Tracker.update`.

| Field | Set by | Notes |
| ----- | ------ | ----- |
| `track_id` | `Tracker.update` (new) | The `tracker_id` from `supervision.ByteTrack`. |
| `cls` | `Tracker.update` (new) | Carried from the first detection in the track. |
| `detections` | `Tracker.update` (new) | Ordered by `frame` ascending. |

### `CardPlay` — `src/crpod/types.py`

The fundamental event the rest of the pipeline consumes. The HF path emits
one per dataset row; the video path emits one per qualifying `Track`
(see R2 in `research.md`).

| Field | Set by (video path) | Notes |
| ----- | ------------------- | ----- |
| `frame` | Track-reduction helper | First frame of the track. |
| `card` | Track-reduction helper | Carried from `Track.cls`. The KataCR class-name → underscore-card-name mapping (TODO.md item) is a separate scope. |
| `x` | Track-reduction helper | First-detection center x, rounded to int. |
| `y` | Track-reduction helper | First-detection center y, rounded to int. |
| `side` | New `crpod.dataset.side.infer_video_side` | `Side.FRIENDLY` if `y > frame_height/2`, else `Side.ENEMY`. |
| `elixir_cost` | Track-reduction helper | Looked up via `card_cost(card)` from `crpod.constants`. Falls back to `0` if the card name isn't in the table — same behavior as the HF path today. |

### `HudState` — `src/crpod/types.py`

Per-frame HUD reading. Produced by `HudReader.read` per Phase 0 R4
(input frame is rescaled to 540×960 before reading).

No fields change shape; the video pipeline calls `HudReader.read` on
every processed frame (or every Nth — see R7's progress note; cadence is
a tuning choice, not a contract change).

### `Replay` — `src/crpod/types.py`

The unified per-match data structure consumed by `analyze_replay`. The
video pipeline assembles one with:

| Field | Value (video path) |
| ----- | ------------------ |
| `replay_id` | `Path(video_path).stem`. Documented in CLI contract. |
| `arena` | `"video"` (a literal sentinel — videos are not associated with an HF arena directory). |
| `plays` | The reduced `CardPlay` stream from R2. |
| `hud` | The full `HudState` stream from `HudReader`. |
| `total_frames` | The iterator's final `out_idx + 1`. |
| `fps` | The CLI's `--target-fps` value (default 10). |

### `Interaction` — `src/crpod/types.py`

Produced by `build_interactions(replay.plays)`. **No video-specific
behavior** — this is exactly the existing logic, exercising the
constitution's "single shared analyze_replay code path" requirement
(Principle II).

### `AnalysisResult` — `src/crpod/pipeline.py`

Final output of `analyze_replay`. Already structured. The video CLI
serializes a subset to `summary.json` per FR-010 / SC-005:

```text
{
  "replay_id": <video stem>,
  "arena": "video",
  "n_plays": int,
  "n_interactions": int,
  "friendly_leak": float,
  "enemy_leak": float,
  "source_video": <abs or relative path>
}
```

`source_video` is a video-path-specific addition — the only schema delta
from the HF analyzer's `summary.json`. SC-005 ("every field present in
the existing analyzer's summary is also present in the video
analyzer's") is satisfied because the video summary is a *superset*, not
a different shape.

## New helper module

### `crpod.dataset.side` (new, ~20 lines)

Exposes one function:

```text
infer_video_side(y: float, frame_height: int) -> Side
```

This is the only piece of new entity-shaping logic in the feature. It is
deliberately *not* a method on a class and *not* a field on `CardPlay` —
it's a stateless utility that the track-reduction helper calls once per
play.

The HF dataset's existing `_infer_side` in `crpod.dataset.huggingface`
stays exactly as-is. FR-007 (don't reuse the HF rule) and FR-014 (don't
break the HF path) are both upheld by keeping the two rules in different
modules.

## State transitions

The CardPlay → Interaction → AnalysisResult pipeline is purely
functional — no mutable state across stages, nothing to reconcile. The
only stateful object in the new code is `supervision.ByteTrack` inside
`Tracker`, which is owned by a single invocation of `Tracker.update`
across one video and discarded after.

## Validation rules

These are derived from the spec's Functional Requirements and applied at
the boundary indicated:

| Rule | Source | Enforced at |
| ---- | ------ | ----------- |
| Video path exists on disk | FR-002 | CLI entry point, before any imports of heavy deps. |
| Weights path exists on disk | FR-002 | CLI entry point, before any imports of heavy deps. |
| Video is decodable | FR-011 (c) | `VideoFrameIterator` raises; CLI catches and exits 1 with stderr message. |
| Detection stream non-empty | FR-011 (d) | `analyze_video`, after the full pass. Exit 1 with stderr message. |
| `--target-fps > 0` | FR-003 | argparse `type=positive_float`. |
| `--out` dir is creatable | FR-010 | `mkdir(parents=True, exist_ok=True)` — same pattern as existing `analyze`. |
| EV model file exists if `--model` supplied | FR-013 | CLI entry point. |
