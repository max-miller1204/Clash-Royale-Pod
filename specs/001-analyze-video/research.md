# Phase 0 Research: `crpod analyze-video`

The feature spec resolved every business/scope question with informed
defaults, so there are no `NEEDS CLARIFICATION` markers to chase. What
remains is a small set of *implementation* decisions where the cleanest path
is not obvious from reading the existing code. This document records those
decisions so the implementer doesn't re-derive them.

## R1. ByteTrack integration via `supervision`

**Decision**: Wrap `supervision.ByteTrack` inside `Tracker.update`. On each
call, build a `supervision.Detections` from the per-frame `Detection`
objects, run `tracker.update_with_detections(detections)`, and accumulate
the resulting `tracker_id`-tagged detections into per-track buckets keyed by
`track_id`. Return a list of `Track` objects whose `detections` list is
ordered by `frame`.

**Rationale**: `supervision.ByteTrack` is the standard third-party wrapper
this project already lists in `pyproject.toml`. Implementing it directly
against the raw ByteTrack repo would re-introduce the same dependency churn
that bit the KataCR weights pivot.

**Alternatives considered**:

- *Re-implement ByteTrack from scratch.* Rejected — out of scope for this
  feature; ByteTrack's Kalman-filter + IoU-association logic is non-trivial
  and the supervision wrapper is the project's chosen abstraction.
- *Skip tracking, emit a `CardPlay` per detection per frame.* Rejected —
  violates FR-005, would emit hundreds of duplicate plays per real card,
  and would make `build_interactions` return garbage.

**Open implementation note**: ByteTrack tuning parameters (track threshold,
match threshold, lost buffer) start at the supervision defaults. If
SC-002 (play count within ±10%) fails on the smoke video, retune before
declaring P1 done.

## R2. Track → CardPlay reduction

**Decision**: One `CardPlay` per `track_id`, anchored at the track's
*first* sighting (frame, center coordinates, class label). Discard tracks
that survive fewer than 2 frames as detection noise.

**Rationale**: A physical card placement is a single event in the player's
mental model. Anchoring on first sighting matches when the player saw the
card appear; using last sighting would shift the play later in time, and
averaging would smear it across the card's whole on-field life. The
2-frame minimum filters single-frame false positives without losing real
fast-traveling units (Princess arrows, Skeletons) — at 10 fps, "real"
units are visible for >10 frames.

**Alternatives considered**:

- *Anchor at the last frame.* Rejected — would disagree with how the HF
  dataset (which records the placement event itself) is structured.
- *Anchor at the centroid of the trajectory.* Rejected — would place a Hog
  Rider in the middle of its lane traversal, not where it was deployed.

## R3. Viewer-aware side inference

**Decision**: For player-perspective video, classify a play as `FRIENDLY`
when the placement Y coordinate is in the bottom half of the frame
(`y > frame_height / 2`) and `ENEMY` otherwise. Encode this rule in a new
`src/crpod/dataset/side.py` module so it does not get confused with the
HF dataset's existing rule.

**Rationale**: The HF dataset is filmed in TV-replay framing (different
camera framing) and uses a tuned `RIVER_Y` constant. Player-perspective
video uses the standard in-game camera, where the river bisects the frame
at roughly the midpoint and the player is always at the bottom. A simple
midpoint split is the smallest correct rule.

**Alternatives considered**:

- *Reuse `_infer_side` from `crpod.dataset.huggingface`.* Rejected by
  FR-007 — the HF rule is calibrated for a different framing.
- *Detect the player's hand strip at the bottom of the frame and key off
  card thumbnails.* Rejected as overkill for v1; it's the right answer
  for the future `crpod live` feature where the rule must work on
  spectator-perspective POVs, but for player POV the midpoint split is
  enough to hit SC-003 (≥95% correct).

## R4. HUD OCR for non-540×960 video

**Decision**: For v1, run HUD OCR using the existing 540×960-tuned
`HudRegions`, but **rescale the input frame to 540×960 before passing it
to `HudReader.read`**. Document that arbitrary-resolution HUD region
calibration is out of scope for this feature.

**Rationale**: The existing OCR rectangles are empirically tuned and the
project does not have a region-calibration tool. Resizing to the tuned
resolution is a one-liner with `cv2.resize` and preserves the existing
contract. If a smoke video has very different aspect ratio
(e.g., 16:9 vs the 9:16 mobile portrait the HUD assumes), elixir-leak
numbers will be unreliable; that limitation is acceptable for v1 and is
already covered by the spec's "OCR-blackout" edge case.

**Alternatives considered**:

- *Auto-detect HUD landmarks per resolution.* Rejected — that's a
  research project of its own and does not block US1 (US1 only requires
  detection + tracking, not OCR; HUD failure degrades elixir-leak
  fields, not the play count).
- *Hard-fail on non-540×960 inputs.* Rejected — too restrictive for the
  pod's primary recording sources (scrcpy at native phone res, OBS
  exports at 1080×1920). Resizing keeps the door open.

## R5. Empty-detection fail-fast

**Decision**: After the full pass through the video, if the detection
stream is empty *or* the tracker emitted zero tracks of length ≥ 2,
return exit code 1 with a stderr message identifying which stage was
empty. Do not pre-emptively bail in the middle of the run.

**Rationale**: A real match's first 1–2 seconds before the king tower
unlocks may legitimately have no detections. Bailing mid-stream risks
false positives. The full-pass cost on a 3-minute video is bounded by
SC-001 (under 60 s), so the user is never "stuck" for long.

**Alternatives considered**:

- *Bail if no detections in the first N frames.* Rejected — too brittle
  on dataset variation.
- *Always exit 0 with `n_plays = 0`.* Rejected by FR-011 scenario (d) —
  a misleading zeroed summary is exactly what the spec says to avoid.

## R6. Output directory convention

**Decision**: Default output directory is
`output/analysis/<video-stem>/`, where `<video-stem>` is
`Path(video_path).stem`. The directory is created if it doesn't exist.
The user can override with `--out DIR`, mirroring the existing `crpod
analyze` flag.

**Rationale**: Stem-based naming is the convention pod members already
use ad-hoc (per `output/analysis/` paths in `docs/TODO.md` and the
existing `analyze` flag). It also avoids collisions when running
multiple videos.

**Alternatives considered**:

- *Always require `--out`.* Rejected — adds friction for the common
  case of "analyze this one file".
- *Timestamped subdirs.* Rejected — makes diffs across runs harder; pod
  members can timestamp manually if they want history.

## R7. Progress reporting cadence

**Decision**: Print a single line to stderr every ~5% of expected frames
(or every 15 seconds wall-clock, whichever fires first), formatted as
`[crpod] frame=NNN/NNN plays=NNN ocr_fail=NN%`. No `tqdm` dependency.

**Rationale**: FR-015 requires "sufficient cadence to confirm progress
without flooding". The existing CLI prints plain JSON to stdout; adding
`tqdm` would couple the CLI to a progress library and pollute non-tty
runs in CI. A bounded set of stderr lines is enough and stays
machine-parseable.

**Alternatives considered**:

- *Use `tqdm` from `ultralytics`.* Rejected — Ultralytics already
  prints its own per-frame chatter when `verbose=True`; we explicitly
  set `verbose=False` in `YoloDetector.infer`. Layering two progress
  bars confuses users.
- *Silent runs.* Rejected — fails FR-015.

## R8. CLI subcommand shape

**Decision**: New subparser registered exactly like the existing
`analyze` and `train` subcommands.

```text
crpod analyze-video VIDEO_PATH
    --weights PATH        # required
    --out DIR             # default: output/analysis/<video-stem>/
    --model PATH          # optional EV model
    --target-fps N        # default: 10 (matches YOLO training)
```

**Rationale**: Mirrors the existing `crpod analyze` shape so users only
have to learn one flag set. `--target-fps` is exposed because the
processing rate is a first-class knob (FR-003), not buried in code.

**Alternatives considered**:

- *Auto-detect target fps.* Rejected — needs to match the rate the
  detection model was trained against. Surfacing it as a flag prevents
  silent miscalibration.

## Out of scope (deferred to follow-up specs)

- Auto-calibrated HUD regions for arbitrary input resolutions.
- Spectator-perspective side inference (will be needed for `crpod live`).
- Multi-match concatenated videos.
- Streaming / live capture sources (covered by future `crpod live`).
- Replacing the elixir-trade EV proxy with a damage-based target.
- Blunder detection on top of EV predictions.
