# Feature Specification: `crpod analyze-video` end-to-end

**Feature Branch**: `001-analyze-video`
**Created**: 2026-04-30
**Status**: Draft
**Input**: User description: "crpod analyze-video"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Analyze a recorded match video and get a summary (Priority: P1)

A pod member has captured (or downloaded) a Clash Royale match as a local
video file from the player's viewer perspective. They want to point the
analyzer at that file and receive the same kind of structured match summary
the project already produces from the HuggingFace TV-replay dataset — without
needing the dataset, the parquet shards, or any manual frame extraction.

**Why this priority**: This is the integration the project has been building
toward. Until a video on disk can flow through the full CV stack
(detection → tracking → OCR) and produce a `summary.json`, the project's CLI
only operates on a single curated dataset. Hitting this milestone unblocks
every downstream goal — modeling, visualization, blunder detection, real-time
mode — because they all consume the same `Replay`/`Interaction` stream
this story produces.

**Independent Test**: Given a recorded match video and trained detection
weights, a single command produces a `summary.json` containing replay
identifier, total play count, total interaction count, friendly elixir
leaked, and enemy elixir leaked. The story is complete when those five
numbers are present, plausible (non-negative; play count > 0 for a normal
match), and the command exits with status 0.

**Acceptance Scenarios**:

1. **Given** a local video of a complete Clash Royale match in viewer
   perspective, **When** the user invokes the analyze-video command with the
   video path and a path to trained detection weights, **Then** the command
   writes a `summary.json` to the configured output directory and prints the
   same summary to stdout.
2. **Given** the user passes a path to a video file that does not exist,
   **When** the command starts, **Then** it exits with a non-zero status and
   a clear error message before any detection or decoding work begins.
3. **Given** the user passes a path to detection weights that does not
   exist, **When** the command starts, **Then** it exits with a non-zero
   status and a clear error message before any video decoding begins.
4. **Given** a valid video and weights but the video produces zero
   detections (wrong content, blank video, mismatched weights), **When** the
   pipeline finishes, **Then** the command exits with a non-zero status and
   a message naming the empty stage rather than emitting a misleading
   summary with zeroed fields.

---

### User Story 2 - Generate the same visualizations from a video (Priority: P2)

After producing a summary from a video, the same pod member wants the
placement heatmap and the elixir-tempo timeseries that the existing
HF-replay analyzer already produces, so that downstream notebooks,
dashboards, and the eventual final report can treat video-sourced and
HF-sourced replays identically.

**Why this priority**: Output parity with the existing `analyze` command is
the difference between "a one-off integration script" and "a real second
ingest path." Without it, every consumer of `output/analysis/` has to learn
that some replays come with viz and some don't. With it, the rest of the
project doesn't need to know which ingest path was used.

**Independent Test**: Running analyze-video against any reasonable match
video yields, in addition to `summary.json`, both a placement heatmap image
and a tempo timeseries image in the configured output directory. The story
passes when both files exist, are non-empty, and are visually
indistinguishable in style from the equivalents produced by the HF replay
analyzer on a different match.

**Acceptance Scenarios**:

1. **Given** a video that produced a non-empty plays list in Story 1,
   **When** the analyze-video command completes, **Then** the output
   directory contains both `placements.png` and `tempo.png`.
2. **Given** the visualization dependency fails or is unavailable, **When**
   the command completes, **Then** `summary.json` is still written and the
   visualization failure is reported as a warning on stderr (consistent with
   the existing analyzer's behavior).

---

### User Story 3 - Apply a trained EV model to a video (Priority: P3)

Once the model and EV stages are wired in, the pod member wants to run a
trained EV model against the interactions extracted from a video, just as
they can today against an HF replay, to get per-interaction EV predictions
alongside the summary.

**Why this priority**: The project's eventual stretch goals (per-card EV,
blunder detection) all consume EV predictions. This story closes the loop
between video ingest and modeling so that those stretch goals work for
arbitrary recorded matches, not only for the curated HF dataset.

**Independent Test**: Given a trained EV model artifact and a video that
yields at least one interaction, the command produces an `ev_predictions`
field (or analogous output) of length equal to the interactions list. The
story passes when supplying a model produces predictions and omitting the
model produces a summary without predictions, in both cases without
errors.

**Acceptance Scenarios**:

1. **Given** a video and a path to a trained EV model, **When** the user
   invokes analyze-video with both, **Then** the resulting summary or a
   sibling artifact lists one EV prediction per interaction.
2. **Given** the user does not supply a model, **When** the command runs,
   **Then** the summary still contains the non-EV fields and no error is
   raised about a missing model.

---

### Edge Cases

- **Video shorter than one interaction window**: the run should complete
  with `n_plays = 0` and `n_interactions = 0` reported, but as a clean
  degenerate result, not as a crash.
- **Video frame rate differs from the analyzer's processing rate**: the
  pipeline must not assume the source fps matches its target rate; it must
  decode at the target rate independent of source rate.
- **Spectator-perspective video** (both decks visible, viewer is not a
  player): the side-inference rule for player-perspective videos will
  classify everything incorrectly; the system must either detect this and
  warn, or document that this case is out of scope for v1.
- **Unsupported codec / truncated file**: must surface a clear "cannot
  decode video" error before the detection model is loaded into memory.
- **Detection weights trained on a different game version than the video**:
  the system has no clean way to detect this, but the failure mode is "very
  few detections" — handled by the empty-detections error in Story 1
  scenario 4.
- **HUD OCR fails on every frame** (stylized text, low resolution): the
  system should still emit a `summary.json` with whatever plays were
  detected, but should report the OCR failure rate so the user knows the
  elixir-leak figures are unreliable.
- **Multi-match concatenated video** (e.g., a YouTube highlight reel): out
  of scope for v1; the system treats the entire input as one match and
  produces a single summary.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST expose a single command-line entry point that
  takes a path to a local video file and a path to trained detection
  weights and produces an analysis summary.
- **FR-002**: The system MUST validate that the video path and the weights
  path both exist on disk before performing any decoding, model loading, or
  network access.
- **FR-003**: The system MUST decode video frames at a fixed processing
  rate independent of the source frame rate, so that videos recorded at
  any common rate produce comparable analyses.
- **FR-004**: The system MUST detect cards on each processed frame using
  the supplied weights.
- **FR-005**: The system MUST track detected entities across frames so
  that one placement of a single card produces exactly one play event,
  not one event per frame the card is visible.
- **FR-006**: The system MUST extract per-frame heads-up display readings
  (own elixir, opponent elixir, match timer, tower health for both sides)
  for as many frames as the heads-up display is visible.
- **FR-007**: The system MUST classify each detected play as friendly
  (the viewer's side) or enemy (the opponent's side) using a viewer-aware
  rule appropriate for player-perspective video; it MUST NOT reuse the
  rule used for the HF dataset, which assumes a different framing.
- **FR-008**: The system MUST assemble a unified replay representation
  containing the full play stream and heads-up-display stream, with the
  same shape that the existing HF-replay analyzer consumes.
- **FR-009**: The system MUST run the same downstream analysis stage that
  the HF-replay analyzer runs, so that interactions, elixir leak, and
  tempo are computed by a single shared code path rather than two
  parallel implementations.
- **FR-010**: The system MUST write a summary file (`summary.json` by
  convention) to a user-configurable output directory, containing at
  minimum the replay identifier, source video path, total play count,
  total interaction count, friendly elixir leaked, and enemy elixir
  leaked.
- **FR-011**: The system MUST exit with a non-zero status and a clear
  message when (a) the video file is missing, (b) the weights file is
  missing, (c) the video cannot be decoded, or (d) the entire detection
  stream is empty (a detection-stage failure rather than a real
  zero-play match).
- **FR-012**: The system SHOULD also produce a placement heatmap image
  and an elixir tempo timeseries image in the same output directory
  whenever the visualization stage is available, and SHOULD treat a
  visualization failure as a non-fatal warning rather than a hard error.
- **FR-013**: The system SHOULD accept an optional path to a trained EV
  model and, when supplied, attach per-interaction EV predictions to the
  produced analysis result.
- **FR-014**: The system MUST NOT change the behavior, output schema, or
  invocation of the existing HF-replay analyzer.
- **FR-015**: The system MUST report progress (frames processed, plays
  detected so far, OCR failure rate) at a cadence sufficient for a user
  to confirm the pipeline is making forward progress on a long video,
  without flooding the terminal on a short one.

### Key Entities *(include if feature involves data)*

- **Video Source**: A path to a local video file representing a single
  Clash Royale match recorded from the player's viewer perspective. The
  contract is "decoded by a standard video reader at any common frame
  rate"; format negotiation, transcoding, and remote URLs are out of
  scope for v1.
- **Detection Weights**: A path to a trained detection model artifact,
  re-trainable in roughly 30 minutes of GPU time per the project's data
  pipeline. The video analyzer is a consumer; it does not retrain or
  validate the weights.
- **EV Model** *(optional)*: A trained expected-value model artifact.
  When present, it scores interactions; when absent, the pipeline runs
  without scoring.
- **Replay**: The unified per-match data structure already defined by
  the project. Both the HF analyzer and the video analyzer produce one;
  downstream stages do not distinguish between them.
- **Card Play**: One card placement event with frame, card identifier,
  position, side classification, and elixir cost.
- **HUD State**: One per-frame heads-up-display reading covering elixir
  for both sides, match timer, and tower health for both sides.
- **Interaction**: A time-windowed bundle of friendly and enemy plays
  with aggregate elixir spent and damage outcomes; the unit the EV
  model scores.
- **Analysis Result**: The end-of-pipeline artifact: the replay, the
  interactions, the per-interaction feature rows, friendly and enemy
  elixir leak totals, the running tempo series, and (optionally) EV
  predictions.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A pod member can analyze a single 3-minute Clash Royale
  match video and receive a complete summary in under 60 seconds on a
  mid-range single-GPU machine, end-to-end.
- **SC-002**: On at least one annotated reference video, the play count
  reported by the analyzer matches the human-counted play count to within
  ±10%.
- **SC-003**: On the same reference video, the friendly-vs-enemy side
  classification matches a human's judgment on at least 95% of the plays
  the analyzer reports.
- **SC-004**: On the same reference video, the friendly elixir leaked and
  enemy elixir leaked figures are each within 1.0 elixir of the
  human-counted ground truth.
- **SC-005**: Every field present in the existing HF analyzer's
  `summary.json` is also present in the video analyzer's `summary.json`,
  so any tool, notebook, or sub-pipeline that consumes one consumes the
  other without modification.
- **SC-006**: When invoked with a missing video, missing weights, or a
  corrupt video, the analyzer surfaces the failure within 2 seconds of
  invocation and never starts the GPU detection stage.

## Assumptions

- The input is a single video file representing one match. Highlight
  reels, multi-match recordings, and live streams are out of scope for
  this feature; live streaming is reserved for the separate `crpod live`
  feature.
- The video is recorded from the **player's** viewer perspective (deck at
  bottom of frame, opponent's deck at top). Spectator and opponent-side
  recordings are out of scope for v1; if they are passed in, the
  side-classification metric (SC-003) will not hold and the user will see
  most plays misclassified.
- The match was played in the Clash Royale season the project's data
  pipeline is pinned to; older-season videos may produce degraded
  detection accuracy and are not validated.
- Trained detection weights exist on disk. (As of this writing they do —
  `output/models/crpod_v1_best.pt` from the KataCR-public-dataset training
  run, mAP@0.5 = 0.885.) Producing or retraining those weights is outside
  this feature's scope.
- The viewer is the bottom-side ("friendly") player. The side-inference
  rule for video is built around that assumption rather than the
  river-Y-coordinate rule used for the HF dataset.
- The downstream analysis stage (`analyze_replay` and the modeling
  pipeline) is treated as a fixed contract by this feature. If those
  stages need to change to support a new EV target, that work is tracked
  separately and is not blocking for this feature.
- Output goes to a local directory under `output/analysis/<video stem>/`
  by default; uploading, persisting to a database, or pushing artifacts
  to a remote store is out of scope for v1.
