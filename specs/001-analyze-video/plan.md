# Implementation Plan: `crpod analyze-video` end-to-end

**Branch**: `001-analyze-video` | **Date**: 2026-04-30 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/001-analyze-video/spec.md`

## Summary

Wire a local video file through the existing CV stack so it produces the same
`AnalysisResult` the HF-replay path already produces. The work is mostly
plumbing: implement `Tracker.update` over `supervision.ByteTrack`, add a
viewer-aware side-inference rule for player-perspective video, replace the
`NotImplementedError` in `analyze_video`, and expose it as a `crpod
analyze-video` CLI subcommand. Three user-story slices ship in priority
order: P1 = `summary.json` end-to-end; P2 = visualization parity; P3 = EV
model scoring on video-derived interactions.

## Technical Context

**Language/Version**: Python 3.11 (pinned by `pyproject.toml` and `flake.nix`).
**Primary Dependencies**: `ultralytics` (YOLOv8), `supervision` (ByteTrack),
`opencv-python` (frame decode + crop), `pytesseract` (HUD OCR), `numpy` —
all already in the project lockfile.
**Storage**: Local filesystem only. Inputs are a video path and a weights
path; outputs land under `output/analysis/<video-stem>/` (gitignored).
**Testing**: `pytest` for pure-logic pieces (side inference, track →
CardPlay reduction, HUD region scaling). Stages requiring GPU + real weights
are exercised via a manual smoke run documented in `quickstart.md`, not unit
tests, in line with constitution Principle IV.
**Target Platform**: Linux/macOS dev shells (Nix flake or Homebrew). GPU is
strongly recommended for SC-001 (60 s on a 3-minute video); CPU works but
will not hit that latency target.
**Project Type**: CLI tool — single Python project layered into the existing
`src/crpod/` package.
**Performance Goals**: SC-001 = 3-minute video → summary in under 60 s on a
single mid-range GPU. SC-006 = fail-fast within 2 s on input errors (no GPU
load, no decoding). FR-015 = visible progress reporting on long runs.
**Constraints**: Must reuse `analyze_replay` end-to-end (FR-009, FR-014).
Must NOT reuse the HF dataset's river-Y side-inference rule (FR-007). Must
validate inputs before any GPU work (FR-002, SC-006). All inter-stage
communication must continue to flow through `src/crpod/types.py`
(constitution Principle II).
**Scale/Scope**: One video per invocation. ~3-minute matches at 10 fps =
~1,800 processed frames. No batching, no live streaming, no remote URLs in
v1.

No `NEEDS CLARIFICATION` markers — the spec resolved every borderline
question with informed defaults documented in its Assumptions section.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Evaluating against `.specify/memory/constitution.md` v1.0.0:

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Reproducible Environments | PASS | Uses only dependencies already pinned in `pyproject.toml` / `uv.lock` and system tools already in `flake.nix`. No new system deps; no ad-hoc `pip install`. |
| II. Pipeline Modularity | PASS | New code lives in the existing `dataset → detection → tracking → ocr → features` subpackages. All inter-stage data flows through `src/crpod/types.py` (`CardPlay`, `HudState`, `Replay`, `Interaction`). No cross-stage shortcuts introduced. |
| III. CLI as Public Interface | PASS | Functionality reaches users only through the new `crpod analyze-video` subcommand; pre-validation of inputs happens before any expensive work (FR-002, SC-006). |
| IV. Pragmatic Testing & Quality Gates | PASS | Pure-logic helpers (side inference, track-reduction, HUD region scaling) get pytest coverage. GPU-bound stages get a documented smoke run, not unit tests. The four CI gates (ruff check, ruff format, mypy src, pytest) remain green. |
| V. MVP-First Scope Discipline | PASS | Feature is explicitly the Option-B end-to-end pipeline from `pod_summary.md`. Real-time mode, blunder detection, and dashboards are out of scope (the spec says so). User stories are independently shippable, with US1 alone delivering the MVP. |

**Verdict**: All gates pass. No entries in Complexity Tracking.

## Project Structure

### Documentation (this feature)

```text
specs/001-analyze-video/
├── plan.md                  # This file
├── spec.md                  # Feature specification
├── research.md              # Phase 0: implementation research notes
├── data-model.md            # Phase 1: entity contracts (re-used from src/crpod/types.py)
├── quickstart.md            # Phase 1: smoke-run recipe
├── contracts/
│   └── cli.md               # Phase 1: CLI contract for `crpod analyze-video`
├── checklists/
│   └── requirements.md      # Spec quality checklist (already passes)
└── tasks.md                 # /speckit-tasks output (NOT created by this command)
```

### Source Code (repository root)

This feature is a single-project addition to the existing `src/crpod/`
layout. No new top-level directories.

```text
src/crpod/
├── __main__.py              # MODIFY: add `analyze-video` subcommand
├── pipeline.py              # MODIFY: replace NotImplementedError in analyze_video;
│                            #         add Replay-from-streams assembly helper
├── tracking/
│   └── bytetrack.py         # MODIFY: implement Tracker.update over supervision.ByteTrack
├── dataset/
│   ├── video.py             # KEEP AS-IS: VideoFrameIterator already works
│   └── side.py              # NEW (small helper): viewer-aware side inference for video,
│                            #      kept separate from huggingface.py's river-Y rule
├── detection/
│   └── yolo.py              # KEEP AS-IS: already accepts the iterator's shape
├── ocr/
│   └── hud.py               # KEEP AS-IS for v1; resolution-scaling note in research.md
└── types.py                 # KEEP AS-IS: existing dataclasses are the contract

tests/
├── test_video_pipeline.py   # NEW: side inference, track→CardPlay reduction,
│                            #      Replay assembly, CLI input validation
└── fixtures/
    └── (existing HUD fixture; no new fixtures committed —
         smoke video lives outside the repo per .gitignore policy)
```

**Structure Decision**: Single-project layout, mutating the existing
`src/crpod/` subpackages. The only new file is `src/crpod/dataset/side.py`,
which deliberately splits the viewer-aware rule from the HF dataset's
`_infer_side` so the two side-inference policies don't get tangled and so
the video path can be tested without pulling in the HF loader.

## Complexity Tracking

> No constitution gate violations — section intentionally empty.
