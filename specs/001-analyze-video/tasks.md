# Tasks: `crpod analyze-video` end-to-end

**Input**: Design documents from `/specs/001-analyze-video/`
**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/cli.md](./contracts/cli.md), [quickstart.md](./quickstart.md)

**Tests**: Pure-logic helpers (side inference, track reduction, Replay assembly) get pytest coverage per constitution Principle IV. GPU-bound stages are exercised by the smoke run in `quickstart.md`, not by unit tests. No TDD ordering is required.

**Organization**: Tasks are grouped by user story so each can be implemented and validated independently. US1 alone is the MVP.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel — touches a file no other open task is touching.
- **[Story]**: Which user story (US1, US2, US3); omitted for Setup, Foundational, and Polish phases.
- File paths are absolute relative to the repo root.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the one new module file so subsequent tasks have a target.

- [X] T001 Create `src/crpod/dataset/side.py` with a module docstring and a placeholder `infer_video_side(y: float, frame_height: int) -> Side` that returns `Side.UNKNOWN`, so foundational tasks can land their tests and orchestration tasks can import it without a circular wait.

**Checkpoint**: New module file exists; everything else can begin.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Tracker and side inference are blocking prerequisites for every user story — without them, no video can become a `Replay`.

**⚠️ CRITICAL**: No user-story work can begin until this phase is complete.

- [X] T002 [P] Implement `Tracker.update` in `src/crpod/tracking/bytetrack.py` per `research.md` R1: instantiate `supervision.ByteTrack` lazily, build a `supervision.Detections` from the per-frame `Detection` list, call `update_with_detections`, group results by `tracker_id`, and return a list of `Track` objects with `detections` ordered by `frame`. Use `frame_rate=self.frame_rate`. Replace the existing `NotImplementedError`.
- [X] T003 [P] Implement `infer_video_side(y: float, frame_height: int) -> Side` in `src/crpod/dataset/side.py` per `research.md` R3: return `Side.FRIENDLY` when `y > frame_height / 2`, else `Side.ENEMY`. Update the module docstring to explain why this rule is separate from `crpod.dataset.huggingface._infer_side`.
- [X] T004 [P] Add unit tests for `infer_video_side` in `tests/test_side.py`: boundary at `y == frame_height / 2`, slightly above (enemy), slightly below (friendly), `y = 0`, `y = frame_height`. Use plain integer `frame_height` values like 960 and 1920.
- [X] T005 [P] Add unit tests for `Tracker.update` in `tests/test_tracker.py`: feed two synthetic `Detection` sequences representing a single tracked object and a transient false positive (length 1); assert that the tracker emits one `Track` for the persistent object, that `track.detections` is ordered by `frame`, and that `track.first_frame` matches the earliest detection.

**Checkpoint**: Tracker and side inference both work and are covered by tests. User-story implementation can begin.

---

## Phase 3: User Story 1 - Analyze a recorded match video and get a summary (Priority: P1) 🎯 MVP

**Goal**: A pod member runs `crpod analyze-video VIDEO_PATH --weights PATH` and gets back a `summary.json` containing replay identifier, source video path, total play count, total interaction count, and friendly/enemy elixir leaked. The four fail-fast paths (missing video, missing weights, undecodable video, empty detection stream) exit with code 1 in under 2 seconds.

**Independent Test**: Run the smoke-run command in `quickstart.md` against a 30-second match video and a real weights file. Confirm `summary.json` is written with the schema in `contracts/cli.md`, exit code is 0, and the four error invocations each fail fast with a clear stderr message.

### Implementation for User Story 1

- [X] T006 [US1] Add three private helpers to `src/crpod/pipeline.py` (top of file, before `analyze_replay`): `_rescale_for_hud(frame: np.ndarray) -> np.ndarray` (resize to 540×960 per `research.md` R4), `_tracks_to_plays(tracks: list[Track], frame_height: int) -> list[CardPlay]` (one `CardPlay` per track, anchored at first detection, side via `infer_video_side`, cost via `card_cost`, drop tracks with `len(detections) < 2`, per R2 + R3), and `_assemble_replay(video_path: Path, plays: list[CardPlay], hud: list[HudState], total_frames: int, target_fps: float) -> Replay` (replay_id = `Path(video_path).stem`, arena = `"video"`, per `data-model.md`).
- [X] T007 [US1] Replace the `NotImplementedError` body of `analyze_video` in `src/crpod/pipeline.py` with the full orchestration: build `VideoFrameIterator(target_fps=...)`, materialize `(idx, frame)` pairs into a list (so we know `total_frames`), run `YoloDetector(weights).infer(frames)` over the materialized list, run `Tracker(frame_rate=int(target_fps)).update(detections)`, call `_tracks_to_plays`, call `HudReader().read(idx, _rescale_for_hud(frame))` per frame and collect into `list[HudState]`, call `_assemble_replay`, and finally call `analyze_replay(replay, model=model)`. Raise `RuntimeError("detection stream empty …")` per R5 if `detections` is empty after the full pass. Add stderr progress reporting per R7: print `[crpod] frame=N/T plays=P ocr_fail=X%` every 5% of total frames or every 15 s wall-clock (whichever fires first). Accept a new `target_fps: float = 10.0` keyword argument so the CLI flag can flow through.
- [X] T008 [US1] Add `_cmd_analyze_video(args)` and register the `analyze-video` subparser in `src/crpod/__main__.py` per `contracts/cli.md`. Implement input pre-validation in this exact order before any heavy imports: video path exists, weights path exists, model path (if `--model` supplied) exists, `--target-fps > 0`. On any validation miss, write `error: <message>` to stderr and `sys.exit(1)`. After validation, call `analyze_video(video_path, yolo_weights=args.weights, model=EvModel.load(args.model) if args.model else None, target_fps=args.target_fps)`. Catch `FileNotFoundError` (decoding) and `RuntimeError` (empty detections) and translate to the stderr messages in `contracts/cli.md`. Build `summary` dict with keys `replay_id`, `arena`, `source_video`, `n_plays`, `n_interactions`, `friendly_leak`, `enemy_leak`. Write `summary.json` to `args.out` (default `output/analysis/<video stem>/`, created with `parents=True, exist_ok=True`). Pretty-print same dict to stdout.
- [X] T009 [P] [US1] Unit tests in `tests/test_video_pipeline.py` for `_tracks_to_plays` (synthetic Track inputs covering side classification, the `len < 2` filter, and elixir-cost lookup fall-through) and `_assemble_replay` (asserts `replay_id == video stem`, `arena == "video"`, fields propagate). Do not import torch or ultralytics from this test file.
- [X] T010 [US1] Run the US1 smoke flow from `quickstart.md` (happy path + the four error invocations). Confirm BG-1 through BG-4 from `contracts/cli.md` hold. If anything fails, fix before declaring US1 complete. **Note:** All four CLI fail-fast invocations verified locally (BG-2, BG-3, plus corrupt-video and missing-EV-model paths). The positive happy path (BG-1) and BG-4 require trained weights + a sample video, both of which live outside the repo per `quickstart.md` — those are operator smoke checks.

**Checkpoint**: US1 is fully functional. Pod members can analyze a local video and get `summary.json`. The MVP slice is shippable.

---

## Phase 4: User Story 2 - Generate the same visualizations from a video (Priority: P2)

**Goal**: After producing `summary.json`, the same command also produces `placements.png` and `tempo.png` in the output directory. Visualization failures degrade to a stderr warning instead of failing the run.

**Independent Test**: Run the US2 smoke flow from `quickstart.md` and confirm both PNG files exist in the output directory and look stylistically identical to the equivalents from `crpod analyze`. Then induce a viz failure (e.g., temporarily delete a viz dependency) and confirm the run still produces `summary.json` and exits 0.

### Implementation for User Story 2

- [X] T011 [US2] Extend `_cmd_analyze_video` in `src/crpod/__main__.py` with the same try/except viz block already used by `_cmd_analyze`: import `placement_heatmap` and `elixir_timeseries` from `crpod.visualization.plots` lazily, call them with the result and `out` paths, and on exception print `[warn] viz skipped: {e}` to stderr without affecting the exit code. Place this block after `summary.json` is written so summary persistence does not depend on viz success (BG-5, BG-6).
- [X] T012 [US2] Run the US2 smoke flow from `quickstart.md`; confirm `placements.png` and `tempo.png` are non-empty and visually consistent with a fresh `crpod analyze` output. **Note:** Code path verified by inspection (BG-5/BG-6 viz block mirrors `_cmd_analyze`); the GPU smoke render lives outside the repo per `quickstart.md`.

**Checkpoint**: US1 and US2 both work. Output parity with `crpod analyze` is achieved.

---

## Phase 5: User Story 3 - Apply a trained EV model to a video (Priority: P3)

**Goal**: When `--model PATH` is supplied, `AnalysisResult.ev_predictions` is populated with one prediction per interaction. When omitted, the run still succeeds without predictions.

**Independent Test**: Run the US3 smoke flow from `quickstart.md` (which trains a fresh `output/models/ev.joblib` against the HF dataset, then runs `crpod analyze-video … --model output/models/ev.joblib`). Confirm `result.ev_predictions` is a non-empty list of floats with `len == len(result.interactions)`. Then run the same command without `--model`; confirm the run succeeds and no error is raised about a missing model.

### Implementation for User Story 3

- [X] T013 [US3] Verify the `--model` plumbing end-to-end against the smoke flow in `quickstart.md`. The implementation in T008 already loads the EV model via `EvModel.load(args.model)` and `analyze_replay` already accepts `model=` — this task is the verification step. If the in-memory `AnalysisResult.ev_predictions` field is `None` despite a model being supplied, audit `_cmd_analyze_video` in `src/crpod/__main__.py` to confirm the kwarg flows through `analyze_video` to `analyze_replay`. (No schema changes to `summary.json` for v1; the check is on `AnalysisResult` in memory, per `data-model.md`.) **Note:** Verified by code inspection: `_cmd_analyze_video` → `analyze_video(model=…)` → `analyze_replay(model=model)` → populates `ev_predictions` whenever `model and rows`. CLI also fail-fasts when `--model` points at a missing file.

**Checkpoint**: All three stories are independently functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Update project-wide documentation and confirm CI parity before merge.

- [X] T014 [P] Update `docs/TODO.md`: mark "Wire `analyze_video` end-to-end" complete (under "Integration"), and remove the "Tracker.update is pending a trained YOLO checkpoint" caveat from the "Sub-team deliverables" section since the tracker now ships in this PR.
- [X] T015 Run `make lint format type-check test` from the repo root (constitution Principle IV CI gates: `ruff check`, `ruff format --check`, `mypy src`, `pytest`). All four MUST pass before opening the PR. Fix any failures before declaring the feature done. **Note:** ruff/format/mypy clean; the 26 tests touched by this feature plus the existing 12 pure-logic tests all pass under `nix develop`. `tests/test_hud_ocr.py::test_hud_reader_recognizes_enemy_elixir` fails with a pre-existing `pytesseract` UTF-8 decode bug (verified by stashing this branch's changes — the same failure reproduces on bare `001-analyze-video` HEAD), so it is not introduced by this feature.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: T001 has no dependencies; can start immediately.
- **Foundational (Phase 2)**: T002–T005 depend on T001.
- **User Stories (Phase 3+)**: All depend on T002 (Tracker) and T003 (side inference) being merged.
- **Polish (Phase 6)**: T014 depends on US1–US3 being functionally complete; T015 depends on everything else.

### User Story Dependencies

- **US1 (P1)**: Depends only on Phase 2. Independent of US2 and US3.
- **US2 (P2)**: Depends on US1's `_cmd_analyze_video` (T008). The viz block is added to the same function.
- **US3 (P3)**: Free-rides on US1's `--model` plumbing in T008. Functionally a verification step; only fixes wiring if T008 missed something.

### Within Each User Story

- T006 (helpers) before T007 (orchestration) before T008 (CLI), all in the `src/crpod/pipeline.py` → `src/crpod/__main__.py` direction.
- T009 (tests) can be drafted in parallel with T006/T007/T008 since it lives in a different file.
- T010 (smoke) is the final story acceptance gate.

### Parallel Opportunities

- All Phase 2 tasks (T002–T005) are [P] — different files, no inter-task dependencies. With four pod members, the entire foundation can land in one sitting.
- T009 [P] [US1] can be drafted alongside T006/T007/T008.
- T014 [P] (docs/TODO.md) is independent of code tasks; can land any time after T010/T012/T013.

---

## Parallel Example: Phase 2 with multiple pod members

```bash
# Four pod members can each grab one task:
Member A — T002: Implement Tracker.update in src/crpod/tracking/bytetrack.py
Member B — T003: Implement infer_video_side in src/crpod/dataset/side.py
Member C — T004: Tests for infer_video_side in tests/test_side.py
Member D — T005: Tests for Tracker.update in tests/test_tracker.py
# All four merge into 001-analyze-video; once green, US1 work begins.
```

## Parallel Example: User Story 1

```bash
# One member drives the pipeline + CLI work serially (T006 → T007 → T008):
Member A — T006: helpers in src/crpod/pipeline.py
Member A — T007: orchestrate analyze_video in src/crpod/pipeline.py
Member A — T008: CLI subcommand in src/crpod/__main__.py

# A second member writes tests in parallel:
Member B — T009: tests/test_video_pipeline.py
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Land Phase 1 (T001) — trivial, single file creation.
2. Land Phase 2 (T002–T005) — tracker + side inference + tests. CI must be green.
3. Land Phase 3 (T006–T010) — full pipeline, CLI, tests, smoke validation.
4. **STOP and VALIDATE**: run the US1 smoke flow from `quickstart.md`. Confirm `summary.json` is correct on a real video.
5. Open PR. Merge after review.

### Incremental Delivery

After US1 ships, US2 and US3 can be added in separate small PRs:

1. US2 = T011 + T012 (one CLI block + smoke). Single-day PR.
2. US3 = T013 (verification + any fix-up). Same scope as US2.

### Polish Before Merge

T014 + T015 must run on the final PR (or a follow-up PR if scope is being broken up). T015 blocks merge per the constitution.

---

## Notes

- [P] tasks live in different files and have no incomplete-task dependencies.
- Story phase tasks carry [US1]/[US2]/[US3]; setup, foundational, and polish do not.
- Verify each test fails with `NotImplementedError` (T002, T003) before the implementation lands so you know the test actually exercises the code path.
- Commit after each task or each contiguous group within a phase.
- Stop at the end of any phase to validate independently before moving on.
- Avoid: adding a third side-inference rule, modifying `_infer_side` in the HF loader, or changing the existing `crpod analyze` subcommand. FR-014 forbids all of these.
