# TODO

Remaining work to finish the Clash Royale Post-Game Analyzer, grouped by what unblocks what.

## Immediate (you, today)

- [x] **Smoke test HF loader listing.** Verified against `chrisrca/clash-royale-tv-replays`: 31 arenas (`arena_01`–`arena_31`, zero-padded lowercase). The project card was right, the schema description was wrong. `arena_15` has 76 replays; replay IDs are UUIDs.
- [x] **Smoke test analysis end-to-end (blocked on YOLO weights).** Pipeline plumbing verified healthy on `arena_15/00a91415-...`: parquet download, frame decode, ultralytics import, and `YOLO(weights)` construction all execute. The run terminates with `FileNotFoundError` at `torch.load` for the weights path, exactly as expected — no other code path issues. NixOS users need `libxcb`/`libGL`/`libstdc++` via `nix-ld` (see README). `_cmd_analyze` and `_cmd_train` now pre-validate `--weights` existence before any HF download.
- [x] **Update `.gitignore`** for the new layout: `output/`, `.venv/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`, `result/` (nix build symlink). Check whether `.claude-flow/` and `.DS_Store` also need rules.
- [x] **Commit flake + src scaffold and open PR.** Stage `flake.nix`, `.envrc`, `src/`, `tests/`, `pyproject.toml`, `Makefile`, `README.md`, `docs/TODO.md`. Do **not** stage `.claude/`, `.claude-flow/`, `.mcp.json`, `.DS_Store`, `.venv/`. Single commit: `feat: nix flake + crpod pipeline scaffold`.

## YOLO training — completed 2026-04-13

**Option A spike (KataCR pretrained weights) — FAILED.** KataCR's `detector1/2_v0.7.13.pt` are pickled against custom `katacr.yolov8.custom_model` classes (require `PYTHONPATH=/home/max/KataCR` + pinned `ultralytics==8.1.24`). Static elements (towers, HUD) transferred but in-field units (cannon, valkyrie, goblins) had zero detections. 2-year staleness + distribution shift = unusable for unit detection.

**Option D (train fresh on KataCR's public dataset) — SUCCEEDED.** KataCR published their annotated training data separately from their broken weights at [Clash-Royale-Detection-Dataset](https://github.com/wty-yy/Clash-Royale-Detection-Dataset): 6,966 frames, 117,294 boxes, ~150 classes, plain YOLO `.txt` format — trainable on stock ultralytics with no hacks. Trained YOLOv8s for 50 epochs on a brev.dev A4000 ($0.20 total).

**Results:** mAP50 = 0.885, mAP50-95 = 0.667 (target was ≥ 0.70). Towers, buildings, in-field troops, health bars all detected on HF dataset frames. Known weakness: card-hand thumbnails at screen edges misclassified as `emote` — irrelevant since pipeline reads placements, not hand contents.

**Artifacts:**
- Trained weights: `output/models/crpod_v1_best.pt` (22 MB, YOLOv8s, ~150 KataCR classes)
- Label converter: `scripts/convert_katacr_labels.py` (12-col → 5-col)
- Overlay inspector: `scripts/detect_overlay.py`
- Brev training script: `scripts/brev_train.sh` (end-to-end clone→convert→train, CUDA 12.8 torch fix baked in)
- Re-trainable anytime for ~$0.20 via `brev create --type hyperstack_A4000`

## Data quality (NOW UNBLOCKED by YOLO weights)

- [x] **Validate side inference heuristic.** Validated visually against arena_05/15/22/28/31 mid-match frames via `scripts/validate_side_inference.py`. Found: HF frames are 540×960, but `constants.py` claimed 480×810, putting `RIVER_Y` at 405 — well above the actual river (frame midpoint = 480). The 75-pixel gap mislabeled enemy-side near-river deploys as friendly. Patched constants to match real frame dimensions (`ARENA_W=540`, `ARENA_H=960`, `RIVER_Y=480`); BRIDGE_LEFT_X/RIGHT_X kept at original values pending a separate lane-zone audit. Re-run of the validation script confirms `RIVER_Y` now lands on the visible bridge across all five arenas.
- [x] **Expand `CARD_COSTS` to all 159 cards.** `src/crpod/constants.py` now covers 121 base cards — all standard troops, buildings, spells, and the current champion roster — sourced from the `cr-csv` mirror of Supercell's `spells_*.csv` with post-2023 rebalances and 2024–2026 additions applied by hand. Evolution variants share the base cost and aren't listed separately. Wave 1b folded in the late additions (`boss_bandit`, `rune_giant`, `spirit_empress`, `terry`, and `mirror`) directly into `CARD_COSTS` and reclassified `goblin_brawler` / `royal_guardian` as spawned subunits in `KATACR_NON_CARD`, so `_KNOWN_UNCONFIRMED_COSTS` is now an empty `frozenset` (kept as a named constant for the validity test in `tests/test_cards.py`). Name mapping landed in `src/crpod/detection/cards.py` — `KATACR_TO_CARD` (125 entries) + `KATACR_NON_CARD` (76 entries) cover all 201 KataCR classes. Snapshot at `src/crpod/detection/katacr_classes.txt` was derived from upstream KataCR's `ClashRoyale_detection.yaml`; if the trained `.pt` ships a different `model.names` list, run `scripts/dump_katacr_classes.py` on a machine with the weights and the coverage tests will surface any mismatch.
- [x] **Replace `elixir_trade` proxy with a real EV target.** Wave 2A landed the princess-tower HP-delta target (`(friendly_left + friendly_right) − (enemy_left + enemy_right)`). Subsequent waves (2B–2J') retrained, expanded to arena_23+ top-ladder cohort, swapped tesseract for HSV-mask HP-bar pixel sampling, added pre-window HP-delta + top-card cross-product + time-pressure mode features. Holdout Spearman ρ ended at **+0.223** on N=4,030 (Δρ vs starting baseline ≈ +0.26). See `docs/ev-validation.md` for the full per-wave trajectory.

## Sub-team deliverables (pod_summary weeks 2-6)

- [x] **Data & Detection — collect YOLO training data.** Superseded by KataCR's public dataset (6,966 frames, 117,294 boxes). No Roboflow annotation needed.
- [x] **Data & Detection — train YOLOv8.** mAP@0.5 = 0.885 (target was ≥ 0.70). Weights at `output/models/crpod_v1_best.pt`.
- [x] **Tracking & Feature Engineering — tune HUD OCR regions.** Measured against `arena_15/00a91415-…` frame 251 at 540×960. `HudRegions` now carries empirical rects for enemy/friendly elixir, match timer, and all four princess-tower HP labels. King-HP rects (which only render when the king is damaged) were re-measured in issue #55 against `arena_15/226fefa9-…` frame 1058; fixture `tests/fixtures/hud/sample_king_damaged_540x960.jpg` + `tests/test_hud_ocr.py::test_king_rects_capture_hp_label_when_damaged` pin the new defaults. `HudReader._read_number` upscales 6× before OCR since the digits are ~20px tall at native res. Fixture `tests/fixtures/hud/sample_540x960.jpg` + `tests/test_hud_ocr.py` assert `pytesseract` reads the enemy elixir digit as `3`.

## Integration (blocked on YOLO + OCR)

- [x] **Wire HF replay path through YOLO.** The HF dataset contains raw frame images, not pre-extracted placements. `_parquet_to_replay` now decodes frames from parquet and runs `YoloDetector` to extract `CardPlay` events. `analyze` and `train` CLI subcommands require `--weights`. Blocked on trained weights landing.
- [x] **Wire `analyze_video` end-to-end.** In `src/crpod/pipeline.py`: `VideoFrameIterator → YoloDetector → Tracker → HudReader → CardPlay/HudState stream → analyze_replay`. The `crpod analyze-video PATH` CLI subcommand pre-validates inputs (video, weights, optional EV model, `--target-fps`) before any heavy import, writes `summary.json` with `replay_id`/`arena`/`source_video`/`n_plays`/`n_interactions`/`friendly_leak`/`enemy_leak`, and degrades viz failures to a stderr warning.

## Real-time mode (scope expansion past the 10-week timeline)

The offline architecture already supports this — the CV stages run ~30-50ms/frame on a mid-range GPU, well under the 100ms budget. What's missing is the streaming glue. YOLO weights are now available (`crpod_v1_best.pt`); still blocked on HUD OCR tuning and ByteTrack integration.

- [ ] **Streaming replay builder.** Add `src/crpod/pipeline/stream.py` with a `StreamingReplay` class that consumes `CardPlay`/`HudState` events as they arrive and yields completed `Interaction`s when the 40-frame window closes. Reuse `running_tempo` and `ElixirLedger` incrementally — neither needs the full play list upfront.
- [ ] **Screen capture source.** Extend `src/crpod/dataset/video.py` so `VideoFrameIterator` accepts an integer device index (`cv2.VideoCapture(0)`) or an scrcpy/mss source in addition to file paths. For a phone game, use scrcpy USB mirroring (KataCR's approach) or an Android emulator on the same machine. macOS needs a capture card or emulator — V4L2 is Linux-only.
- [ ] **Live side labeling.** The HF-dataset `_infer_side` y-split won't work live — you're viewing your own POV, both sides visible. Key off UI elements instead: your deck sits at the bottom of the frame, opponent's at the top. Detect card thumbnails in the hand strip to disambiguate who played what.
- [ ] **Rolling EV inference.** Wire the trained `EvModel` into the stream loop — on each completed interaction, run `predict` and emit a live signal (stdout, WebSocket, or overlay). LightGBM single-row inference is ~1ms, well under budget.
- [ ] **`crpod live` CLI subcommand.** Wraps the stream pipeline: `crpod live --source 0 --model output/models/ev.joblib`. Prints rolling tempo, per-interaction EV, and blunder flags in real time.
- [ ] **Latency budget audit.** Profile the full pipeline on target hardware. KataCR hit ~120ms/decision + 240ms feature fusion on an RTX 4060 — be skeptical of "<100ms" claims until measured. Record p50/p95/p99 per stage.
- [ ] **Overlay renderer (stretch).** OBS browser source or an always-on-top window showing live tempo graph + EV readouts. Only after the core pipeline is solid.

**Caveat:** `pod_summary.md` scopes the project as offline post-game analysis. Real-time is a genuine expansion — viable with this codebase as the foundation, but it pushes the deliverable past the 10-week timeline unless some offline features get dropped. Recommend shipping offline first (weeks 1-6), training YOLO in parallel (weeks 2-5), then building `crpod live` in weeks 7-9.

## Stretch goals

- [x] **Blunder detection.** Wave 3 shipped `crpod.analysis.blunders.detect_blunders` — flags plays whose predicted EV is more than 1σ below the per-card training-fold median. CLI wires it into both `analyze` and `analyze-video` and writes `output/analysis/<id>/blunders.json` when `--model` is supplied. Coverage in `tests/test_blunders.py` (zero / multi / under-5-samples / unknown-card / zero-std edge cases, threshold strictness, length mismatch).
- [x] **Self-contained HTML report.** Wave 4A shipped `crpod.visualization.report.render_report` — emits `report.html` next to `summary.json` with summary fields, top-5 blunders table, and base64-embedded placement / tempo PNGs. Opens offline. Coverage in `tests/test_report.py`.
- [x] **Latency-budget audit.** Wave 4B added `crpod.instrumentation.Timer` + `scripts/latency_audit.py` + `scripts/latency_report.py`. Baseline measurements landed in `docs/latency-budget.md` (CPU floor across 3× 30 s clips, YOLO dominates).
- [x] **arena_15 smoke harness.** Wave 4C added `scripts/smoke_arena15.sh` that walks all 76 replays end-to-end. 8/8 sample run on CPU passes; full-pool run is brev-friendly. Remaining edge cases tracked in `docs/known-issues.md`.
- [x] **GitHub Actions CI.** `.github/workflows/ci.yml` runs `ruff check`, `ruff format --check`, `mypy src`, and `pytest` on PRs to `main` via `cachix/install-nix-action` + `nix develop --command`, so CI matches the local flake. All four checks pass locally on this branch.
- [ ] **Wave 2K — model class A/B + hyperparameter sweep.** Branch `swarm/wave-2k-model-class-ab` is scaffolded with the CV-sweep script. Awaiting a brev run (cost ≈ $10 / 14 h on A6000+). The next signal-quality lift if `+0.223` ρ isn't enough.

## Suggested order

**YOLO weights are no longer the blocker** — `crpod_v1_best.pt` is trained and validated. The new critical path is wiring the weights into the analysis pipeline:

1. **Name mapping + CARD_COSTS expansion** — must happen before any analysis produces meaningful output. KataCR class names (`the-log`, `spear-goblin`) need to map to `CARD_COSTS` keys. Expand costs to cover ~150 classes.
2. **Validate side inference** — run the detector on a few replays, check `_infer_side` y-split.
3. **ByteTrack tracker** — implement `Tracker.update` so detections become tracked objects across frames.
4. **HUD OCR tuning** — independent of the above, can proceed in parallel.
5. **Wire `analyze_video` end-to-end** — blocked on tracker + OCR.
6. **EV target replacement** — once the analysis pipeline produces real data, replace the elixir-trade proxy.
7. **Blunder detection + CI** — stretch goals, after core pipeline ships.
