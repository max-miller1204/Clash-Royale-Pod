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

- [ ] **Validate side inference heuristic.** The dataset doesn't label friendly vs enemy — `_infer_side` in `src/crpod/dataset/huggingface.py` splits on `y > RIVER_Y` using detection center coordinates. Now that weights are available, inspect placement distributions across a few replays to confirm which side is the recorder. Fix if wrong.
- [x] **Expand `CARD_COSTS` to all 159 cards.** `src/crpod/constants.py` now covers 116 base cards — all standard troops, buildings, spells, and 8 champions — sourced from the `cr-csv` mirror of Supercell's `spells_*.csv` with post-2023 rebalances and 2024–2026 additions applied by hand. Evolution variants share the base cost and aren't listed separately. A handful of very recent champions (boss_bandit, rune_giant, spirit_empress, terry) are still omitted — they'll fall back to `default=3` until their costs are confirmed. Still TODO: name-mapping layer for KataCR class names (e.g. `the-log`, `spear-goblin`) → underscore convention (`log`, `spear_goblins`) so `card_cost()` lookups actually hit.
- [ ] **Replace `elixir_trade` proxy with a real EV target.** `crpod train` currently uses elixir trade as the label — that's a starting proxy, not true EV. Add damage approximation (frame-to-frame placement proximity) or wire in a tower HP signal. See `pod_summary.md` Option B.

## Sub-team deliverables (pod_summary weeks 2-6)

- [x] **Data & Detection — collect YOLO training data.** Superseded by KataCR's public dataset (6,966 frames, 117,294 boxes). No Roboflow annotation needed.
- [x] **Data & Detection — train YOLOv8.** mAP@0.5 = 0.885 (target was ≥ 0.70). Weights at `output/models/crpod_v1_best.pt`. Still TODO: implement `Tracker.update` in `src/crpod/tracking/bytetrack.py` wrapping `supervision.ByteTrack`.
- [x] **Tracking & Feature Engineering — tune HUD OCR regions.** Measured against `arena_15/00a91415-…` frame 251 at 540×960. `HudRegions` now carries empirical rects for enemy/friendly elixir, match timer, and all four princess-tower HP labels (king HPs remain rough guesses — they only render when damaged). `HudReader._read_number` upscales 6× before OCR since the digits are ~20px tall at native res. Fixture `tests/fixtures/hud/sample_540x960.jpg` + `tests/test_hud_ocr.py` assert `pytesseract` reads the enemy elixir digit as `3`.

## Integration (blocked on YOLO + OCR)

- [x] **Wire HF replay path through YOLO.** The HF dataset contains raw frame images, not pre-extracted placements. `_parquet_to_replay` now decodes frames from parquet and runs `YoloDetector` to extract `CardPlay` events. `analyze` and `train` CLI subcommands require `--weights`. Blocked on trained weights landing.
- [ ] **Wire `analyze_video` end-to-end.** In `src/crpod/pipeline.py`: `VideoFrameIterator → YoloDetector → Tracker → HudReader → CardPlay/HudState stream → analyze_replay`. Add a `crpod analyze-video PATH` CLI subcommand.

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

- [ ] **Blunder detection.** Top 3 worst plays per match. After the EV model is trained, compare predicted EV vs median EV for each card; plays > 1σ below are blunders. Emit `output/analysis/blunders.json`.
- [x] **GitHub Actions CI.** `.github/workflows/ci.yml` runs `ruff check`, `ruff format --check`, `mypy src`, and `pytest` on PRs to `main` via `cachix/install-nix-action` + `nix develop --command`, so CI matches the local flake. All four checks pass locally on this branch.

## Suggested order

**YOLO weights are no longer the blocker** — `crpod_v1_best.pt` is trained and validated. The new critical path is wiring the weights into the analysis pipeline:

1. **Name mapping + CARD_COSTS expansion** — must happen before any analysis produces meaningful output. KataCR class names (`the-log`, `spear-goblin`) need to map to `CARD_COSTS` keys. Expand costs to cover ~150 classes.
2. **Validate side inference** — run the detector on a few replays, check `_infer_side` y-split.
3. **ByteTrack tracker** — implement `Tracker.update` so detections become tracked objects across frames.
4. **HUD OCR tuning** — independent of the above, can proceed in parallel.
5. **Wire `analyze_video` end-to-end** — blocked on tracker + OCR.
6. **EV target replacement** — once the analysis pipeline produces real data, replace the elixir-trade proxy.
7. **Blunder detection + CI** — stretch goals, after core pipeline ships.
