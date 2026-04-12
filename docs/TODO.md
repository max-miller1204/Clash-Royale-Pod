# TODO

Remaining work to finish the Clash Royale Post-Game Analyzer, grouped by what unblocks what.

## Immediate (you, today)

- [x] **Smoke test HF loader listing.** Verified against `chrisrca/clash-royale-tv-replays`: 31 arenas (`arena_01`–`arena_31`, zero-padded lowercase). The project card was right, the schema description was wrong. `arena_15` has 76 replays; replay IDs are UUIDs.
- [x] **Smoke test analysis end-to-end (blocked on YOLO weights).** Pipeline plumbing verified healthy on `arena_15/00a91415-...`: parquet download, frame decode, ultralytics import, and `YOLO(weights)` construction all execute. The run terminates with `FileNotFoundError` at `torch.load` for the weights path, exactly as expected — no other code path issues. NixOS users need `libxcb`/`libGL`/`libstdc++` via `nix-ld` (see README). `_cmd_analyze` and `_cmd_train` now pre-validate `--weights` existence before any HF download.
- [x] **Update `.gitignore`** for the new layout: `output/`, `.venv/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`, `result/` (nix build symlink). Check whether `.claude-flow/` and `.DS_Store` also need rules.
- [x] **Commit flake + src scaffold and open PR.** Stage `flake.nix`, `.envrc`, `src/`, `tests/`, `pyproject.toml`, `Makefile`, `README.md`, `docs/TODO.md`. Do **not** stage `.claude/`, `.claude-flow/`, `.mcp.json`, `.DS_Store`, `.venv/`. Single commit: `feat: nix flake + crpod pipeline scaffold`.

## Data quality (blocked on YOLO weights for the HF path)

- [ ] **Validate side inference heuristic.** The dataset doesn't label friendly vs enemy — `_infer_side` in `src/crpod/dataset/huggingface.py` splits on `y > RIVER_Y` using detection center coordinates. Once YOLO weights land, inspect placement distributions across a few replays to confirm which side is the recorder. Fix if wrong.
- [ ] **Expand `CARD_COSTS` to all 159 cards.** `src/crpod/constants.py` covers ~40. Pull the full card → cost map from RoyaleAPI or the CR wiki. Unknown cards fall back to `default=3` which biases every EV calculation.
- [ ] **Replace `elixir_trade` proxy with a real EV target.** `crpod train` currently uses elixir trade as the label — that's a starting proxy, not true EV. Add damage approximation (frame-to-frame placement proximity) or wire in a tower HP signal. See `pod_summary.md` Option B.

## Sub-team deliverables (pod_summary weeks 2-6)

- [ ] **Data & Detection — collect YOLO training data.** 500+ frames annotated in Roboflow with troop/spell/structure bboxes in COCO format. Blocks the entire custom-replay path.
- [ ] **Data & Detection — train YOLOv8.** Target mAP@0.5 ≥ 0.70. Save weights to `output/models/yolo.pt`. Then implement `Tracker.update` in `src/crpod/tracking/bytetrack.py` wrapping `supervision.ByteTrack`.
- [x] **Tracking & Feature Engineering — tune HUD OCR regions.** Measured against `arena_15/00a91415-…` frame 251 at 540×960. `HudRegions` now carries empirical rects for enemy/friendly elixir, match timer, and all four princess-tower HP labels (king HPs remain rough guesses — they only render when damaged). `HudReader._read_number` upscales 6× before OCR since the digits are ~20px tall at native res. Fixture `tests/fixtures/hud/sample_540x960.jpg` + `tests/test_hud_ocr.py` assert `pytesseract` reads the enemy elixir digit as `3`.

## Integration (blocked on YOLO + OCR)

- [x] **Wire HF replay path through YOLO.** The HF dataset contains raw frame images, not pre-extracted placements. `_parquet_to_replay` now decodes frames from parquet and runs `YoloDetector` to extract `CardPlay` events. `analyze` and `train` CLI subcommands require `--weights`. Blocked on trained weights landing.
- [ ] **Wire `analyze_video` end-to-end.** In `src/crpod/pipeline.py`: `VideoFrameIterator → YoloDetector → Tracker → HudReader → CardPlay/HudState stream → analyze_replay`. Add a `crpod analyze-video PATH` CLI subcommand.

## Real-time mode (scope expansion past the 10-week timeline)

The offline architecture already supports this — the CV stages run ~30-50ms/frame on a mid-range GPU, well under the 100ms budget. What's missing is the streaming glue. Blocked on YOLO weights (#12) and HUD OCR tuning (#13).

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

`Smoke test listing → .gitignore → commit PR` gets the scaffold landed before anyone else pulls. **YOLO weights are now the critical-path blocker** — both the HF-dataset path and the custom-video path require them. Once the Data & Detection team delivers `yolo.pt`, the HF-dataset path (side inference → CARD_COSTS → EV target → blunder detection) unblocks. OCR tuning and `analyze_video` wiring are independent of HF analysis and can proceed in parallel once weights exist.
