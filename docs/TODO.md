# TODO

Remaining work to finish the Clash Royale Post-Game Analyzer, grouped by what unblocks what.

## Immediate (you, today)

- [ ] **Smoke test HF loader listing.** Run `uv run crpod list-replays` against `chrisrca/clash-royale-tv-replays` to confirm arena naming — the dataset schema claimed 3 arena values but the card described 31, verify the real layout before the team relies on it.
- [ ] **Smoke test analysis end-to-end (blocked on YOLO weights).** The HF dataset contains raw frame images (`frame_id`, `image`, `hash`), not pre-extracted card placements. Running `uv run crpod analyze arena_15 <replay_id> --weights <path>` requires trained YOLO weights to detect cards in each frame. Once weights land, inspect `summary.json`, `placements.png`, `tempo.png`.
- [ ] **Update `.gitignore`** for the new layout: `output/`, `.venv/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`, `result/` (nix build symlink). Check whether `.claude-flow/` and `.DS_Store` also need rules.
- [ ] **Commit flake + src scaffold and open PR.** Stage `flake.nix`, `.envrc`, `src/`, `tests/`, `pyproject.toml`, `Makefile`, `README.md`, `docs/TODO.md`. Do **not** stage `.claude/`, `.claude-flow/`, `.mcp.json`, `.DS_Store`, `.venv/`. Single commit: `feat: nix flake + crpod pipeline scaffold`.

## Data quality (blocked on YOLO weights for the HF path)

- [ ] **Validate side inference heuristic.** The dataset doesn't label friendly vs enemy — `_infer_side` in `src/crpod/dataset/huggingface.py` splits on `y > RIVER_Y` using detection center coordinates. Once YOLO weights land, inspect placement distributions across a few replays to confirm which side is the recorder. Fix if wrong.
- [ ] **Expand `CARD_COSTS` to all 159 cards.** `src/crpod/constants.py` covers ~40. Pull the full card → cost map from RoyaleAPI or the CR wiki. Unknown cards fall back to `default=3` which biases every EV calculation.
- [ ] **Replace `elixir_trade` proxy with a real EV target.** `crpod train` currently uses elixir trade as the label — that's a starting proxy, not true EV. Add damage approximation (frame-to-frame placement proximity) or wire in a tower HP signal. See `pod_summary.md` Option B.

## Sub-team deliverables (pod_summary weeks 2-6)

- [ ] **Data & Detection — collect YOLO training data.** 500+ frames annotated in Roboflow with troop/spell/structure bboxes in COCO format. Blocks the entire custom-replay path.
- [ ] **Data & Detection — train YOLOv8.** Target mAP@0.5 ≥ 0.70. Save weights to `output/models/yolo.pt`. Then implement `Tracker.update` in `src/crpod/tracking/bytetrack.py` wrapping `supervision.ByteTrack`.
- [ ] **Tracking & Feature Engineering — tune HUD OCR regions.** `HudRegions` in `src/crpod/ocr/hud.py` has placeholder pixel rects. Sample 10 frames from the HF dataset, measure the actual elixir/timer/tower HP regions on 540×960, replace the defaults. Add a test that asserts `pytesseract` recognizes the elixir digit for a saved fixture frame.

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
- [ ] **GitHub Actions CI.** `.github/workflows/ci.yml` running `ruff`, `mypy`, `pytest` on PRs via `cachix/install-nix-action` + `nix develop --command` so CI matches the local flake.

## Suggested order

`Smoke test listing → .gitignore → commit PR` gets the scaffold landed before anyone else pulls. **YOLO weights are now the critical-path blocker** — both the HF-dataset path and the custom-video path require them. Once the Data & Detection team delivers `yolo.pt`, the HF-dataset path (side inference → CARD_COSTS → EV target → blunder detection) unblocks. OCR tuning and `analyze_video` wiring are independent of HF analysis and can proceed in parallel once weights exist.
