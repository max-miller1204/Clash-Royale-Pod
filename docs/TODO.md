# TODO

Remaining work to finish the Clash Royale Post-Game Analyzer, grouped by what unblocks what.

## Immediate (you, today)

- [x] **Smoke test HF loader listing.** Verified against `chrisrca/clash-royale-tv-replays`: 31 arenas (`arena_01`–`arena_31`, zero-padded lowercase). The project card was right, the schema description was wrong. `arena_15` has 76 replays; replay IDs are UUIDs.
- [x] **Smoke test analysis end-to-end (blocked on YOLO weights).** Pipeline plumbing verified healthy on `arena_15/00a91415-...`: parquet download, frame decode, ultralytics import, and `YOLO(weights)` construction all execute. The run terminates with `FileNotFoundError` at `torch.load` for the weights path, exactly as expected — no other code path issues. NixOS users need `libxcb`/`libGL`/`libstdc++` via `nix-ld` (see README). Minor follow-up: pre-validate `--weights` existence in `_cmd_analyze` before downloading 469 MB from HF.
- [x] **Update `.gitignore`** for the new layout: `output/`, `.venv/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`, `result/` (nix build symlink). Check whether `.claude-flow/` and `.DS_Store` also need rules.
- [x] **Commit flake + src scaffold and open PR.** Stage `flake.nix`, `.envrc`, `src/`, `tests/`, `pyproject.toml`, `Makefile`, `README.md`, `docs/TODO.md`. Do **not** stage `.claude/`, `.claude-flow/`, `.mcp.json`, `.DS_Store`, `.venv/`. Single commit: `feat: nix flake + crpod pipeline scaffold`.

## Option A spike — KataCR pretrained weights (CLOSED 2026-04-13)

**Verdict: unusable for unit detection, but the spike redirected us to a much better option (see next section).**

**What was tried:** downloaded KataCR's `detector1_v0.7.13.pt` and `detector2_v0.7.13.pt` (Google Drive links in `/home/max/KataCR/README_en.md:63`, ~632 MB each). Ran `scripts/detect_overlay.py` against one `arena_15` replay with `PYTHONPATH=/home/max/KataCR` (the checkpoints are pickled against KataCR's custom `katacr.yolov8.custom_model` classes and fail to deserialize without it).

**What worked:** static screen elements — all 4 princess towers, both king towers, elixir counters, health bars above units. This is unsurprising because these classes sit at fixed relative coordinates every frame.

**What failed:**
- **In-field units not localized.** Cannon, valkyrie, goblins on the board — no bboxes at all, not even wrong ones. Detector's `bar` class picks up the health bars but the unit bodies are not boxed. So there's no localization signal to bootstrap annotation from.
- **New card art breaks it.** Pancake tower cosmetic skin boxed as a troop; king-tower decorations from new skins boxed as "goblins"; "dagger-duchess-tower-bar" false-positive on board decorations. Expected 2-year staleness but the extent confirms transfer is broken on dynamic elements.
- **KataCR modified ultralytics internals.** Custom model classes in the checkpoint require their pinned `ultralytics==8.1.24`, risking conflicts with anything newer. Even if we hacked compatibility, we would inherit their fork as a maintenance burden.

**Why we're NOT salvaging this as a HUD-only detector:** towers and elixir UI sit at fixed pixel coordinates. The existing `HudRegions` pixel-rect approach in `src/crpod/ocr/hud.py` handles them with ~10 lines of array slicing. Adding a dual-detector runtime + `PYTHONPATH=/home/max/KataCR` hack + 938 MB of weights to replace fixed-coordinate cropping is architectural overkill. The only reason to run YOLO on the HUD is if positions were *not* fixed — they are.

**Cleanup — delete exactly these to retire the spike:**
- `scripts/detect_overlay.py`
- `output/models/detector1_v0.7.13.pt`, `output/models/detector2_v0.7.13.pt` (~938 MB combined)
- `output/overlays/` (spike render PNGs)

No `src/crpod/` code was touched; no rollback work in the library. Keep this section as the postmortem until the spike is cleaned up.

## Critical path — train fresh YOLO on KataCR's public dataset (Option D)

**Status:** unblocked. This is the new critical path. Supersedes the earlier "collect 500 frames + annotate in Roboflow" plan (Option C), which is no longer needed for v1.

**The find:** KataCR's training DATA is public and separate from the broken weights. Cloned to `Clash-Royale-Detection-Dataset/` at the worktree root. Contents:

- **6,939 labeled real-video frames** under `images/part2/` across 28 video sessions (`OYASSU_*`, `WTY_*`, `lan77_*`, 2021–2024). Manifest at `images/part2/annotation.txt`.
- **1,816-frame mini subset** at `images/part2/train_annotation_mini.txt` for quick training iterations.
- **566×896 resolution**, very close to the HF dataset's 540×960 (same mobile portrait ~2:3 aspect). Ultralytics resizes to 640 during training, so the gap is negligible.
- **~150 classes** with per-class friendly/enemy counts tracked in `version_info/annotation_v0.14.csv`. Strong coverage for common cards (`queen-tower` 13k/13k, `skeleton` 2.7k/780, `hog-rider` 1.2k/60, `musketeer` 2.5k/31) and thin tails for rare ones (some at 0–30 examples — those will be weak in the trained model but that's fine, rare cards matter less for EV signal).
- **154 sliced sprite directories** under `images/segment/` for synthetic data generation via `generator.py` — optional augmentation for rare classes later.
- **No custom ultralytics classes in the label format** — annotations are plain YOLO `.txt` files, trainable on modern ultralytics directly. No `PYTHONPATH` hack.

**Label format caveat:** KataCR's `.txt` files have 12 values per line instead of YOLOv8's 5: `class cx cy w h belonging 0 0 0 0 0 0`. The extra columns encode which side the unit belongs to (friendly/enemy) plus 6 reserved zeros. Standard ultralytics training ignores anything past column 5, but some versions warn or error — safest to pre-process into a copy with only the first 5 columns. One small script.

**Staleness:** latest session is `WTY_20240406`, so ~1 year of new cards/skins are missing (berserker, newer tower skins, newer evolutions). For v1 this is acceptable — the HF dataset is old enough that most frames pre-date the gap anyway. For v2, supplement with targeted HF annotation on whatever new cards the trained detector whiffs on.

**Concrete steps — order matters:**

1. **Gitignore the dataset clone.** Add `Clash-Royale-Detection-Dataset/` to `.gitignore` before anything gets staged. It's ~700 MB.
2. **Label-format converter.** Write `scripts/convert_katacr_labels.py` that walks `Clash-Royale-Detection-Dataset/images/part2/**/*.txt` and writes a parallel tree under `output/katacr_yolo/labels/` with only the first 5 columns. Symlink or copy JPGs alongside.
3. **Build a `data.yaml` for modern ultralytics.** Class indices need to match KataCR's `label_list.py`. Start from `/home/max/KataCR/katacr/yolov8/ClashRoyale_detection.yaml` (already in YOLOv8 format, 150 classes) — copy it, point `path`/`train`/`val` at the converted layout, drop the `pad_*` classes which are placeholder reserved slots.
4. **Train a small model as a smoke test.** `yolo train data=output/katacr_yolo/data.yaml model=yolov8n.pt epochs=10 imgsz=640` on the mini subset (1,816 frames). Target: any reasonable mAP to prove the pipeline end-to-end, not final quality. Budget: one evening on local GPU, or rent an A100 hour.
5. **Test the resulting weights against HF frames** using `scripts/detect_overlay.py` on the same `arena_15` replay as the Option A spike. Compare visually: do units now get boxes? If yes → full training run. If no → the resolution/distribution gap is real and needs domain adaptation.
6. **Full training run** on all 6,939 frames once the smoke test passes. Save to `output/models/crpod_yolov8n.pt`. This is the weights file the pipeline ships with.
7. **Optional: domain-adapt on 50–100 HF frames** if step 5 showed a gap. Annotate in Roboflow, fine-tune for a few epochs on top of the KataCR-trained weights. Two orders of magnitude less annotation work than the original 500-frame plan.
8. **Wire into the HF pipeline** — this is already done. `YoloDetector` reads class names from the checkpoint, so whatever taxonomy you train with just appears. The only plumbing left is the name-mapping for `CARD_COSTS` lookups (`spear-goblin` → `spear_goblins`, `the-log` → `log`) and a decision on how to handle the 100+ classes KataCR labels that aren't in your `CARD_COSTS` table.

**What the taxonomy work from earlier becomes:** obsolete for v1. The 40-class plan (30 cards + 3 unknowns + 2 towers + 3 HUD) was designed for the 500-frame budget. With 150 classes and 6,939 frames available for free, just use KataCR's taxonomy as-is. The `CARD_COSTS` expansion task in the data-quality section is still worth doing — expand it to cover KataCR's full card list, not your original 30, so downstream EV calculations get real costs for every class the new detector produces.

**Cleanup if Option D also fails** (e.g., training diverges or weights don't transfer to HF frames):
- `Clash-Royale-Detection-Dataset/` (cloned repo, ~700 MB)
- `output/katacr_yolo/` (converted labels)
- `output/models/crpod_yolov8n.pt` (trained weights)
- `scripts/convert_katacr_labels.py`
- This section

At that point the fallback is the original Option C (500 Roboflow frames from scratch), with the earlier 40-class taxonomy work still valid.

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
