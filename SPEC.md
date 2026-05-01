# SPEC: Finish Clash Royale Post-Game Analyzer

## Context

The pod has shipped the offline analysis pipeline end-to-end (PRs #19–#23). `crpod analyze-video VIDEO --weights PATH` now produces `summary.json` plus visualizations. Underlying ML is trained: YOLOv8s detector (mAP@0.5 = 0.885), HUD OCR for elixir/timer, ByteTrack for tracking, LightGBM for EV prediction.

What's missing for production-grade pod use:
1. The EV model is trained on `interaction.elixir_trade` (`src/crpod/modeling/ev.py:28`, `src/crpod/__main__.py:145`) — a starting proxy, not real EV. Predictions don't translate to actionable insight.
2. No blunder detection — the headline pod-facing deliverable.
3. The HF-dataset side-inference heuristic in `src/crpod/dataset/huggingface.py` is unverified. If wrong, the EV model is trained on mislabeled data.
4. Seven champion costs (`mirror`, `goblin_brawler`, `royal_guardian`, `boss_bandit`, `rune_giant`, `spirit_empress`, `terry`) sit in `_KNOWN_UNCONFIRMED_COSTS`; matches with these cards have wrong elixir math.
5. Production-grade requires a latency audit, smoke across all 76 arena_15 replays, polished output the pod can consume, and doc cleanup.

**Goal:** a tool the pod runs on its own match videos and gets back useful blunder callouts.

## Scope

**In scope**
- Validation + fix of HF-dataset side inference (`_infer_side`)
- Confirmation of 7 unconfirmed champion costs into `CARD_COSTS`
- YOLO detection-quality audit on the pod's own match videos (not just HF)
- Replace `elixir_trade` EV target with tower-HP-delta from HUD OCR
- Re-train `EvModel` on the new target with held-out validation
- Statistical blunder detection: plays >1σ below per-card median EV
- One-page self-contained HTML report bundling summary + blunders + viz
- Latency budget audit (p50/p95/p99 per pipeline stage)
- Smoke run across all 76 arena_15 replays + edge-case fixes
- README + quickstart + `docs/TODO.md` polish

**Out of scope**
- Real-time mode (`crpod live`, streaming pipeline, screen capture)
- Overlay renderer (OBS browser source, always-on-top window)
- Cross-arena evaluation beyond arena_15 (the EV model trains on this distribution; cross-arena is research follow-up)
- Re-training the YOLO detector or adding new classes
- New CLI subcommands beyond what's already shipped (`analyze`, `analyze-video`, `train`)

## Design

### Pipeline after this spec ships

```
crpod analyze-video VIDEO --weights MODEL --model EV
  ↓
VideoFrameIterator → YoloDetector → Tracker → HudReader (now exposes tower HPs)
  ↓
CardPlay + HudState stream
  ↓
analyze_replay → Interactions w/ tower_hp_delta features
  ↓
EvModel.predict   (trained on tower-HP-delta target; carries per-card median table)
  ↓
detect_blunders   (statistical, per-card median ± σ from training fold)
  ↓
output/analysis/<id>/
    summary.json           (existing schema)
    blunders.json          (NEW: list of Blunder records)
    placements.png         (existing)
    tempo.png              (existing)
    report.html            (NEW: self-contained one-page report)
```

### Key design decisions
- **EV target = tower HP delta over the interaction window.** Direct measurement of the thing that wins games. HUD princess-tower regions are already tuned in `HudRegions`. Training rows where any required HUD reading is `None` are dropped (drop rate logged at train time).
- **Blunder rule = per-card median EV − σ, threshold 1σ.** Card-aware (a -2 elixir trade is normal for Knight, awful for Mega Knight). Median + std are computed over the training fold and persisted inside the `EvModel` artifact, so inference doesn't need access to training data. Cards with fewer than 5 training samples are excluded from blunder calls.
- **Output is files-only, plus an HTML report.** No CLI behavior change (no stdout flood); the HTML is openable offline in any browser.
- **HF-dataset side inference is the EV model's lifeblood.** If wave 1A finds the heuristic wrong, fix it before wave 2 retrains the model. This is the only cross-wave dependency that isn't strictly serial-by-merge.

## Verification

End-to-end smoke that proves we're done:

1. `crpod train --weights output/models/crpod_v1_best.pt --out output/models/ev.joblib` succeeds and prints holdout MAE + Spearman correlation against the tower-HP-delta target.
2. `crpod analyze-video <pod-replay>.mp4 --weights ... --model output/models/ev.joblib` produces:
   - `summary.json` (existing schema)
   - `blunders.json` — list of `{play_idx, card, ev_predicted, per_card_median, sigma_below}` for plays >1σ below
   - `placements.png`, `tempo.png` (existing)
   - `report.html` — single-page bundle, opens in Chrome/Safari without console errors
3. `docs/latency-budget.md` shows per-stage p50/p95/p99 on a 30-second video.
4. `scripts/smoke_arena15.sh` runs all 76 arena_15 replays end-to-end; failure rate < 5%; remaining failures explained in `docs/known-issues.md`.
5. `make lint format type-check test` passes.
6. README quickstart is copy-paste runnable; `docs/TODO.md` reflects every completed item.

## Waves

### Wave 1 — Data Quality (parallel)

**Scaffold:** none — three independent leaves.

**Interface contracts locked:** none new this wave.

| Branch | Scope | Done-when |
|---|---|---|
| `swarm/finish-project-wave-1a-side-inference` | Run `scripts/validate_side_inference.py` against ≥10 arena_15 replays. Eyeball placement distributions. If `_infer_side` in `src/crpod/dataset/huggingface.py` is wrong, fix it. Commit findings to `docs/data-quality.md` either way. | `docs/data-quality.md` documents observed friendly/enemy split per inspected replay; if the heuristic was wrong, a passing test in `tests/test_hf_replay.py` covers the corrected boundary. |
| `swarm/finish-project-wave-1b-champion-costs` | Source the 7 unconfirmed champion costs from Supercell's published spell data. Add to `CARD_COSTS` in `src/crpod/constants.py`. Remove from `_KNOWN_UNCONFIRMED_COSTS`. | All 7 cards present in `CARD_COSTS`; `_KNOWN_UNCONFIRMED_COSTS` is empty; `tests/test_cards.py` covers the new entries. |
| `swarm/finish-project-wave-1c-yolo-pod-audit` | Run `crpod_v1_best.pt` over 3–5 of the pod's own match videos. Annotate detection quality per pod-deck card. Write `docs/yolo-pod-audit.md` with findings. No code changes unless a hard bug surfaces. | `docs/yolo-pod-audit.md` lists detection quality per pod-deck card; if any card shows critical failure, a follow-up issue is opened (not fixed here). |

**Sequencing:** all three leaves are independent; merge order is arbitrary. Wave 2 cannot start until **1A** merges (EV retraining depends on side-inference correctness); 1B and 1C may merge in parallel with wave 2.

### Wave 2 — Tower-HP EV Target (mostly serial)

**Scaffold (commit before dispatch):**
- Extend `HudState` in `src/crpod/types.py` with `friendly_king_hp`, `friendly_left_princess_hp`, `friendly_right_princess_hp`, `enemy_king_hp`, `enemy_left_princess_hp`, `enemy_right_princess_hp` (all `int | None`).
- Extend `HudReader.read` in `src/crpod/ocr/hud.py` to populate the new fields using existing princess-tower regions in `HudRegions`. King-HP fields stay `None` for now (rough rects per `docs/TODO.md:37`).
- Add `src/crpod/features/ev_target.py` (new) with `def tower_hp_delta(window: list[HudState]) -> dict[str, int | None]` — returns per-side, per-tower HP swing across the window; `None` if either bookend is unreadable.
- Extend `Interaction` in `src/crpod/types.py` with `tower_hp_delta: dict[str, int | None]` field, populated in `src/crpod/features/interactions.py`.

**Interface contracts locked in scaffold commit:** `tower_hp_delta` return shape and `Interaction.tower_hp_delta` field type are final — both wave-2 leaves consume them.

| Branch | Scope | Done-when |
|---|---|---|
| `swarm/finish-project-wave-2a-ev-label` | In `src/crpod/__main__.py:_cmd_train` (the `targets.append(...)` site at :145), replace `interaction.elixir_trade` with `(friendly_princess_hp_delta − enemy_princess_hp_delta)` summed across both princess towers. King HP excluded. Drop training rows where any required HUD reading was `None`; log the drop rate. | `crpod train` produces a model; training script logs sample count and drop rate; `tests/test_ev_target.py` covers the label math on synthetic `HudState` sequences. |
| `swarm/finish-project-wave-2b-retrain-validate` | Re-train `EvModel` on the new target. Hold out 20% of arena_15 replays as a validation split. Report holdout MAE + Spearman correlation in `docs/ev-validation.md`. Compute per-card `(median, std)` from the training fold; persist inside the `EvModel` artifact (new `EvModel.per_card_stats: dict[str, tuple[float, float]]` attribute) for wave 3 to consume. | New `output/models/ev.joblib` exists; `docs/ev-validation.md` reports holdout metrics; `EvModel.load(...).per_card_stats` returns the persisted dict. |

**Sequencing:** 2A must merge before 2B (2B trains on 2A's labels). Both depend on the wave-2 scaffold commit.

### Wave 3 — Blunder Detection (scaffold + parallel)

**Scaffold (commit before dispatch):**
- Add `Blunder` dataclass in `src/crpod/types.py`: `play_idx: int`, `card: str`, `ev_predicted: float`, `per_card_median: float`, `sigma_below: float`.
- Create `src/crpod/analysis/blunders.py` (new module + `__init__.py`) with `def detect_blunders(plays: list[CardPlay], ev_predictions: list[float], per_card_stats: dict[str, tuple[float, float]]) -> list[Blunder]` skeleton (raises `NotImplementedError`). The signature is the contract.
- Extend `AnalysisResult` in `src/crpod/pipeline.py` with `blunders: list[Blunder]` field (default empty list).

**Interface contracts locked in scaffold commit:** `detect_blunders` signature + `Blunder` dataclass shape are final.

| Branch | Scope | Done-when |
|---|---|---|
| `swarm/finish-project-wave-3a-statistical-detector` | Implement `detect_blunders` per spec: for each play with an EV prediction, look up `(median, std)` for that card; flag if `(median − ev) / std > 1.0`. Skip cards with fewer than 5 training samples (already excluded from `per_card_stats` by wave 2B). Return list ordered worst-first. | Function returns ordered `list[Blunder]`; pure-logic; no I/O. |
| `swarm/finish-project-wave-3b-cli-wiring` | In `src/crpod/__main__.py`, after `analyze_replay` returns, call `detect_blunders` (when EV model loaded), populate `result.blunders`, write `blunders.json` next to `summary.json`. Apply to both `_cmd_analyze` and `_cmd_analyze_video`. | Both subcommands write `blunders.json` when `--model` supplied; absent when `--model` omitted. |
| `swarm/finish-project-wave-3c-tests` | `tests/test_blunders.py`: synthetic `CardPlay` + EV inputs with known z-scores. Cover: zero-blunder match, multi-blunder match, card with <5 samples (excluded), card not in median table (excluded). | All four cases covered; runs without GPU/torch imports. |

**Sequencing:** 3A and 3C may run in parallel after the scaffold; 3B depends on 3A's implementation merging. Order: (3A ‖ 3C) → 3B.

### Wave 4 — Production Polish (parallel)

**Scaffold (commit before dispatch):**
- Add `src/crpod/visualization/report.py` (new) with `def render_report(result: AnalysisResult, out_dir: Path) -> None` raising `NotImplementedError`.
- Add `src/crpod/instrumentation/__init__.py` and `src/crpod/instrumentation/timing.py` (new) with a `Timer` context manager exposing `Timer.record(stage: str)`.

**Interface contracts locked in scaffold commit:** `render_report` signature and `Timer.record` API.

| Branch | Scope | Done-when |
|---|---|---|
| `swarm/finish-project-wave-4a-html-report` | Implement `render_report`: single-page HTML with summary fields, top-N blunders table, embedded `placements.png` + `tempo.png` (relative paths or base64). No external CSS/JS deps; opens offline. Wire into both `_cmd_analyze` and `_cmd_analyze_video` after `summary.json` is written. | `report.html` lands in output dir for any successful analyze run; renders in Chrome/Safari without console errors. |
| `swarm/finish-project-wave-4b-latency-audit` | Wrap each pipeline stage (decode, YOLO, tracker, HUD, EV, blunders, report) with `Timer.record`. Run on 3 representative videos. Write `docs/latency-budget.md` with per-stage p50/p95/p99 + total wall-clock; flag any stage over the budget stated in `pod_summary.md`. | `docs/latency-budget.md` exists with per-stage table and identifies any over-budget stage. |
| `swarm/finish-project-wave-4c-smoke-arena15` | Add `scripts/smoke_arena15.sh` that runs `crpod analyze` over all 76 arena_15 replays. Capture failures. Fix until failure rate < 5%. Document remaining failures in `docs/known-issues.md`. | Script exits 0 with failure rate < 5%; remaining failures explained in `docs/known-issues.md`. |
| `swarm/finish-project-wave-4d-docs` | Update `README.md` quickstart for the new outputs (`blunders.json`, `report.html`). Update `docs/TODO.md` to reflect everything in waves 1–3 complete. | README quickstart is copy-paste runnable; TODO has no stale `[x]`/`[ ]` mismatches. |

**Sequencing:** all four leaves are independent. 4D should land last because it touches docs that other leaves' findings might revise.

## Execution

Delivery mode: fork-pr
PR unit: wave
Base strategy: upstream-trunk
Branch naming: swarm/{slug}-wave-{n}
Fork remote: origin
Upstream remote: origin

_Wave 1 executed 2026-05-01 on branch swarm/finish-project-wave-1; chunks 1a-side-inference, 1b-champion-costs, 1c-yolo-pod-audit; PR https://github.com/max-miller1204/Clash-Royale-Pod/pull/24_

_Wave 2 executed 2026-05-01 on branch swarm/finish-project-wave-2; chunks 2a-ev-label, 2b-retrain-validate; PR https://github.com/max-miller1204/Clash-Royale-Pod/pull/26_

_Wave 2C executed 2026-05-01 on branch swarm/finish-project-wave-2c-hud-on-hf; chunks 2c-hud-on-hf (structural fix only — `Replay.hud` now populated from HF parquet); PR https://github.com/max-miller1204/Clash-Royale-Pod/pull/27 (stacked on #26). Real `docs/ev-validation.md` metrics deferred to wave 2D (brev retrain run)._

_Wave 2D executed 2026-05-01 on branch swarm/finish-project-wave-2d-brev-metrics; chunks 2d-brev-metrics (smoke run on H100; 47/47 rows dropped because tesseract 4.1.1 cannot read CR's stylised in-game HP digits — 0/583 frames had all four princess HPs readable). Diagnosis pinned in `docs/ev-validation.md`. Real metrics still pending — needs wave 2E to replace the OCR digit reader (HP-bar pixel sampling recommended). PR pending._
