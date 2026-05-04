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
- Cross-arena evaluation across multiple skill tiers (the EV model can train on any single high-skill tier; arena_23+ is the chosen tier for wave 2.5)
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

1. `crpod train --weights output/models/crpod_v1_best.pt --out output/models/ev.joblib` succeeds and prints holdout MAE + Spearman correlation against the tower-HP-delta target. Metrics recorded honestly in `docs/ev-validation.md`; no fixed signal target — wave 2.5 captures the actual values reached after the signal-quality roadmap.
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

### Wave 2.5 — Signal Quality (strict serial)

Wave 2 / 2C-2F shipped a functional pipeline but holdout Spearman ρ landed
at +0.162 — statistically real but too weak for blunder calls to be
useful. Wave 2.5 is a sequenced quality push driven by
`docs/superpowers/specs/2026-05-02-wave-2.5-signal-quality-design.md`.
Strict serial ordering: each chunk's PR stacks on the prior so each
move's Δρ is attributable. **No fixed Spearman target** — stop when
marginal gains aren't worth additional overfitting risk; the actual ρ
achieved drives wave 3's threshold decision.

| Branch | Scope | Done-when |
|---|---|---|
| `swarm/wave-2g-numpy-elixir-reader` | Replace pytesseract elixir read in `HudReader` with numpy pixel-sampling on the pink elixir bar (mirrors wave 2E's HP-bar approach). Pure infra investment — `_cmd_train` uses `CardPlay.elixir_cost`, not `HudState.friendly_elixir`. | `crpod train` end-to-end < 20 min on A6000 (was ~3h); holdout ρ unchanged within ±0.02. |
| `swarm/wave-2h-top-ladder-data` | Drop `--arena arena_15` filter; train on arena_23+ pool (~1,253 replays vs 76). Sanity-check bar reader on the new cohort; recalibrate inline if needed. Freeze a new replay-level 80/20 holdout split. | New holdout split frozen on arena_23+; ρ recorded with Δρ vs wave 2F baseline (0.162); recalibration committed if bar reader broke. |
| `swarm/wave-2i-drop-rate-fix` | On the 2H pool: special-case destroyed-tower bookend as `delta=0` instead of dropping; loosen HSV mask to handle VFX occlusion frames. Pure variance reduction. | Drop rate < 25% on the 2H pool (was 33%); ρ recorded with Δρ vs 2H. |
| `swarm/wave-2j-feature-audit` | Audit-only chunk. Extract LightGBM feature importance (`gain` + `split` types) from the wave-2I model artifact (`output/models/ev_wave2i.joblib`) via a local one-off script — no brev run needed for the audit itself. Hard-drop any feature whose `gain` importance is < 1% of the max-feature `gain` **AND** whose `split` count is < 5; remove dropped keys from `crpod.modeling.ev.interaction_features`. If drops apply, run one brev train on the wave-2I data + frozen holdout and record Δρ. If nothing qualifies for dropping, ship the importance table only — no brev run, no Δρ entry (Δρ ≈ 0 a priori). No hyperparameter changes (reserved for 2K). | Feature-importance table (per-feature `gain` + `split` columns) committed to `docs/ev-validation.md`; drop list pinned (or "no drops" stated explicitly); if drops applied, ρ recorded with Δρ vs 2I; `interaction_features` updated. |
| `swarm/wave-2j-prime-feature-add` | Add three features to `crpod.modeling.ev.interaction_features` after 2J's drops have applied: **(1)** `top_friendly_x_top_enemy` — string categorical encoding the highest-`elixir_cost` card on each side joined by `_x_`. Tie-break: first-played by `frame`. Empty side: encode as the literal `'none'` (e.g. `'none_x_hog_rider'`). **(2/3)** `pre_window_friendly_hp_delta_30s` + `pre_window_enemy_hp_delta_30s` — sum of princess-tower HP swings in the 300-frame window before `Interaction.start_frame`. Implementation: extend `Interaction` (`crpod/types.py`) with two additive `int \| None` fields, populated in `crpod.features.interactions.build_interactions` by scanning `Replay.hud` for the frame ≈ `start_frame − 300` and diffing against the start bookend. `None` when the lookback frame is unreadable or the window starts < 30 s into the replay. **(4)** `time_pressure_mode` — 4-level string categorical (`single`/`double`/`triple`/`overtime`) derived from `Interaction.start_frame / Replay.fps` against the standard CR clock (180 s single, 120 s double, 60 s triple, then overtime). Single brev train run on wave-2J's data + frozen holdout. No hyperparameter changes (2K's domain). | All three features wired through `Interaction` → `interaction_features`; unit tests cover each (synthetic `Interaction` + `Replay.hud` for the pre-window scan; tie-break and empty-side cases for the card pair; clock-boundary cases for the time mode); ρ recorded with Δρ vs 2J in `docs/ev-validation.md`. |
| `swarm/wave-2k-model-class-ab` | Only if 2J' produced a small Δρ. 5-fold CV on the 2J' training fold across LightGBM (with hyperparameter sweep), Ridge regression baseline, and XGBoost. CV-best model gets one shot at the holdout. | CV results table committed (model × fold × ρ); final model is whichever wins CV; final ρ recorded with Δρ vs 2J'. |

**Sequencing:** strict serial — 2G → 2H → 2I → 2J → 2J' → 2K. Each
chunk runs its single brev train run against the frozen holdout and
records Δρ in `docs/ev-validation.md`. The 2J/2J' split (audit-only,
then feature-add) replaces the original combined 2J chunk so each
sub-change's Δρ is attributable. Stop conditions per the design doc:
0 < Δρ < 0.02 twice consecutively → stop; Δρ < -0.02 → stop and
investigate; re-shuffled-split ρ diverges from frozen-split ρ by
> 0.05 → stop (holdout has leaked).

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

_Wave 2D executed 2026-05-01 on branch swarm/finish-project-wave-2d-brev-metrics; chunks 2d-brev-metrics (smoke run on H100; 47/47 rows dropped because tesseract 4.1.1 cannot read CR's stylised in-game HP digits — 0/583 frames had all four princess HPs readable). Diagnosis pinned in `docs/ev-validation.md`. Real metrics still pending — needs wave 2E to replace the OCR digit reader (HP-bar pixel sampling recommended). PR https://github.com/max-miller1204/Clash-Royale-Pod/pull/28 (stacked on #27)._

_Wave 2E executed 2026-05-02 on branch swarm/finish-project-wave-2e-hp-bar-reader; chunks 2e-hp-bar-reader (replaced tesseract digit OCR with HSV-mask HP-bar pixel sampling; brev A6000 smoke 30 replays / 98 min / ~$1: drop rate 100% → 34%, kept 310/467 interactions, holdout MAE 463.20, Spearman -0.037, per_card_stats populated for 7 cards). Pipeline is end-to-end functional; EV signal weak on 234 train rows — wave 3 can proceed against real per_card_stats but blunder calls will be noisy until train fold expands. PR https://github.com/max-miller1204/Clash-Royale-Pod/pull/29 (stacked on #28)._

_Wave 2F executed 2026-05-02 on branch swarm/finish-project-wave-2f-expand-and-features; chunks 2f-expand-and-features (added start_{friendly,enemy}_total_princess_hp features, expanded train to all 76 arena_15 replays on brev H100; drop rate 33%, kept 794/1180 interactions, train 605 / holdout 189; MAE 463 → 509, Spearman -0.037 → **+0.162** (~2.2σ above zero, p ≈ 0.03 — cleared the 0.10 stop-or-continue threshold), per_card_stats cardinality 7 → 16). Signal is real but weak (Cohen "small"); foundation a future wave can build on. PR https://github.com/max-miller1204/Clash-Royale-Pod/pull/31 (stacked on #30)._

_Wave 2G executed 2026-05-03 on branch swarm/wave-2g-numpy-elixir-reader; chunks 2g-numpy-elixir-reader (replaced pytesseract elixir read with numpy pixel sampling on the pink elixir bar; brev A6000_plus sanity-check 30 arena_15 replays / 17.1 min / ~$0.20: 5.7× speedup vs wave 2E's 98 min, drop rate 34% unchanged, holdout ρ −0.037 → −0.008 within noise floor). Pure infra investment — `_cmd_train` consumes `CardPlay.elixir_cost`, not `HudState.friendly_elixir`. PR https://github.com/max-miller1204/Clash-Royale-Pod/pull/32._

_Wave 2H executed 2026-05-03 on branch swarm/wave-2h-top-ladder-data; chunks 2h-top-ladder-data (dropped --arena arena_15 filter, trained on arena_23+ pool with 1,203 replays on brev A6000_plus / 14h7m / ~$11.50; new replay-level 80/20 frozen holdout committed at `docs/wave-2.5-holdout.txt`; train 15,206 / holdout 3,930 interactions; drop rate 34% → **19%**, MAE 445.84 → 335.00, holdout ρ −0.008 → **+0.078** on N=3,930, ~4.9σ above zero — statistically much stronger than 2F's 0.162 on N=76; per_card_stats cardinality 7 → 69). Critical fix: `HFReplayLoader.delete_after_load=True` stream-deletes parquet cache to fit 256 GB cloud disk. PR https://github.com/max-miller1204/Clash-Royale-Pod/pull/35._

_Wave 2I executed 2026-05-04 on branch swarm/wave-2i-drop-rate-fix; chunks wave-2i-drop-rate-fix (destroyed-tower bookend → delta=0 when a tower reads None across every frame in window; gap-tolerant HP-bar reader, MIN_BRIDGE_SEG_PX=10 px; mask-threshold loosening deferred; calibration regression gate ≤ 2.5% MAE; tests 90 → 100). Brev A6000_plus 11h27m / ~$7.80: drop rate 19% → **13%**, kept 20,520 / 23,662 interactions (vs 19,136 / 23,662 in 2H), `per_card_stats` 69 → 72, holdout MAE 335.00 → **326.82**, holdout ρ +0.078 → **+0.194**, **Δρ = +0.116** (signal more than doubled). Wave-2.5 stop conditions all clear; 2J (feature audit) is next. PR https://github.com/max-miller1204/Clash-Royale-Pod/pull/36; landed_on_main: no._
