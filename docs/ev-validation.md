# EV Model Validation — Wave 2B / 2E

## Target

`EvModel` is now trained against the **princess-tower HP delta** target
(`(friendly_left + friendly_right) - (enemy_left + enemy_right)` per
`_training_target` in `src/crpod/__main__.py`), replacing the prior
`elixir_trade` proxy. Sign convention: positive = our towers came out
ahead in the interaction window.

## Holdout protocol

- Split unit: **replay**, not row. All interactions from one replay land
  in the same fold so adjacent-frame leakage doesn't inflate the score.
- 80% train / 20% holdout; deterministic via `random.Random(0)` shuffle.
- `per_card_stats` (median, std of training-fold targets, anchored on
  `Interaction.friendly_plays[0].card`) is computed from the **training
  fold only**. Cards with fewer than five training-fold samples are
  excluded so wave 3's blunder rule reduces to a key check.
- Holdout metrics: MAE = `mean(|pred - true|)`; Spearman correlation via
  `scipy.stats.spearmanr`.

## How to reproduce

```bash
nix develop -c bash -c '
  uv run crpod train \
    --weights output/models/crpod_v1_best.pt \
    --out output/models/ev.joblib \
    --max-replays 50
'
```

`_cmd_train` prints, in order: drop-rate (HUD-unreadable rows, from
wave 2A), train sample count, holdout sample count, holdout MAE,
holdout Spearman, and the saved-model path. The artifact lands at
`output/models/ev.joblib`; `output/` is gitignored, so the binary
never enters version control.

## Run results

> **Status: end-to-end metrics produced.** Wave 2E replaced the
> tesseract digit OCR with HP-bar pixel sampling and unblocked
> training. Drop rate fell from 100% (wave 2D) to 34%, holdout MAE =
> 463.20 HP, holdout Spearman ρ = -0.037 (≈ 0). Earlier
> tesseract-can't-read-the-digits diagnosis archived in PR #28.

### Wave-2E run (smoke, arena_15, 30 replays)

| Field                        | Value                                                                |
| ---------------------------- | -------------------------------------------------------------------- |
| Branch                       | `swarm/finish-project-wave-2e-hp-bar-reader`                         |
| Brev instance                | `hyperstack_A6000` ($0.60/hr; A6000 48GB, 28 vCPU, 100GB disk)       |
| Driver / CUDA                | shadecloud A6000 / CUDA 12.6 wheels                                  |
| Python / torch               | CPython 3.11.15 venv / `torch==2.11.0+cu126`                         |
| HUD reader                   | HP-bar pixel sampling for the four princess-HP fields; tesseract 4.1.1 retained for elixir |
| Invocation                   | `crpod train --weights output/models/crpod_v1_best.pt --out output/models/ev.joblib --arena arena_15 --max-replays 30` |
| Wall-clock                   | ≈ 1 h 38 min (00:30:04Z → 02:08:55Z, 2026-05-02)                     |
| Replays processed            | 30 (arena_15)                                                        |
| Frames with HUD-OCR exception| 0 (`ocr_fail=0%` reported throughout)                                |
| Total interactions seen      | 467                                                                  |
| Dropped (unreadable HUD)     | **157 / 467 (34%)** — well under the 80% blocker threshold           |
| Kept sample count            | **310** (467 - 157)                                                  |
| Train / holdout split        | 234 interactions from 24 replays / 76 interactions from 6 replays    |
| Holdout MAE                  | **463.20** (HP units; princess full-HP scale ≈ 3052)                 |
| Holdout Spearman ρ           | **-0.037** (≈ 0 — model is barely better than constant prediction)   |
| `per_card_stats`             | 7 cards with ≥5 train-fold samples                                   |

The full `crpod train` log was captured at `/home/shadeform/smoke.log`
on the brev box and copied to `/tmp/wave2e-smoke.log` locally for
reference. Key terminal lines:

```text
dropped 157/467 training rows (34% — unreadable HUD)
training on 234 interactions from 24 replays (holdout 76 interactions from 6 replays)
per_card_stats: 7 cards with ≥5 train samples
holdout MAE: 463.20
holdout Spearman: -0.037
saved model → output/models/ev.joblib
```

`output/` is gitignored, so the `ev.joblib` artifact lives only on
the brev box (deleted on tear-down) and locally if the operator
copies it back.

### HP-bar reader: how it works

`crpod.ocr.hud.HudReader._read_hp_bar` replaces the tesseract digit
OCR for the four princess-HP fields (`friendly_left`,
`friendly_right`, `enemy_left`, `enemy_right`). Elixir keeps the
tesseract path — it works on the clean white-on-purple digits.

Algorithm:

1. Crop the bar rectangle from `HudRegions.{friendly,enemy}_{left,right}_hp_bar`.
   Friendly bars are at y ∈ [683, 690); enemy bars are mirrored to
   y ∈ [263, 270). x-range matches the digit-overlay rect (60-180 or
   360-480 depending on tower).
2. Apply a side-specific BGR mask: friendly bars match
   `b > 130 ∧ b - r > 25 ∧ b - g > 0` (bright cyan-blue); enemy bars
   match `r > 130 ∧ r - g > 25 ∧ r - b > 0` (bright pink-red). The
   masks reject the gold king-level crown badge and the dark
   unfilled portion of the bar.
3. Compute the longest horizontal run of masked pixels in the rect.
4. Convert run length to HP via per-side scales: 56.5 HP/px friendly,
   50.5 HP/px enemy. Calibrated from `tests/fixtures/hud/sample_540x960.jpg`
   (arena_15 / 00a91415-... frame 251), where the four ground-truth
   HPs (1446, 3052, 2423, 1810) recover within 2.5%.
5. Return `None` for an empty run (tower destroyed, mask noise) or a
   run wider than `MAX_PLAUSIBLE_BAR_PX = 75` (e.g., a frame-wide VFX
   flash). The training loop drops `None` rows.

Per-side scales differ by ~12% because the bright bar segment fades
earlier on the friendly badge than the enemy one. The constants are
hardcoded in `src/crpod/ocr/hud.py` rather than exposed as
constructor args; if a future arena uses different bar styling,
re-calibrate from a sampled frame and update both the rects and the
HP/px constants.

### Why the drop rate is 34%, not <5%

Three sources of drop, in rough order of impact:

1. **Frames where the digit overlay is occluded by VFX.** Princess
   towers periodically fire splash projectiles whose impact effects
   overlay the badge. The bar pixels under the splash get masked out
   transiently and the read returns either a small run (low HP) or
   `None` (run too short to register). `_training_target` requires
   *both* bookend frames of the interaction window to have all four
   towers readable, so a single occluded bookend drops the whole row.
2. **Tower-destroyed frames late in the game.** Once a princess tower
   is destroyed (HP = 0), the badge collapses to a stump and there
   is no bar to sample. Returning `None` is correct here, but a
   destroyed-tower bookend does drop the interaction row. A wave-3
   improvement would special-case "tower already destroyed at start
   of window" as `delta = 0` for that tower.
3. **Calibration error at very low HP.** The bar gets noisy when fill
   width drops below ~10 px (1 px ≈ 50 HP, so HP < 500 is in the
   noise floor). For now this just adds a few HP of measurement
   error per row, not a drop, but tower-destruction events near the
   end of the window can flip a "low HP" read into `None`.

34% is the empirical floor on this fixture set. Below 80% is the
chunk's "ship" threshold and we cleared it by 46 points.

### Why holdout Spearman is ≈ 0

The model is barely correlated with the holdout target. Likely
causes, in priority order:

1. **Sample size**: 234 train / 76 holdout, with 9 features, is
   small. LightGBM logged `No further splits with positive gain`
   repeatedly during boosting — the model converged to near-constant
   predictions.
2. **Feature signal**: the EV-feature builder
   (`crpod.features.ev_target.build`) emits 9 hand-engineered
   features per interaction. Several rows likely have zero
   `tower_hp_delta` (no tower took damage in the window), which
   compresses the target distribution and hurts both MAE and
   Spearman.
3. **Bar-reader noise**: HP reads are accurate to ~2.5% on the
   fixture, but the per-frame variance in the smoke run is
   plausibly higher because of VFX occlusion. Even small noise on
   the bookends dilutes the delta signal.

MAE = 463 HP versus a per-card target std of 500-760 HP indicates the
model is essentially predicting the median. Wave-3 work should
expand the replay count (76 replays available; we ran on 30 to keep
wall-clock tight) and consider richer features before the EV signal
is meaningful enough to drive a blunder rule. The structural
deliverable — a non-zero kept-sample count and end-to-end metrics —
is in place.

### Bonus YOLO-class warning

The smoke run also emitted one `Unknown KataCR class
'dagger-duchess-tower' — likely a card missing from KATACR_TO_CARD or
a non-card class missing from KATACR_NON_CARD. Detection dropped.`
warning. That's a one-card omission in the YOLO-to-Card mapping (the
Dagger Duchess seems to ship with a tower variant the mapping
doesn't know about) and unrelated to the HUD-OCR drop; pin it as a
small wave-3 cleanup, not a blocker.

### Brev environment runbook

Re-running this on a brev GPU box keeps tripping on a tower of
version mismatches; here is the minimum incantation that worked,
captured so the next operator doesn't burn another hour on it:

```bash
# 1. Provision (H100 not needed — workload is OCR-bound, see below).
brev create wave2d-metrics --type hyperstack_A6000

# 2. Setup on the box (all as `shadeform`):
sudo apt-get install -y -qq tesseract-ocr libgl1 libglib2.0-0 ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
git clone https://github.com/<user>/Clash-Royale-Pod.git crpod && cd crpod
git checkout <branch>

# 3. *** CRITICAL *** — pin Python 3.11 BEFORE syncing.
#    `uv sync` from the bare `torch>=2.4` constraint in pyproject.toml
#    resolves to Python 3.14 + torch 2.11.0+cu130, which fails to
#    initialise CUDA against the hyperstack box's CUDA-12.8 driver
#    *and* has no cu126 wheel for cp314 (cu126 only ships cp39-cp313).
uv python install 3.11
uv venv --python 3.11
uv sync --python 3.11

# 4. Force-reinstall torch onto the cu126 wheel index.
#    `uv pip install --reinstall` is required — the cached cu130
#    wheel will be picked otherwise.
uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu126 \
    torch torchvision

# 5. Verify — should print `2.11.0+cu126 True NVIDIA <GPU>`.
.venv/bin/python -c \
  'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'

# 6. Run via `.venv/bin/crpod` directly, NOT `uv run crpod`. `uv run`
#    re-syncs and clobbers the cu126 install back to cu130.
mkdir -p output/models
scp <local>:.../crpod_v1_best.pt output/models/crpod_v1_best.pt
tmux new-session -d -s train ".venv/bin/crpod train ..."
```

Two things that surprised wave-2D and may surprise wave-2E:

- **`brev ls` reports `UNHEALTHY / BUILDING / NOT READY` indefinitely
  for the hyperstack containers** — this is the dashboard's container
  probe, not the underlying VM. SSH (`brev shell` / `brev exec` /
  direct `ssh wave2d-metrics`) works fine despite the dashboard
  saying otherwise.
- **GPU choice doesn't matter for this workload yet.** YOLO inference
  finishes in seconds on H100/A6000 alike; `_parquet_to_replay` then
  spends 5-10 min per replay running tesseract single-threaded on the
  CPU loop in `huggingface.py`. The H100 sits at 0% utilisation
  during OCR. Until OCR is parallelised or replaced, an A6000 (or
  even a CPU-only box) is more cost-efficient than an H100.
- **`brev` CLI auth tokens expire mid-session.** If `brev exec`
  starts erroring with `auth.shouldLogin`, fall through to direct
  `ssh wave2d-metrics` (the SSH config brev installed on first
  contact is still valid) until the user re-runs `brev login`.

### Top-10 most-frequent cards (training fold)

Anchor card = `Interaction.friendly_plays[0].card`. Cards with fewer
than five training-fold samples are excluded from `per_card_stats`
(seven cards qualified). Three more cards (`elixir_golem`, `log`,
`minions`) sit at n=4 and are listed below the cut for reference.

| Rank | Card           | n_samples | median target | std target |
| ---- | -------------- | --------- | ------------- | ---------- |
| 1    | `skeletons`    | 80        | +0.0          | 603.8      |
| 2    | `goblins`      | 20        | -85.0         | 581.3      |
| 3    | `musketeer`    | 18        | +0.0          | 521.6      |
| 4    | `ice_spirit`   | 14        | +0.0          | 761.0      |
| 5    | `tesla`        | 11        | +0.0          | 593.0      |
| 6    | `cannon`       | 10        | +0.0          | 612.6      |
| 7    | `ice_golem`    | 8         | +0.0          | 760.0      |
| —    | `elixir_golem` | 4         | excluded (<5) | excluded   |
| —    | `log`          | 4         | excluded (<5) | excluded   |
| —    | `minions`      | 4         | excluded (<5) | excluded   |

Six of seven kept cards have median = 0 HP — the median anchor play
of those cards landed in a window where no tower took damage. Only
`goblins` shows a non-zero median (-85 HP), suggesting the goblin
spawn slightly correlates with conceding tower HP. Std values are
500-760 HP across the board; combined with MAE = 463, the model's
holdout error is ≈ 1 std, which is the "predicting the mean"
regime.

### Interpretation

The wave-2E HP-bar reader unblocks the EV training pipeline end to
end: 30 arena_15 replays now produce 310 trainable interactions
(was 0), `EvModel` loads and saves, `per_card_stats` has seven
qualifying cards, and the holdout split is non-degenerate. That is
the structural deliverable.

The model itself is weak (Spearman ≈ 0, MAE ≈ 1 std). This is
*expected* at 234 train rows on a 9-feature LightGBM with sparse
non-zero targets, and is the floor for wave-3 work to build on, not
a regression. Concretely:

- the **`per_card_stats` map** is the wave-3A blunder-rule input and
  is now populated for the seven most-frequent arena_15 anchors.
  `tests/test_ev_model.py` continues to pass against the new model
  artifact.
- the **scale calibration** is per-side-empirical, not theoretical.
  If wave-3 surfaces a card whose median target looks suspiciously
  scaled, re-run `_read_hp_bar` against a hand-labelled frame set
  before suspecting the model.
- a **stronger signal** likely needs (a) more replays — we processed
  30 of 76 available; (b) a soft-fallback target that uses readable
  towers when one bookend is occluded, recovering some of the 34%
  drop; or (c) a richer feature set than the current 9. Each of
  those is a future-wave change, not blocked by this chunk.

## Wave-3 contract published by this chunk

`EvModel.load(path).per_card_stats` returns
`dict[str, tuple[float, float]]`:

- key — card name string, matches `CardPlay.card`.
- value — `(median, std)` of training-fold tower-HP-delta targets for
  interactions whose anchor friendly play is that card.
- cards with `<5` training-fold samples are absent from the dict, so
  wave 3A's "skip cards with insufficient samples" rule is a key
  membership check.

`tests/test_ev_model.py::test_save_load_round_trip` pins the
joblib round-trip; `test_compute_per_card_stats_excludes_low_sample_cards`
pins the `<5` exclusion.
