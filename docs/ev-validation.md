# EV Model Validation — Wave 2B

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

> **Status: structural deliverable only — metrics blocked by a
> tesseract-vs-stylised-HUD-digits failure in the OCR pipeline. See
> "Diagnosis: tesseract can't read CR's stylised HP digits" below for
> the empirical evidence and the wave-2E/3 follow-up needed to
> unblock.**

### Wave-2D run (smoke, arena_15, 3 replays)

| Field                        | Value                                                                |
| ---------------------------- | -------------------------------------------------------------------- |
| Branch                       | `swarm/finish-project-wave-2d-brev-metrics`                          |
| Brev instance                | `hyperstack_H100` ($2.28/hr; H100 PCIe 80GB, 28 vCPU, 181 GiB RAM)   |
| Driver / CUDA                | NVIDIA 570.195.03 / CUDA 12.8                                        |
| Python / torch               | CPython 3.11.15 venv / `torch==2.11.0+cu126`                         |
| Tesseract                    | 4.1.1 (Ubuntu 22.04 `tesseract-ocr` apt package)                     |
| Invocation                   | `crpod train --weights output/models/crpod_v1_best.pt --out output/models/ev-smoke.joblib --arena arena_15 --max-replays 3` |
| Wall-clock                   | 23 min 40 s (22:49:59Z → 23:13:39Z, 2026-05-01)                      |
| Replays processed            | 3 (arena_15 IDs `00a91415…`, `02c3eb19…`, `0364a998…`)               |
| Frames decoded               | 2 373 (583 + 1 024 + 766)                                            |
| Frames with HUD-OCR exception| 0 (`ocr_fail=0%` reported throughout)                                |
| Total interactions seen      | 47                                                                   |
| Dropped (unreadable HUD)     | **47 / 47 (100%)**                                                   |
| Kept sample count            | **0** — `_cmd_train` exited with `no training data collected`         |
| Train / holdout split        | n/a (no rows survived to be split)                                   |
| Holdout MAE                  | n/a                                                                  |
| Holdout Spearman ρ           | n/a                                                                  |
| `per_card_stats`             | n/a                                                                  |

The full smoke `crpod train` log was captured at
`/home/shadeform/smoke.log` on the brev box. The terminal lines were:

```text
dropped 47/47 training rows (100% — unreadable HUD)
no training data collected
```

The `output/models/ev-smoke.joblib` artifact was never written
(`_cmd_train` returns 1 before reaching `model.save`), so there is
no model from this run.

### Diagnosis: tesseract can't read CR's stylised HP digits

The wave-2A "empty `Replay.hud=[]`" failure documented in earlier
revisions of this file is *fixed* — wave-2C wired `HudReader` into
`_parquet_to_replay`, so `Replay.hud` is now populated for every
decoded frame and `ocr_fail=0%` (no `HudReader.read` exceptions). The
remaining 100% drop is a different failure further down the pipeline:
the OCR returns `None` *silently* for almost every princess-HP region
because tesseract can't recognise the in-game digit glyphs. Per
`HudReader._read_number`, an empty or non-numeric string returns
`None` without raising, so the `ocr_fail` counter stays at 0% even
when most reads are actually empty.

A throw-away inspector (`HFReplayLoader.load(...)` on
`arena_15/00a91415-49c6-4773-a000-f87722361130`, 583 frames; then a
per-field non-`None` count over `replay.hud`) produced the following
breakdown — re-derive in two minutes if needed:

| Region                 | Non-`None` frames | Rate    |
| ---------------------- | ----------------- | ------- |
| `friendly_left`  (fL)  | 30 / 583          | 5.1%    |
| `friendly_right` (fR)  | 14 / 583          | 2.4%    |
| `enemy_left`     (eL)  | 37 / 583          | 6.3%    |
| `enemy_right`    (eR)  | 42 / 583          | 7.2%    |
| **all four**           | **0 / 583**       | **0%**  |

`tower_hp_delta(window)` requires the bookend `HudState` of the
window to have all four princess HPs non-`None`; even one `None`
makes the corresponding `delta` entry `None`, which makes
`_training_target` return `None`, which makes the row drop. With
0 / 583 frames satisfying the all-four-readable condition, every
interaction window in this replay drops by construction. The other
two smoke replays show the same pattern, so 47 / 47 dropped is the
floor, not a fluke.

A direct tesseract probe on three known-good frames (early / mid /
late, same replay) confirms the OCR is the failure point rather than
the regions:

```text
--- early (idx=5, frame_id=6, shape=(960, 540, 3)) ---
  friendly_elixir:      digit-whitelist=''   no-whitelist=''
  friendly_tower_left:  digit-whitelist=''   no-whitelist='aha ; y'
  friendly_tower_right: digit-whitelist=''   no-whitelist=''
  enemy_tower_left:     digit-whitelist=''   no-whitelist=''
  enemy_tower_right:    digit-whitelist=''   no-whitelist=''
```

Visual inspection of those same crops (`/home/shadeform/region_crops/`
on brev, e.g. `early_friendly_tower_left.png`) shows clearly readable
HP digits to a human — `2786`, `2423`, `1675`, etc. — yet tesseract
4.1.1 returns either an empty string or unrelated noise (`'aha ; y'`,
`'es'`, `'ais | i\''`). Even `friendly_elixir`, which displays a
clean `10` in pink, is read as empty. The `--psm 7` mode plus a
6×-cubic upscale aren't enough to recover the stylised, anti-aliased
in-game font.

**This is not a HudRegions calibration bug** — the regions overlap
the right pixels. **This is not a wave-2C wiring bug** — `HudState`
flows end-to-end and the OCR is being called per frame. The bug is in
the OCR pipeline (`crpod.ocr.hud.HudReader._read_number`), and fixing
it requires `src/` changes that are explicitly out of scope for this
chunk per `CHUNK.md` ("Out of scope (do NOT touch): Any `src/`
file"). It belongs to a wave-2E "OCR-pipeline-actually-reads-HP"
chunk or to wave 3.

Concrete options for that follow-up:

1. **Sample HP from the depleting tower bar (graphical), not the
   digit overlay (textual)**. The bar is persistent and rendered as a
   solid coloured strip whose pixel-length tracks HP. A ~10-line
   numpy threshold-and-count is more robust than tesseract on
   stylised digits and survives the case where the digit overlay
   isn't drawn (e.g. tower destroyed, late game).
2. **If the digit overlay is the source of truth**: stronger
   pre-processing (HSV mask on the gold/red digit colour →
   morphological clean → only then upscale → pass `outputbase` PNG
   to tesseract). Bench against a held-out frame set with
   ground-truth labels before re-running training.
3. **Soft-fallback in `_training_target`**: compute the delta over
   the towers that *are* readable instead of demanding all four. Less
   principled (the EV target shape changes per row) but recovers some
   training signal in the meantime.

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
than five training-fold samples are excluded from `per_card_stats`.

| Rank | Card | n_samples | median target | std target |
| ---- | ---- | --------- | ------------- | ---------- |
| —    | n/a  | n/a       | n/a           | n/a        |

`per_card_stats` is empty because no training rows survived the drop
filter. The wave-3A blunder rule reduces to a vacuous key check —
every card is "absent from `per_card_stats`" and therefore skipped.
Cannot land wave 3 on top of this without first fixing the OCR
blocker.

### Interpretation

There is no holdout signal to interpret. The model was not trained
because zero rows survived the all-four-towers-readable filter. We
therefore cannot say whether the tower-HP-delta target is
meaningfully better than the original `elixir_trade` proxy from this
run; that comparison needs a wave-2E pass that lands a working OCR
(or HP-bar) reader, after which `crpod train` can be re-invoked on
this branch unchanged. The wave-2A label arithmetic, the wave-2B
holdout protocol, and the wave-2C HUD wiring are all in place — the
only missing piece is recovering numeric princess HP from the frame
pixels.

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
