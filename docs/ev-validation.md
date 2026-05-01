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

> **Status: structural deliverable only — metrics pending a real
> training run.**
>
> Two environment gaps blocked executing the run inside this worktree:
>
> 1. **`scikit-learn` is not in `pyproject.toml`.** LightGBM's
>    `LGBMRegressor` (sklearn API) raises `LightGBMError: scikit-learn
>    is required for lightgbm.sklearn` at fit time. `pyproject.toml` is
>    outside this chunk's owned files (per `CHUNK.md` ground rule 2),
>    so I'm surfacing rather than silently editing.
> 2. **Local box has very low compute power.** Per `CLAUDE.md`, heavy
>    pipelines (YOLO inference on HF replay frames) should run on
>    `brev`. Even a `--max-replays 5` smoke run on macOS CPU is
>    untested and likely impractical.
>
> The chunk's compute notes call out three lanes: (a) reduced
> `--max-replays` smoke run, (b) brev for the full retrain, (c) stop
> and surface. We're at lane (c) until the user picks (a) or (b).

Once a run lands, fill in:

| Metric                    | Value |
| ------------------------- | ----- |
| Replays processed         | TBD   |
| Drop rate (unreadable HUD)| TBD   |
| Total interactions        | TBD   |
| Train sample count        | TBD   |
| Holdout sample count      | TBD   |
| Holdout MAE               | TBD   |
| Holdout Spearman ρ        | TBD   |
| Run mode (full / smoke)   | TBD   |

### Top-10 most-frequent cards (training fold)

Anchor card = `Interaction.friendly_plays[0].card`. Cards with fewer
than five training-fold samples are excluded from `per_card_stats`.

| Rank | Card | n_samples | median target | std target |
| ---- | ---- | --------- | ------------- | ---------- |
| 1    | TBD  | TBD       | TBD           | TBD        |
| ...  | ...  | ...       | ...           | ...        |

### Interpretation

> One paragraph here once metrics land: is the new tower-HP-delta
> target better than the old `elixir_trade` proxy? Specifically — is
> holdout Spearman positive and meaningfully above zero, and is MAE
> small relative to the natural per-tower HP scale (princess towers
> start at ~3000 HP, so MAE in the few-hundreds range would be
> respectable; MAE comparable to that scale would mean the model is
> roughly guessing the mean)?

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
