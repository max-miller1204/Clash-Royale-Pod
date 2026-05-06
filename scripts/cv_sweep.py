"""5-fold CV across LightGBM / Ridge / XGBoost — wave 2K harness.

Reads the joblib written by `crpod train --save-data ...`, runs CV on
the training fold across the candidate grid, picks the CV winner by
mean fold-ρ (tie-break: mean MAE), one-shot fits the winner on the
full training fold and reports holdout ρ + Δρ vs wave-2J' (+0.223).
Also runs a re-shuffled-split sanity check for holdout-leak detection.

Usage:
    uv run python scripts/cv_sweep.py output/models/wave2k_data.joblib \\
        --out-csv cv_results.csv \\
        --out-model output/models/ev_wave2k.joblib

    # Smoke (one config per class, ~1 minute on the train fold):
    uv run python scripts/cv_sweep.py path/to/data.joblib --smoke

Holdout discipline: the frozen holdout is touched ONCE — at the
final-fit step after the CV winner is picked. Do not re-run with a
different config on the same data; that defeats the holdout.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

from crpod.modeling.ev import EvModel, _frame_with_categoricals, compute_per_card_stats

WAVE_2J_PRIME_RHO = 0.223
LEAK_GAP_THRESHOLD = 0.05
REGRESSION_THRESHOLD = -0.02
RIDGE_CARDINALITY_CAP = 500
RARE_CATEGORY_MIN_COUNT = 5


def _lgb_grid() -> list[dict]:
    return [
        {
            "n_estimators": n,
            "learning_rate": lr,
            "num_leaves": nl,
            "min_child_samples": md,
        }
        for n, lr, nl, md in itertools.product(
            [100, 200, 400, 800],
            [0.02, 0.05, 0.1],
            [15, 31, 63],
            [10, 20, 50],
        )
    ]


def _ridge_grid() -> list[dict]:
    return [{"alpha": a} for a in [0.1, 1.0, 10.0, 100.0]]


def _xgb_grid() -> list[dict]:
    return [
        {
            "n_estimators": n,
            "learning_rate": lr,
            "max_depth": d,
            "min_child_weight": w,
        }
        for n, lr, d, w in itertools.product(
            [100, 200, 400, 800],
            [0.02, 0.05, 0.1],
            [4, 6, 8],
            [1, 5, 10],
        )
    ]


def _score(y_true: list[float], y_pred) -> tuple[float, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    rho = float(spearmanr(y_pred_arr, y_true_arr).statistic)
    mae = float(np.mean(np.abs(y_pred_arr - y_true_arr)))
    return rho, mae


def _fit_lgb(X_fit: pd.DataFrame, y_fit: list[float], **kwargs):
    import lightgbm as lgb

    cat = [c for c, dt in X_fit.dtypes.items() if str(dt) == "category"]
    model = lgb.LGBMRegressor(verbosity=-1, **kwargs)
    model.fit(X_fit, y_fit, categorical_feature=cat)
    return model


def _fit_xgb(X_fit: pd.DataFrame, y_fit: list[float], **kwargs):
    import xgboost as xgb

    model = xgb.XGBRegressor(
        enable_categorical=True,
        tree_method="hist",
        verbosity=0,
        n_jobs=-1,
        **kwargs,
    )
    model.fit(X_fit, y_fit)
    return model


def _bucket_rare_top_card(X_fit: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    """Lump rare top_friendly_x_top_enemy categories into '__other__'.

    Wave 2K Ridge gotcha: high-cardinality one-hot blows up dimensionality.
    Cap at RIDGE_CARDINALITY_CAP levels by lumping anything with
    < RARE_CATEGORY_MIN_COUNT train-fold occurrences.
    """
    col = "top_friendly_x_top_enemy"
    if col not in X_fit.columns:
        return X_fit, set()
    nunique = X_fit[col].nunique()
    if nunique <= RIDGE_CARDINALITY_CAP:
        return X_fit, set(X_fit[col].astype(object).unique())

    counts = X_fit[col].value_counts()
    keep_levels = set(counts[counts >= RARE_CATEGORY_MIN_COUNT].index)
    keep_levels.add("__other__")
    out = X_fit.copy()
    obj = out[col].astype(object)
    out[col] = obj.where(obj.isin(keep_levels), "__other__").astype("category")
    return out, keep_levels


def _apply_top_card_buckets(X: pd.DataFrame, keep_levels: set[str]) -> pd.DataFrame:
    col = "top_friendly_x_top_enemy"
    if col not in X.columns or not keep_levels:
        return X
    out = X.copy()
    obj = out[col].astype(object)
    out[col] = obj.where(obj.isin(keep_levels), "__other__").astype("category")
    return out


def _ridge_design(
    X_fit: pd.DataFrame, X_val: pd.DataFrame
) -> tuple[csr_matrix, csr_matrix]:
    """One-hot the categorical cols (post-bucketing); passthrough numerics.

    Numerics are NaN-filled with 0; Ridge can't ingest NaN and
    pre_window_*_hp_delta_30s is NaN-when-missing in the raw rows.
    """
    X_fit_b, keep = _bucket_rare_top_card(X_fit)
    X_val_b = _apply_top_card_buckets(X_val, keep)

    cat_cols = [c for c, dt in X_fit_b.dtypes.items() if str(dt) == "category"]
    num_cols = [c for c in X_fit_b.columns if c not in cat_cols]

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    enc.fit(X_fit_b[cat_cols].astype(str))
    fit_cat = enc.transform(X_fit_b[cat_cols].astype(str))
    val_cat = enc.transform(X_val_b[cat_cols].astype(str))

    fit_num = X_fit_b[num_cols].fillna(0.0).to_numpy(dtype=float)
    val_num = X_val_b[num_cols].fillna(0.0).to_numpy(dtype=float)

    return hstack([fit_cat, csr_matrix(fit_num)]).tocsr(), hstack(
        [val_cat, csr_matrix(val_num)]
    ).tocsr()


def _cv_lgb(X: pd.DataFrame, y: list[float], kf: KFold, cfg: dict) -> tuple[list, list]:
    rhos, maes = [], []
    for fold, (tr, va) in enumerate(kf.split(X)):
        Xf, Xv = X.iloc[tr], X.iloc[va]
        yf = [y[i] for i in tr]
        yv = [y[i] for i in va]
        model = _fit_lgb(Xf, yf, **cfg)
        rho, mae = _score(yv, model.predict(Xv))
        rhos.append(rho)
        maes.append(mae)
        print(f"    lgb cfg={cfg} fold={fold} rho={rho:+.4f} mae={mae:.2f}", flush=True)
    return rhos, maes


def _cv_ridge(X: pd.DataFrame, y: list[float], kf: KFold, cfg: dict) -> tuple[list, list]:
    rhos, maes = [], []
    for fold, (tr, va) in enumerate(kf.split(X)):
        Xf, Xv = X.iloc[tr], X.iloc[va]
        yf = [y[i] for i in tr]
        yv = [y[i] for i in va]
        Xf_d, Xv_d = _ridge_design(Xf, Xv)
        model = Ridge(**cfg)
        model.fit(Xf_d, yf)
        rho, mae = _score(yv, model.predict(Xv_d))
        rhos.append(rho)
        maes.append(mae)
        print(f"    ridge cfg={cfg} fold={fold} rho={rho:+.4f} mae={mae:.2f}", flush=True)
    return rhos, maes


def _cv_xgb(X: pd.DataFrame, y: list[float], kf: KFold, cfg: dict) -> tuple[list, list]:
    rhos, maes = [], []
    for fold, (tr, va) in enumerate(kf.split(X)):
        Xf, Xv = X.iloc[tr], X.iloc[va]
        yf = [y[i] for i in tr]
        yv = [y[i] for i in va]
        model = _fit_xgb(Xf, yf, **cfg)
        rho, mae = _score(yv, model.predict(Xv))
        rhos.append(rho)
        maes.append(mae)
        print(f"    xgb cfg={cfg} fold={fold} rho={rho:+.4f} mae={mae:.2f}", flush=True)
    return rhos, maes


def _final_predict(
    cls: str,
    cfg: dict,
    X_train: pd.DataFrame,
    y_train: list[float],
    X_target: pd.DataFrame,
):
    """Refit on (X_train, y_train), predict on X_target. Returns (model, preds)."""
    if cls == "lightgbm":
        model = _fit_lgb(X_train, y_train, **cfg)
        return model, model.predict(X_target)
    if cls == "ridge":
        Xf, Xt = _ridge_design(X_train, X_target)
        model = Ridge(**cfg)
        model.fit(Xf, y_train)
        return model, model.predict(Xt)
    if cls == "xgboost":
        model = _fit_xgb(X_train, y_train, **cfg)
        return model, model.predict(X_target)
    raise RuntimeError(f"unknown model class: {cls}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", type=Path, help="path to wave2k_data.joblib")
    ap.add_argument("--out-csv", type=Path, default=Path("cv_results.csv"))
    ap.add_argument(
        "--out-model", type=Path, default=Path("output/models/ev_wave2k.joblib")
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="run one config per class, useful for local testing",
    )
    args = ap.parse_args(argv)

    payload = joblib.load(args.data)
    train_rows = payload["train_rows"]
    train_targets = list(payload["train_targets"])
    train_interactions = payload["train_interactions"]
    holdout_rows = payload["holdout_rows"]
    holdout_targets = list(payload["holdout_targets"])

    X_train = _frame_with_categoricals(train_rows)
    X_holdout = _frame_with_categoricals(holdout_rows)
    y_train = [float(v) for v in train_targets]
    y_holdout = [float(v) for v in holdout_targets]

    print(
        f"loaded: train={X_train.shape} y_train={len(y_train)} "
        f"holdout={X_holdout.shape} y_holdout={len(y_holdout)}",
        flush=True,
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    lgb_grid = _lgb_grid()
    ridge_grid = _ridge_grid()
    xgb_grid = _xgb_grid()
    if args.smoke:
        lgb_grid = lgb_grid[:1]
        ridge_grid = ridge_grid[:1]
        xgb_grid = xgb_grid[:1]

    rows: list[dict] = []

    print(f"\n=== LightGBM sweep ({len(lgb_grid)} configs) ===", flush=True)
    for i, cfg in enumerate(lgb_grid):
        print(f"[lgb {i + 1}/{len(lgb_grid)}] {cfg}", flush=True)
        rhos, maes = _cv_lgb(X_train, y_train, kf, cfg)
        rows.append(
            {
                "model_class": "lightgbm",
                "config_repr": repr(cfg),
                "config": cfg,
                "mean_rho": float(np.mean(rhos)),
                "std_rho": float(np.std(rhos)),
                "mean_mae": float(np.mean(maes)),
            }
        )

    print(f"\n=== Ridge sweep ({len(ridge_grid)} configs) ===", flush=True)
    for i, cfg in enumerate(ridge_grid):
        print(f"[ridge {i + 1}/{len(ridge_grid)}] {cfg}", flush=True)
        rhos, maes = _cv_ridge(X_train, y_train, kf, cfg)
        rows.append(
            {
                "model_class": "ridge",
                "config_repr": repr(cfg),
                "config": cfg,
                "mean_rho": float(np.mean(rhos)),
                "std_rho": float(np.std(rhos)),
                "mean_mae": float(np.mean(maes)),
            }
        )

    print(f"\n=== XGBoost sweep ({len(xgb_grid)} configs) ===", flush=True)
    for i, cfg in enumerate(xgb_grid):
        print(f"[xgb {i + 1}/{len(xgb_grid)}] {cfg}", flush=True)
        rhos, maes = _cv_xgb(X_train, y_train, kf, cfg)
        rows.append(
            {
                "model_class": "xgboost",
                "config_repr": repr(cfg),
                "config": cfg,
                "mean_rho": float(np.mean(rhos)),
                "std_rho": float(np.std(rhos)),
                "mean_mae": float(np.mean(maes)),
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model_class", "config_repr", "mean_rho", "std_rho", "mean_mae"])
        for r in rows:
            writer.writerow(
                [
                    r["model_class"],
                    r["config_repr"],
                    f"{r['mean_rho']:.6f}",
                    f"{r['std_rho']:.6f}",
                    f"{r['mean_mae']:.6f}",
                ]
            )
    print(f"\nCV table → {args.out_csv}", flush=True)

    # Pick winner: max mean_ρ; tie-break by min mean_MAE.
    best = max(rows, key=lambda r: (r["mean_rho"], -r["mean_mae"]))
    print(
        f"\nCV winner: {best['model_class']} {best['config_repr']}  "
        f"mean_ρ={best['mean_rho']:+.4f} ± {best['std_rho']:.4f}, "
        f"mean_MAE={best['mean_mae']:.2f}",
        flush=True,
    )

    cls = best["model_class"]
    cfg = best["config"]

    # ONE shot at the frozen holdout.
    winner, holdout_preds = _final_predict(cls, cfg, X_train, y_train, X_holdout)
    final_rho, final_mae = _score(y_holdout, holdout_preds)
    delta_rho = final_rho - WAVE_2J_PRIME_RHO
    print("\n=== Frozen-holdout one-shot ===", flush=True)
    print(f"Final holdout MAE: {final_mae:.2f}", flush=True)
    print(f"Final holdout Spearman ρ: {final_rho:+.4f}", flush=True)
    print(f"Δρ vs wave 2J' (+0.223): {delta_rho:+.4f}", flush=True)
    if delta_rho < REGRESSION_THRESHOLD:
        print(
            f"⚠️  Δρ < {REGRESSION_THRESHOLD} — REGRESSION; "
            "do not ship without operator review.",
            flush=True,
        )

    # Save artifact.
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    per_card_stats = compute_per_card_stats(train_interactions, y_train)
    if cls == "lightgbm":
        ev = EvModel(model=winner, per_card_stats=per_card_stats)
        ev.save(args.out_model)
    else:
        joblib.dump(
            {
                "model_class": cls,
                "config": cfg,
                "model": winner,
                "per_card_stats": per_card_stats,
            },
            args.out_model,
        )
    print(f"saved final model → {args.out_model}", flush=True)

    # Re-shuffled-split sanity check (different seed; same winner config).
    print("\n=== Re-shuffled-split sanity check ===", flush=True)
    rng = np.random.default_rng(seed=20260505)
    rows_all = list(train_rows) + list(holdout_rows)
    y_all = list(y_train) + list(y_holdout)
    n = len(y_all)
    perm = rng.permutation(n)
    split = int(0.8 * n)
    tr_idx, ho_idx = perm[:split], perm[split:]
    X_tr2 = _frame_with_categoricals([rows_all[i] for i in tr_idx])
    X_ho2 = _frame_with_categoricals([rows_all[i] for i in ho_idx])
    y_tr2 = [y_all[i] for i in tr_idx]
    y_ho2 = [y_all[i] for i in ho_idx]
    _, preds2 = _final_predict(cls, cfg, X_tr2, y_tr2, X_ho2)
    rho2, mae2 = _score(y_ho2, preds2)
    gap = abs(rho2 - final_rho)
    print(f"Frozen-split ρ: {final_rho:+.4f}", flush=True)
    print(f"Re-shuffled ρ:  {rho2:+.4f}  (MAE {mae2:.2f})", flush=True)
    print(f"Gap:            {gap:.4f}  threshold={LEAK_GAP_THRESHOLD}", flush=True)
    if gap > LEAK_GAP_THRESHOLD:
        print(
            f"⚠️  Re-shuffled-split divergence > {LEAK_GAP_THRESHOLD} — "
            "possible holdout leak; STOP and report.",
            flush=True,
        )
    else:
        print("✓ Within leak threshold.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
