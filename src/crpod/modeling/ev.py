"""Expected-value model.

Training target: per-interaction princess-tower HP delta — see `_training_target`
in `crpod.__main__`. Features: cards played, placement zones, tempo, elixir
state at interaction start.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crpod.features.placement import zone_of
from crpod.types import Interaction


def interaction_features(interaction: Interaction) -> dict[str, Any]:
    """Flatten an Interaction into a feature row."""
    friendly_cards = [p.card for p in interaction.friendly_plays]
    enemy_cards = [p.card for p in interaction.enemy_plays]
    zones = [zone_of(p.x, p.y).value for p in interaction.friendly_plays]
    return {
        "n_friendly_cards": len(friendly_cards),
        "n_enemy_cards": len(enemy_cards),
        "friendly_elixir_spent": interaction.friendly_elixir_spent,
        "enemy_elixir_spent": interaction.enemy_elixir_spent,
        "elixir_trade": interaction.elixir_trade,
        "friendly_cards": ",".join(friendly_cards),
        "enemy_cards": ",".join(enemy_cards),
        "friendly_zones": ",".join(zones),
        "duration_frames": interaction.end_frame - interaction.start_frame,
    }


def compute_per_card_stats(
    interactions: Sequence[Interaction],
    targets: Sequence[float],
    min_samples: int = 5,
) -> dict[str, tuple[float, float]]:
    """Group targets by anchor friendly card and return (median, std) per card.

    The anchor is `interaction.friendly_plays[0].card`; rows with empty
    `friendly_plays` contribute no per-card sample. Cards with fewer than
    `min_samples` training-fold rows are excluded.
    """
    by_card: dict[str, list[float]] = {}
    for interaction, target in zip(interactions, targets, strict=True):
        if not interaction.friendly_plays:
            continue
        anchor = interaction.friendly_plays[0].card
        by_card.setdefault(anchor, []).append(float(target))
    return {
        card: (statistics.median(values), statistics.pstdev(values))
        for card, values in by_card.items()
        if len(values) >= min_samples
    }


@dataclass
class EvModel:
    """Thin LightGBM wrapper. Training/predict are lazy — the pipeline can
    import and flow data through features even without LightGBM installed."""

    model: Any = None
    per_card_stats: dict[str, tuple[float, float]] = field(default_factory=dict)

    def fit(self, rows: list[dict[str, Any]], target: list[float]) -> None:
        import lightgbm as lgb
        import pandas as pd

        df = pd.DataFrame(rows)
        categorical = [c for c in df.columns if df[c].dtype == object]
        for c in categorical:
            df[c] = df[c].astype("category")
        self.model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05)
        self.model.fit(df, target, categorical_feature=categorical)

    def predict(self, rows: list[dict[str, Any]]) -> list[float]:
        if self.model is None:
            raise RuntimeError("EvModel.fit must be called before predict")
        import pandas as pd

        df = pd.DataFrame(rows)
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype("category")
        return list(self.model.predict(df))

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("nothing to save")
        import joblib

        joblib.dump(
            {"model": self.model, "per_card_stats": self.per_card_stats},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> EvModel:
        import joblib

        payload = joblib.load(path)
        # Tolerate the pre-2b artifact format (raw LGBMRegressor) for any
        # checkpoint saved before per_card_stats was wired up.
        if isinstance(payload, dict) and "model" in payload:
            raw_stats = payload.get("per_card_stats") or {}
            return cls(
                model=payload["model"],
                per_card_stats={
                    card: (float(v[0]), float(v[1])) for card, v in raw_stats.items()
                },
            )
        return cls(model=payload)
