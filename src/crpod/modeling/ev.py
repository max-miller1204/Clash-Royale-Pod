"""Expected-value model.

Training target: per-interaction elixir trade + damage delta (a proxy for
win-probability shift). Features: cards played, placement zones, tempo,
elixir state at interaction start.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class EvModel:
    """Thin LightGBM wrapper. Training/predict are lazy — the pipeline can
    import and flow data through features even without LightGBM installed."""

    model: Any = None

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

        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path) -> EvModel:
        import joblib

        return cls(model=joblib.load(path))
