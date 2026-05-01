"""Round-trip tests for `EvModel.per_card_stats`.

The wave-3 blunder rule keys off `per_card_stats` to skip cards with too
few training-fold samples. These tests pin down the contract:

- per-card stats are populated from the **anchor** friendly play
  (`Interaction.friendly_plays[0].card`).
- cards with fewer than 5 training-fold samples are excluded.
- the dict round-trips through `save`/`load`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crpod.modeling.ev import (
    EvModel,
    compute_per_card_stats,
    interaction_features,
)
from crpod.types import CardPlay, Interaction, Side


def _interaction(card: str, frame: int) -> Interaction:
    play = CardPlay(frame=frame, card=card, x=270, y=480, side=Side.FRIENDLY)
    return Interaction(
        start_frame=frame,
        end_frame=frame + 30,
        friendly_plays=(play,),
        enemy_plays=(),
        friendly_elixir_spent=play.elixir_cost,
        enemy_elixir_spent=0,
    )


def test_compute_per_card_stats_excludes_low_sample_cards() -> None:
    interactions = [_interaction("knight", i) for i in range(5)] + [
        _interaction("musketeer", 100),
        _interaction("musketeer", 130),
    ]
    targets = [10.0, 20.0, 30.0, 40.0, 50.0, 5.0, 15.0]

    stats = compute_per_card_stats(interactions, targets)

    assert "knight" in stats
    assert "musketeer" not in stats  # only 2 samples
    median, std = stats["knight"]
    assert median == pytest.approx(30.0)
    assert std > 0


def test_compute_per_card_stats_skips_empty_friendly_plays() -> None:
    empty = Interaction(
        start_frame=0,
        end_frame=30,
        friendly_plays=(),
        enemy_plays=(),
        friendly_elixir_spent=0,
        enemy_elixir_spent=0,
    )
    interactions = [empty] + [_interaction("knight", i) for i in range(5)]
    targets = [99.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    stats = compute_per_card_stats(interactions, targets)

    assert "knight" in stats
    assert stats["knight"][0] == pytest.approx(3.0)


def test_save_load_round_trip(tmp_path: Path) -> None:
    # `importorskip` only catches ImportError; lightgbm on macOS without libomp
    # raises OSError during dlopen. LightGBM's sklearn API also needs scikit-learn
    # at fit time and raises LightGBMError if it's missing.
    try:
        import lightgbm  # noqa: F401
    except (ImportError, OSError) as e:
        pytest.skip(f"lightgbm unavailable: {e}")
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")

    interactions = [_interaction("knight", i) for i in range(6)] + [
        _interaction("musketeer", 200),
        _interaction("musketeer", 230),
    ]
    targets = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 1.0, 2.0]
    rows = [interaction_features(i) for i in interactions]

    model = EvModel()
    model.fit(rows, targets)
    model.per_card_stats = compute_per_card_stats(interactions, targets)
    assert "knight" in model.per_card_stats
    assert "musketeer" not in model.per_card_stats

    out = tmp_path / "ev.joblib"
    model.save(out)

    loaded = EvModel.load(out)
    assert loaded.per_card_stats.keys() == model.per_card_stats.keys()
    for card, (median, std) in model.per_card_stats.items():
        loaded_median, loaded_std = loaded.per_card_stats[card]
        assert loaded_median == pytest.approx(median)
        assert loaded_std == pytest.approx(std)
    # value type contract: tuple[float, float]
    sample = next(iter(loaded.per_card_stats.values()))
    assert isinstance(sample, tuple)
    assert len(sample) == 2
    assert all(isinstance(v, float) for v in sample)
