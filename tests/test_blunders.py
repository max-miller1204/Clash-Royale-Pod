"""Wave 3C — pure-logic tests for `detect_blunders`.

The blunder rule (from `crpod.analysis.blunders`):

    (per_card_median − predicted_ev) / per_card_std > 1.0

i.e. a play is a blunder when the model's predicted outcome is more than
1σ worse than what the card typically returns over the training fold.

These tests synthesize `CardPlay` + EV inputs with hand-computed z-scores
so the threshold logic is pinned without needing a trained model.
"""

from __future__ import annotations

from crpod.analysis.blunders import detect_blunders
from crpod.types import CardPlay, Side


def _play(card: str, frame: int) -> CardPlay:
    return CardPlay(frame=frame, card=card, x=270, y=480, side=Side.FRIENDLY)


def test_zero_blunder_match_returns_empty_list() -> None:
    """Every play sits at-or-above the per-card median → no blunders."""
    plays = [_play("knight", 10), _play("musketeer", 60)]
    ev_predictions = [50.0, 100.0]  # both at median
    per_card_stats = {
        "knight": (50.0, 10.0),
        "musketeer": (100.0, 25.0),
    }
    assert detect_blunders(plays, ev_predictions, per_card_stats) == []


def test_multi_blunder_match_orders_worst_first() -> None:
    """Two blunders → sigma_below descending; third play stays clean."""
    plays = [
        _play("knight", 10),
        _play("hog_rider", 60),
        _play("fireball", 110),
    ]
    # knight median 50, std 10 → ev 30 = 2.0σ below
    # hog_rider median 200, std 50 → ev 100 = 2.0σ below… tie. ev 50 = 3.0σ below.
    # fireball median 80, std 20 → ev 75 = 0.25σ below (under threshold)
    ev_predictions = [30.0, 50.0, 75.0]
    per_card_stats = {
        "knight": (50.0, 10.0),
        "hog_rider": (200.0, 50.0),
        "fireball": (80.0, 20.0),
    }
    out = detect_blunders(plays, ev_predictions, per_card_stats)
    assert len(out) == 2
    assert out[0].card == "hog_rider"
    assert out[0].sigma_below == 3.0
    assert out[0].play_idx == 1
    assert out[0].ev_predicted == 50.0
    assert out[0].per_card_median == 200.0
    assert out[1].card == "knight"
    assert out[1].sigma_below == 2.0
    assert out[1].play_idx == 0


def test_threshold_is_strictly_greater_than_one_sigma() -> None:
    """Exactly-1σ-below should NOT trigger; just-over should. The rule
    is `> 1.0`, not `>= 1.0` — matches the spec language."""
    plays = [_play("knight", 10), _play("musketeer", 20)]
    ev_predictions = [40.0, 75.0]  # knight = exactly 1σ, musketeer = 1σ + ε
    per_card_stats = {
        "knight": (50.0, 10.0),
        "musketeer": (100.0, 24.0),  # (100-75)/24 ≈ 1.041 > 1
    }
    out = detect_blunders(plays, ev_predictions, per_card_stats)
    assert [b.card for b in out] == ["musketeer"]


def test_card_with_under_5_samples_is_excluded() -> None:
    """Per spec: cards with <5 train-fold samples are already filtered
    out of `per_card_stats` upstream — the detector treats a missing
    card key as "skip this play, not enough data to judge."""
    plays = [_play("knight", 10), _play("mega_knight", 60)]
    ev_predictions = [30.0, 0.0]
    # mega_knight intentionally absent — wave 2B excludes it (fewer than 5)
    per_card_stats = {"knight": (50.0, 10.0)}
    out = detect_blunders(plays, ev_predictions, per_card_stats)
    assert [b.card for b in out] == ["knight"]
    assert all(b.card != "mega_knight" for b in out)


def test_card_not_in_median_table_is_skipped_silently() -> None:
    """A card the model has literally never seen also gets skipped — same
    contract as the <5-samples case; the detector can't distinguish them
    and shouldn't try."""
    plays = [_play("unknown_champion", 10)]
    ev_predictions = [-9999.0]
    per_card_stats = {"knight": (50.0, 10.0)}
    assert detect_blunders(plays, ev_predictions, per_card_stats) == []


def test_zero_std_card_is_skipped() -> None:
    """Edge case — a card with a degenerate (zero-std) training distribution
    would divide by zero. The function skips rather than emitting `inf`
    sigma values."""
    plays = [_play("constant_card", 10)]
    ev_predictions = [-1000.0]
    per_card_stats = {"constant_card": (0.0, 0.0)}
    assert detect_blunders(plays, ev_predictions, per_card_stats) == []


def test_length_mismatch_raises_value_error() -> None:
    """plays and ev_predictions must align."""
    import pytest

    plays = [_play("knight", 10), _play("hog_rider", 60)]
    with pytest.raises(ValueError):
        detect_blunders(plays, [30.0], {"knight": (50.0, 10.0)})


def test_blunder_play_idx_indexes_into_input_plays() -> None:
    """`play_idx` is the position in the input `plays` list — callers in
    `__main__.py` remap to their own coordinate space, but the function's
    contract is "index into what you passed me."""
    plays = [
        _play("knight", 10),
        _play("knight", 60),
        _play("knight", 110),
    ]
    ev_predictions = [60.0, 0.0, 55.0]
    per_card_stats = {"knight": (50.0, 10.0)}
    out = detect_blunders(plays, ev_predictions, per_card_stats)
    assert len(out) == 1
    assert out[0].play_idx == 1
