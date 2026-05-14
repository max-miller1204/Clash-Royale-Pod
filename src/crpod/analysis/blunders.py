"""Statistical blunder detection — wave 3A.

Rule (from SPEC.md): a play is a blunder when

    (per_card_median − predicted_ev) / per_card_std > 1.0

i.e. the model thinks the outcome this play is about to get is more than
one standard deviation worse than the typical outcome for this specific
card. The per-card (median, std) table comes from the *training fold*
and is persisted inside the `EvModel` artifact (`per_card_stats`), so
inference doesn't need access to training data.

Cards with fewer than 5 training samples are pre-filtered out of
`per_card_stats` (wave 2B), so a missing-card lookup here means the same
thing as "not enough data to judge" and the play is skipped.
"""

from __future__ import annotations

from collections.abc import Sequence

from crpod.types import Blunder, CardPlay

_SIGMA_THRESHOLD = 1.0


def detect_blunders(
    plays: Sequence[CardPlay],
    ev_predictions: Sequence[float],
    per_card_stats: dict[str, tuple[float, float]],
) -> list[Blunder]:
    """Flag plays whose predicted EV sits > 1σ below the per-card median.

    `plays` and `ev_predictions` must be index-aligned and same length.
    `per_card_stats[card]` is `(median, std)` from the training fold.

    Returns a list ordered worst-first (largest sigma below the median
    first). Plays whose card is missing from `per_card_stats`, or whose
    std is zero (degenerate — would divide by zero), are skipped.
    """
    if len(plays) != len(ev_predictions):
        raise ValueError(
            f"plays/ev_predictions length mismatch: {len(plays)} vs {len(ev_predictions)}"
        )

    blunders: list[Blunder] = []
    for idx, (play, ev) in enumerate(zip(plays, ev_predictions, strict=True)):
        stats = per_card_stats.get(play.card)
        if stats is None:
            continue
        median, std = stats
        if std <= 0.0:
            continue
        sigma_below = (median - ev) / std
        if sigma_below > _SIGMA_THRESHOLD:
            blunders.append(
                Blunder(
                    play_idx=idx,
                    card=play.card,
                    ev_predicted=float(ev),
                    per_card_median=float(median),
                    sigma_below=float(sigma_below),
                )
            )

    blunders.sort(key=lambda b: b.sigma_below, reverse=True)
    return blunders
