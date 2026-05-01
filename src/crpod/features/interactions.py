"""Group plays into interaction windows.

An 'interaction' is a short span where one side plays a card and the other
responds. The canonical example is hog-tesla: hog drops at frame N, tesla
drops at frame N+8, both resolve by N+40 → one interaction, 4-4 elixir trade.
"""

from __future__ import annotations

from crpod.constants import card_cost
from crpod.features.ev_target import tower_hp_delta
from crpod.types import CardPlay, HudState, Interaction, Side

DEFAULT_WINDOW_FRAMES: int = 40  # ~4 seconds at 10 fps


def _hud_window(hud: list[HudState], start_frame: int, end_frame: int) -> list[HudState]:
    return [s for s in hud if start_frame <= s.frame <= end_frame]


def build_interactions(
    plays: list[CardPlay],
    window: int = DEFAULT_WINDOW_FRAMES,
    hud: list[HudState] | None = None,
) -> list[Interaction]:
    """Greedy windowing: each play opens an interaction if none is active.

    When `hud` is provided, each interaction's `tower_hp_delta` is populated
    from the HudState bookends inside that interaction's frame window.
    """
    if not plays:
        return []

    ordered = sorted(plays, key=lambda p: p.frame)
    out: list[Interaction] = []
    i = 0
    while i < len(ordered):
        anchor = ordered[i]
        end = anchor.frame + window
        bucket: list[CardPlay] = [anchor]
        j = i + 1
        while j < len(ordered) and ordered[j].frame <= end:
            bucket.append(ordered[j])
            j += 1

        friendly = tuple(p for p in bucket if p.side is Side.FRIENDLY)
        enemy = tuple(p for p in bucket if p.side is Side.ENEMY)
        end_frame = max(p.frame for p in bucket)
        delta: dict[str, int | None] = {}
        if hud is not None:
            delta = tower_hp_delta(_hud_window(hud, anchor.frame, end_frame))
        out.append(
            Interaction(
                start_frame=anchor.frame,
                end_frame=end_frame,
                friendly_plays=friendly,
                enemy_plays=enemy,
                friendly_elixir_spent=sum(p.elixir_cost or card_cost(p.card) for p in friendly),
                enemy_elixir_spent=sum(p.elixir_cost or card_cost(p.card) for p in enemy),
                tower_hp_delta=delta,
            )
        )
        i = j
    return out
