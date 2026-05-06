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
PRE_WINDOW_LOOKBACK_SECONDS: float = 30.0


def _hud_window(hud: list[HudState], start_frame: int, end_frame: int) -> list[HudState]:
    return [s for s in hud if start_frame <= s.frame <= end_frame]


def build_interactions(
    plays: list[CardPlay],
    window: int = DEFAULT_WINDOW_FRAMES,
    hud: list[HudState] | None = None,
    fps: float | None = None,
) -> list[Interaction]:
    """Greedy windowing: each play opens an interaction if none is active.

    When `hud` is provided, each interaction's `tower_hp_delta` is populated
    from the HudState bookends inside that interaction's frame window.
    When `fps` is provided alongside `hud`, the wave-2J' pre-window HP delta
    fields are populated by diffing the start-frame princess HPs against the
    nearest HUD reading at-or-before `start_frame − 30 s`.
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
        sfl: int | None = None
        sfr: int | None = None
        sel: int | None = None
        ser: int | None = None
        if hud is not None:
            hud_slice = _hud_window(hud, anchor.frame, end_frame)
            delta = tower_hp_delta(hud_slice)
            if hud_slice:
                start = hud_slice[0]
                sfl = start.friendly_left_princess_hp
                sfr = start.friendly_right_princess_hp
                sel = start.enemy_left_princess_hp
                ser = start.enemy_right_princess_hp

        pre_friendly: int | None = None
        pre_enemy: int | None = None
        start_seconds: float | None = None
        if fps is not None and fps > 0:
            start_seconds = anchor.frame / fps
            lookback_frames = int(round(PRE_WINDOW_LOOKBACK_SECONDS * fps))
            lookback_frame = anchor.frame - lookback_frames
            if (
                lookback_frame >= 0
                and hud is not None
                and sfl is not None
                and sfr is not None
                and sel is not None
                and ser is not None
            ):
                candidates = [s for s in hud if s.frame <= lookback_frame]
                if candidates:
                    lb = max(candidates, key=lambda s: s.frame)
                    if (
                        lb.friendly_left_princess_hp is not None
                        and lb.friendly_right_princess_hp is not None
                    ):
                        pre_friendly = (sfl + sfr) - (
                            lb.friendly_left_princess_hp + lb.friendly_right_princess_hp
                        )
                    if (
                        lb.enemy_left_princess_hp is not None
                        and lb.enemy_right_princess_hp is not None
                    ):
                        pre_enemy = (sel + ser) - (
                            lb.enemy_left_princess_hp + lb.enemy_right_princess_hp
                        )

        out.append(
            Interaction(
                start_frame=anchor.frame,
                end_frame=end_frame,
                friendly_plays=friendly,
                enemy_plays=enemy,
                friendly_elixir_spent=sum(p.elixir_cost or card_cost(p.card) for p in friendly),
                enemy_elixir_spent=sum(p.elixir_cost or card_cost(p.card) for p in enemy),
                tower_hp_delta=delta,
                start_friendly_left_princess_hp=sfl,
                start_friendly_right_princess_hp=sfr,
                start_enemy_left_princess_hp=sel,
                start_enemy_right_princess_hp=ser,
                pre_window_friendly_hp_delta_30s=pre_friendly,
                pre_window_enemy_hp_delta_30s=pre_enemy,
                start_seconds=start_seconds,
            )
        )
        i = j
    return out
