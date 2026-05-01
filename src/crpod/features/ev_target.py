"""Tower-HP-delta EV target.

Replaces the original `interaction.elixir_trade` proxy with a direct
measurement of HP swing across the interaction window. Used by the EV
training loop in `crpod.__main__._cmd_train` (wave 2A) and consumed by the
EV-validation step (wave 2B).

Sign convention: every delta is `end_hp - start_hp`. Negative means HP was
lost over the window. Friendly losing tower HP is bad; enemy losing tower HP
is good. The EV target sums (friendly_delta - enemy_delta) across the two
princess towers per side.
"""

from __future__ import annotations

from crpod.types import HudState

_TOWER_KEYS: tuple[str, ...] = (
    "friendly_left",
    "friendly_right",
    "enemy_left",
    "enemy_right",
)


def _hp_for(state: HudState, key: str) -> int | None:
    if key == "friendly_left":
        return state.friendly_left_princess_hp
    if key == "friendly_right":
        return state.friendly_right_princess_hp
    if key == "enemy_left":
        return state.enemy_left_princess_hp
    if key == "enemy_right":
        return state.enemy_right_princess_hp
    raise KeyError(key)


def tower_hp_delta(window: list[HudState]) -> dict[str, int | None]:
    """Per-tower princess HP swing across the window.

    Returns one entry per tower keyed by `{friendly,enemy}_{left,right}`.
    Each value is `end_hp - start_hp` for that tower, or `None` if either
    bookend HUD reading was unreadable. An empty `window` yields `None` for
    every tower.
    """
    if not window:
        return dict.fromkeys(_TOWER_KEYS)
    start = window[0]
    end = window[-1]
    out: dict[str, int | None] = {}
    for key in _TOWER_KEYS:
        s = _hp_for(start, key)
        e = _hp_for(end, key)
        out[key] = (e - s) if (s is not None and e is not None) else None
    return out
