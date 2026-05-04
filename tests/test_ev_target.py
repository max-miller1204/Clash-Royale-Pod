"""Unit tests for the princess-tower HP-delta EV target.

Pure-Python construction of HudState/Interaction so this runs without GPU,
YOLO, or tesseract — exercises only the label math used by `_cmd_train`.
"""

from __future__ import annotations

from crpod.__main__ import _training_target
from crpod.features.ev_target import tower_hp_delta
from crpod.types import HudState, Interaction


def _hud(
    frame: int,
    fl: int | None = 1000,
    fr: int | None = 1000,
    el: int | None = 1000,
    er: int | None = 1000,
) -> HudState:
    return HudState(
        frame=frame,
        friendly_elixir=0.0,
        enemy_elixir=None,
        friendly_left_princess_hp=fl,
        friendly_right_princess_hp=fr,
        enemy_left_princess_hp=el,
        enemy_right_princess_hp=er,
    )


def _interaction(window: list[HudState]) -> Interaction:
    return Interaction(
        start_frame=window[0].frame,
        end_frame=window[-1].frame,
        friendly_plays=(),
        enemy_plays=(),
        friendly_elixir_spent=0,
        enemy_elixir_spent=0,
        tower_hp_delta=tower_hp_delta(window),
    )


def test_friendly_left_loss_yields_negative_target() -> None:
    window = [
        _hud(0, fl=1000, fr=1000, el=1000, er=1000),
        _hud(40, fl=800, fr=1000, el=1000, er=1000),
    ]
    assert _training_target(_interaction(window)) == -200.0


def test_enemy_right_loss_yields_positive_target() -> None:
    window = [
        _hud(0, fl=1000, fr=1000, el=1000, er=1000),
        _hud(40, fl=1000, fr=1000, el=1000, er=700),
    ]
    assert _training_target(_interaction(window)) == 300.0


def test_mixed_damage_combines_per_spec_formula() -> None:
    # friendly_left -200, friendly_right -100, enemy_left -50, enemy_right -300
    # → (-200 + -100) - (-50 + -300) = +50
    window = [
        _hud(0, fl=1000, fr=1000, el=1000, er=1000),
        _hud(40, fl=800, fr=900, el=950, er=700),
    ]
    assert _training_target(_interaction(window)) == 50.0


def test_unreadable_end_tower_signals_drop() -> None:
    window = [
        _hud(0, fl=1000, fr=1000, el=1000, er=1000),
        _hud(40, fl=None, fr=900, el=950, er=700),
    ]
    assert _training_target(_interaction(window)) is None


def test_unreadable_start_tower_signals_drop() -> None:
    window = [
        _hud(0, fl=1000, fr=None, el=1000, er=1000),
        _hud(40, fl=800, fr=900, el=950, er=700),
    ]
    assert _training_target(_interaction(window)) is None


def test_no_damage_yields_zero_target() -> None:
    window = [
        _hud(0, fl=1000, fr=1000, el=1000, er=1000),
        _hud(40, fl=1000, fr=1000, el=1000, er=1000),
    ]
    assert _training_target(_interaction(window)) == 0.0


def test_destroyed_tower_window_all_none_yields_zero_delta() -> None:
    """Wave 2I: a tower whose HUD reads None across every frame in the
    window is treated as destroyed (princess towers don't unfaint), so
    its delta is 0 instead of None — the interaction is no longer dropped.
    """
    window = [
        _hud(0, fl=1000, fr=1000, el=None, er=1000),
        _hud(20, fl=1000, fr=1000, el=None, er=1000),
        _hud(40, fl=1000, fr=1000, el=None, er=1000),
    ]
    delta = tower_hp_delta(window)
    assert delta["enemy_left"] == 0
    assert delta["friendly_left"] == 0
    assert delta["friendly_right"] == 0
    assert delta["enemy_right"] == 0


def test_destroyed_tower_does_not_drop_training_row() -> None:
    """Wave 2I: a destroyed enemy tower (None-throughout) plus real damage
    on other towers yields a valid training target reflecting the live
    towers' deltas. Without the fix this row would drop.
    """
    window = [
        _hud(0, fl=1000, fr=1000, el=None, er=1000),
        _hud(20, fl=1000, fr=1000, el=None, er=1000),
        _hud(40, fl=900, fr=1000, el=None, er=950),
    ]
    # friendly_left -100, friendly_right 0, enemy_left 0 (destroyed),
    # enemy_right -50 → (-100 + 0) - (0 + -50) = -50.
    assert _training_target(_interaction(window)) == -50.0


def test_occluded_bookend_with_readable_mid_still_drops() -> None:
    """Wave 2I: if a tower's bookend is None but the window has any frame
    with a real read for that tower, treat it as transient occlusion (not
    destroyed) and keep dropping the row — matches pre-2I behaviour for
    the VFX-occlusion case which Change 2 (gap-tolerant reader) handles.
    """
    window = [
        _hud(0, fl=1000, fr=1000, el=None, er=1000),
        _hud(20, fl=1000, fr=1000, el=2000, er=1000),
        _hud(40, fl=900, fr=1000, el=None, er=1000),
    ]
    assert _training_target(_interaction(window)) is None


def test_destroyed_tower_does_not_rescue_other_towers_occlusion() -> None:
    """Wave 2I: the destroyed-tower fix targets only the destroyed tower.
    If another tower's bookend is occluded the row still drops — the fix
    is selective, not a blanket loosening of the bookend rule.
    """
    window = [
        _hud(0, fl=1000, fr=1000, el=None, er=1000),
        _hud(20, fl=1000, fr=1000, el=None, er=1000),
        _hud(40, fl=None, fr=1000, el=None, er=1000),  # friendly_left occluded at end
    ]
    assert _training_target(_interaction(window)) is None
