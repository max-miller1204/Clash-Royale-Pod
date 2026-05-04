from crpod.features.interactions import build_interactions
from crpod.types import CardPlay, HudState, Side


def _play(frame: int, card: str, side: Side, cost: int) -> CardPlay:
    return CardPlay(frame=frame, card=card, x=200, y=400, side=side, elixir_cost=cost)


def _hud(
    frame: int,
    *,
    fl: int | None = 1000,
    fr: int | None = 1000,
    el: int | None = 1000,
    er: int | None = 1000,
) -> HudState:
    return HudState(
        frame=frame,
        friendly_elixir=5.0,
        enemy_elixir=5.0,
        friendly_left_princess_hp=fl,
        friendly_right_princess_hp=fr,
        enemy_left_princess_hp=el,
        enemy_right_princess_hp=er,
    )


def test_hog_tesla_single_interaction():
    plays = [
        _play(0, "hog_rider", Side.FRIENDLY, 4),
        _play(8, "tesla", Side.ENEMY, 4),
    ]
    interactions = build_interactions(plays)
    assert len(interactions) == 1
    i = interactions[0]
    assert i.friendly_elixir_spent == 4
    assert i.enemy_elixir_spent == 4
    assert i.elixir_trade == 0


def test_positive_trade_when_opponent_overcommits():
    plays = [
        _play(0, "skeletons", Side.FRIENDLY, 1),
        _play(5, "fireball", Side.ENEMY, 4),
    ]
    interactions = build_interactions(plays)
    assert interactions[0].elixir_trade == 3


def test_empty_input_returns_empty():
    assert build_interactions([]) == []


def test_distant_plays_split_into_separate_interactions():
    plays = [
        _play(0, "hog_rider", Side.FRIENDLY, 4),
        _play(200, "hog_rider", Side.FRIENDLY, 4),
    ]
    assert len(build_interactions(plays)) == 2


# Wave 2J': pre-window HP delta + start_seconds.


def test_pre_window_delta_none_when_fps_missing():
    plays = [_play(400, "hog_rider", Side.FRIENDLY, 4)]
    hud = [_hud(400)]
    [interaction] = build_interactions(plays, hud=hud)
    assert interaction.pre_window_friendly_hp_delta_30s is None
    assert interaction.pre_window_enemy_hp_delta_30s is None
    assert interaction.start_seconds is None


def test_pre_window_delta_none_when_window_starts_before_replay():
    # fps=10 → lookback_frames=300; anchor.frame=100 → lookback_frame=-200 < 0
    plays = [_play(100, "hog_rider", Side.FRIENDLY, 4)]
    hud = [_hud(100, fl=900, fr=900, el=900, er=900)]
    [interaction] = build_interactions(plays, hud=hud, fps=10.0)
    assert interaction.pre_window_friendly_hp_delta_30s is None
    assert interaction.pre_window_enemy_hp_delta_30s is None
    # start_seconds still derives even when the lookback path bails.
    assert interaction.start_seconds == 10.0


def test_pre_window_delta_correct_sign_for_synthetic_drop():
    # Friendly side: start=1000, lookback (sum)=1500 → delta = -500.
    # Enemy side: start=1000, lookback (sum)=900 → delta = +100.
    plays = [_play(400, "hog_rider", Side.FRIENDLY, 4)]
    hud = [
        _hud(100, fl=750, fr=750, el=450, er=450),
        _hud(400, fl=500, fr=500, el=500, er=500),
    ]
    [interaction] = build_interactions(plays, hud=hud, fps=10.0)
    assert interaction.pre_window_friendly_hp_delta_30s == -500
    assert interaction.pre_window_enemy_hp_delta_30s == 100


def test_pre_window_delta_none_when_lookback_hud_unreadable():
    # Friendly lookback bookend has unreadable left tower → friendly delta None.
    # Enemy lookback bookend is fully readable → enemy delta computes.
    plays = [_play(400, "hog_rider", Side.FRIENDLY, 4)]
    hud = [
        _hud(100, fl=None, fr=750, el=400, er=400),
        _hud(400, fl=500, fr=500, el=300, er=300),
    ]
    [interaction] = build_interactions(plays, hud=hud, fps=10.0)
    assert interaction.pre_window_friendly_hp_delta_30s is None
    assert interaction.pre_window_enemy_hp_delta_30s == (300 + 300) - (400 + 400)


def test_start_seconds_populates_from_fps():
    plays = [_play(200, "hog_rider", Side.FRIENDLY, 4)]
    [interaction] = build_interactions(plays, fps=10.0)
    assert interaction.start_seconds == 20.0
