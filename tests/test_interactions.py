from crpod.features.interactions import build_interactions
from crpod.types import CardPlay, Side


def _play(frame: int, card: str, side: Side, cost: int) -> CardPlay:
    return CardPlay(frame=frame, card=card, x=200, y=400, side=side, elixir_cost=cost)


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
