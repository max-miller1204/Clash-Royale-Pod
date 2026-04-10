from crpod.features.elixir import ElixirLedger, elixir_leak, running_tempo
from crpod.types import CardPlay, Side


def _play(frame: int, card: str, side: Side, cost: int) -> CardPlay:
    return CardPlay(frame=frame, card=card, x=200, y=400, side=side, elixir_cost=cost)


def test_ledger_spends_and_caps_at_ten():
    ledger = ElixirLedger(start_elixir=10.0)
    ledger.spend(_play(0, "fireball", Side.FRIENDLY, 4))
    assert ledger.level == 6.0
    assert ledger.spent == 4


def test_ledger_leaks_when_overfilled():
    ledger = ElixirLedger(start_elixir=10.0)
    ledger.advance_to(1000)
    assert ledger.leaked > 0
    assert ledger.level == 10.0


def test_tempo_cumulative_differential():
    plays = [
        _play(0, "hog_rider", Side.FRIENDLY, 4),
        _play(10, "tesla", Side.ENEMY, 4),
        _play(50, "fireball", Side.FRIENDLY, 4),
    ]
    tempo = running_tempo(plays)
    assert tempo[-1][1] == 4
    assert tempo[1][1] == 0


def test_elixir_leak_per_side():
    plays = [
        _play(0, "skeletons", Side.FRIENDLY, 1),
        _play(5, "goblins", Side.ENEMY, 2),
    ]
    friendly_leak = elixir_leak(plays, total_frames=5000, side=Side.FRIENDLY)
    enemy_leak = elixir_leak(plays, total_frames=5000, side=Side.ENEMY)
    assert friendly_leak > 0
    assert enemy_leak > 0
    assert friendly_leak > enemy_leak
