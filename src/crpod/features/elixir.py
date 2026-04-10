"""Elixir accounting — tempo, leak, per-side ledger.

Pure functions over CardPlay lists. No external deps so it's trivially
testable without the ML stack.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from crpod.constants import (
    ELIXIR_REGEN_FRAMES_PER_UNIT,
    MAX_ELIXIR,
    card_cost,
)
from crpod.types import CardPlay, Side


@dataclass
class ElixirLedger:
    """Tracks a single side's elixir bar over the course of a match.

    Elixir regenerates at 1 unit per `ELIXIR_REGEN_FRAMES_PER_UNIT` frames
    (double/triple speed phases handled by the caller via `regen_multiplier`).
    """

    start_elixir: float = 5.0
    regen_multiplier: float = 1.0
    _last_frame: int = 0
    _level: float = 5.0
    _leaked: float = 0.0
    _spent: int = 0

    def __post_init__(self) -> None:
        self._level = self.start_elixir

    def advance_to(self, frame: int) -> None:
        if frame <= self._last_frame:
            return
        elapsed = frame - self._last_frame
        regen = (elapsed / ELIXIR_REGEN_FRAMES_PER_UNIT) * self.regen_multiplier
        new_level = self._level + regen
        if new_level > MAX_ELIXIR:
            self._leaked += new_level - MAX_ELIXIR
            new_level = MAX_ELIXIR
        self._level = new_level
        self._last_frame = frame

    def spend(self, play: CardPlay) -> None:
        self.advance_to(play.frame)
        cost = play.elixir_cost or card_cost(play.card)
        self._level = max(0.0, self._level - cost)
        self._spent += cost

    @property
    def leaked(self) -> float:
        return self._leaked

    @property
    def spent(self) -> int:
        return self._spent

    @property
    def level(self) -> float:
        return self._level


def elixir_leak(plays: Iterable[CardPlay], total_frames: int, side: Side) -> float:
    """Total elixir wasted by `side` across the match."""
    ledger = ElixirLedger()
    for p in plays:
        if p.side is side:
            ledger.spend(p)
    ledger.advance_to(total_frames)
    return ledger.leaked


def running_tempo(plays: list[CardPlay]) -> list[tuple[int, int]]:
    """Returns [(frame, friendly_spent - enemy_spent)] — positive = we're spending faster.

    'Tempo' in the pod_summary sense is the cumulative elixir differential;
    being positive means you're forcing the opponent to react.
    """
    out: list[tuple[int, int]] = []
    diff = 0
    for p in sorted(plays, key=lambda x: x.frame):
        cost = p.elixir_cost or card_cost(p.card)
        if p.side is Side.FRIENDLY:
            diff += cost
        elif p.side is Side.ENEMY:
            diff -= cost
        out.append((p.frame, diff))
    return out
