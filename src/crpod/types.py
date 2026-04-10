"""Shared dataclasses for pipeline stages.

These are the interface contracts between stages. Adding a field here is a
breaking change — prefer extending.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Side(StrEnum):
    FRIENDLY = "friendly"
    ENEMY = "enemy"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CardPlay:
    """A single card placement event in a match."""

    frame: int
    card: str
    x: int
    y: int
    side: Side = Side.UNKNOWN
    elixir_cost: int = 0


@dataclass(frozen=True)
class HudState:
    """Per-frame HUD reading."""

    frame: int
    friendly_elixir: float
    enemy_elixir: float | None
    friendly_king_hp: int | None
    enemy_king_hp: int | None
    friendly_princess_hp: tuple[int | None, int | None] = (None, None)
    enemy_princess_hp: tuple[int | None, int | None] = (None, None)


@dataclass
class Replay:
    """A full match replay — the unit the pipeline consumes."""

    replay_id: str
    arena: str
    plays: list[CardPlay] = field(default_factory=list)
    hud: list[HudState] = field(default_factory=list)
    total_frames: int = 0
    fps: float = 10.0


@dataclass(frozen=True)
class Interaction:
    """A time-windowed interaction: your plays vs theirs, with outcomes."""

    start_frame: int
    end_frame: int
    friendly_plays: tuple[CardPlay, ...]
    enemy_plays: tuple[CardPlay, ...]
    friendly_elixir_spent: int
    enemy_elixir_spent: int
    damage_dealt: int = 0
    damage_taken: int = 0

    @property
    def elixir_trade(self) -> int:
        """Positive = we came out ahead on elixir."""
        return self.enemy_elixir_spent - self.friendly_elixir_spent
