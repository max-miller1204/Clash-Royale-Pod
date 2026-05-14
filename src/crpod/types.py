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
    """Per-frame HUD reading.

    Tower-HP fields are flat (left/right rather than tuples) so the EV target
    builder in `crpod.features.ev_target` can name each tower in its output
    dict without index ceremony. King HP is read separately because
    `HudRegions.{friendly,enemy}_king` is still rough — see `docs/TODO.md`.
    """

    frame: int
    friendly_elixir: float
    enemy_elixir: float | None
    friendly_king_hp: int | None = None
    enemy_king_hp: int | None = None
    friendly_left_princess_hp: int | None = None
    friendly_right_princess_hp: int | None = None
    enemy_left_princess_hp: int | None = None
    enemy_right_princess_hp: int | None = None


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
    # Per-side, per-tower princess HP swing across the interaction window.
    # Keys: "friendly_left", "friendly_right", "enemy_left", "enemy_right".
    # Value sign convention: end_hp - start_hp (negative = HP lost).
    # `None` for any tower whose start- or end-frame HUD reading was unreadable.
    tower_hp_delta: dict[str, int | None] = field(default_factory=dict)
    # Absolute princess HP at the start-frame HUD bookend, per tower. `None`
    # for towers whose start-frame reading was unreadable. Wave 2F additive
    # context for the EV feature builder; pre-2F call sites that don't pass
    # HUD continue to leave these unset.
    start_friendly_left_princess_hp: int | None = None
    start_friendly_right_princess_hp: int | None = None
    start_enemy_left_princess_hp: int | None = None
    start_enemy_right_princess_hp: int | None = None
    # Wave 2J': pre-window HP-swing context. Sum of princess-tower HP delta
    # across the 30 s preceding the interaction's start_frame (start_HP minus
    # lookback_HP, so negative = the side conceded HP in the lookback window).
    # `None` when fps was not provided to build_interactions, when the lookback
    # frame falls before the start of the replay, or when either bookend HUD
    # read is unreadable.
    pre_window_friendly_hp_delta_30s: int | None = None
    pre_window_enemy_hp_delta_30s: int | None = None
    # Wave 2J': start time in seconds, derived from start_frame / fps. Lets
    # `interaction_features` compute time_pressure_mode without needing access
    # to the Replay. `None` when fps was not provided.
    start_seconds: float | None = None

    @property
    def elixir_trade(self) -> int:
        """Positive = we came out ahead on elixir."""
        return self.enemy_elixir_spent - self.friendly_elixir_spent


@dataclass(frozen=True)
class Blunder:
    """A play whose predicted EV sits more than 1σ below the per-card median.

    Emitted by `crpod.analysis.blunders.detect_blunders`. `sigma_below` is
    the unsigned magnitude in standard deviations: 2.3 means "the model
    predicts an outcome 2.3σ worse than this card's typical outcome from
    the training fold." Always positive — non-blunder plays are filtered
    out before construction.
    """

    play_idx: int
    card: str
    ev_predicted: float
    per_card_median: float
    sigma_below: float
