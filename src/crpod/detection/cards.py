"""KataCR class-name → CardPlay boundary.

The trained YOLO model emits class strings in KataCR's convention
(hyphenated, often singular: ``the-log``, ``spear-goblin``). This module
translates them to the project's canonical underscore convention used
by ``CARD_COSTS``, and drops non-card classes (towers, HP bars,
projectiles, UI) before they become phantom CardPlay events.

Two parallel structures keep cost data and name aliasing independent:
``KATACR_TO_CARD`` aliases names; ``CARD_COSTS`` (in ``constants``)
keeps costs. Adding a card means one row in each.
"""

from __future__ import annotations

import logging
from typing import Final

from crpod.constants import RIVER_Y, card_cost
from crpod.detection.yolo import Detection
from crpod.types import CardPlay, Side

_log = logging.getLogger(__name__)
_warned: set[str] = set()

# Champions whose costs Supercell hasn't published yet. They may appear
# as values in KATACR_TO_CARD without a matching CARD_COSTS row.
_KNOWN_UNCONFIRMED_CHAMPIONS: Final[frozenset[str]] = frozenset(
    {
        "boss_bandit",
        "rune_giant",
        "spirit_empress",
        "terry",
    }
)

# KataCR class name -> canonical CARD_COSTS key. Populated in Task 9.
KATACR_TO_CARD: dict[str, str] = {}

# KataCR class names that are not card placements (towers, HP bars,
# projectiles, UI elements). Populated in Task 9.
KATACR_NON_CARD: frozenset[str] = frozenset()


def _infer_side(y: int) -> Side:
    if y < 0:
        return Side.UNKNOWN
    return Side.FRIENDLY if y >= RIVER_Y else Side.ENEMY


def _warn_unknown(cls: str) -> None:
    if cls in _warned:
        return
    _warned.add(cls)
    _log.warning(
        "Unknown KataCR class %r — likely a card missing from KATACR_TO_CARD "
        "or a non-card class missing from KATACR_NON_CARD. Detection dropped.",
        cls,
    )


def to_card_play(det: Detection) -> CardPlay | None:
    """Convert a raw detection to a CardPlay, or None if not a card play."""
    if det.cls in KATACR_TO_CARD:
        canonical = KATACR_TO_CARD[det.cls]
        cx, cy = det.center
        return CardPlay(
            frame=det.frame,
            card=canonical,
            x=int(cx),
            y=int(cy),
            side=_infer_side(int(cy)),
            elixir_cost=card_cost(canonical),
        )
    if det.cls in KATACR_NON_CARD:
        return None
    _warn_unknown(det.cls)
    return None
