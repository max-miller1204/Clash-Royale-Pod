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

# Historically held card names whose elixir cost Supercell hadn't yet
# published. All entries have since been resolved (see CARD_COSTS) or
# reclassified as spawned subunits in KATACR_NON_CARD. Kept as a named
# (empty) constant for the validity test in tests/test_cards.py.
_KNOWN_UNCONFIRMED_COSTS: Final[frozenset[str]] = frozenset()

# KataCR class name -> canonical CARD_COSTS key. Generated against
# ``katacr_classes.txt`` (derived from upstream KataCR's public
# ClashRoyale_detection.yaml). Covers all 151 real classes; the 50
# pad_* placeholders go in KATACR_NON_CARD.
KATACR_TO_CARD: dict[str, str] = {
    # the- prefix
    "the-log": "log",
    # Singular -> plural (KataCR labels each unit individually)
    "skeleton": "skeletons",
    "goblin": "goblins",
    "spear-goblin": "spear_goblins",
    "bat": "bats",
    "barbarian": "barbarians",
    "wall-breaker": "wall_breakers",
    "archer": "archers",
    "minion": "minions",
    "royal-recruit": "royal_recruits",
    "guard": "guards",
    "skeleton-dragon": "skeleton_dragons",
    "zappy": "zappies",
    "elite-barbarian": "elite_barbarians",
    "royal-hog": "royal_hogs",
    # Multi-state classes -> single base card
    "elixir-golem-big": "elixir_golem",
    "elixir-golem-mid": "elixir_golem",
    "elixir-golem-small": "elixir_golem",
    "phoenix-big": "phoenix",
    "phoenix-egg": "phoenix",
    "phoenix-small": "phoenix",
    "rascal-boy": "rascals",
    "rascal-girl": "rascals",
    # Punctuation differences
    "x-bow": "xbow",
    # Evolution variants -> base card (evolutions share the base cost)
    "archer-evolution": "archers",
    "barbarian-evolution": "barbarians",
    "bat-evolution": "bats",
    "bomber-evolution": "bomber",
    "firecracker-evolution": "firecracker",
    "ice-spirit-evolution": "ice_spirit",
    "knight-evolution": "knight",
    "mortar-evolution": "mortar",
    "royal-giant-evolution": "royal_giant",
    "royal-recruit-evolution": "royal_recruits",
    "skeleton-evolution": "skeletons",
    "tesla-evolution": "tesla",
    "valkyrie-evolution": "valkyrie",
    "wall-breaker-evolution": "wall_breakers",
    "mirror": "mirror",
    # Standard cards (kebab-case in KataCR -> underscore in CARD_COSTS)
    "archer-queen": "archer_queen",
    "arrows": "arrows",
    "baby-dragon": "baby_dragon",
    "balloon": "balloon",
    "bandit": "bandit",
    "barbarian-barrel": "barbarian_barrel",
    "barbarian-hut": "barbarian_hut",
    "battle-healer": "battle_healer",
    "battle-ram": "battle_ram",
    "bomb-tower": "bomb_tower",
    "bomber": "bomber",
    "bowler": "bowler",
    "cannon": "cannon",
    "cannon-cart": "cannon_cart",
    "clone": "clone",
    "dark-prince": "dark_prince",
    "dart-goblin": "dart_goblin",
    "earthquake": "earthquake",
    "electro-dragon": "electro_dragon",
    "electro-giant": "electro_giant",
    "electro-spirit": "electro_spirit",
    "electro-wizard": "electro_wizard",
    "elixir-collector": "elixir_collector",
    "executioner": "executioner",
    "fire-spirit": "fire_spirit",
    "fireball": "fireball",
    "firecracker": "firecracker",
    "fisherman": "fisherman",
    "flying-machine": "flying_machine",
    "freeze": "freeze",
    "furnace": "furnace",
    "giant": "giant",
    "giant-skeleton": "giant_skeleton",
    "giant-snowball": "giant_snowball",
    "goblin-barrel": "goblin_barrel",
    "goblin-cage": "goblin_cage",
    "goblin-drill": "goblin_drill",
    "goblin-giant": "goblin_giant",
    "goblin-hut": "goblin_hut",
    "golden-knight": "golden_knight",
    "golem": "golem",
    "graveyard": "graveyard",
    "heal-spirit": "heal_spirit",
    "hog-rider": "hog_rider",
    "hunter": "hunter",
    "ice-golem": "ice_golem",
    "ice-spirit": "ice_spirit",
    "ice-wizard": "ice_wizard",
    "inferno-dragon": "inferno_dragon",
    "inferno-tower": "inferno_tower",
    "knight": "knight",
    "lava-hound": "lava_hound",
    "lightning": "lightning",
    "little-prince": "little_prince",
    "lumberjack": "lumberjack",
    "magic-archer": "magic_archer",
    "mega-knight": "mega_knight",
    "mega-minion": "mega_minion",
    "mighty-miner": "mighty_miner",
    "miner": "miner",
    "mini-pekka": "mini_pekka",
    "monk": "monk",
    "mortar": "mortar",
    "mother-witch": "mother_witch",
    "musketeer": "musketeer",
    "night-witch": "night_witch",
    "pekka": "pekka",
    "poison": "poison",
    "prince": "prince",
    "princess": "princess",
    "rage": "rage",
    "ram-rider": "ram_rider",
    "rocket": "rocket",
    "royal-delivery": "royal_delivery",
    "royal-ghost": "royal_ghost",
    "royal-giant": "royal_giant",
    "skeleton-barrel": "skeleton_barrel",
    "skeleton-king": "skeleton_king",
    "sparky": "sparky",
    "tesla": "tesla",
    "tombstone": "tombstone",
    "tornado": "tornado",
    "valkyrie": "valkyrie",
    "witch": "witch",
    "wizard": "wizard",
    "zap": "zap",
}

# KataCR class names that are not card placements: towers, HP bars,
# UI elements, projectiles, mid-fight spawn-derived units (a `goblin`
# from a goblin-barrel is the unit, not a fresh card play), and the
# `pad_*` placeholders KataCR included in their yaml for class-id
# alignment.
KATACR_NON_CARD: frozenset[str] = frozenset(
    {
        # HUD / cosmetic / UI
        "axe",
        "bar",
        "bar-level",
        "bomb",
        "cannoneer-tower",
        "clock",
        "dirt",
        "elixir",
        "emote",
        "evolution-symbol",
        "goblin-ball",
        # Spawned subunits (cage/ability outputs, not card plays) —
        # same pattern as golemite/lava-pup/hog below.
        "goblin-brawler",
        "golemite",
        "hog",
        "royal-guardian",
        "ice-spirit-evolution-symbol",
        "king-tower",
        "king-tower-bar",
        "lava-pup",
        "queen-tower",
        "selected",
        "skeleton-king-bar",
        "skeleton-king-skill",
        "tesla-evolution-shock",
        "text",
        "tower-bar",
        # Padding placeholders (no training examples)
        "pad_0",
        "pad_1",
        "pad_2",
        "pad_3",
        "pad_4",
        "pad_5",
        "pad_6",
        "pad_7",
        "pad_8",
        "pad_9",
        "pad_10",
        "pad_11",
        "pad_12",
        "pad_13",
        "pad_14",
        "pad_15",
        "pad_16",
        "pad_17",
        "pad_18",
        "pad_19",
        "pad_20",
        "pad_21",
        "pad_22",
        "pad_23",
        "pad_24",
        "pad_25",
        "pad_26",
        "pad_27",
        "pad_28",
        "pad_29",
        "pad_30",
        "pad_31",
        "pad_32",
        "pad_33",
        "pad_34",
        "pad_35",
        "pad_36",
        "pad_37",
        "pad_38",
        "pad_39",
        "pad_40",
        "pad_41",
        "pad_42",
        "pad_43",
        "pad_44",
        "pad_45",
        "pad_46",
        "pad_47",
        "pad_48",
        "pad_belong",
    }
)


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
