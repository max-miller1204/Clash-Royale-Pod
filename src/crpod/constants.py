"""Game constants — card costs, arena dimensions, etc.

The HF replay dataset uses a 480x810 play area (x in [-1,479], y in [-1,810]).
Card elixir costs sourced from the public Clash Royale wiki; expand as needed.
"""

from __future__ import annotations

ARENA_W: int = 480
ARENA_H: int = 810
RIVER_Y: int = ARENA_H // 2
BRIDGE_LEFT_X: int = 90
BRIDGE_RIGHT_X: int = 390

MAX_ELIXIR: int = 10
ELIXIR_REGEN_FRAMES_PER_UNIT: int = 28

# Non-exhaustive — extend as the pipeline encounters unknown cards.
# Values are base elixir costs.
CARD_COSTS: dict[str, int] = {
    "knight": 3,
    "archers": 3,
    "goblins": 2,
    "spear_goblins": 2,
    "minions": 3,
    "minion_horde": 5,
    "skeletons": 1,
    "bats": 2,
    "barbarians": 5,
    "giant": 5,
    "hog_rider": 4,
    "balloon": 5,
    "musketeer": 4,
    "mini_pekka": 4,
    "pekka": 7,
    "golem": 8,
    "lava_hound": 7,
    "valkyrie": 4,
    "witch": 5,
    "wizard": 5,
    "fireball": 4,
    "arrows": 3,
    "zap": 2,
    "log": 2,
    "rocket": 6,
    "lightning": 6,
    "poison": 4,
    "freeze": 4,
    "tornado": 3,
    "cannon": 3,
    "tesla": 4,
    "inferno_tower": 5,
    "bomb_tower": 4,
    "mortar": 4,
    "xbow": 6,
    "elixir_collector": 6,
    "prince": 5,
    "dark_prince": 4,
    "electro_wizard": 4,
    "ice_wizard": 3,
    "princess": 3,
    "miner": 3,
    "goblin_barrel": 3,
}


def card_cost(card: str, default: int = 3) -> int:
    return CARD_COSTS.get(card.lower().replace(" ", "_"), default)
