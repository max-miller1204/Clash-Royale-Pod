"""Game constants — card costs, arena dimensions, etc.

The HF replay dataset uses a 480x810 play area (x in [-1,479], y in [-1,810]).
Card elixir costs are base deploy costs, sourced from Supercell's
`spells_characters`/`spells_buildings`/`spells_other` CSVs (via the
cr-csv mirror) with post-2023 additions and rebalances applied by hand.
Evolution variants share the base card's cost and so are not listed
separately. Unknown cards fall back to `default` in `card_cost`, which
is a known source of bias in EV calculations — prefer adding the card
here over relying on the fallback.
"""

from __future__ import annotations

ARENA_W: int = 480
ARENA_H: int = 810
RIVER_Y: int = ARENA_H // 2
BRIDGE_LEFT_X: int = 90
BRIDGE_RIGHT_X: int = 390

MAX_ELIXIR: int = 10
ELIXIR_REGEN_FRAMES_PER_UNIT: int = 28

CARD_COSTS: dict[str, int] = {
    # Common troops
    "knight": 3,
    "archers": 3,
    "bomber": 2,
    "goblins": 2,
    "spear_goblins": 2,
    "minions": 3,
    "minion_horde": 5,
    "skeletons": 1,
    "bats": 2,
    "barbarians": 5,
    "elite_barbarians": 6,
    "royal_giant": 6,
    "royal_recruits": 7,
    "royal_hogs": 5,
    "firecracker": 3,
    "dart_goblin": 3,
    "ice_spirit": 1,
    "fire_spirit": 2,
    "electro_spirit": 1,
    "heal_spirit": 1,
    "rascals": 5,
    "skeleton_barrel": 3,
    "skeleton_dragons": 4,
    "suspicious_bush": 2,
    "berserker": 3,
    "goblin_demolisher": 4,
    # Rare troops
    "giant": 5,
    "musketeer": 4,
    "mini_pekka": 4,
    "valkyrie": 4,
    "hog_rider": 4,
    "wizard": 5,
    "three_musketeers": 9,
    "battle_ram": 4,
    "ice_golem": 2,
    "mega_minion": 3,
    "flying_machine": 4,
    "zappies": 4,
    "elixir_golem": 3,
    "battle_healer": 4,
    # Epic troops
    "pekka": 7,
    "prince": 5,
    "giant_skeleton": 6,
    "baby_dragon": 4,
    "skeleton_army": 3,
    "witch": 5,
    "balloon": 5,
    "golem": 8,
    "dark_prince": 4,
    "guards": 3,
    "goblin_gang": 3,
    "executioner": 5,
    "bowler": 5,
    "electro_dragon": 5,
    "cannon_cart": 5,
    "hunter": 4,
    "goblin_giant": 6,
    "electro_giant": 7,
    "wall_breakers": 2,
    "mega_knight": 7,
    # Legendary troops
    "ice_wizard": 3,
    "princess": 3,
    "lava_hound": 7,
    "miner": 3,
    "sparky": 6,
    "lumberjack": 4,
    "inferno_dragon": 4,
    "electro_wizard": 4,
    "night_witch": 4,
    "bandit": 3,
    "royal_ghost": 3,
    "ram_rider": 5,
    "magic_archer": 4,
    "fisherman": 3,
    "mother_witch": 4,
    "phoenix": 4,
    # Champions
    "mighty_miner": 4,
    "skeleton_king": 4,
    "archer_queen": 5,
    "golden_knight": 4,
    "monk": 5,
    "little_prince": 3,
    "goblin_machine": 5,
    "goblinstein": 5,
    # Buildings
    "cannon": 3,
    "tesla": 4,
    "inferno_tower": 5,
    "bomb_tower": 4,
    "mortar": 4,
    "xbow": 6,
    "elixir_collector": 6,
    "tombstone": 3,
    "furnace": 4,
    "goblin_hut": 5,
    "barbarian_hut": 6,
    "goblin_cage": 4,
    "goblin_drill": 4,
    # Spells
    "fireball": 4,
    "arrows": 3,
    "zap": 2,
    "log": 2,
    "rocket": 6,
    "lightning": 6,
    "poison": 4,
    "freeze": 4,
    "tornado": 3,
    "rage": 2,
    "clone": 3,
    "graveyard": 5,
    "earthquake": 3,
    "barbarian_barrel": 2,
    "giant_snowball": 2,
    "royal_delivery": 3,
    "goblin_barrel": 3,
    "goblin_curse": 3,
    "void": 3,
}


def card_cost(card: str, default: int = 3) -> int:
    return CARD_COSTS.get(card.lower().replace(" ", "_"), default)
