from crpod.constants import CARD_COSTS, card_cost, normalize_card_name


def test_kebab_to_underscore_with_the_prefix():
    assert card_cost("the-log") == 2


def test_singular_to_plural_alias():
    assert card_cost("spear-goblin") == 2
    assert card_cost("skeleton") == 1
    assert card_cost("bat") == 2


def test_kebab_basic():
    assert card_cost("mini-pekka") == 4


def test_existing_underscore_form_still_resolves():
    assert card_cost("knight") == 3


def test_case_insensitive():
    assert card_cost("Knight") == 3


def test_unknown_falls_back_to_default():
    assert card_cost("nonexistent-card") == 3


def test_normalize_is_pure_and_returns_key_form():
    assert normalize_card_name("The-Log") == "log"
    assert normalize_card_name("Spear-Goblin") == "spear_goblins"
    assert normalize_card_name("  mini-pekka  ") == "mini_pekka"


def test_phoenix_states_all_map_to_base_card():
    base = CARD_COSTS["phoenix"]
    for state in ("phoenix-big", "phoenix-egg", "phoenix-small"):
        assert card_cost(state) == base


def test_xbow_alias():
    assert card_cost("x-bow") == CARD_COSTS["xbow"]


def test_summoned_units_inherit_parent_cost():
    assert "goblin_brawler" in CARD_COSTS
    assert "royal_guardian" in CARD_COSTS
    assert card_cost("goblin-brawler") == CARD_COSTS["goblin_cage"]
    assert card_cost("royal-guardian") == CARD_COSTS["little_prince"]
