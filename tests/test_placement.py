from crpod.constants import ARENA_W, RIVER_Y
from crpod.features.placement import PlacementZone, zone_of


def test_enemy_back_for_low_y():
    assert zone_of(ARENA_W // 2, 10) is PlacementZone.ENEMY_BACK


def test_enemy_bridge_region():
    assert zone_of(ARENA_W // 2, RIVER_Y - 10) is PlacementZone.ENEMY_BRIDGE


def test_friendly_bridge_region():
    assert zone_of(ARENA_W // 2, RIVER_Y + 10) is PlacementZone.FRIENDLY_BRIDGE


def test_friendly_back_center():
    assert zone_of(ARENA_W // 2, 800) is PlacementZone.CENTER
