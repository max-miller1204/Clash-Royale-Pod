"""Placement zoning — discretize (x, y) into strategic regions."""

from __future__ import annotations

from enum import StrEnum

from crpod.constants import ARENA_W, BRIDGE_LEFT_X, BRIDGE_RIGHT_X, RIVER_Y


class PlacementZone(StrEnum):
    FRIENDLY_BACK = "friendly_back"
    FRIENDLY_BRIDGE = "friendly_bridge"
    ENEMY_BRIDGE = "enemy_bridge"
    ENEMY_BACK = "enemy_back"
    LEFT_LANE = "left_lane"
    RIGHT_LANE = "right_lane"
    CENTER = "center"


def zone_of(x: int, y: int) -> PlacementZone:
    """Map an (x, y) placement to a discrete zone.

    The pod_summary EV framework uses this as the multiplier key —
    bridge placements score differently than back placements.
    """
    if y < RIVER_Y // 2:
        return PlacementZone.ENEMY_BACK
    if y < RIVER_Y:
        return PlacementZone.ENEMY_BRIDGE
    if y < RIVER_Y + (RIVER_Y // 2):
        return PlacementZone.FRIENDLY_BRIDGE
    if x < BRIDGE_LEFT_X:
        return PlacementZone.LEFT_LANE
    if x > BRIDGE_RIGHT_X:
        return PlacementZone.RIGHT_LANE
    if abs(x - ARENA_W // 2) < 40:
        return PlacementZone.CENTER
    return PlacementZone.FRIENDLY_BACK
