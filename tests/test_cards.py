"""Tests for the KataCR class-name → CardPlay boundary."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from crpod.constants import CARD_COSTS
from crpod.detection.cards import (
    _KNOWN_UNCONFIRMED_CHAMPIONS,
    KATACR_NON_CARD,
    KATACR_TO_CARD,
    to_card_play,
)
from crpod.detection.yolo import Detection
from crpod.types import CardPlay, Side


def _det(cls: str, *, frame: int = 42, cx: float = 240.0, cy: float = 600.0) -> Detection:
    """Build a Detection centered at (cx, cy). FRIENDLY side if cy >= 405."""
    half = 25.0
    return Detection(
        frame=frame,
        cls=cls,
        confidence=0.9,
        xyxy=(cx - half, cy - half, cx + half, cy + half),
    )


KATACR_CLASSES_FILE = (
    Path(__file__).parent.parent / "src" / "crpod" / "detection" / "katacr_classes.txt"
)


def _load_snapshot() -> list[str]:
    if not KATACR_CLASSES_FILE.exists():
        pytest.skip(
            f"{KATACR_CLASSES_FILE} not present — run "
            "scripts/dump_katacr_classes.py on a machine with the "
            "trained weights to generate it."
        )
    return [line.strip() for line in KATACR_CLASSES_FILE.read_text().splitlines() if line.strip()]


class TestToCardPlay:
    def test_known_card_returns_cardplay_with_canonical_name(self, monkeypatch):
        monkeypatch.setitem(KATACR_TO_CARD, "the-log", "log")
        play = to_card_play(_det("the-log", cy=600.0))
        assert isinstance(play, CardPlay)
        assert play.card == "log"
        assert play.elixir_cost == 2  # log is 2 elixir in CARD_COSTS
        assert play.frame == 42
        assert play.side == Side.FRIENDLY  # cy=600 > RIVER_Y=405
        assert play.x == 240
        assert play.y == 600

    def test_known_card_enemy_side(self, monkeypatch):
        monkeypatch.setitem(KATACR_TO_CARD, "the-log", "log")
        play = to_card_play(_det("the-log", cy=200.0))
        assert play is not None
        assert play.side == Side.ENEMY

    def test_non_card_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "crpod.detection.cards.KATACR_NON_CARD",
            frozenset({"king-tower"}),
        )
        assert to_card_play(_det("king-tower")) is None

    def test_unknown_class_returns_none_and_warns_once(self, monkeypatch, caplog):
        # Ensure the class is not classified as either card or non-card.
        monkeypatch.setitem(KATACR_TO_CARD, "the-log", "log")  # unrelated entry
        monkeypatch.setattr("crpod.detection.cards.KATACR_NON_CARD", frozenset())
        # Reset the warn-once dedup set so the test is hermetic.
        monkeypatch.setattr("crpod.detection.cards._warned", set())

        with caplog.at_level(logging.WARNING, logger="crpod.detection.cards"):
            assert to_card_play(_det("brand-new-class")) is None
            assert to_card_play(_det("brand-new-class")) is None  # dedup

        warnings = [r for r in caplog.records if "brand-new-class" in r.getMessage()]
        assert len(warnings) == 1, f"expected one warning, got {len(warnings)}"


class TestMappingValidity:
    def test_every_alias_target_has_known_cost_or_is_unconfirmed_champion(self):
        """Each value in KATACR_TO_CARD must be a key in CARD_COSTS, OR one
        of the 4 champions whose costs Supercell hasn't published yet."""
        bad = []
        for katacr_name, canonical in KATACR_TO_CARD.items():
            if canonical in CARD_COSTS:
                continue
            if canonical in _KNOWN_UNCONFIRMED_CHAMPIONS:
                continue
            bad.append(f"{katacr_name!r} -> {canonical!r}")
        assert not bad, "KATACR_TO_CARD has aliases pointing nowhere:\n  " + "\n  ".join(bad)

    def test_no_overlap_between_card_and_non_card(self):
        overlap = set(KATACR_TO_CARD.keys()) & KATACR_NON_CARD
        assert not overlap, f"classes in both card and non-card sets: {overlap}"


class TestSnapshotCoverage:
    def test_every_snapshot_class_is_mapped(self):
        classes = set(_load_snapshot())
        classified = set(KATACR_TO_CARD.keys()) | KATACR_NON_CARD
        unclassified = classes - classified
        assert not unclassified, (
            "snapshot has classes not in KATACR_TO_CARD or KATACR_NON_CARD:\n  "
            + "\n  ".join(sorted(unclassified))
        )

    def test_no_mapping_entry_for_class_outside_snapshot(self):
        classes = set(_load_snapshot())
        classified = set(KATACR_TO_CARD.keys()) | KATACR_NON_CARD
        stale = classified - classes
        assert not stale, (
            "mapping has entries not in the snapshot (stale or typo):\n  "
            + "\n  ".join(sorted(stale))
        )
