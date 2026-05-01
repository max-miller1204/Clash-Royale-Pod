"""Tests for the KataCR class-name → CardPlay boundary."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from crpod.constants import CARD_COSTS
from crpod.detection.cards import (
    _KNOWN_UNCONFIRMED_COSTS,
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
        assert play.side == Side.FRIENDLY  # cy=600 > RIVER_Y=480
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
    def test_every_alias_target_has_known_cost(self):
        """Every canonical target in KATACR_TO_CARD must have a row in
        CARD_COSTS. The historical 'unconfirmed' escape hatch is gone."""
        bad = [
            f"{katacr_name!r} -> {canonical!r}"
            for katacr_name, canonical in KATACR_TO_CARD.items()
            if canonical not in CARD_COSTS
        ]
        assert not bad, "KATACR_TO_CARD has aliases pointing nowhere:\n  " + "\n  ".join(bad)

    def test_known_unconfirmed_costs_is_empty(self):
        """Sentinel: the set must stay empty. If a new card appears whose
        cost is genuinely unknown, add it to CARD_COSTS with a sourced
        value rather than re-populating this set."""
        assert frozenset() == _KNOWN_UNCONFIRMED_COSTS

    def test_no_overlap_between_card_and_non_card(self):
        overlap = set(KATACR_TO_CARD.keys()) & KATACR_NON_CARD
        assert not overlap, f"classes in both card and non-card sets: {overlap}"

    @pytest.mark.parametrize(
        ("card", "expected_cost"),
        [
            ("boss_bandit", 6),
            ("rune_giant", 4),
            ("spirit_empress", 6),
            ("terry", 4),
            ("mirror", 1),
        ],
    )
    def test_wave_1b_costs_are_present(self, card, expected_cost):
        """Costs sourced in the Wave 1B commit (see commit message for
        per-card citations). Spirit Empress stores the headline 6-elixir
        cost; Mirror stores the +1 surcharge from spells_other.csv."""
        assert card in CARD_COSTS, f"{card} missing from CARD_COSTS"
        assert CARD_COSTS[card] == expected_cost, (
            f"{card} cost mismatch: CARD_COSTS={CARD_COSTS[card]} expected={expected_cost}"
        )

    @pytest.mark.parametrize("subunit", ["goblin-brawler", "royal-guardian"])
    def test_spawned_subunits_are_non_card(self, subunit):
        """Goblin Brawler (spawned by Goblin Cage) and Royal Guardian
        (spawned by Little Prince's Royal Rescue ability) are not card
        plays — they should be filtered out alongside golemite/lava-pup/
        hog rather than producing CardPlay events."""
        assert subunit in KATACR_NON_CARD
        assert subunit not in KATACR_TO_CARD


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
