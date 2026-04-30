# KataCR Name Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Translate the trained YOLO model's KataCR class strings into canonical `CARD_COSTS` keys, and drop non-card detections (towers, HP bars, projectiles, UI) before they become phantom 3-cost `CardPlay` events.

**Architecture:** New module `src/crpod/detection/cards.py` owns the boundary. Two parallel data structures — `KATACR_TO_CARD: dict[str, str]` (alias map) and `KATACR_NON_CARD: frozenset[str]` (filter set) — feed a single `to_card_play(det) -> CardPlay | None` helper that replaces the existing private `_detection_to_card_play` in `huggingface.py`. Unknown classes warn-once and return `None`. A checked-in snapshot `katacr_classes.txt` (dumped from `crpod_v1_best.pt`'s `model.names`) anchors a coverage test that fails if classes are added/removed without the mapping being updated.

**Tech Stack:** Python 3.11, dataclasses, ultralytics (only in dump script), pytest, ruff. No new runtime dependencies.

**Source spec:** `docs/superpowers/specs/2026-04-30-katacr-name-mapping-design.md` (commit `4c71fdb`).

**Phase split:** Tasks 1–7 land the structural change locally without needing the trained `.pt`. Task 8 is a one-shot dump on a remote machine that has the weights. Task 9 fills in the full mapping data once the snapshot exists.

---

## File Structure

**Create:**
- `src/crpod/detection/cards.py` — boundary module: `KATACR_TO_CARD`, `KATACR_NON_CARD`, `_KNOWN_UNCONFIRMED_CHAMPIONS`, `_warn_unknown`, `_infer_side`, `to_card_play`.
- `tests/test_cards.py` — coverage, validity, behavioral, warn-once tests.
- `scripts/dump_katacr_classes.py` — one-shot tool to extract `model.names` from a trained `.pt` and write `src/crpod/detection/katacr_classes.txt`.
- `src/crpod/detection/katacr_classes.txt` — checked-in snapshot of class names (Phase 2; absent until Task 8 runs).

**Modify:**
- `src/crpod/dataset/huggingface.py` — remove `_infer_side` (lines 30–33) and `_detection_to_card_play` (lines 36–45); update `_parquet_to_replay` (line 125) to use `to_card_play` and filter `None`s.

---

## Conventions

- Run tests: `uv run pytest tests/test_cards.py -v`
- Run all tests: `uv run pytest -q`
- Format: `uv run ruff format <files>` (CI enforces this)
- Lint: `uv run ruff check <files>`
- Type check: `uv run mypy src`
- Commit format: conventional (`feat:`, `refactor:`, `test:`, `docs:`, `chore:`).
- The 4 known-unconfirmed champions are: `boss_bandit`, `rune_giant`, `spirit_empress`, `terry`. They may appear in `KATACR_TO_CARD` values without a matching `CARD_COSTS` entry — the validity test must allow this.

---

### Task 1: Create the cards module skeleton

**Files:**
- Create: `src/crpod/detection/cards.py`

- [ ] **Step 1: Create the module with empty mappings and the typed signatures**

```python
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
_KNOWN_UNCONFIRMED_CHAMPIONS: Final[frozenset[str]] = frozenset({
    "boss_bandit",
    "rune_giant",
    "spirit_empress",
    "terry",
})

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
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `uv run python -c "from crpod.detection.cards import to_card_play, KATACR_TO_CARD, KATACR_NON_CARD; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Format and lint**

Run: `uv run ruff format src/crpod/detection/cards.py && uv run ruff check src/crpod/detection/cards.py`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/crpod/detection/cards.py
git commit -m "feat: add empty cards module skeleton

KATACR_TO_CARD and KATACR_NON_CARD start empty; to_card_play returns
None for everything until Task 9 populates the mappings. Module
structure and type contracts land first so tests and the call-site
refactor can build on them."
```

---

### Task 2: Behavioral tests for `to_card_play`

**Files:**
- Create: `tests/test_cards.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the KataCR class-name → CardPlay boundary."""

from __future__ import annotations

import logging

import pytest

from crpod.detection.cards import (
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
        monkeypatch.setattr(
            "crpod.detection.cards.KATACR_NON_CARD", frozenset()
        )
        # Reset the warn-once dedup set so the test is hermetic.
        monkeypatch.setattr("crpod.detection.cards._warned", set())

        with caplog.at_level(logging.WARNING, logger="crpod.detection.cards"):
            assert to_card_play(_det("brand-new-class")) is None
            assert to_card_play(_det("brand-new-class")) is None  # dedup

        warnings = [r for r in caplog.records if "brand-new-class" in r.getMessage()]
        assert len(warnings) == 1, f"expected one warning, got {len(warnings)}"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cards.py -v`
Expected: tests pass for known-card behavior (we already implemented `to_card_play` in Task 1). If a test fails, fix `cards.py` not the test.

- [ ] **Step 3: If any test fails, fix `src/crpod/detection/cards.py`**

Common cause: `_infer_side` boundary off-by-one. The expected behavior: `y >= RIVER_Y` (= 405) means FRIENDLY (recorder's side, bottom half of a 810-tall arena), `y < RIVER_Y` means ENEMY, `y < 0` means UNKNOWN.

- [ ] **Step 4: Run all tests pass**

Run: `uv run pytest tests/test_cards.py -v`
Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_cards.py
git commit -m "test: behavioral tests for to_card_play

Covers known-card translation, side inference, non-card filtering,
and warn-once on unknown classes."
```

---

### Task 3: Validity test — every alias points to a real cost

**Files:**
- Modify: `tests/test_cards.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cards.py`:

```python
from crpod.constants import CARD_COSTS
from crpod.detection.cards import _KNOWN_UNCONFIRMED_CHAMPIONS


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
        assert not bad, (
            "KATACR_TO_CARD has aliases pointing nowhere:\n  "
            + "\n  ".join(bad)
        )

    def test_no_overlap_between_card_and_non_card(self):
        overlap = set(KATACR_TO_CARD.keys()) & KATACR_NON_CARD
        assert not overlap, f"classes in both card and non-card sets: {overlap}"
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_cards.py::TestMappingValidity -v`
Expected: both tests PASS (KATACR_TO_CARD is empty, so there's nothing to be invalid). They will gain teeth in Task 9 when entries are added.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cards.py
git commit -m "test: validity tests for KATACR mapping data

Every KATACR_TO_CARD value must be a CARD_COSTS key or a known
unconfirmed champion. Card and non-card sets must be disjoint."
```

---

### Task 4: Coverage test — snapshot ↔ mapping must match

**Files:**
- Modify: `tests/test_cards.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cards.py`:

```python
from pathlib import Path

KATACR_CLASSES_FILE = (
    Path(__file__).parent.parent
    / "src"
    / "crpod"
    / "detection"
    / "katacr_classes.txt"
)


def _load_snapshot() -> list[str]:
    if not KATACR_CLASSES_FILE.exists():
        pytest.skip(
            f"{KATACR_CLASSES_FILE} not present — run "
            "scripts/dump_katacr_classes.py on a machine with the "
            "trained weights to generate it."
        )
    return [
        line.strip()
        for line in KATACR_CLASSES_FILE.read_text().splitlines()
        if line.strip()
    ]


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
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_cards.py::TestSnapshotCoverage -v`
Expected: both tests SKIP with the message about running the dump script.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cards.py
git commit -m "test: snapshot coverage tests for KATACR mapping

Skipped until katacr_classes.txt is generated. Once present, asserts
the mapping covers every class exactly (no orphans, no stale rows)."
```

---

### Task 5: Wire `to_card_play` into the HF replay path

**Files:**
- Modify: `src/crpod/dataset/huggingface.py:30-45,125`

- [ ] **Step 1: Replace the inline helpers with the new boundary**

In `src/crpod/dataset/huggingface.py`, remove `_infer_side` (lines 30–33) and `_detection_to_card_play` (lines 36–45). Update the import block at the top of the file to drop unused symbols (`RIVER_Y`, `card_cost`) and add the new helper:

```python
# at the top, replace the existing crpod imports with:
from crpod.detection.cards import to_card_play
from crpod.detection.yolo import Detection, YoloDetector
from crpod.types import CardPlay, HudState, Replay, Side
```

(Drop `from crpod.constants import RIVER_Y, card_cost` entirely — both moved into `cards.py`.)

Then replace the body of `_parquet_to_replay` at line 125:

```python
def _parquet_to_replay(path: Path, arena: str, replay_id: str, detector: YoloDetector) -> Replay:
    detections = detector.infer(_decode_frames(path))
    plays = [p for p in (to_card_play(d) for d in detections) if p is not None]
    total_frames = max((p.frame for p in plays), default=0)
    return Replay(
        replay_id=replay_id,
        arena=arena,
        plays=plays,
        hud=[],
        total_frames=total_frames,
        fps=10.0,
    )
```

- [ ] **Step 2: Verify huggingface.py still imports cleanly**

Run: `uv run python -c "from crpod.dataset.huggingface import _parquet_to_replay; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Audit other imports of the removed symbols**

Run: `uv run grep -rn "_detection_to_card_play\|from crpod.dataset.huggingface import.*_infer_side" src tests scripts`
Expected: no matches (both helpers were file-private with `_` prefix).

If any matches exist, replace them with `from crpod.detection.cards import to_card_play` (or the right helper) and adjust the usage.

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass (the snapshot-coverage tests skip; everything else passes).

- [ ] **Step 5: Type check**

Run: `uv run mypy src`
Expected: no errors. If `Side`, `CardPlay`, or `HudState` show up as unused in the import block, drop them.

- [ ] **Step 6: Format**

Run: `uv run ruff format src/crpod/dataset/huggingface.py src/crpod/detection/cards.py tests/test_cards.py`

- [ ] **Step 7: Commit**

```bash
git add src/crpod/dataset/huggingface.py
git commit -m "refactor: route HF detections through to_card_play boundary

_parquet_to_replay now drops detections that aren't card plays
instead of constructing phantom CardPlay events from towers and
HP bars. With KATACR_TO_CARD empty, this temporarily produces zero
plays per replay — Task 9 populates the mapping."
```

---

### Task 6: One-shot dump script for the class snapshot

**Files:**
- Create: `scripts/dump_katacr_classes.py`

- [ ] **Step 1: Create the script**

```python
"""Dump the trained YOLO model's class names to a checked-in snapshot.

Run once on a machine that has output/models/crpod_v1_best.pt. The
output file (src/crpod/detection/katacr_classes.txt) is the canonical
list of class strings the deployed weights can emit. Tests assert
that KATACR_TO_CARD ∪ KATACR_NON_CARD covers it exactly.

Re-run when the model is retrained.

Usage:
    uv run python scripts/dump_katacr_classes.py
    uv run python scripts/dump_katacr_classes.py --weights other.pt --out other.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_WEIGHTS = Path("output/models/crpod_v1_best.pt")
DEFAULT_OUT = Path("src/crpod/detection/katacr_classes.txt")


def dump(weights: Path, out: Path) -> int:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = model.names  # dict[int, str] — keyed by class id, in order
    if not isinstance(names, dict):
        raise RuntimeError(f"expected model.names to be a dict, got {type(names)!r}")

    ordered = [names[i] for i in sorted(names.keys())]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ordered) + "\n")
    return len(ordered)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    if not args.weights.exists():
        raise SystemExit(
            f"weights not found: {args.weights}\n"
            "Run this script on a machine that has the trained .pt "
            "(e.g. the brev instance from scripts/brev_train.sh)."
        )

    n = dump(args.weights, args.out)
    print(f"wrote {n} class names to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it parses and shows --help**

Run: `uv run python scripts/dump_katacr_classes.py --help`
Expected: argparse usage printed; no import error from ultralytics (it's a soft import inside `dump`, so `--help` works without the .pt).

- [ ] **Step 3: Format and lint**

Run: `uv run ruff format scripts/dump_katacr_classes.py && uv run ruff check scripts/dump_katacr_classes.py`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/dump_katacr_classes.py
git commit -m "feat: dump_katacr_classes.py — extract model.names to snapshot

One-shot tool for refreshing src/crpod/detection/katacr_classes.txt
from the trained weights. Run on a machine that has the .pt; commit
the resulting text file."
```

---

### Task 7: Document the workflow + checkpoint

**Files:**
- Modify: `docs/TODO.md` — flip the relevant TODO entry

- [ ] **Step 1: Update docs/TODO.md**

Find the line:

```
Still TODO: name-mapping layer for KataCR class names (e.g. `the-log`, `spear-goblin`) → underscore convention (`log`, `spear_goblins`) so `card_cost()` lookups actually hit.
```

Replace with:

```
Name mapping landed in `src/crpod/detection/cards.py` (Phase 1). Phase 2 is populating `KATACR_TO_CARD` and `KATACR_NON_CARD` against the snapshot — run `scripts/dump_katacr_classes.py` on a machine with `output/models/crpod_v1_best.pt`, commit the resulting `src/crpod/detection/katacr_classes.txt`, then fill in the mapping rows until `pytest tests/test_cards.py` passes (no skips).
```

- [ ] **Step 2: Run the full test + typecheck once more**

Run: `uv run pytest -q && uv run mypy src && uv run ruff check`
Expected: tests pass (with snapshot-coverage tests skipped), no type errors, no lint errors.

- [ ] **Step 3: Commit**

```bash
git add docs/TODO.md
git commit -m "docs: update TODO — Phase 1 of KataCR name mapping landed"
```

**Checkpoint:** Phase 1 is complete. The boundary, tests, and call-site refactor are in. The HF replay path now produces zero plays until Task 8 + Task 9 populate the mapping data — this is the intended state and is honest about the underlying gap (previously it produced bogus 3-cost plays). Open a PR for Tasks 1–7 if working in a branch.

---

### Task 8: Generate the snapshot (remote machine)

**Files:**
- Create: `src/crpod/detection/katacr_classes.txt`

This task runs on a machine with `output/models/crpod_v1_best.pt`. From the local dev machine, this is typically a brev instance.

- [ ] **Step 1: Locate or provision a machine with the weights**

If the brev instance from `scripts/brev_train.sh` is still alive, ssh in. Otherwise, re-create one and rsync the weights from cloud storage (or re-train — ~$0.20).

- [ ] **Step 2: Run the dump script**

```bash
cd Clash-Royale-Pod
uv run python scripts/dump_katacr_classes.py
```

Expected: `wrote NNN class names to src/crpod/detection/katacr_classes.txt` (NNN should be ~150 per `pod_summary.md`).

- [ ] **Step 3: Bring the snapshot file back to the local machine**

Easiest: scp it back, or commit on the remote and pull on local.

- [ ] **Step 4: Verify locally**

Run: `wc -l src/crpod/detection/katacr_classes.txt`
Expected: ~150 lines.

- [ ] **Step 5: Run the previously-skipped coverage tests**

Run: `uv run pytest tests/test_cards.py::TestSnapshotCoverage -v`
Expected: tests now run (no skip). They will FAIL because `KATACR_TO_CARD` and `KATACR_NON_CARD` are still empty and the snapshot has ~150 entries. The failure message lists every unclassified class — that list is the input to Task 9.

- [ ] **Step 6: Commit the snapshot**

```bash
git add src/crpod/detection/katacr_classes.txt
git commit -m "chore: snapshot KataCR class names from crpod_v1_best.pt

Anchor file for the KATACR_TO_CARD / KATACR_NON_CARD coverage tests."
```

---

### Task 9: Populate the mapping

**Files:**
- Modify: `src/crpod/detection/cards.py` — fill in `KATACR_TO_CARD` and `KATACR_NON_CARD`.
- (Optional) Modify: `src/crpod/constants.py` — add any cards present in KataCR but missing from `CARD_COSTS`.

- [ ] **Step 1: Get the unclassified list**

Run: `uv run pytest tests/test_cards.py::TestSnapshotCoverage::test_every_snapshot_class_is_mapped -v`
Expected: failure with a sorted list of unclassified class names. Copy that list — it's your worklist.

- [ ] **Step 2: Triage each class into one of three buckets**

For each name in the worklist, decide:

- **Card** — anything that's a deck slot. Examples: `the-log`, `spear-goblin`, `mini-pekka`, `goblin-barrel`, `dark-prince`. Add a row to `KATACR_TO_CARD`. The value must be a key in `CARD_COSTS` (already exists or you add it). Examples of likely transformations: hyphen → underscore (`mini-pekka` → `mini_pekka`), and singular → plural where `CARD_COSTS` uses plural (`spear-goblin` → `spear_goblins`, `goblin` → `goblins`, `bat` → `bats`, `skeleton` → `skeletons`, `archer` → `archers`, `barbarian` → `barbarians`, `minion` → `minions`).
- **Non-card** — towers, HP bars, projectiles, UI elements, emote, evolution-cycle indicator. Examples: `king-tower`, `queen-tower`, `princess-tower`, `king-tower-bar`, `cannonball`, `arrow`, `emote`, `clock`. Add the name to `KATACR_NON_CARD`.
- **Card we don't have a `CARD_COSTS` row for** — add the row to `CARD_COSTS` first (with the canonical underscore name and Supercell-published cost), then add the alias to `KATACR_TO_CARD`. For the 4 known-unconfirmed champions (`boss_bandit`, `rune_giant`, `spirit_empress`, `terry`), add the alias only — the validity test allows this.

KataCR's spawn-derived classes (e.g. `goblin-from-barrel`, `skeleton-from-graveyard`) are *not* card plays — they're mid-fight units the detector sees after a card resolves. Put them in `KATACR_NON_CARD`. The `goblin_barrel` / `graveyard` cards themselves (the deploy event) are separate classes if KataCR labels them.

- [ ] **Step 3: Edit `src/crpod/detection/cards.py`**

Replace the empty initializers:

```python
KATACR_TO_CARD: dict[str, str] = {
    "the-log": "log",
    "spear-goblin": "spear_goblins",
    # ... (one row per card class from the worklist)
}

KATACR_NON_CARD: frozenset[str] = frozenset({
    "king-tower",
    "queen-tower",
    # ... (one entry per non-card class from the worklist)
})
```

Group entries with comment headers (`# Spells`, `# Buildings`, `# UI / HUD`) for readability — match the structure of `CARD_COSTS` in `constants.py`.

- [ ] **Step 4: Run the coverage tests**

Run: `uv run pytest tests/test_cards.py -v`
Expected: all tests pass — no skips, no failures.

If coverage fails, the failure message lists what's still unclassified. Iterate.
If validity fails, you typo'd a `CARD_COSTS` key — fix the alias target.

- [ ] **Step 5: Sanity check on real data**

Run: `uv run crpod analyze --weights output/models/crpod_v1_best.pt --arena arena_15 --replay 00a91415-...` (use a known-good replay ID).
Expected: the produced replay has a non-zero number of plays with realistic distribution of card names. Spot-check a handful by name; the elixir trade in interactions should no longer be uniformly biased.

- [ ] **Step 6: Format, lint, type check**

Run: `uv run ruff format src/crpod/detection/cards.py && uv run ruff check && uv run mypy src`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/crpod/detection/cards.py src/crpod/constants.py
git commit -m "feat: populate KATACR_TO_CARD and KATACR_NON_CARD

NNN cards mapped to canonical CARD_COSTS keys; MMM non-card classes
(towers, HP bars, projectiles, UI, spawn-derived units) filtered.
Coverage and validity tests all pass.

Card costs added to constants.py for KataCR cards previously missing:
[list cards added]
"
```

- [ ] **Step 8: Update `docs/TODO.md` to mark the item complete**

Replace the Phase 1/Phase 2 status note with a completion note matching the existing style of completed items.

---

## Self-Review

**Spec coverage:**
- Spec § "Module layout" → Tasks 1, 5.
- Spec § "Source of truth: katacr_classes.txt" → Tasks 6, 8.
- Spec § "Caller change" → Task 5.
- Spec § "Unknown-class handling" (warn-once) → Tasks 1, 2.
- Spec § "Testing" — Coverage → Task 4. Validity → Task 3. Behavioral → Task 2.
- Spec § "Out of scope" (4 unconfirmed champions) → Task 3 (validity test allows them).
- Spec § "Risks" (.pt isn't local; coverage test skip) → Task 4 implements skip; Task 8 explicitly resolves it.
- Spec § "Implementation notes" (mechanical row population) → Task 9.

**Placeholder scan:** No "TBD" / "fill in later" without concrete guidance. The two places I lean on the implementer's judgement (which exact mapping rows to add in Task 9, what the analyze command's exact arguments are in Task 9 step 5) are unavoidable — they depend on data that doesn't exist locally — but each is accompanied by concrete decision criteria, examples, and a verification command.

**Type consistency:**
- `Detection` (frame: int, cls: str, confidence: float, xyxy: tuple) — used consistently in Tasks 1, 2.
- `CardPlay` (frame, card, x, y, side, elixir_cost) — matches `src/crpod/types.py:20-28`.
- `Side` enum (FRIENDLY/ENEMY/UNKNOWN) — matches `src/crpod/types.py:13-16`.
- `to_card_play(det) -> CardPlay | None` — same signature in module (Task 1) and call site (Task 5) and tests (Task 2).
- `KATACR_TO_CARD: dict[str, str]` and `KATACR_NON_CARD: frozenset[str]` — same types throughout.

No issues found.
