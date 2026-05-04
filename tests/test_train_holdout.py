"""Wave 2H: --min-arena filtering + --frozen-holdout split discipline.

These tests cover the helpers that let `crpod train` (a) restrict the HF
replay pool to a top-skill cohort and (b) reuse a committed holdout list
across chunks 2I-2K so the spec's strict-serial Δρ attribution stays
honest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crpod.__main__ import (
    _arena_index,
    _filter_min_arena,
    _parse_frozen_holdout,
    _write_frozen_holdout,
)


def test_arena_index_extracts_trailing_int() -> None:
    assert _arena_index("arena_15") == 15
    assert _arena_index("arena_23") == 23
    assert _arena_index("arena_31") == 31


def test_arena_index_raises_on_unparseable_name() -> None:
    with pytest.raises(ValueError):
        _arena_index("not-an-arena")


def test_filter_min_arena_keeps_geq_threshold() -> None:
    pool = [
        ("arena_15", "r1"),
        ("arena_22", "r2"),
        ("arena_23", "r3"),
        ("arena_24", "r4"),
        ("arena_31", "r5"),
    ]
    assert _filter_min_arena(pool, 23) == [
        ("arena_23", "r3"),
        ("arena_24", "r4"),
        ("arena_31", "r5"),
    ]


def test_filter_min_arena_none_returns_all() -> None:
    pool = [("arena_15", "r1"), ("arena_23", "r2")]
    assert _filter_min_arena(pool, None) == pool


def test_filter_min_arena_empty_when_threshold_too_high() -> None:
    pool = [("arena_15", "r1"), ("arena_23", "r2")]
    assert _filter_min_arena(pool, 99) == []


def test_parse_frozen_holdout_round_trip(tmp_path: Path) -> None:
    """A holdout file emitted by `_write_frozen_holdout` parses back to the
    same set of (arena, replay_id) keys."""
    bundle = [
        ("arena_23", "abc-def-ghi"),
        ("arena_24", "jkl-mno-pqr"),
        ("arena_31", "stu-vwx-yz0"),
    ]
    path = tmp_path / "holdout.txt"
    _write_frozen_holdout(path, bundle)
    assert _parse_frozen_holdout(path) == set(bundle)


def test_parse_frozen_holdout_skips_blank_and_comment_lines(tmp_path: Path) -> None:
    """Comments (`#`) and blank lines are ignored — committed holdout files
    can carry a header explaining provenance without polluting the keys."""
    path = tmp_path / "holdout.txt"
    path.write_text(
        "# wave 2.5 frozen holdout — arena_23+ pool\n"
        "# generated 2026-05-03 from Random(0)\n"
        "\n"
        "arena_23 abc-def-ghi\n"
        "arena_24 jkl-mno-pqr\n"
        "\n"
    )
    assert _parse_frozen_holdout(path) == {
        ("arena_23", "abc-def-ghi"),
        ("arena_24", "jkl-mno-pqr"),
    }


def test_parse_frozen_holdout_rejects_malformed_line(tmp_path: Path) -> None:
    """A non-comment line that doesn't have exactly two whitespace-separated
    tokens must surface a clear error — silently dropping rows would
    corrupt the train/holdout split without the user noticing."""
    path = tmp_path / "holdout.txt"
    path.write_text("arena_23 abc-def-ghi\nmalformed\n")
    with pytest.raises(ValueError, match="malformed"):
        _parse_frozen_holdout(path)


def test_write_frozen_holdout_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "holdout.txt"
    _write_frozen_holdout(path, [("arena_23", "r1")])
    assert path.exists()
    assert "arena_23 r1" in path.read_text()
