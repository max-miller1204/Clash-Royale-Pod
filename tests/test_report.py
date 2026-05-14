"""Wave 4A — `render_report` smoke and shape tests.

The report is text-only HTML (no DOM parsing) and writes to a tmp dir,
so these tests just assert: the file is written, the basic shape is
right, blunder rows appear / absent as expected, and missing images
degrade gracefully.
"""

from __future__ import annotations

from pathlib import Path

from crpod.pipeline import AnalysisResult
from crpod.types import Blunder, Replay
from crpod.visualization.report import render_report


def _result(*, blunders: list[Blunder]) -> AnalysisResult:
    replay = Replay(replay_id="r1", arena="arena_15")
    return AnalysisResult(
        replay=replay,
        interactions=[],
        feature_rows=[],
        friendly_leak=1.5,
        enemy_leak=3.0,
        tempo=[],
        ev_predictions=None,
        blunders=blunders,
    )


def test_renders_a_self_contained_html_file(tmp_path: Path) -> None:
    out = render_report(_result(blunders=[]), tmp_path)
    assert out == tmp_path / "report.html"
    text = out.read_text()
    assert "<!doctype html>" in text
    assert "<title>crpod report" in text
    # The report references its replay-id and arena fields.
    assert "r1" in text
    assert "arena_15" in text


def test_empty_blunder_table_shows_placeholder(tmp_path: Path) -> None:
    out = render_report(_result(blunders=[]), tmp_path)
    assert "No blunders flagged" in out.read_text()


def test_blunder_rows_render_with_card_and_sigma(tmp_path: Path) -> None:
    blunders = [
        Blunder(
            play_idx=3,
            card="hog_rider",
            ev_predicted=-120.5,
            per_card_median=80.0,
            sigma_below=4.01,
        ),
        Blunder(
            play_idx=7,
            card="knight",
            ev_predicted=-50.0,
            per_card_median=10.0,
            sigma_below=1.5,
        ),
    ]
    out = render_report(_result(blunders=blunders), tmp_path)
    text = out.read_text()
    assert "hog_rider" in text
    assert "knight" in text
    assert "4.01σ" in text
    assert "1.50σ" in text
    # Sigma-below sort order: hog_rider's row should appear before knight's.
    assert text.index("hog_rider") < text.index("knight")


def test_missing_plot_images_degrade_to_placeholder(tmp_path: Path) -> None:
    """No placements.png / tempo.png in the out_dir → in-line placeholder
    text, not an exception."""
    out = render_report(_result(blunders=[]), tmp_path)
    text = out.read_text()
    assert "placement heatmap unavailable" in text
    assert "elixir tempo unavailable" in text


def test_image_files_get_embedded_as_base64(tmp_path: Path) -> None:
    """When placements.png / tempo.png exist, they get inlined as
    `data:image/png;base64,...` URIs — no external file refs."""
    placements = tmp_path / "placements.png"
    tempo = tmp_path / "tempo.png"
    # 1-byte PNG-ish stub — render_report doesn't inspect the bytes.
    placements.write_bytes(b"\x89PNG\r\n\x1a\n")
    tempo.write_bytes(b"\x89PNG\r\n\x1a\n")

    out = render_report(_result(blunders=[]), tmp_path)
    text = out.read_text()
    assert text.count('src="data:image/png;base64,') == 2
    # No relative-path img refs leak in.
    assert 'src="placements.png"' not in text
    assert 'src="tempo.png"' not in text


def test_blunder_count_card_appears_in_summary(tmp_path: Path) -> None:
    blunders = [
        Blunder(
            play_idx=0,
            card="c",
            ev_predicted=0.0,
            per_card_median=0.0,
            sigma_below=1.1,
        )
    ]
    out = render_report(_result(blunders=blunders), tmp_path)
    text = out.read_text()
    # Summary grid renders the count.
    assert "Blunders" in text
