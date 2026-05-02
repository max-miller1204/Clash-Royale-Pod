"""HUD reader regression tests.

Two regimes:

- Real-frame tests on `tests/fixtures/hud/sample_540x960.jpg` (a frame
  sampled from `chrisrca/clash-royale-tv-replays` arena_15 / 00a91415-...
  frame 251). Ground truth visible in the fixture: friendly elixir 2,
  enemy elixir 3, friendly princess HP left/right = 1446/3052, enemy
  princess HP left/right = 2423/1810. After wave 2G these tests are
  pure numpy + opencv — no tesseract dependency.
- Synthetic-frame tests for `_read_hp_bar` and `_read_elixir_bar`. These
  build a 540x960 numpy array with a known coloured bar of known pixel
  length and verify the conversion.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from crpod.ocr.hud import (
    BAR_PX_PER_ELIXIR,
    ENEMY_HP_PER_BAR_PX,
    FRIENDLY_HP_PER_BAR_PX,
    HudReader,
    HudRegions,
    _longest_horizontal_run,
)

FIXTURE = Path(__file__).parent / "fixtures" / "hud" / "sample_540x960.jpg"


def _load_frame() -> np.ndarray:
    assert FIXTURE.exists(), f"missing fixture {FIXTURE}"
    frame = cv2.imread(str(FIXTURE))
    assert frame is not None, f"failed to decode {FIXTURE}"
    assert frame.shape[:2] == (960, 540), f"expected 540x960, got {frame.shape[:2]}"
    return frame


def test_hud_reader_reads_fixture_elixir() -> None:
    """Both elixir bars resolve via numpy pixel-sampling — no tesseract.

    Fixture: friendly digit shows "2" (bar ~87 px), enemy "3" (bar ~135 px).
    With BAR_PX_PER_ELIXIR ≈ 44 these round to 2.0 and 3.0 respectively.
    """
    frame = _load_frame()
    state = HudReader().read(frame_idx=0, frame=frame)
    assert state.enemy_elixir == 3.0
    assert state.friendly_elixir == 2.0


def test_hud_regions_cover_frame() -> None:
    frame = _load_frame()
    h, w = frame.shape[:2]
    regions = HudRegions()
    for name, rect in vars(regions).items():
        x1, y1, x2, y2 = rect
        assert 0 <= x1 < x2 <= w, f"{name} x out of bounds: {rect}"
        assert 0 <= y1 < y2 <= h, f"{name} y out of bounds: {rect}"


def test_hp_bar_reader_recovers_fixture_hps() -> None:
    """The four bar reads on the fixture should land within ±5% of truth.

    Calibration tolerance is generous on purpose — the per-side HP-per-px
    scale is approximate and we only need ratiometric fidelity for the
    EV target. Spearman is scale-invariant; MAE shifts by a constant.
    """
    frame = _load_frame()
    reader = HudReader()
    expected = {
        "friendly_left_princess_hp": 1446,
        "friendly_right_princess_hp": 3052,
        "enemy_left_princess_hp": 2423,
        "enemy_right_princess_hp": 1810,
    }
    state = reader.read(frame_idx=0, frame=frame)
    for field, truth in expected.items():
        got = getattr(state, field)
        assert got is not None, f"{field} returned None"
        rel = abs(got - truth) / truth
        assert rel < 0.05, f"{field}: got {got}, expected {truth}, |Δ|/truth = {rel:.1%}"


def _synthetic_frame_with_bar(
    rect: tuple[int, int, int, int],
    bar_len: int,
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    """Build a 540x960 black frame with a coloured bar inside `rect`."""
    frame = np.zeros((960, 540, 3), dtype=np.uint8)
    x1, y1, _x2, y2 = rect
    if bar_len > 0:
        frame[y1:y2, x1 : x1 + bar_len] = color_bgr
    return frame


def test_read_hp_bar_friendly_synthetic_known_length() -> None:
    """A 30-pixel cyan bar in the friendly_left region should read as
    round(30 * FRIENDLY_HP_PER_BAR_PX) HP."""
    reader = HudReader()
    rect = HudRegions().friendly_left_hp_bar
    frame = _synthetic_frame_with_bar(rect, bar_len=30, color_bgr=(240, 200, 120))
    state = reader.read(frame_idx=0, frame=frame)
    expected = round(30 * FRIENDLY_HP_PER_BAR_PX)
    assert state.friendly_left_princess_hp == expected


def test_read_hp_bar_enemy_synthetic_known_length() -> None:
    """A 40-pixel pink bar in the enemy_right region should read as
    round(40 * ENEMY_HP_PER_BAR_PX) HP."""
    reader = HudReader()
    rect = HudRegions().enemy_right_hp_bar
    frame = _synthetic_frame_with_bar(rect, bar_len=40, color_bgr=(80, 60, 220))
    state = reader.read(frame_idx=0, frame=frame)
    expected = round(40 * ENEMY_HP_PER_BAR_PX)
    assert state.enemy_right_princess_hp == expected


def test_read_hp_bar_destroyed_tower_returns_none() -> None:
    """A black region (no bar at all) should yield None — the EV target
    loop drops these rows rather than feeding a bogus 0 into training."""
    reader = HudReader()
    frame = np.zeros((960, 540, 3), dtype=np.uint8)
    state = reader.read(frame_idx=0, frame=frame)
    assert state.friendly_left_princess_hp is None
    assert state.friendly_right_princess_hp is None
    assert state.enemy_left_princess_hp is None
    assert state.enemy_right_princess_hp is None


def test_read_hp_bar_implausibly_long_run_returns_none() -> None:
    """A run wider than `MAX_PLAUSIBLE_BAR_PX` (e.g., a frame-wide cyan
    background) is not a real tower bar and must be rejected."""
    reader = HudReader()
    rect = HudRegions().friendly_left_hp_bar
    frame = _synthetic_frame_with_bar(rect, bar_len=120, color_bgr=(240, 200, 120))
    state = reader.read(frame_idx=0, frame=frame)
    assert state.friendly_left_princess_hp is None


def test_read_hp_bar_wrong_color_returns_none() -> None:
    """A red bar in a friendly region (or vice versa) must NOT be read
    as friendly HP — the side-specific colour mask should reject it."""
    reader = HudReader()
    rect = HudRegions().friendly_left_hp_bar
    # Red fill in a friendly slot.
    frame = _synthetic_frame_with_bar(rect, bar_len=30, color_bgr=(40, 40, 220))
    state = reader.read(frame_idx=0, frame=frame)
    assert state.friendly_left_princess_hp is None


def test_read_elixir_bar_friendly_synthetic_known_length() -> None:
    """A 88-px pink bar in the friendly elixir region rounds to 2 elixir."""
    reader = HudReader()
    rect = HudRegions().friendly_elixir_bar
    frame = _synthetic_frame_with_bar(rect, bar_len=88, color_bgr=(180, 60, 220))
    state = reader.read(frame_idx=0, frame=frame)
    assert state.friendly_elixir == round(88 / BAR_PX_PER_ELIXIR)


def test_read_elixir_bar_enemy_synthetic_known_length() -> None:
    """A 132-px pink bar in the enemy elixir region rounds to 3 elixir."""
    reader = HudReader()
    rect = HudRegions().enemy_elixir_bar
    frame = _synthetic_frame_with_bar(rect, bar_len=132, color_bgr=(180, 60, 220))
    state = reader.read(frame_idx=0, frame=frame)
    assert state.enemy_elixir == float(round(132 / BAR_PX_PER_ELIXIR))


def test_read_elixir_bar_empty_region_yields_zero_or_none() -> None:
    """Empty bar (0 elixir) → friendly stays float 0.0; enemy reports None
    so the HF loader's drop-on-None policy still distinguishes 'unreadable'
    from a real 0-elixir state. Black frame is the empty case."""
    reader = HudReader()
    frame = np.zeros((960, 540, 3), dtype=np.uint8)
    state = reader.read(frame_idx=0, frame=frame)
    assert state.friendly_elixir == 0.0
    assert state.enemy_elixir is None


def test_read_elixir_bar_wrong_color_does_not_read() -> None:
    """A cyan bar (HP-bar colour) in the elixir region must not be read as
    elixir — the pink mask should reject it."""
    reader = HudReader()
    rect = HudRegions().friendly_elixir_bar
    frame = _synthetic_frame_with_bar(rect, bar_len=88, color_bgr=(240, 200, 120))
    state = reader.read(frame_idx=0, frame=frame)
    assert state.friendly_elixir == 0.0


def test_longest_horizontal_run_basic() -> None:
    mask = np.array(
        [
            [True, True, False, True, True, True, False],
            [False, True, True, False, False, False, False],
        ]
    )
    assert _longest_horizontal_run(mask) == 3


def test_longest_horizontal_run_empty() -> None:
    assert _longest_horizontal_run(np.zeros((0, 0), dtype=bool)) == 0
    assert _longest_horizontal_run(np.zeros((4, 5), dtype=bool)) == 0
