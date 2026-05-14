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
# Second fixture used by the king-HP rect regression tests: both kings sit
# damaged so their HP labels actually render. Sampled from arena_15 /
# 226fefa9-… frame 1058 — five frames before the truly-final row of the
# replay's parquet (the last few rows are typically post-match overlays).
KING_DAMAGED_FIXTURE = (
    Path(__file__).parent / "fixtures" / "hud" / "sample_king_damaged_540x960.jpg"
)


def _load_frame() -> np.ndarray:
    assert FIXTURE.exists(), f"missing fixture {FIXTURE}"
    frame = cv2.imread(str(FIXTURE))
    assert frame is not None, f"failed to decode {FIXTURE}"
    assert frame.shape[:2] == (960, 540), f"expected 540x960, got {frame.shape[:2]}"
    return frame


def _load_king_damaged_frame() -> np.ndarray:
    assert KING_DAMAGED_FIXTURE.exists(), f"missing fixture {KING_DAMAGED_FIXTURE}"
    frame = cv2.imread(str(KING_DAMAGED_FIXTURE))
    assert frame is not None, f"failed to decode {KING_DAMAGED_FIXTURE}"
    assert frame.shape[:2] == (960, 540), f"expected 540x960, got {frame.shape[:2]}"
    return frame


def _yellow_badge_pct(crop: np.ndarray) -> float:
    """Fraction of pixels matching the king HP label's gold-crown badge.

    Bright saturated yellow: R high, G mid-high (≤ R), B low. This catches
    the level-badge crown and the bar outline that frame the HP digits, both
    of which only render once the king has taken damage.
    """
    b = crop[..., 0].astype(np.int32)
    g = crop[..., 1].astype(np.int32)
    r = crop[..., 2].astype(np.int32)
    mask = (r > 200) & (g > 150) & (g < 230) & (b < 100)
    return float(mask.mean())


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


def test_read_hp_bar_recovers_vfx_gap_friendly() -> None:
    """Wave 2I: a bar split by a 3-px black gap (modelling a splash-VFX
    overlay) should reunite as a single longer run, not read as just the
    longer half. Both halves are ≥ MIN_BRIDGE_SEG_PX so the gap-tolerant
    reader bridges them.
    """
    reader = HudReader()
    rect = HudRegions().friendly_left_hp_bar
    x1, y1, _x2, y2 = rect
    cyan = (240, 200, 120)
    frame = np.zeros((960, 540, 3), dtype=np.uint8)
    # Two cyan halves separated by a 3-px black VFX gap.
    frame[y1:y2, x1 : x1 + 22] = cyan
    frame[y1:y2, x1 + 25 : x1 + 50] = cyan
    state = reader.read(frame_idx=0, frame=frame)
    # Without the bridge: max(22, 25) = 25 → 1412 HP.
    # With the bridge: 50 → round(50 * FRIENDLY_HP_PER_BAR_PX).
    expected = round(50 * FRIENDLY_HP_PER_BAR_PX)
    assert state.friendly_left_princess_hp == expected


def test_read_hp_bar_recovers_vfx_gap_enemy() -> None:
    """Same VFX-gap bridging behaviour for the enemy (red-mask) side."""
    reader = HudReader()
    rect = HudRegions().enemy_right_hp_bar
    x1, y1, _x2, y2 = rect
    pink = (80, 60, 220)
    frame = np.zeros((960, 540, 3), dtype=np.uint8)
    frame[y1:y2, x1 : x1 + 18] = pink
    frame[y1:y2, x1 + 21 : x1 + 38] = pink
    state = reader.read(frame_idx=0, frame=frame)
    expected = round(38 * ENEMY_HP_PER_BAR_PX)
    assert state.enemy_right_princess_hp == expected


def test_read_hp_bar_does_not_bridge_short_noise_blobs() -> None:
    """Wave 2I: a small (<MIN_BRIDGE_SEG_PX) bar-coloured blob next to the
    main bar must NOT be merged in. This pins the calibration-preserving
    behaviour of the gap-tolerant reader against the fixture-frame style
    of decorative noise.
    """
    reader = HudReader()
    rect = HudRegions().friendly_left_hp_bar
    x1, y1, _x2, y2 = rect
    cyan = (240, 200, 120)
    frame = np.zeros((960, 540, 3), dtype=np.uint8)
    # Main 30-px bar plus a 7-px stray blob 3 px to the right.
    frame[y1:y2, x1 : x1 + 30] = cyan
    frame[y1:y2, x1 + 33 : x1 + 40] = cyan
    state = reader.read(frame_idx=0, frame=frame)
    # The 7-px blob is below MIN_BRIDGE_SEG_PX (10), so the read stays
    # at 30 px instead of inflating to 40.
    expected = round(30 * FRIENDLY_HP_PER_BAR_PX)
    assert state.friendly_left_princess_hp == expected


def test_hp_bar_reader_calibration_within_2_5_percent() -> None:
    """Wave 2I sanity check: the docstringed ≤2.5% MAE on the fixture
    frame in `hud.py` is preserved after the gap-tolerant run finder
    landed. Tighter than the ≤5% bound in
    `test_hp_bar_reader_recovers_fixture_hps` (which exists as a looser
    smoke test); this is the calibration regression gate.
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
        assert rel <= 0.025, (
            f"{field}: got {got}, expected {truth}, |Δ|/truth = {rel:.1%} "
            f"(>2.5% — the gap-tolerant run finder shifted calibration)"
        )


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
    """Pre-2I: longest unbroken run in each row, max over rows. Wave 2I's
    gap-tolerant bridging only kicks in when the segment-to-merge is
    ≥ MIN_BRIDGE_SEG_PX (10), so these short toy rows fall back to the
    simple longest-run behaviour.
    """
    mask = np.array(
        [
            [True, True, False, True, True, True, False],
            [False, True, True, False, False, False, False],
        ]
    )
    assert _longest_horizontal_run(mask) == 3


def test_longest_horizontal_run_bridges_vfx_gap() -> None:
    """Wave 2I: two ≥10-px True segments separated by a ≤3-px False gap
    are bridged — emulates a splash projectile masking a few bar pixels.
    """
    # 11 True, 3 False, 12 True, then 5 False. Bridged length = 26.
    row = np.concatenate(
        [
            np.ones(11, dtype=bool),
            np.zeros(3, dtype=bool),
            np.ones(12, dtype=bool),
            np.zeros(5, dtype=bool),
        ]
    )
    mask = row.reshape(1, -1)
    assert _longest_horizontal_run(mask) == 26


def test_longest_horizontal_run_does_not_bridge_short_segment() -> None:
    """Wave 2I: a short (<MIN_BRIDGE_SEG_PX) segment near a long run must
    not be merged — this is the noise-rejection guard that preserves the
    fixture's ≤2.5% calibration.
    """
    # 30 True, 3 False, 7 True. Short blob (7 px) → not bridged → 30.
    row = np.concatenate(
        [
            np.ones(30, dtype=bool),
            np.zeros(3, dtype=bool),
            np.ones(7, dtype=bool),
        ]
    )
    mask = row.reshape(1, -1)
    assert _longest_horizontal_run(mask) == 30


def test_longest_horizontal_run_empty() -> None:
    assert _longest_horizontal_run(np.zeros((0, 0), dtype=bool)) == 0
    assert _longest_horizontal_run(np.zeros((4, 5), dtype=bool)) == 0


def test_king_rects_within_frame_bounds() -> None:
    """Sanity: both king rects fit inside the 540x960 frame."""
    regions = HudRegions()
    for name, rect in (
        ("friendly_king", regions.friendly_king),
        ("enemy_king", regions.enemy_king),
    ):
        x1, y1, x2, y2 = rect
        assert 0 <= x1 < x2 <= 540, f"{name} x out of bounds: {rect}"
        assert 0 <= y1 < y2 <= 960, f"{name} y out of bounds: {rect}"


def test_king_rects_capture_hp_label_when_damaged() -> None:
    """The updated `friendly_king` / `enemy_king` rects should land on the
    in-game HP label whenever the king has taken damage.

    The label is a gold crown badge plus a yellow-trimmed HP bar; both render
    only when the king is damaged. On the king-damaged fixture, the gold
    ratio inside each rect should clear an absolute floor AND exceed the
    same-coordinates ratio on the undamaged fixture (where the rect covers
    arena/card-hand background). The floor + monotonicity gate is enough to
    fail if the rect drifts off the label without overfitting to the precise
    pixel count: a soft-king replay with HP near 0 still has the gold bar
    outline + badge crown to anchor on, even though the white bar fill is
    nearly empty.
    """
    damaged = _load_king_damaged_frame()
    undamaged = _load_frame()
    regions = HudRegions()
    for name, rect in (
        ("friendly_king", regions.friendly_king),
        ("enemy_king", regions.enemy_king),
    ):
        x1, y1, x2, y2 = rect
        dmg_pct = _yellow_badge_pct(damaged[y1:y2, x1:x2])
        und_pct = _yellow_badge_pct(undamaged[y1:y2, x1:x2])
        assert dmg_pct >= 0.06, (
            f"{name} damaged-frame gold ratio {dmg_pct:.1%} below 6% — "
            f"rect {rect} likely drifted off the HP label"
        )
        assert dmg_pct > und_pct, (
            f"{name} damaged ratio {dmg_pct:.1%} ≤ undamaged {und_pct:.1%} — "
            f"rect picks up the same amount of gold regardless of damage state, "
            f"so it isn't anchored on the in-game HP label"
        )
