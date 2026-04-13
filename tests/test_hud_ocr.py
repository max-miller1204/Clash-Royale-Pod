"""HUD OCR regression tests.

The fixture `tests/fixtures/hud/sample_540x960.jpg` is a frame sampled from
`chrisrca/clash-royale-tv-replays` (arena_15 / 00a91415-... frame 251). The
top HUD shows the enemy elixir counter at "3" and the match timer at "1:40".
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from crpod.ocr.hud import HudReader, HudRegions

pytest.importorskip("pytesseract")

FIXTURE = Path(__file__).parent / "fixtures" / "hud" / "sample_540x960.jpg"


def _load_frame():
    assert FIXTURE.exists(), f"missing fixture {FIXTURE}"
    frame = cv2.imread(str(FIXTURE))
    assert frame is not None, f"failed to decode {FIXTURE}"
    assert frame.shape[:2] == (960, 540), f"expected 540x960, got {frame.shape[:2]}"
    return frame


def test_hud_reader_recognizes_enemy_elixir():
    frame = _load_frame()
    state = HudReader().read(frame_idx=0, frame=frame)
    assert state.enemy_elixir == 3.0


def test_hud_regions_cover_frame():
    frame = _load_frame()
    h, w = frame.shape[:2]
    regions = HudRegions()
    for name, rect in vars(regions).items():
        x1, y1, x2, y2 = rect
        assert 0 <= x1 < x2 <= w, f"{name} x out of bounds: {rect}"
        assert 0 <= y1 < y2 <= h, f"{name} y out of bounds: {rect}"
