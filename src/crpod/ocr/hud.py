"""HUD reading: tesseract for elixir, HP-bar pixel sampling for tower HP.

The princess-tower HP digits are rendered in a stylised in-game font that
tesseract 4.1.1 cannot read (wave 2D smoke: 0/583 frames had all four
princess HPs readable). This module switched to graphical bar-fill
sampling for the four princess-HP fields — a horizontal strip of bright
cyan-blue (friendly) or pink-red (enemy) whose pixel length tracks HP.
Elixir continues to use tesseract because the elixir digit is a clean
white glyph on a flat purple background and reads reliably.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np

from crpod.types import HudState

# Princess HP / pixel calibration (chrisrca/clash-royale-tv-replays
# arena_15 / 00a91415-... frame 251, 540x960). Held towers map to:
#   friendly_right 3052 HP → 54 bar pixels  (3052/54 ≈ 56.5 HP/px)
#   friendly_left  1446 HP → 25 bar pixels  (1446/25 ≈ 57.8 HP/px)
#   enemy_left     2423 HP → 48 bar pixels  (2423/48 ≈ 50.5 HP/px)
#   enemy_right    1810 HP → 35 bar pixels  (1810/35 ≈ 51.7 HP/px)
# Per-side scales differ ~12% because the bright bar segment fades earlier
# on the friendly badge than the enemy one. Calibration error is ≤2.5% on
# all four fixture towers — well below the noise floor of training MAE.
FRIENDLY_HP_PER_BAR_PX = 56.5
ENEMY_HP_PER_BAR_PX = 50.5
# A full lvl-14 princess bar is ~60 px; clip implausibly long runs (e.g.,
# a stray VFX flash in a non-tower region) at ~25% above that threshold.
MAX_PLAUSIBLE_BAR_PX = 75


@dataclass(frozen=True)
class HudRegions:
    """Pixel rectangles (x1, y1, x2, y2) for HUD elements on a 540x960 frame.

    Measured against the HF `chrisrca/clash-royale-tv-replays` dataset, which
    ships third-person TV-replay footage at 540x960. Re-measure if feeding a
    different source.
    """

    enemy_elixir: tuple[int, int, int, int] = (15, 10, 65, 50)
    friendly_elixir: tuple[int, int, int, int] = (15, 900, 65, 945)
    timer: tuple[int, int, int, int] = (450, 140, 540, 180)
    enemy_tower_left: tuple[int, int, int, int] = (60, 240, 180, 275)
    enemy_tower_right: tuple[int, int, int, int] = (360, 240, 480, 275)
    friendly_tower_left: tuple[int, int, int, int] = (60, 678, 180, 712)
    friendly_tower_right: tuple[int, int, int, int] = (360, 678, 480, 712)
    friendly_king: tuple[int, int, int, int] = (220, 810, 320, 855)
    enemy_king: tuple[int, int, int, int] = (220, 115, 320, 155)
    # HP-bar pixel-sampling rects: thin horizontal strips covering the
    # bright bar fill segment of each princess badge. Friendly bars sit
    # in the upper part of the badge (y=683-690); enemy bars are mirrored
    # to the lower part (y=263-270). x-range is the full digit-overlay
    # rect — the bar's left edge starts after the king-level crown badge,
    # which the BGR mask cleanly excludes (gold ≠ cyan/red).
    friendly_left_hp_bar: tuple[int, int, int, int] = (60, 683, 180, 690)
    friendly_right_hp_bar: tuple[int, int, int, int] = (360, 683, 480, 690)
    enemy_left_hp_bar: tuple[int, int, int, int] = (60, 263, 180, 270)
    enemy_right_hp_bar: tuple[int, int, int, int] = (360, 263, 480, 270)


class HudReader:
    def __init__(self, regions: HudRegions | None = None) -> None:
        self.regions = regions or HudRegions()
        self._pytesseract = None

    def _lazy_load(self) -> None:
        if self._pytesseract is not None:
            return
        import pytesseract

        self._pytesseract = pytesseract

    def read(self, frame_idx: int, frame: np.ndarray) -> HudState:
        self._lazy_load()
        friendly_elixir = self._read_number(frame, self.regions.friendly_elixir)
        enemy_elixir = self._read_number(frame, self.regions.enemy_elixir)
        friendly_left = self._read_hp_bar(frame, self.regions.friendly_left_hp_bar, "friendly")
        friendly_right = self._read_hp_bar(frame, self.regions.friendly_right_hp_bar, "friendly")
        enemy_left = self._read_hp_bar(frame, self.regions.enemy_left_hp_bar, "enemy")
        enemy_right = self._read_hp_bar(frame, self.regions.enemy_right_hp_bar, "enemy")
        return HudState(
            frame=frame_idx,
            friendly_elixir=float(friendly_elixir or 0),
            enemy_elixir=float(enemy_elixir) if enemy_elixir is not None else None,
            friendly_king_hp=None,
            enemy_king_hp=None,
            friendly_left_princess_hp=friendly_left,
            friendly_right_princess_hp=friendly_right,
            enemy_left_princess_hp=enemy_left,
            enemy_right_princess_hp=enemy_right,
        )

    def _read_number(self, frame: np.ndarray, region: tuple[int, int, int, int]) -> int | None:
        assert self._pytesseract is not None
        x1, y1, x2, y2 = region
        crop = frame[y1:y2, x1:x2]
        # The HUD digits are ~20px tall at 540x960 — upscale before OCR so
        # Tesseract's feature extractor has enough resolution to work with.
        upscaled = cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
        # Pass a realpath-resolved file path rather than a numpy array: the
        # nix-built tesseract on macOS can't follow the /tmp -> /private/tmp
        # symlink that pytesseract's default tempfile flow lands on.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        try:
            cv2.imwrite(tmp_path, upscaled)
            txt = self._pytesseract.image_to_string(
                os.path.realpath(tmp_path),
                config="--psm 7 -c tessedit_char_whitelist=0123456789",
            ).strip()
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_path)
        try:
            return int(txt) if txt else None
        except ValueError:
            return None

    def _read_hp_bar(
        self, frame: np.ndarray, region: tuple[int, int, int, int], side: str
    ) -> int | None:
        """Sample the bright fill of a princess-tower HP bar.

        Counts the longest horizontal run of bar-coloured pixels inside
        `region` and converts it to HP via a side-specific HP-per-pixel
        scale. Returns `None` for a vanished or implausibly long run so
        the EV target loop drops the row instead of consuming a bogus HP.
        """
        x1, y1, x2, y2 = region
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        b = crop[..., 0].astype(np.int32)
        g = crop[..., 1].astype(np.int32)
        r = crop[..., 2].astype(np.int32)
        if side == "friendly":
            # Bright cyan-blue fill: B dominant, well above the dark
            # unfilled portion of the badge (B≈110 there, vs ≥130 in the
            # bar). The B>R margin rejects the gold king-level crown.
            mask = (b > 130) & (b - r > 25) & (b - g > 0)
            scale = FRIENDLY_HP_PER_BAR_PX
        elif side == "enemy":
            # Bright pink-red fill: R dominant. Same shape, mirrored hue.
            mask = (r > 130) & (r - g > 25) & (r - b > 0)
            scale = ENEMY_HP_PER_BAR_PX
        else:
            raise ValueError(f"unknown side: {side!r}")
        run = _longest_horizontal_run(mask)
        if run == 0:
            return None
        if run > MAX_PLAUSIBLE_BAR_PX:
            return None
        return round(run * scale)


def _longest_horizontal_run(mask: np.ndarray) -> int:
    """Longest run of True values along the last axis, max over rows.

    Pure-Python loop because the rect is tiny (~120×7 ≈ 840 cells).
    """
    if mask.size == 0:
        return 0
    best = 0
    for row in mask:
        run = 0
        for v in row:
            if v:
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
    return best
