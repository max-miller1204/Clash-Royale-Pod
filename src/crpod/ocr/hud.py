"""HUD reading via Tesseract plus simple color heuristics for HP bars."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from crpod.types import HudState


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
        return HudState(
            frame=frame_idx,
            friendly_elixir=float(friendly_elixir or 0),
            enemy_elixir=float(enemy_elixir) if enemy_elixir is not None else None,
            friendly_king_hp=None,
            enemy_king_hp=None,
        )

    def _read_number(self, frame: np.ndarray, region: tuple[int, int, int, int]) -> int | None:
        assert self._pytesseract is not None
        x1, y1, x2, y2 = region
        crop = frame[y1:y2, x1:x2]
        # The HUD digits are ~20px tall at 540x960 — upscale before OCR so
        # Tesseract's feature extractor has enough resolution to work with.
        upscaled = cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
        txt = self._pytesseract.image_to_string(
            upscaled, config="--psm 7 -c tessedit_char_whitelist=0123456789"
        ).strip()
        try:
            return int(txt) if txt else None
        except ValueError:
            return None
