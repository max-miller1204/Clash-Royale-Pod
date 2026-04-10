"""HUD reading via Tesseract plus simple color heuristics for HP bars."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crpod.types import HudState


@dataclass(frozen=True)
class HudRegions:
    """Pixel rectangles for HUD elements on a 540x960 frame.

    Coordinates are placeholders tuned for the HF TV-replay resolution —
    re-measure if feeding a different source.
    """

    friendly_elixir: tuple[int, int, int, int] = (200, 880, 500, 920)
    timer: tuple[int, int, int, int] = (240, 40, 300, 80)
    friendly_king: tuple[int, int, int, int] = (230, 780, 310, 820)
    enemy_king: tuple[int, int, int, int] = (230, 50, 310, 90)


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
        elixir = self._read_number(frame, self.regions.friendly_elixir)
        return HudState(
            frame=frame_idx,
            friendly_elixir=float(elixir or 0),
            enemy_elixir=None,
            friendly_king_hp=None,
            enemy_king_hp=None,
        )

    def _read_number(self, frame: np.ndarray, region: tuple[int, int, int, int]) -> int | None:
        assert self._pytesseract is not None
        x1, y1, x2, y2 = region
        crop = frame[y1:y2, x1:x2]
        txt = self._pytesseract.image_to_string(
            crop, config="--psm 7 -c tessedit_char_whitelist=0123456789"
        ).strip()
        try:
            return int(txt) if txt else None
        except ValueError:
            return None
