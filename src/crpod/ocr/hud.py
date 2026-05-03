"""HUD reading via numpy pixel sampling.

All four princess HP bars and both elixir counters are read from
horizontal-fill bands using BGR colour masks plus a longest-run scan —
no tesseract subprocess. Wave 2G dropped the tesseract elixir read
because it was the wall-clock bottleneck for training (~270k subprocess
spawns per replay sweep). The princess-HP digits had already moved off
tesseract in wave 2E (the in-game digit font is unreadable to tesseract
4.1.1).
"""

from __future__ import annotations

from dataclasses import dataclass

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

# Elixir bar calibration (same fixture frame). The elixir bar is the
# horizontal pink/magenta strip running across the top (enemy) and bottom
# (friendly) edges of the HUD. Same colour both sides.
#   friendly bar 2 elixir → 87 bar pixels (in the 2-pixel-tall sample band)
#   enemy bar    3 elixir → 135 bar pixels
# Both round-trip cleanly at ~44 px per elixir. The bar fills smoothly so
# round() of `run / 44` recovers the integer elixir reading the digit
# overlay would show.
BAR_PX_PER_ELIXIR = 44.0


@dataclass(frozen=True)
class HudRegions:
    """Pixel rectangles (x1, y1, x2, y2) for HUD elements on a 540x960 frame.

    Measured against the HF `chrisrca/clash-royale-tv-replays` dataset, which
    ships third-person TV-replay footage at 540x960. Re-measure if feeding a
    different source.
    """

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
    # Elixir-bar pixel-sampling rects. x starts at 60 to skip the rounded
    # digit badge on the left edge (which is the same pink colour as the
    # bar fill); the bar itself extends right of the badge to roughly the
    # full HUD width. The y band is tight on the bright fill strip in the
    # top/bottom HUD edges.
    enemy_elixir_bar: tuple[int, int, int, int] = (60, 22, 540, 32)
    friendly_elixir_bar: tuple[int, int, int, int] = (60, 928, 540, 940)


class HudReader:
    def __init__(self, regions: HudRegions | None = None) -> None:
        self.regions = regions or HudRegions()

    def read(self, frame_idx: int, frame: np.ndarray) -> HudState:
        friendly_elixir = self._read_elixir_bar(frame, self.regions.friendly_elixir_bar)
        enemy_elixir = self._read_elixir_bar(frame, self.regions.enemy_elixir_bar)
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

    def _read_elixir_bar(self, frame: np.ndarray, region: tuple[int, int, int, int]) -> int | None:
        """Sample the bright pink elixir bar.

        Returns the integer elixir reading (0..10) the digit overlay would
        show, or `None` for a fully empty bar so the HF loader can flag the
        frame as unreadable. The mask matches the bar's pink/magenta fill
        (R dominant over G, B noticeably present — distinguishes from the
        red HP-bar fill which has B near zero).
        """
        x1, y1, x2, y2 = region
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        b = crop[..., 0].astype(np.int32)
        g = crop[..., 1].astype(np.int32)
        r = crop[..., 2].astype(np.int32)
        # Pink/magenta fill: R bright, much above G, with significant B
        # content (the b > 80 floor is what separates this from the red
        # HP-bar fill, which has B near zero).
        mask = (r > 150) & (r - g > 40) & (b > 80)
        run = _longest_horizontal_run(mask)
        if run == 0:
            return None
        elixir = round(run / BAR_PX_PER_ELIXIR)
        # Clamp to the in-game elixir cap; a run wider than ~10 elixir is
        # a calibration error, not a real reading.
        if elixir > 10:
            return 10
        return elixir


def _longest_horizontal_run(mask: np.ndarray) -> int:
    """Longest run of True values along the last axis, max over rows.

    Vectorised via an edge-diff over each row (concat-pad with zeros so
    runs touching the edges are detected). The fallback for an all-False
    row is 0; the fallback for an empty mask is 0.
    """
    if mask.size == 0:
        return 0
    best = 0
    for row in mask:
        # Find rising and falling edges in the row by diffing a 0-padded
        # int8 view. starts = idx where 0→1; ends = idx where 1→0.
        padded = np.concatenate(([0], row.astype(np.int8), [0]))
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        if starts.size == 0:
            continue
        run = int((ends - starts).max())
        if run > best:
            best = run
    return best
