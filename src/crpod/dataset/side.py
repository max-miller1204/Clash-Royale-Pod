"""Viewer-aware side inference for player-perspective video.

Player-perspective video uses the standard in-game camera, where the
recorder is always at the bottom of the frame and the river bisects the
frame at roughly the midpoint. This is intentionally separate from
`crpod.dataset.huggingface._infer_side`, which is calibrated against
TV-replay framing using a fixed `RIVER_Y` constant. Keeping the two
rules in different modules so the HF path is not perturbed when the
video path's heuristic evolves.
"""

from __future__ import annotations

from crpod.types import Side


def infer_video_side(y: float, frame_height: int) -> Side:
    """Classify a placement as friendly (bottom half) or enemy (top half).

    The recorder is the friendly player by convention for v1 player POV.
    """
    return Side.FRIENDLY if y > frame_height / 2 else Side.ENEMY
