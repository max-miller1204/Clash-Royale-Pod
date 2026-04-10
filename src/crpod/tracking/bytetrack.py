"""ByteTrack wrapper — converts per-frame detections into stable tracks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from crpod.detection.yolo import Detection


@dataclass
class Track:
    track_id: int
    cls: str
    detections: list[Detection] = field(default_factory=list)

    @property
    def first_frame(self) -> int:
        return self.detections[0].frame

    @property
    def last_frame(self) -> int:
        return self.detections[-1].frame

    def trajectory(self) -> list[tuple[float, float]]:
        return [d.center for d in self.detections]


class Tracker:
    """Wraps supervision.ByteTrack.

    Implementation detail left stubbed until detection weights exist.
    """

    def __init__(self, frame_rate: int = 10) -> None:
        self.frame_rate = frame_rate
        self._tracker = None

    def update(self, detections: Sequence[Detection]) -> list[Track]:
        raise NotImplementedError(
            "Tracker.update is pending a trained YOLO checkpoint. "
            "For the HF dataset path, placements are already structured — "
            "use crpod.dataset.huggingface instead."
        )
