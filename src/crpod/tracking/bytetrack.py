"""ByteTrack wrapper — converts per-frame detections into stable tracks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
    """Wraps `supervision.ByteTrack`.

    The tracker is stateful across frames in a single video, so we
    instantiate it lazily on first `update` and feed detections one frame
    at a time even though the public API takes a flat sequence.
    """

    def __init__(self, frame_rate: int = 10) -> None:
        self.frame_rate = frame_rate
        self._tracker: Any = None

    def _lazy_load(self) -> None:
        if self._tracker is not None:
            return
        from supervision import ByteTrack

        self._tracker = ByteTrack(frame_rate=self.frame_rate)

    def update(self, detections: Sequence[Detection]) -> list[Track]:
        from supervision import Detections

        self._lazy_load()
        assert self._tracker is not None

        # Group input detections by frame so ByteTrack receives one
        # `Detections` batch per frame, in temporal order.
        per_frame: dict[int, list[Detection]] = defaultdict(list)
        for d in detections:
            per_frame[d.frame].append(d)

        # ByteTrack needs integer class IDs; we operate on string class
        # names. Maintain a local interning table so tracks carry the
        # original string back out.
        cls_to_id: dict[str, int] = {}
        id_to_cls: dict[int, str] = {}

        def _intern(cls: str) -> int:
            if cls not in cls_to_id:
                cls_to_id[cls] = len(cls_to_id)
                id_to_cls[cls_to_id[cls]] = cls
            return cls_to_id[cls]

        # Bucket tracked detections per tracker_id and remember the order
        # they were observed (so `Track.detections` is naturally frame-
        # ascending).
        buckets: dict[int, list[Detection]] = defaultdict(list)
        first_cls: dict[int, str] = {}

        for frame in sorted(per_frame):
            frame_dets = per_frame[frame]
            xyxy = np.array([d.xyxy for d in frame_dets], dtype=np.float32)
            confidence = np.array([d.confidence for d in frame_dets], dtype=np.float32)
            class_id = np.array([_intern(d.cls) for d in frame_dets], dtype=int)
            sv_dets = Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            tracked = self._tracker.update_with_detections(sv_dets)
            if tracked.tracker_id is None:
                continue
            for i, tid in enumerate(tracked.tracker_id):
                tid_int = int(tid)
                cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
                cls_name = id_to_cls.get(cls_id, "")
                x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                buckets[tid_int].append(
                    Detection(frame=frame, cls=cls_name, confidence=conf, xyxy=(x1, y1, x2, y2))
                )
                first_cls.setdefault(tid_int, cls_name)

        return [
            Track(track_id=tid, cls=first_cls[tid], detections=sorted(dets, key=lambda d: d.frame))
            for tid, dets in buckets.items()
        ]
