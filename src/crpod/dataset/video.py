"""Raw video ingest — the 'custom replay' path through YOLO+ByteTrack+OCR."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VideoFrameIterator:
    """Yields (frame_index, np.ndarray BGR) at a target fps."""

    path: Path
    target_fps: float = 10.0

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        import cv2

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {self.path}")
        source_fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        step = max(1, round(source_fps / self.target_fps))
        idx = 0
        out_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                if idx % step == 0:
                    yield out_idx, frame
                    out_idx += 1
                idx += 1
        finally:
            cap.release()
