"""Thin wrapper around ultralytics YOLO."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Detection:
    frame: int
    cls: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class YoloDetector:
    """Loads a trained YOLO checkpoint and runs it on frames."""

    def __init__(self, weights: Path, conf: float = 0.25) -> None:
        self.weights = Path(weights)
        self.conf = conf
        self._model: Any = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        self._model = YOLO(str(self.weights))

    def infer(self, frames: Iterable[tuple[int, np.ndarray]]) -> list[Detection]:
        self._lazy_load()
        assert self._model is not None
        out: list[Detection] = []
        for idx, frame in frames:
            results = self._model.predict(frame, conf=self.conf, verbose=False)
            for r in results:
                names = r.names
                for box in r.boxes:
                    cls_id = int(box.cls.item())
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                    out.append(
                        Detection(
                            frame=idx,
                            cls=names[cls_id],
                            confidence=float(box.conf.item()),
                            xyxy=(x1, y1, x2, y2),
                        )
                    )
        return out
