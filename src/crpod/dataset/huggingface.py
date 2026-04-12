"""Loader for chrisrca/clash-royale-tv-replays.

The dataset ships per-replay parquet files with raw frame images:
    frame_id: int64, image: struct<bytes: binary, path: string>, hash: string

Card placements are NOT pre-extracted — YOLO detection must be run on the
decoded frame images to produce CardPlay events.

Side assignment heuristic: HF dataset captures TV-royale third-person view
where both players are visible but side isn't labeled. We infer side from
y-coordinate (y > RIVER_Y → friendly_back half → friendly side). Replace
with per-replay ground truth if you find it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crpod.constants import RIVER_Y, card_cost
from crpod.detection.yolo import Detection, YoloDetector
from crpod.types import CardPlay, Replay, Side

DATASET_ID = "chrisrca/clash-royale-tv-replays"


def _infer_side(y: int) -> Side:
    if y < 0:
        return Side.UNKNOWN
    return Side.FRIENDLY if y >= RIVER_Y else Side.ENEMY


def _detection_to_card_play(det: Detection) -> CardPlay:
    cx, cy = det.center
    return CardPlay(
        frame=det.frame,
        card=det.cls,
        x=int(cx),
        y=int(cy),
        side=_infer_side(int(cy)),
        elixir_cost=card_cost(det.cls),
    )


@dataclass
class HFReplayLoader:
    """Streams replays from the HF hub.

    Downloads are cached via huggingface_hub's default cache dir. Set
    `token` if the dataset becomes gated.

    Requires `yolo_weights` pointing at a trained YOLO checkpoint to
    extract card placements from the raw frame images.
    """

    yolo_weights: Path | None = None
    cache_dir: Path | None = None
    token: str | None = None
    yolo_conf: float = 0.25

    def list_replays(self, arena: str | None = None) -> list[tuple[str, str]]:
        """Return (arena, replay_id) pairs available in the dataset."""
        from huggingface_hub import HfApi

        api = HfApi(token=self.token)
        files = api.list_repo_files(DATASET_ID, repo_type="dataset")
        out: list[tuple[str, str]] = []
        for f in files:
            if not f.endswith("/frames.parquet"):
                continue
            parts = f.split("/")
            if len(parts) < 3:
                continue
            a, r = parts[-3], parts[-2]
            if arena is None or a == arena:
                out.append((a, r))
        return out

    def load(self, arena: str, replay_id: str) -> Replay:
        if self.yolo_weights is None:
            raise ValueError(
                "YOLO weights are required to analyze HF replays. The dataset "
                "contains raw frame images — card placements must be extracted "
                "via YOLO detection. Pass --weights <path> to the CLI."
            )
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=DATASET_ID,
            filename=f"{arena}/{replay_id}/frames.parquet",
            repo_type="dataset",
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            token=self.token,
        )
        detector = YoloDetector(self.yolo_weights, conf=self.yolo_conf)
        return _parquet_to_replay(Path(path), arena=arena, replay_id=replay_id, detector=detector)

    def stream(self, arena: str | None = None) -> Iterator[Replay]:
        for a, r in self.list_replays(arena=arena):
            yield self.load(a, r)


def _decode_frames(path: Path) -> Iterator[tuple[int, np.ndarray]]:
    """Read the parquet and yield (frame_id, BGR ndarray) pairs."""
    import cv2
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["frame_id", "image"])
    frame_ids = table.column("frame_id").to_pylist()
    images = table.column("image").to_pylist()

    for frame_id, img in zip(frame_ids, images, strict=True):
        raw = img["bytes"]
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            yield int(frame_id), bgr


def _parquet_to_replay(path: Path, arena: str, replay_id: str, detector: YoloDetector) -> Replay:
    detections = detector.infer(_decode_frames(path))
    plays = [_detection_to_card_play(d) for d in detections]
    total_frames = max((p.frame for p in plays), default=0)
    return Replay(
        replay_id=replay_id,
        arena=arena,
        plays=plays,
        hud=[],
        total_frames=total_frames,
        fps=10.0,
    )


def load_replay(arena: str, replay_id: str, yolo_weights: Path, **kwargs: object) -> Replay:
    """Convenience wrapper."""
    return HFReplayLoader(yolo_weights=yolo_weights, **kwargs).load(arena, replay_id)  # type: ignore[arg-type]
