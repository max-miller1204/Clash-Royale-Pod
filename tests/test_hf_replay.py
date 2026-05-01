from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pytest

from crpod.dataset import huggingface as hf
from crpod.detection.yolo import Detection, YoloDetector


def _det(frame: int, cls: str, x: float, y: float) -> Detection:
    """40x60 box centered at (x, y) — large enough for ByteTrack IoU matching."""
    return Detection(
        frame=frame,
        cls=cls,
        confidence=0.9,
        xyxy=(x - 20, y - 30, x + 20, y + 30),
    )


class _StubDetector:
    """Returns a fixed detection list regardless of input frames."""

    def __init__(self, dets: list[Detection]) -> None:
        self._dets = dets

    def infer(self, frames: Iterable[tuple[int, object]]) -> list[Detection]:
        list(frames)  # exhaust in case _decode_frames is real
        return self._dets


@pytest.fixture
def silenced_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace _decode_frames with a no-op so tests don't need a real parquet."""
    monkeypatch.setattr(hf, "_decode_frames", lambda path: iter([]))


def test_parquet_to_replay_collapses_persistent_object(silenced_decode: None) -> None:
    """A unit detected across many frames should produce ONE CardPlay, not one per frame."""
    detections = [_det(i, "knight", x=200 + i * 2, y=400) for i in range(15)]
    detector = _StubDetector(detections)

    replay = hf._parquet_to_replay(
        Path("ignored.parquet"),
        arena="arena_15",
        replay_id="r1",
        detector=cast(YoloDetector, detector),
    )

    knight_plays = [p for p in replay.plays if p.card == "knight"]
    assert len(knight_plays) == 1, f"expected 1 knight CardPlay, got {len(knight_plays)}"


def test_parquet_to_replay_drops_singleton_noise(silenced_decode: None) -> None:
    """A single-frame stray detection shouldn't survive as its own CardPlay.

    Uses the KATACR class name "skeleton" (singular) — which `to_card_play`
    maps to canonical card "skeletons" (plural) — so under the broken impl
    the singleton would emit a CardPlay; under the fix the <2-frame track
    filter drops it.
    """
    persistent = [_det(i, "knight", x=200 + i * 2, y=400) for i in range(15)]
    stray = [_det(7, "skeleton", x=480, y=120)]
    detector = _StubDetector(persistent + stray)

    replay = hf._parquet_to_replay(
        Path("ignored.parquet"),
        arena="arena_15",
        replay_id="r1",
        detector=cast(YoloDetector, detector),
    )

    cards = {p.card for p in replay.plays}
    assert "skeletons" not in cards, f"singleton stray leaked into plays: {cards}"


def test_parquet_to_replay_preserves_katacr_mapping(silenced_decode: None) -> None:
    """Hyphenated KATACR class names should map to canonical underscored card names."""
    detections = [_det(i, "the-log", x=300, y=500) for i in range(10)]
    detector = _StubDetector(detections)

    replay = hf._parquet_to_replay(
        Path("ignored.parquet"),
        arena="arena_15",
        replay_id="r1",
        detector=cast(YoloDetector, detector),
    )

    log_plays = [p for p in replay.plays if p.card == "log"]
    assert len(log_plays) == 1, (
        f"expected exactly 1 'log' CardPlay (KATACR mapping + tracker collapse), "
        f"got {len(log_plays)}: {[p.card for p in replay.plays]}"
    )
