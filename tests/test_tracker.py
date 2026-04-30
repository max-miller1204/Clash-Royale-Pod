from __future__ import annotations

from crpod.detection.yolo import Detection
from crpod.tracking.bytetrack import Tracker


def _det(frame: int, cls: str, x: float, y: float) -> Detection:
    # 40x60 box centered at (x, y) — large enough that ByteTrack's
    # IoU-association over consecutive frames matches it to itself.
    return Detection(
        frame=frame,
        cls=cls,
        confidence=0.9,
        xyxy=(x - 20, y - 30, x + 20, y + 30),
    )


def test_persistent_object_yields_one_track():
    # A single object slowly drifting right across many frames should
    # collapse to one Track. ByteTrack needs ~3 consecutive matches
    # before it confirms an id, so we feed it more than that.
    detections = [_det(i, "knight", x=200 + i * 2, y=400) for i in range(15)]
    tracks = Tracker(frame_rate=10).update(detections)

    assert len(tracks) == 1
    track = tracks[0]
    assert track.detections == sorted(track.detections, key=lambda d: d.frame)
    assert track.first_frame == track.detections[0].frame
    assert track.cls == "knight"


def test_transient_false_positive_does_not_emit_track():
    # A persistent object plus a single-frame stray detection far away.
    # ByteTrack requires multiple confirmations before promoting to a
    # confirmed track, so the singleton should not produce its own Track.
    detections = [_det(i, "knight", x=200 + i * 2, y=400) for i in range(15)]
    detections.append(_det(7, "skeletons", x=480, y=120))

    tracks = Tracker(frame_rate=10).update(detections)
    track_classes = {t.cls for t in tracks}
    assert "skeletons" not in track_classes
