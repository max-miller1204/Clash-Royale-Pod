"""Tests for the pure-logic helpers used by `analyze_video`.

Deliberately avoid importing torch / ultralytics / opencv: these tests
should run in any environment that has the project's pure-Python stack.
"""

from __future__ import annotations

from pathlib import Path

from crpod.detection.yolo import Detection
from crpod.pipeline import _assemble_replay, _tracks_to_plays
from crpod.tracking.bytetrack import Track
from crpod.types import CardPlay, HudState, Side


def _det(frame: int, cls: str, x: float, y: float) -> Detection:
    return Detection(
        frame=frame,
        cls=cls,
        confidence=0.9,
        xyxy=(x - 10, y - 15, x + 10, y + 15),
    )


def test_tracks_to_plays_anchors_at_first_detection():
    track = Track(
        track_id=1,
        cls="hog_rider",
        detections=[
            _det(10, "hog_rider", 240, 700),
            _det(11, "hog_rider", 245, 690),
            _det(12, "hog_rider", 250, 680),
        ],
    )
    plays = _tracks_to_plays([track], frame_height=960)
    assert len(plays) == 1
    play = plays[0]
    assert play.frame == 10
    assert play.x == 240
    assert play.y == 700
    assert play.card == "hog_rider"
    # known cost lookup
    assert play.elixir_cost == 4


def test_tracks_to_plays_classifies_friendly_when_below_midpoint():
    track = Track(
        track_id=1,
        cls="knight",
        detections=[_det(0, "knight", 200, 700), _det(1, "knight", 200, 700)],
    )
    plays = _tracks_to_plays([track], frame_height=960)
    assert plays[0].side is Side.FRIENDLY


def test_tracks_to_plays_classifies_enemy_when_above_midpoint():
    track = Track(
        track_id=1,
        cls="knight",
        detections=[_det(0, "knight", 200, 200), _det(1, "knight", 200, 200)],
    )
    plays = _tracks_to_plays([track], frame_height=960)
    assert plays[0].side is Side.ENEMY


def test_tracks_to_plays_drops_singleton_tracks():
    track = Track(
        track_id=1,
        cls="knight",
        detections=[_det(0, "knight", 200, 700)],
    )
    assert _tracks_to_plays([track], frame_height=960) == []


def test_tracks_to_plays_falls_back_to_default_cost_for_unknown_card():
    track = Track(
        track_id=1,
        cls="not_a_real_card",
        detections=[
            _det(0, "not_a_real_card", 200, 700),
            _det(1, "not_a_real_card", 200, 700),
        ],
    )
    play = _tracks_to_plays([track], frame_height=960)[0]
    # `card_cost` falls back to its default when the card is unknown —
    # we only assert it's a non-negative int rather than hard-coding the
    # default value, since changing the default is a one-line tweak.
    assert isinstance(play.elixir_cost, int)
    assert play.elixir_cost >= 0


def test_assemble_replay_uses_video_stem_as_replay_id():
    plays = [CardPlay(frame=0, card="knight", x=100, y=700, side=Side.FRIENDLY, elixir_cost=3)]
    hud: list[HudState] = []
    replay = _assemble_replay(
        video_path=Path("/some/dir/match_2026_04_30.mp4"),
        plays=plays,
        hud=hud,
        total_frames=1800,
        target_fps=10.0,
    )
    assert replay.replay_id == "match_2026_04_30"
    assert replay.arena == "video"
    assert replay.plays is plays
    assert replay.total_frames == 1800
    assert replay.fps == 10.0
