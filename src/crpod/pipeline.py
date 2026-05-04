"""End-to-end orchestrator: replay → events → features → model → viz.

Two ingest paths:

1. `analyze_hf_replay` — raw frame images from the HF TV-replay dataset.
   Runs YOLO detection on decoded frames to extract card placements.
2. `analyze_video` — raw mp4. Runs YOLO → ByteTrack → HUD OCR to reconstruct
   the same CardPlay/HudState stream, then joins path 1 downstream.

Both converge on the same `AnalysisResult` produced by the shared feature
and EV stages.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from crpod.constants import card_cost
from crpod.dataset.huggingface import HFReplayLoader
from crpod.dataset.side import infer_video_side
from crpod.dataset.video import VideoFrameIterator
from crpod.detection.yolo import YoloDetector
from crpod.features.elixir import elixir_leak, running_tempo
from crpod.features.interactions import build_interactions
from crpod.modeling.ev import EvModel, interaction_features
from crpod.ocr.hud import HudReader
from crpod.tracking.bytetrack import Track, Tracker
from crpod.types import CardPlay, HudState, Interaction, Replay, Side

_HUD_W = 540
_HUD_H = 960


def _rescale_for_hud(frame: np.ndarray) -> np.ndarray:
    """Resize an arbitrary-resolution frame to the 540x960 the HUD regions assume."""
    import cv2

    return cv2.resize(frame, (_HUD_W, _HUD_H), interpolation=cv2.INTER_AREA)


def _tracks_to_plays(tracks: list[Track], frame_height: int) -> list[CardPlay]:
    """Reduce tracks to one `CardPlay` per persistent object.

    Per `research.md` R2: anchor at the track's first sighting and discard
    tracks shorter than 2 frames as detection noise.
    """
    plays: list[CardPlay] = []
    for track in tracks:
        if len(track.detections) < 2:
            continue
        first = track.detections[0]
        cx, cy = first.center
        plays.append(
            CardPlay(
                frame=first.frame,
                card=first.cls,
                x=int(round(cx)),
                y=int(round(cy)),
                side=infer_video_side(cy, frame_height),
                elixir_cost=card_cost(first.cls),
            )
        )
    return plays


def _assemble_replay(
    video_path: Path,
    plays: list[CardPlay],
    hud: list[HudState],
    total_frames: int,
    target_fps: float,
) -> Replay:
    return Replay(
        replay_id=Path(video_path).stem,
        arena="video",
        plays=plays,
        hud=hud,
        total_frames=total_frames,
        fps=target_fps,
    )


@dataclass
class AnalysisResult:
    replay: Replay
    interactions: list[Interaction]
    feature_rows: list[dict[str, Any]]
    friendly_leak: float
    enemy_leak: float
    tempo: list[tuple[int, int]]
    ev_predictions: list[float] | None = None


def analyze_replay(replay: Replay, model: EvModel | None = None) -> AnalysisResult:
    interactions = build_interactions(replay.plays, hud=replay.hud, fps=replay.fps)
    rows = [interaction_features(i) for i in interactions]
    predictions = model.predict(rows) if (model and rows) else None
    return AnalysisResult(
        replay=replay,
        interactions=interactions,
        feature_rows=rows,
        friendly_leak=elixir_leak(replay.plays, replay.total_frames, Side.FRIENDLY),
        enemy_leak=elixir_leak(replay.plays, replay.total_frames, Side.ENEMY),
        tempo=running_tempo(replay.plays),
        ev_predictions=predictions,
    )


def analyze_hf_replay(
    arena: str,
    replay_id: str,
    yolo_weights: Path,
    model: EvModel | None = None,
) -> AnalysisResult:
    loader = HFReplayLoader(yolo_weights=yolo_weights)
    replay = loader.load(arena, replay_id)
    return analyze_replay(replay, model=model)


def analyze_video(
    video_path: Path,
    yolo_weights: Path,
    model: EvModel | None = None,
    target_fps: float = 10.0,
) -> AnalysisResult:
    """Run a local match video through the full CV pipeline.

    Materializes decoded frames into memory, runs YOLO + ByteTrack +
    HUD OCR, reduces tracks to `CardPlay` events, and hands a `Replay`
    off to `analyze_replay`. Progress is logged to stderr per
    `research.md` R7.
    """
    iterator = VideoFrameIterator(path=Path(video_path), target_fps=target_fps)
    frames: list[tuple[int, np.ndarray]] = list(iterator)
    if not frames:
        raise RuntimeError("detection stream empty — check weights match the game version")

    total_frames = len(frames)
    frame_height = frames[0][1].shape[0]

    detector = YoloDetector(yolo_weights)
    detections = detector.infer(frames)
    if not detections:
        raise RuntimeError("detection stream empty — check weights match the game version")

    tracker = Tracker(frame_rate=int(target_fps))
    tracks = tracker.update(detections)
    plays = _tracks_to_plays(tracks, frame_height=frame_height)

    hud_reader = HudReader()
    hud_states: list[HudState] = []
    progress_every = max(1, total_frames // 20)  # ~5%
    last_log_wall = time.monotonic()
    ocr_failures = 0
    for processed, (idx, frame) in enumerate(frames, start=1):
        try:
            hud_state = hud_reader.read(idx, _rescale_for_hud(frame))
        except Exception:
            ocr_failures += 1
            hud_state = HudState(
                frame=idx,
                friendly_elixir=0.0,
                enemy_elixir=None,
                friendly_king_hp=None,
                enemy_king_hp=None,
            )
        hud_states.append(hud_state)
        now = time.monotonic()
        if processed % progress_every == 0 or now - last_log_wall >= 15.0:
            ocr_pct = int(round(100 * ocr_failures / processed))
            print(
                f"[crpod] frame={processed}/{total_frames} plays={len(plays)} ocr_fail={ocr_pct}%",
                file=sys.stderr,
                flush=True,
            )
            last_log_wall = now

    replay = _assemble_replay(
        video_path=Path(video_path),
        plays=plays,
        hud=hud_states,
        total_frames=total_frames,
        target_fps=target_fps,
    )
    return analyze_replay(replay, model=model)
