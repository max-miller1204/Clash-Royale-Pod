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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crpod.dataset.huggingface import HFReplayLoader
from crpod.features.elixir import elixir_leak, running_tempo
from crpod.features.interactions import build_interactions
from crpod.modeling.ev import EvModel, interaction_features
from crpod.types import Interaction, Replay, Side


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
    interactions = build_interactions(replay.plays)
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
) -> AnalysisResult:
    """Runs the full CV pipeline. Requires trained YOLO weights.

    Left as an integration shim — weights don't exist until the Data &
    Detection sub-team trains them (pod_summary weeks 2-4).
    """
    raise NotImplementedError(
        "analyze_video requires trained YOLO weights and the OCR region "
        "tuning pass. Run analyze_hf_replay against the HF dataset until "
        "weights land."
    )
