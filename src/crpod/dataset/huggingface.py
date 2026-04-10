"""Loader for chrisrca/clash-royale-tv-replays.

The dataset ships per-replay parquet files with (card, x, y, frame, arena)
rows — structured card placements already extracted. This bypasses YOLO for
dataset replays; YOLO is only needed for custom video ingest.

Schema (per row): card: str, png_bytes: bytes, x: int16, y: int16,
arena: str, replay: str, frame: int16.

Side assignment heuristic: HF dataset captures TV-royale third-person view
where both players are visible but side isn't labeled. We infer side from
y-coordinate (y > RIVER_Y → friendly_back half → friendly side). Replace
with per-replay ground truth if you find it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from crpod.constants import RIVER_Y, card_cost
from crpod.types import CardPlay, Replay, Side

DATASET_ID = "chrisrca/clash-royale-tv-replays"


def _infer_side(y: int) -> Side:
    if y < 0:
        return Side.UNKNOWN
    return Side.FRIENDLY if y >= RIVER_Y else Side.ENEMY


@dataclass
class HFReplayLoader:
    """Streams replays from the HF hub.

    Downloads are cached via huggingface_hub's default cache dir. Set
    `token` if the dataset becomes gated.
    """

    cache_dir: Path | None = None
    token: str | None = None

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
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=DATASET_ID,
            filename=f"{arena}/{replay_id}/frames.parquet",
            repo_type="dataset",
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            token=self.token,
        )
        return _parquet_to_replay(Path(path), arena=arena, replay_id=replay_id)

    def stream(self, arena: str | None = None) -> Iterator[Replay]:
        for a, r in self.list_replays(arena=arena):
            yield self.load(a, r)


def _parquet_to_replay(path: Path, arena: str, replay_id: str) -> Replay:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["card", "x", "y", "frame"])
    cards = table.column("card").to_pylist()
    xs = table.column("x").to_pylist()
    ys = table.column("y").to_pylist()
    frames = table.column("frame").to_pylist()

    plays: list[CardPlay] = []
    for card, x, y, frame in zip(cards, xs, ys, frames, strict=True):
        if card is None:
            continue
        plays.append(
            CardPlay(
                frame=int(frame),
                card=str(card),
                x=int(x),
                y=int(y),
                side=_infer_side(int(y)),
                elixir_cost=card_cost(str(card)),
            )
        )

    total_frames = max((p.frame for p in plays), default=0)
    return Replay(
        replay_id=replay_id,
        arena=arena,
        plays=plays,
        hud=[],
        total_frames=total_frames,
        fps=10.0,
    )


def load_replay(arena: str, replay_id: str, **kwargs: object) -> Replay:
    """Convenience wrapper."""
    return HFReplayLoader(**kwargs).load(arena, replay_id)  # type: ignore[arg-type]
