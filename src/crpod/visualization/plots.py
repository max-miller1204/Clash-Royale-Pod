"""Matplotlib/seaborn plots for post-game reports."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from crpod.types import CardPlay, Replay


def placement_heatmap(plays: Iterable[CardPlay], out: Path) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    xs = np.array([p.x for p in plays if p.x >= 0])
    ys = np.array([p.y for p in plays if p.y >= 0])
    if xs.size == 0:
        raise ValueError("no positive placements to plot")

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.hist2d(xs, ys, bins=(24, 40), cmap="magma")
    ax.set_xlim(0, 480)
    ax.set_ylim(810, 0)
    ax.set_title("Placement heatmap")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def elixir_timeseries(tempo: list[tuple[int, int]], out: Path) -> Path:
    import matplotlib.pyplot as plt

    frames = [t[0] for t in tempo]
    diffs = [t[1] for t in tempo]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(frames, diffs)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("frame")
    ax.set_ylabel("friendly − enemy elixir spent")
    ax.set_title("Running tempo")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def ev_breakdown(replay: Replay, card_evs: dict[str, float], out: Path) -> Path:
    import matplotlib.pyplot as plt

    cards = sorted(card_evs, key=lambda c: card_evs[c], reverse=True)
    values = [card_evs[c] for c in cards]
    fig, ax = plt.subplots(figsize=(8, max(3, len(cards) * 0.3)))
    ax.barh(cards, values)
    ax.set_title(f"Per-card EV — {replay.replay_id}")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
