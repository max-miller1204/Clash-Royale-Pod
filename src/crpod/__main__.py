"""crpod CLI.

Usage:
    uv run crpod list-replays [--arena ARENA]
    uv run crpod analyze ARENA REPLAY_ID [--out DIR]
    uv run crpod train --out MODEL_PATH [--arena ARENA] [--max-replays N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from crpod.dataset.huggingface import HFReplayLoader
from crpod.modeling.ev import EvModel
from crpod.pipeline import analyze_hf_replay, analyze_replay


def _cmd_list(args: argparse.Namespace) -> int:
    loader = HFReplayLoader()  # no weights needed for listing
    for arena, replay_id in loader.list_replays(arena=args.arena):
        print(f"{arena}\t{replay_id}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    model = EvModel.load(args.model) if args.model else None
    result = analyze_hf_replay(
        args.arena, args.replay_id, yolo_weights=args.weights, model=model
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "replay_id": result.replay.replay_id,
        "arena": result.replay.arena,
        "n_plays": len(result.replay.plays),
        "n_interactions": len(result.interactions),
        "friendly_leak": round(result.friendly_leak, 2),
        "enemy_leak": round(result.enemy_leak, 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    try:
        from crpod.visualization.plots import elixir_timeseries, placement_heatmap

        placement_heatmap(result.replay.plays, out / "placements.png")
        elixir_timeseries(result.tempo, out / "tempo.png")
    except Exception as e:
        print(f"[warn] viz skipped: {e}", file=sys.stderr)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    loader = HFReplayLoader(yolo_weights=args.weights)
    rows: list[dict] = []
    targets: list[float] = []
    available = loader.list_replays(arena=args.arena)[: args.max_replays]
    for arena, replay_id in available:
        replay = loader.load(arena, replay_id)
        result = analyze_replay(replay)
        for interaction, row in zip(result.interactions, result.feature_rows, strict=True):
            rows.append(row)
            targets.append(float(interaction.elixir_trade))
    if not rows:
        print("no training data collected", file=sys.stderr)
        return 1
    print(f"training on {len(rows)} interactions from {len(available)} replays")
    model = EvModel()
    model.fit(rows, targets)
    model.save(Path(args.out))
    print(f"saved model → {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crpod")
    sub = p.add_subparsers(dest="cmd", required=True)

    lst = sub.add_parser("list-replays", help="list available HF replays")
    lst.add_argument("--arena", default=None)
    lst.set_defaults(func=_cmd_list)

    ana = sub.add_parser("analyze", help="analyze one HF replay")
    ana.add_argument("arena")
    ana.add_argument("replay_id")
    ana.add_argument("--weights", required=True, type=Path, help="path to trained YOLO weights")
    ana.add_argument("--out", default="output/analysis")
    ana.add_argument("--model", default=None, type=Path)
    ana.set_defaults(func=_cmd_analyze)

    tr = sub.add_parser("train", help="train EV model on HF replays")
    tr.add_argument("--weights", required=True, type=Path, help="path to trained YOLO weights")
    tr.add_argument("--out", required=True, type=Path)
    tr.add_argument("--arena", default=None)
    tr.add_argument("--max-replays", type=int, default=50)
    tr.set_defaults(func=_cmd_train)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
