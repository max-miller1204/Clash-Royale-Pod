"""crpod CLI.

Usage:
    uv run crpod list-replays [--arena ARENA]
    uv run crpod analyze ARENA REPLAY_ID [--out DIR]
    uv run crpod analyze-video VIDEO_PATH --weights PATH [--out DIR] [--model PATH] [--target-fps N]
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


def _positive_float(value: str) -> float:
    f = float(value)
    if f <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return f


def _cmd_list(args: argparse.Namespace) -> int:
    loader = HFReplayLoader()  # no weights needed for listing
    for arena, replay_id in loader.list_replays(arena=args.arena):
        print(f"{arena}\t{replay_id}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    if not args.weights.exists():
        sys.exit(f"weights file not found: {args.weights}")
    model = EvModel.load(args.model) if args.model else None
    result = analyze_hf_replay(args.arena, args.replay_id, yolo_weights=args.weights, model=model)
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


def _cmd_analyze_video(args: argparse.Namespace) -> int:
    video_path = Path(args.video_path)
    weights_path = Path(args.weights)
    model_path = Path(args.model) if args.model else None

    # Pre-validate paths before any heavy imports so the four
    # 2-second fail-fast paths in `contracts/cli.md` are honored.
    if not video_path.exists():
        print(f"error: video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    if not weights_path.exists():
        print(f"error: weights file not found: {weights_path}", file=sys.stderr)
        sys.exit(1)
    if model_path is not None and not model_path.exists():
        print(f"error: EV model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    if args.target_fps <= 0:
        print("error: --target-fps must be > 0", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out) if args.out else Path("output/analysis") / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    from crpod.pipeline import analyze_video

    model = EvModel.load(model_path) if model_path else None
    try:
        result = analyze_video(
            video_path,
            yolo_weights=weights_path,
            model=model,
            target_fps=args.target_fps,
        )
    except FileNotFoundError as e:
        if "cannot open video" in str(e):
            print(
                f"error: cannot decode video (codec or truncated file): {video_path}",
                file=sys.stderr,
            )
        else:
            print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    summary = {
        "replay_id": result.replay.replay_id,
        "arena": result.replay.arena,
        "source_video": str(video_path),
        "n_plays": len(result.replay.plays),
        "n_interactions": len(result.interactions),
        "friendly_leak": round(result.friendly_leak, 2),
        "enemy_leak": round(result.enemy_leak, 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    try:
        from crpod.visualization.plots import elixir_timeseries, placement_heatmap

        placement_heatmap(result.replay.plays, out_dir / "placements.png")
        elixir_timeseries(result.tempo, out_dir / "tempo.png")
    except Exception as e:
        print(f"[warn] viz skipped: {e}", file=sys.stderr)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    if not args.weights.exists():
        sys.exit(f"weights file not found: {args.weights}")
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

    av = sub.add_parser("analyze-video", help="analyze a local match video file")
    av.add_argument("video_path", type=Path, help="path to a local video file")
    av.add_argument("--weights", required=True, type=Path, help="path to trained YOLO weights")
    av.add_argument(
        "--out",
        default=None,
        type=Path,
        help="output directory (default: output/analysis/<video-stem>/)",
    )
    av.add_argument("--model", default=None, type=Path, help="optional EV model (.joblib)")
    av.add_argument(
        "--target-fps",
        type=_positive_float,
        default=10.0,
        help="frames per second to decimate the video to (default: 10)",
    )
    av.set_defaults(func=_cmd_analyze_video)

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
