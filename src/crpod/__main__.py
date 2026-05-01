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
import random
import sys
from pathlib import Path

from crpod.dataset.huggingface import HFReplayLoader
from crpod.modeling.ev import EvModel, compute_per_card_stats
from crpod.pipeline import analyze_hf_replay, analyze_replay
from crpod.types import Interaction

_HOLDOUT_SEED = 0
_HOLDOUT_FRACTION = 0.2


def _training_target(interaction: Interaction) -> float | None:
    """Princess-tower HP-delta EV target for one interaction.

    Returns `None` when any of the four princess deltas is unreadable so
    callers can drop the row.
    """
    delta = interaction.tower_hp_delta
    fl = delta.get("friendly_left")
    fr = delta.get("friendly_right")
    el = delta.get("enemy_left")
    er = delta.get("enemy_right")
    if fl is None or fr is None or el is None or er is None:
        return None
    return float((fl + fr) - (el + er))


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
        print(f"error: weights file not found: {args.weights}", file=sys.stderr)
        sys.exit(1)
    if args.model is not None and not args.model.exists():
        print(f"error: EV model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)
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
        print(f"error: weights file not found: {args.weights}", file=sys.stderr)
        sys.exit(1)
    if not args.out.parent.exists():
        print(f"error: output directory does not exist: {args.out.parent}", file=sys.stderr)
        sys.exit(1)
    if args.max_replays <= 0:
        print("error: --max-replays must be > 0", file=sys.stderr)
        sys.exit(1)
    loader = HFReplayLoader(yolo_weights=args.weights)
    # Each entry is a per-replay bundle so we can split at the replay level
    # and avoid adjacent-frame leakage across folds.
    per_replay: list[tuple[list[dict], list[float], list[Interaction]]] = []
    seen = 0
    dropped = 0
    available = loader.list_replays(arena=args.arena)[: args.max_replays]
    for arena, replay_id in available:
        replay = loader.load(arena, replay_id)
        result = analyze_replay(replay)
        rep_rows: list[dict] = []
        rep_targets: list[float] = []
        rep_interactions: list[Interaction] = []
        for interaction, row in zip(result.interactions, result.feature_rows, strict=True):
            seen += 1
            target = _training_target(interaction)
            if target is None:
                dropped += 1
                continue
            rep_rows.append(row)
            rep_targets.append(target)
            rep_interactions.append(interaction)
        if rep_rows:
            per_replay.append((rep_rows, rep_targets, rep_interactions))
    if seen:
        pct = round(100 * dropped / seen)
        print(
            f"dropped {dropped}/{seen} training rows ({pct}% — unreadable HUD)",
            file=sys.stderr,
        )
    if not per_replay:
        print("no training data collected", file=sys.stderr)
        return 1

    rng = random.Random(_HOLDOUT_SEED)
    shuffled = list(per_replay)
    rng.shuffle(shuffled)
    n_replays = len(shuffled)
    if n_replays >= 2:
        split_idx = max(1, min(n_replays - 1, round(n_replays * (1 - _HOLDOUT_FRACTION))))
    else:
        split_idx = n_replays  # nothing to hold out — single replay
    train_bundle = shuffled[:split_idx]
    holdout_bundle = shuffled[split_idx:]

    train_rows: list[dict] = []
    train_targets: list[float] = []
    train_interactions: list[Interaction] = []
    for r, t, i in train_bundle:
        train_rows.extend(r)
        train_targets.extend(t)
        train_interactions.extend(i)
    holdout_rows: list[dict] = []
    holdout_targets: list[float] = []
    for r, t, _ in holdout_bundle:
        holdout_rows.extend(r)
        holdout_targets.extend(t)

    print(
        f"training on {len(train_rows)} interactions from {len(train_bundle)} replays "
        f"(holdout {len(holdout_rows)} interactions from {len(holdout_bundle)} replays)"
    )
    model = EvModel()
    model.fit(train_rows, train_targets)
    model.per_card_stats = compute_per_card_stats(train_interactions, train_targets)
    print(f"per_card_stats: {len(model.per_card_stats)} cards with ≥5 train samples")

    # Top-10 anchor cards by training-fold sample count (for docs/ev-validation.md).
    counts: dict[str, int] = {}
    for interaction in train_interactions:
        if not interaction.friendly_plays:
            continue
        counts[interaction.friendly_plays[0].card] = (
            counts.get(interaction.friendly_plays[0].card, 0) + 1
        )
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("top-10 anchor cards by train-fold n_samples:")
    for card, n in top:
        if card in model.per_card_stats:
            median, std = model.per_card_stats[card]
            print(f"  {card}: n={n} median={median:+.1f} std={std:.1f}")
        else:
            print(f"  {card}: n={n} (excluded — <5 samples)")

    if holdout_rows:
        from scipy.stats import spearmanr

        preds = model.predict(holdout_rows)
        mae = sum(abs(p - t) for p, t in zip(preds, holdout_targets, strict=True)) / len(
            holdout_targets
        )
        spearman_result = spearmanr(preds, holdout_targets)
        rho = float(spearman_result.statistic)
        print(f"holdout MAE: {mae:.2f}")
        print(f"holdout Spearman: {rho:.3f}")
    else:
        print("holdout MAE: n/a (single-replay training set)", file=sys.stderr)
        print("holdout Spearman: n/a (single-replay training set)", file=sys.stderr)

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
