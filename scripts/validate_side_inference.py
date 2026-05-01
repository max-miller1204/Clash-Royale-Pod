"""Validate the `_infer_side` heuristic against real HF detections.

Runs YOLO on one mid-match frame from each of several replays and renders
an overlay PNG showing:
- bboxes color-coded by inferred side (FRIENDLY = green, ENEMY = red,
  UNKNOWN = gray)
- the `RIVER_Y` constant as a horizontal line (the actual decision boundary)
- the frame midpoint as a second horizontal line (where the river *should*
  fall if `RIVER_Y` matches the frame size)
- tower detections labeled, since their fixed positions anchor the call

The eyeball test: do tower detections cluster on the side they belong to,
and does `RIVER_Y` fall on the visual river? If yes, `_infer_side` is
validated. If `RIVER_Y` is off the river, the constant or frame size is
wrong. If towers straddle the boundary inconsistently across replays,
escalate to per-replay calibration.

Usage:
    uv run python scripts/validate_side_inference.py \\
        --weights output/models/crpod_v1_best.pt \\
        --out output/side_validation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download

from crpod.constants import RIVER_Y
from crpod.dataset.huggingface import DATASET_ID, _decode_frames
from crpod.detection.cards import _infer_side
from crpod.detection.yolo import Detection, YoloDetector
from crpod.types import Side

DEFAULT_REPLAYS: list[tuple[str, str]] = [
    # (arena, replay_id) — picked across arenas to surface camera variation if any
    ("arena_05", ""),
    ("arena_15", ""),
    ("arena_22", ""),
    ("arena_28", ""),
    ("arena_31", ""),
]

SIDE_COLORS: dict[Side, tuple[int, int, int]] = {
    Side.FRIENDLY: (0, 200, 0),  # green
    Side.ENEMY: (0, 0, 220),  # red
    Side.UNKNOWN: (160, 160, 160),  # gray
}

TOWER_KEYWORDS = ("tower", "king")


def _is_tower(cls: str) -> bool:
    return any(k in cls.lower() for k in TOWER_KEYWORDS)


def _list_first_replay(arena: str, token: str | None = None) -> str:
    """Return the first replay_id available for `arena`."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    files = api.list_repo_files(DATASET_ID, repo_type="dataset")
    for f in files:
        if not f.endswith("/frames.parquet"):
            continue
        parts = f.split("/")
        if len(parts) < 3:
            continue
        a, r = parts[-3], parts[-2]
        if a == arena:
            return r
    raise SystemExit(f"no replays found for {arena}")


def _pick_mid_frame(parquet_path: Path) -> tuple[int, np.ndarray]:
    """Decode all frames and return (frame_id, image) at the middle index."""
    frames = list(_decode_frames(parquet_path))
    if not frames:
        raise SystemExit(f"no decodable frames in {parquet_path}")
    return frames[len(frames) // 2]


def _draw(img: np.ndarray, dets: list[Detection]) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    # Horizontal reference lines
    midpoint_y = h // 2
    cv2.line(out, (0, RIVER_Y), (w, RIVER_Y), (255, 200, 0), 2)
    cv2.putText(
        out,
        f"RIVER_Y={RIVER_Y}",
        (5, RIVER_Y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 200, 0),
        1,
        cv2.LINE_AA,
    )
    if midpoint_y != RIVER_Y:
        cv2.line(out, (0, midpoint_y), (w, midpoint_y), (255, 0, 255), 1)
        cv2.putText(
            out,
            f"frame_mid={midpoint_y} (h={h})",
            (5, midpoint_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        "FRIENDLY",
        (5, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        SIDE_COLORS[Side.FRIENDLY],
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "ENEMY",
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        SIDE_COLORS[Side.ENEMY],
        2,
        cv2.LINE_AA,
    )

    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        cy = (y1 + y2) // 2
        side = _infer_side(cy)
        color = SIDE_COLORS[side]
        thickness = 3 if _is_tower(d.cls) else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{d.cls}{'*' if _is_tower(d.cls) else ''}"
        cv2.putText(
            out,
            label,
            (x1, max(y1 - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    return out


def _verdict_line(dets: list[Detection], img_shape: tuple[int, int]) -> str:
    h, w = img_shape
    towers = [d for d in dets if _is_tower(d.cls)]
    enemy_towers = sum(1 for d in towers if _infer_side(int(d.center[1])) is Side.ENEMY)
    friendly_towers = sum(1 for d in towers if _infer_side(int(d.center[1])) is Side.FRIENDLY)
    return (
        f"frame={h}x{w}  RIVER_Y={RIVER_Y}  midpoint={h // 2}  "
        f"towers: {friendly_towers} friendly / {enemy_towers} enemy / {len(towers)} total"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("output/side_validation"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument(
        "--arena",
        action="append",
        help="repeat to override default arena set (uses first replay in each)",
    )
    p.add_argument(
        "--replay",
        nargs=2,
        metavar=("ARENA", "REPLAY_ID"),
        action="append",
        help="explicit (arena, replay_id) pair; repeat for multiple",
    )
    args = p.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"weights not found: {args.weights}")

    args.out.mkdir(parents=True, exist_ok=True)

    if args.replay:
        replays = [(a, r) for a, r in args.replay]
    else:
        arenas = args.arena or [a for a, _ in DEFAULT_REPLAYS]
        replays = [(a, _list_first_replay(a)) for a in arenas]

    detector = YoloDetector(args.weights, conf=args.conf)
    print(f"validating side inference on {len(replays)} replays")
    print(f"RIVER_Y={RIVER_Y}, weights={args.weights}, conf={args.conf}")
    print()

    for arena, replay_id in replays:
        print(f"==> {arena} / {replay_id}")
        parquet = Path(
            hf_hub_download(
                repo_id=DATASET_ID,
                filename=f"{arena}/{replay_id}/frames.parquet",
                repo_type="dataset",
            )
        )
        frame_id, img = _pick_mid_frame(parquet)
        h, w = img.shape[:2]
        dets = detector.infer(iter([(frame_id, img)]))

        overlay = _draw(img, dets)
        out_path = args.out / f"{arena}_{replay_id[:8]}_frame{frame_id:05d}.png"
        cv2.imwrite(str(out_path), overlay)

        print(f"    {_verdict_line(dets, (h, w))}")
        print(f"    -> {out_path}")
        print()


if __name__ == "__main__":
    main()
