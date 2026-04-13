"""Spike: run a YOLO checkpoint on HF replay frames and save overlay PNGs.

Usage:
    uv run python scripts/detect_overlay.py \
        --weights detector1_v0.7.13.pt \
        --arena arena_15 \
        --replay 00a91415-... \
        --frames 0,30,60,120,240 \
        --out output/overlays
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download

from crpod.dataset.huggingface import DATASET_ID, _decode_frames
from crpod.detection.yolo import Detection, YoloDetector


def _draw(img: np.ndarray, dets: list[Detection]) -> np.ndarray:
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d.cls} {d.confidence:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--arena", required=True)
    p.add_argument("--replay", required=True)
    p.add_argument("--frames", default="0,30,60,120,240",
                   help="comma-separated frame indices to render")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", type=Path, default=Path("output/overlays"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    wanted = {int(x) for x in args.frames.split(",")}

    parquet_path = Path(hf_hub_download(
        repo_id=DATASET_ID,
        filename=f"{args.arena}/{args.replay}/frames.parquet",
        repo_type="dataset",
    ))

    frames = [(i, f) for i, f in _decode_frames(parquet_path) if i in wanted]
    if not frames:
        raise SystemExit(f"no frames matched {wanted} in {parquet_path}")

    detector = YoloDetector(args.weights, conf=args.conf)
    detections = detector.infer(iter(frames))
    by_frame: dict[int, list[Detection]] = {}
    for d in detections:
        by_frame.setdefault(d.frame, []).append(d)

    for idx, img in frames:
        overlay = _draw(img, by_frame.get(idx, []))
        out_path = args.out / f"{args.arena}_{args.replay[:8]}_frame{idx:05d}.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"{out_path}  ({len(by_frame.get(idx, []))} detections)")


if __name__ == "__main__":
    main()
