"""Audit `output/models/crpod_v1_best.pt` on the pod's own match videos.

The model was trained on HF + KataCR data (540x960 frames). The pod plays the
real game on real devices, which produces different aspect ratios, skins,
emote overlays, and screen elements. This script does not retrain anything —
it just runs the model over a few sampled frames per video and prints a
per-class detection summary so we can see which cards survive the
distribution shift and which silently fail.

Usage:
    uv run python scripts/audit_pod_videos.py \\
        --weights output/models/crpod_v1_best.pt \\
        --videos data/pod-videos \\
        --out /tmp/yolo_audit_out \\
        --fps 1.0 \\
        --conf 0.25

Outputs:
- `<out>/<video_stem>__detections.csv` — one row per detection
- `<out>/<video_stem>__per_class.csv` — aggregate stats per class
- `<out>/<video_stem>__frame_<N>.jpg` — annotated samples (every Kth frame)
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# `pad_*` classes are KataCR mapping filler — they cannot legitimately
# appear on a real Clash Royale screen, so they are surfaced separately
# as a sanity check rather than mixed into the per-card report.
PAD_PREFIX = "pad_"


def _iter_frames(path: Path, target_fps: float) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    step = max(1, round(src_fps / target_fps))
    out: list[tuple[int, np.ndarray]] = []
    idx = 0
    out_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            out.append((out_idx, frame))
            out_idx += 1
        idx += 1
    cap.release()
    return out


def _draw(
    frame: np.ndarray,
    boxes: list[tuple[str, float, tuple[float, float, float, float]]],
) -> np.ndarray:
    img = frame.copy()
    for cls, conf, (x1, y1, x2, y2) in boxes:
        color = (0, 255, 0) if conf >= 0.5 else (0, 165, 255)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{cls} {conf:.2f}"
        cv2.putText(img, label, (int(x1), max(int(y1) - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def audit(
    weights: Path,
    video: Path,
    out_dir: Path,
    target_fps: float,
    conf: float,
    sample_every: int,
) -> None:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = model.names
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = video.stem.replace(" ", "_")
    det_csv = out_dir / f"{stem}__detections.csv"
    cls_csv = out_dir / f"{stem}__per_class.csv"

    frames = _iter_frames(video, target_fps)
    n_frames = len(frames)
    print(f"[{video.name}] sampled {n_frames} frames at ~{target_fps} fps")

    per_class_confs: dict[str, list[float]] = defaultdict(list)
    per_class_frames: dict[str, set[int]] = defaultdict(set)

    with det_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "cls", "confidence", "x1", "y1", "x2", "y2"])

        for i, (fidx, frame) in enumerate(frames):
            results = model.predict(frame, conf=conf, verbose=False)
            boxes_for_draw: list[tuple[str, float, tuple[float, float, float, float]]] = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls.item())
                    cls = str(names[cls_id])
                    c = float(box.conf.item())
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                    w.writerow([
                        fidx, cls, f"{c:.4f}",
                        f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                    ])
                    per_class_confs[cls].append(c)
                    per_class_frames[cls].add(fidx)
                    boxes_for_draw.append((cls, c, (x1, y1, x2, y2)))

            if i % sample_every == 0:
                annotated = _draw(frame, boxes_for_draw)
                cv2.imwrite(str(out_dir / f"{stem}__frame_{fidx:04d}.jpg"), annotated)

    with cls_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "cls", "n_detections", "n_frames", "frame_coverage_pct",
            "mean_conf", "median_conf", "max_conf", "is_pad",
        ])
        rows = []
        for cls, confs in per_class_confs.items():
            n = len(confs)
            nf = len(per_class_frames[cls])
            cov = 100.0 * nf / max(n_frames, 1)
            mean_c = float(np.mean(confs))
            median_c = float(np.median(confs))
            max_c = float(np.max(confs))
            is_pad = cls.startswith(PAD_PREFIX)
            rows.append((cls, n, nf, cov, mean_c, median_c, max_c, is_pad))
        rows.sort(key=lambda r: (-r[1]))
        for r in rows:
            cls, n, nf, cov, mn, md, mx, ispad = r
            w.writerow([
                cls, n, nf,
                f"{cov:.1f}", f"{mn:.3f}", f"{md:.3f}", f"{mx:.3f}",
                "1" if ispad else "0",
            ])

    print(f"[{video.name}] wrote {det_csv.name}, {cls_csv.name}")
    print(f"[{video.name}] top 15 classes by detection count:")
    for r in rows[:15]:
        cls, n, nf, cov, mn, md, mx, ispad = r
        tag = "  (PAD)" if ispad else ""
        print(
            f"  {cls:30s}  n={n:5d}  frames={nf:4d} ({cov:5.1f}%)  "
            f"mean={mn:.2f}  med={md:.2f}  max={mx:.2f}{tag}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--videos", type=Path, required=True, help="directory of pod videos")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument(
        "--sample-every", type=int, default=20,
        help="save an annotated frame every Nth sampled frame (1 fps default => every 20s)",
    )
    ap.add_argument("--glob", default="*.MOV")
    args = ap.parse_args()

    vids = sorted(args.videos.glob(args.glob))
    if not vids:
        raise SystemExit(f"no videos matched {args.videos}/{args.glob}")
    args.out.mkdir(parents=True, exist_ok=True)
    for v in vids:
        audit(args.weights, v, args.out, args.fps, args.conf, args.sample_every)


if __name__ == "__main__":
    main()
