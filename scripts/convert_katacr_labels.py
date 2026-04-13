"""Convert KataCR detection labels to standard 5-column YOLOv8 format.

KataCR's .txt files encode 12 values per line:
    class cx cy w h belonging 0 0 0 0 0 0

Ultralytics YOLOv8 expects 5:
    class cx cy w h

This script walks Clash-Royale-Detection-Dataset/images/part2/, mirrors the
directory tree under output/katacr_yolo/ in ultralytics manifest layout
(images/ and labels/ siblings), symlinks the JPGs to avoid duplicating
647 MB of data, writes 5-column label files, and produces train/val
manifests plus a data.yaml ready for `yolo train`.

Usage:
    uv run python scripts/convert_katacr_labels.py
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml


def convert_label(src: Path, dst: Path) -> int:
    """Write src's first-5-column form to dst. Returns number of boxes."""
    out_lines: list[str] = []
    for line in src.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        out_lines.append(" ".join(parts[:5]))
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    return len(out_lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        type=Path,
        default=Path("Clash-Royale-Detection-Dataset/images/part2"),
    )
    p.add_argument(
        "--katacr-yaml",
        type=Path,
        default=Path("/home/max/KataCR/katacr/yolov8/ClashRoyale_detection.yaml"),
    )
    p.add_argument("--out", type=Path, default=Path("output/katacr_yolo"))
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src_root: Path = args.src.resolve()
    out_root: Path = args.out.resolve()
    images_out = out_root / "images"
    labels_out = out_root / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Only .txt files that are nested under a subdir (skip top-level manifests).
    txt_paths = [p for p in sorted(src_root.rglob("*.txt")) if p.parent != src_root]

    records: list[Path] = []
    total_boxes = 0
    skipped_no_jpg = 0

    for src_txt in txt_paths:
        src_jpg = src_txt.with_suffix(".jpg")
        if not src_jpg.exists():
            skipped_no_jpg += 1
            continue

        rel = src_txt.relative_to(src_root)
        dst_txt = labels_out / rel
        dst_jpg = (images_out / rel).with_suffix(".jpg")
        dst_txt.parent.mkdir(parents=True, exist_ok=True)
        dst_jpg.parent.mkdir(parents=True, exist_ok=True)

        total_boxes += convert_label(src_txt, dst_txt)

        if dst_jpg.is_symlink() or dst_jpg.exists():
            dst_jpg.unlink()
        dst_jpg.symlink_to(src_jpg)
        records.append(dst_jpg)

    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * args.val_frac))
    val, train = records[:n_val], records[n_val:]

    (out_root / "train.txt").write_text("\n".join(str(p) for p in train) + "\n")
    (out_root / "val.txt").write_text("\n".join(str(p) for p in val) + "\n")

    katacr_cfg = yaml.safe_load(args.katacr_yaml.read_text())
    data_yaml = {
        "path": str(out_root),
        "train": "train.txt",
        "val": "val.txt",
        "names": katacr_cfg["names"],
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    print(f"converted {len(records)} frames  ({total_boxes} boxes)")
    print(f"skipped (no matching .jpg): {skipped_no_jpg}")
    print(f"train: {len(train)}  val: {len(val)}")
    print(f"data.yaml: {out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
