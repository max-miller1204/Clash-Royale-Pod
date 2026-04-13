#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

echo "=== [1/5] Installing dependencies ==="
# Install PyTorch for the instance's CUDA version first, then ultralytics.
# Hyperstack A4000 ships driver 570 / CUDA 12.8 — default pip pulls cu130
# which doesn't work. cu128 matches the driver.
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -q ultralytics pyyaml

echo "=== [2/5] Cloning KataCR detection dataset ==="
cd ~
if [ ! -d Clash-Royale-Detection-Dataset ]; then
    git clone --depth 1 https://github.com/wty-yy/Clash-Royale-Detection-Dataset.git
fi
SRC=~/Clash-Royale-Detection-Dataset/images/part2
OUT=~/yolo_train

echo "=== [3/5] Converting 12-col labels to 5-col YOLOv8 format ==="
python3 -c "
import os, random, yaml
from pathlib import Path

src = Path('$SRC').resolve()
out = Path('$OUT').resolve()
images_out = out / 'images'
labels_out = out / 'labels'
images_out.mkdir(parents=True, exist_ok=True)
labels_out.mkdir(parents=True, exist_ok=True)

txt_paths = [p for p in sorted(src.rglob('*.txt')) if p.parent != src]
records, total_boxes, skipped = [], 0, 0
for src_txt in txt_paths:
    src_jpg = src_txt.with_suffix('.jpg')
    if not src_jpg.exists():
        skipped += 1
        continue
    rel = src_txt.relative_to(src)
    dst_txt = labels_out / rel
    dst_jpg = (images_out / rel).with_suffix('.jpg')
    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    dst_jpg.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for line in src_txt.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            lines.append(' '.join(parts[:5]))
    dst_txt.write_text('\n'.join(lines) + ('\n' if lines else ''))
    total_boxes += len(lines)
    if dst_jpg.is_symlink() or dst_jpg.exists():
        dst_jpg.unlink()
    dst_jpg.symlink_to(src_jpg)
    records.append(str(dst_jpg))

rng = random.Random(42)
rng.shuffle(records)
n_val = max(1, int(len(records) * 0.1))
val, train = records[:n_val], records[n_val:]
(out / 'train.txt').write_text('\n'.join(train) + '\n')
(out / 'val.txt').write_text('\n'.join(val) + '\n')

katacr_yaml = src / 'ClashRoyale_detection.yaml'
cfg = yaml.safe_load(katacr_yaml.read_text())
data = {'path': str(out), 'train': 'train.txt', 'val': 'val.txt', 'names': cfg['names']}
(out / 'data.yaml').write_text(yaml.safe_dump(data, sort_keys=False))
print(f'converted {len(records)} frames  ({total_boxes} boxes)')
print(f'train: {len(train)}  val: {len(val)}')
"

echo "=== [4/5] Training YOLOv8s — 50 epochs ==="
yolo train \
    data=$OUT/data.yaml \
    model=yolov8s.pt \
    epochs=50 \
    imgsz=640 \
    batch=32 \
    device=0 \
    project=$OUT/runs \
    name=crpod_v1 \
    exist_ok=True

echo "=== [5/5] Done! ==="
echo "Weights at: $OUT/runs/crpod_v1/weights/best.pt"
ls -lh $OUT/runs/crpod_v1/weights/
nvidia-smi
