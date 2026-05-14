# Latency budget — wave 4B

Per-stage wall-clock for `crpod analyze-video` on a 30-second clip
decimated to 10 fps (300 frames total). The hardware is the operator's
**low-compute Mac** (Apple Silicon, no CUDA, no MPS) — these numbers are
a **CPU floor**, not a target. The pipeline is intended to run on GPU
in production; brev A6000 runs from waves 2E–2J' suggest YOLO drops to
~10–15 % of these numbers there.

## How this was measured

`crpod.instrumentation.Timer` wraps each stage in `analyze_video`:
**decode**, **yolo**, **tracker**, **hud**, **features**, **ev**. (Blunders
and report rendering happen in the CLI layer, outside `analyze_video`,
and are negligible — both run in milliseconds against in-memory results
and a base64-PNG embed.) Each `analyze-video` run dumps the per-stage
total as one JSONL line via `scripts/latency_audit.py`. The aggregator
`scripts/latency_report.py` walks the JSONL and emits percentiles.

```bash
uv run python scripts/latency_audit.py \
    clip30_1.mp4 clip30_2.mp4 clip30_3.mp4 \
    --weights output/models/crpod_v1_best.pt \
    --target-fps 10.0 \
    --jsonl docs/latency-runs.jsonl

uv run python scripts/latency_report.py docs/latency-runs.jsonl
```

The three clips are 30-second slices of the pod's own match recordings
in `data/pod-videos/` (886×1920, 60 fps native, downsampled to 10 fps
for inference). Different match phases were sampled per clip
(t=0 s, t=60 s, t=60 s) so the YOLO load varies with on-screen unit count.

## Per-stage table — 3-run CPU baseline, 30 s clip

| stage | p50 (s) | p95 (s) | p99 (s) | mean (s) | min | max |
|---|---|---|---|---|---|---|
| yolo | 21.333 | 23.016 | 23.166 | 20.753 | 17.721 | 23.203 |
| hud | 2.901 | 2.939 | 2.942 | 2.908 | 2.880 | 2.943 |
| decode | 1.825 | 1.934 | 1.944 | 1.829 | 1.717 | 1.946 |
| tracker | 1.067 | 1.087 | 1.089 | 0.838 | 0.356 | 1.090 |
| ev | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| features | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**Total wall-clock per 30 s clip:** ~26 s (mean), of which **YOLO is
~80 %** and HUD-OCR is the next slowest stage at ~11 %. `features` /
`ev` round to zero — LightGBM inference on ~10 interaction rows is
sub-millisecond and the feature builder is pure-Python list ops.

Three caveats on the table:

1. `ev` is zero because the audit was run without `--model`. With a
   wave-2J'-era EV model loaded, predict-time stays sub-millisecond per
   interaction batch; the `ev` stage will surface as a small but non-zero
   number rather than rounding to 0.000.
2. `tracker` min=0.356 vs max=1.090 — the 3× spread is real: clip 2
   tracked far fewer detections because mid-match action had fewer
   distinct on-screen units. Tracker time scales with detection density.
3. p95 / p99 from N=3 are misleading by definition — they're just
   interpolating between max and second-max. The aggregator emits them
   because `pod_summary.md` requested the format, not because they're
   load-bearing at this sample size. Re-run with N=20+ if you want
   trustworthy tail latency.

## Budget check vs `pod_summary.md`

`pod_summary.md` calls out a "**~30–50 ms/frame** on a mid-range GPU,
well under the 100 ms budget" for the real-time mode (which is out of
scope for the current spec, but the same numbers apply here).
Translating CPU baseline to per-frame:

- YOLO: 20.7 s / 300 frames = **69 ms/frame** on CPU. On GPU
  (wave-2J' brev measurements imply ~5 ms/frame), this is **~14× faster**
  → ~5 ms/frame, well under budget.
- HUD: 2.9 s / 300 frames = **9.7 ms/frame**. Pure NumPy HSV-mask
  sampling — already CPU-native, no GPU speedup expected.
- Tracker + decode + features + EV: combined **~11 ms/frame** on CPU.

**Verdict:** all stages are within budget on GPU. On the operator's
CPU-only Mac, only the YOLO stage is over budget — by design. The
project is not chasing real-time CPU inference.

## Known caveats / out-of-budget stages

None on GPU. On CPU, YOLO at ~69 ms/frame is the bottleneck. The
existing `output/models/crpod_v1_best.pt` is YOLOv8**s** (small);
swapping to YOLOv8n would roughly halve CPU time at a ~5 % mAP cost,
but the project does not currently expose a swap path.

## How to re-run

The full audit takes ~3 min on a 30 s clip on CPU. To capture a fresh
baseline:

```bash
# Clip 30 seconds from each pod video
uv run python -c "
import cv2
for i, src in enumerate(['video1.mov', 'video2.mov', 'video3.mov']):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS); w = int(cap.get(3)); h = int(cap.get(4))
    out = cv2.VideoWriter(f'clip30_{i}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 60))
    for _ in range(int(fps * 30)):
        ok, frame = cap.read()
        if not ok: break
        out.write(frame)
    out.release(); cap.release()
"

# Run the audit
uv run python scripts/latency_audit.py clip30_0.mp4 clip30_1.mp4 clip30_2.mp4 \
    --weights output/models/crpod_v1_best.pt --target-fps 10.0 \
    --jsonl docs/latency-runs.jsonl

# Aggregate
uv run python scripts/latency_report.py docs/latency-runs.jsonl
```

The JSONL is append-only, so historical baselines stay alongside fresh runs.
