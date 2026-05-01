# Data Quality: HF-dataset Side Inference

## Question

`src/crpod/dataset/huggingface.py` decodes raw frame images from
`chrisrca/clash-royale-tv-replays` and runs YOLO over them to extract
`CardPlay` events. The dataset does not label which side (friendly /
enemy) each detection belongs to. We infer side from the y-coordinate
of the detection center via `_infer_side` in
`src/crpod/detection/cards.py`:

```python
def _infer_side(y: int) -> Side:
    if y < 0:
        return Side.UNKNOWN
    return Side.FRIENDLY if y >= RIVER_Y else Side.ENEMY
```

with `RIVER_Y = ARENA_H // 2 = 480` (frame midpoint, since HF frames
are 540×960). The EV model trains on this dataset, so a wrong split
mislabels every training row.

This doc records the wave-1A audit: do the inferred sides match what a
human sees in the frame?

## Method

`scripts/validate_side_inference.py` was extended with a `--per-arena
N` flag. We then ran:

```bash
uv run python scripts/validate_side_inference.py \
  --weights /Users/max/Clash-Royale-Pod/output/models/crpod_v1_best.pt \
  --out output/side_validation \
  --arena arena_15 --per-arena 10
```

For each replay the script:

1. downloads `arena_15/<replay_id>/frames.parquet` from the HF hub,
2. picks the frame at index `len(frames) // 2` (mid-match — both
   players are typically on the board),
3. runs YOLO at conf=0.25,
4. classifies every detection's center via `_infer_side` and renders an
   overlay PNG: green box = FRIENDLY, red = ENEMY, gray = UNKNOWN. The
   blue horizontal line is `RIVER_Y`; magenta is the frame midpoint
   (they coincide here).

The eyeball test: (a) does the blue line fall on the visible river, and
(b) do enemy princess towers cluster above the line, friendly princess
towers below?

## Results — 10 arena_15 replays

All ten frames are 540×960 (`frame=960x540` in script output =
`(h, w)`), `RIVER_Y=480`, midpoint=480.

| # | replay_id (prefix) | frame | friendly / enemy / unknown | tower friendly / enemy / total | overlay PNG | eyeball |
|---|---|---|---|---|---|---|
| 1 | `00a91415` | 292 | 15 / 11 / 0 | 5 / 5 / 10 | `arena_15_00a91415_frame00292.png` | river line on river; enemy princess towers (red) at y≈210-280, friendly (green) at y≈680-740. Friendly knight at the bottom-half lane correctly green. |
| 2 | `02c3eb19` | 513 | 13 / 11 / 0 | 6 / 5 / 11 | `arena_15_02c3eb19_frame00513.png` | river line on river; symmetric tower detection; friendly green box around our king tower at y≈800. |
| 3 | `0364a998` | 384 | 13 / 6 / 0 | 6 / 2 / 8 | `arena_15_0364a998_frame00384.png` | tower count is asymmetric (only 2 enemy towers detected), but every detection is on the *correct* side of the line. Asymmetry is a YOLO recall thing, not a side-classification bug. |
| 4 | `07eba4b4` | 294 | 11 / 11 / 0 | 7 / 5 / 12 | `arena_15_07eba4b4_frame00294.png` | enemy princess towers + crown HUD all red; friendly side all green; line on river. |
| 5 | `18ebf665` | 396 | 9 / 9 / 0 | 5 / 6 / 11 | `arena_15_18ebf665_frame00396.png` | "60 SECONDS LEFT" overlay text sits on the river line but does not produce any rogue detection. Tower split balanced. |
| 6 | `1b27e668` | 317 | 15 / 8 / 0 | 7 / 4 / 11 | `arena_15_1b27e668_frame00317.png` | friendly side has more activity at the snapshot moment (deck swap mid-match). All units correctly green; enemy princess towers correctly red. |
| 7 | `1dc26cb7` | 312 | 10 / 9 / 0 | 5 / 5 / 10 | `arena_15_1dc26cb7_frame00312.png` | clean balanced split; tower detection symmetric. |
| 8 | `226fefa9` | 532 | 11 / 10 / 0 | 5 / 4 / 9 | `arena_15_226fefa9_frame00532.png` | friendly units crossing the bridge to attack — the y=480 boundary lands above the bridge crossings, so units on/just past the bridge are correctly labeled by which side of the river they sit on. |
| 9 | `27125b76` | 429 | 12 / 14 / 0 | 6 / 6 / 12 | `arena_15_27125b76_frame00429.png` | symmetric (6/6 towers), 60-seconds-left overlay; line on river. |
| 10 | `295a4ed7` | 345 | 11 / 19 / 0 | 5 / 5 / 10 | `arena_15_295a4ed7_frame00345.png` | busiest frame (30 detections); enemy push in progress so naturally more enemy units. All correctly classified. |

### Aggregate (10 replays, mid-match frame each)

- Detections: **228 total** across 10 frames (avg ~23 per frame).
- Side split: **120 friendly / 108 enemy / 0 unknown** — 53% / 47%, in
  the ballpark of the 50/50 expected by symmetry of the camera framing.
- Towers: **57 friendly / 47 enemy / 104 total** — both sides see
  roughly the expected 5-7 tower detections per frame (1 king + 2
  princess per side, plus occasional duplicate detections of the
  6 visible towers and HUD-band crown icons that match `cls.lower()`'s
  `tower`/`king` keyword).
- **No detection ever landed on the wrong side** in human eyeball
  inspection of the 10 overlays.

## Verdict

**The `_infer_side` heuristic is correct on arena_15.** The
`RIVER_Y=480` constant (set in commit `b99b1bc`) lands on the visible
river in every inspected frame, and YOLO detections cluster on the
correct side of that line in 10/10 replays.

**No code change required.** `src/crpod/dataset/huggingface.py`,
`src/crpod/detection/cards.py:_infer_side`, and
`src/crpod/constants.py:RIVER_Y` are all left untouched. Wave 2's EV
re-training can proceed on this dataset with confidence that the side
labels are accurate.

### Caveats / limits of this audit

- We sampled one mid-match frame per replay. Camera framing in the HF
  dataset appears static across a single replay, but we did not
  validate per-frame across the full timeline of any replay. If a
  future arena introduces dynamic camera offset (e.g. screen
  shake / zoom), `RIVER_Y` would need re-validation there.
- Audit was scoped to `arena_15` per the SPEC; the constants comment
  in `src/crpod/constants.py` already notes a previous five-arena
  pass (arena_05/15/22/28/31) at the same RIVER_Y=480.
- Detections at y < 0 (degenerate bbox) get UNKNOWN. We saw zero of
  these in 228 detections.
- `BRIDGE_LEFT_X` / `BRIDGE_RIGHT_X` are out of scope for this audit;
  the pre-existing comment in `constants.py` flags those for separate
  re-validation.
