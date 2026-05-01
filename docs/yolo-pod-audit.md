# YOLO Pod-Video Detection Audit

**Model:** `output/models/crpod_v1_best.pt` (YOLOv8s, 201 classes — 155 real
Clash Royale classes + 44 `pad_*` filler from the KataCR class mapping)
**Footage:** 3 pod screen-recordings of player **Max**, captured 2026-05-01,
886 × 1920 portrait at 60 fps, 3–4 min each. Located under
`data/pod-videos/` (gitignored, large MOVs).
**Sampling:** 1 frame/s, default `conf=0.25`. ~180–240 frames per match.
A second pass at `conf=0.05` (essentially the noise floor) was used as a
sanity check on "the model fails to detect this card" claims — see the
"Sub-floor verification" section below.
**Reproducer:** `scripts/audit_pod_videos.py` (commits the script, not the
output). Raw per-detection CSV + annotated sample frames live in
`/tmp/yolo_audit_run/`.

> This is a research/documentation chunk, not a code fix. Nothing about
> the model, constants, or pipeline was changed. Findings here should
> drive a follow-up issue *if* the EV path on pod videos depends on the
> classes flagged below.

---

## TL;DR

The detector behaves like a model trained on the HF/KataCR distribution
when it is shown a different distribution: **HUD and tower geometry
transfer well, swarm troops transfer okay, and the pod's signature
big-troop cards (mega-knight, mini-pekka, wizard, giant-skeleton)
do not appear to fire at all on these recordings** — even when those
cards are clearly visible in the player's hand and almost certainly
deployed within each ~3-min match.

Spells (tornado, giant-snowball, fireball) are also effectively
invisible to the audit at 1 fps because their on-screen footprint lasts
~0.5 s — but `fireball` produced one detection at conf 0.93, hinting
the model can see it when sampled near peak.

The aspect ratio gap matters: HF training frames are 540 × 960
(0.5625), pod recordings are 886 × 1920 (0.461). Ultralytics letterboxes
during inference, but the resulting in-arena unit sizes do not match
the training distribution, which is consistent with the pattern we see
(towers fine, large troops fail).

---

## Pod's deck per match

Reconstructed by visually inspecting the bottom card-tray HUD across
~5 sampled timestamps per video. Where a card icon was ambiguous I
flag it with `?`.

### Match 1 (`ScreenRecording_05-01-2026 03-56-58_1.MOV`, 236 s)
1. **mega-knight** (7) — visible in `Next:` slot at 35 s, 70 s, 117 s, 198 s
2. **mini-pekka** (4)
3. **mega-minion** (3) — pod owner correction; previously mis-identified
   as `knight` in earlier drafts based on the masked-helmet icon
4. **giant-snowball** (2)
5. **tornado** (3)
6. **bomber** (2) — pod-owner confirmed
7. **goblin-drill** (4) — pod-owner confirmed
8. one slot I could not nail down across the sampled trays

### Match 2 (`ScreenRecording_05-01-2026 04-01-24_1.MOV`, 181 s)
Same deck as Match 1 (same player, deck appears unchanged).

### Match 3 (`ScreenRecording_05-01-2026 04-04-50_1.MOV`, 229 s)
Different archetype — wizard / fireball / heavy beatdown:
1. **wizard** (5)
2. **mini-pekka** (4)
3. masked-helmet 3-cost slot — previously labeled `knight`; pod owner
   does not run knight, so this is most likely **mega-minion** (the
   same correction as the Match 1/2 deck) but should be reconfirmed
4. **fireball** (4)
5. **giant-skeleton** (6)
6. **zappies** (4)
7. an orange "welder/cossack" 4-cost — possibly **mighty-miner** or
   **goblin-machine** (uncertain)
8. a 2-cost redhead-with-axe — likely a champion **ability** card slot
   (cost shown is the ability cost, not the base deploy cost)

---

## Per-card detection findings

The table aggregates across all 3 matches. `n` = total detections at
`conf>=0.25` across the ~640 sampled frames; `coverage` = fraction of
frames in which the class appeared at least once; `mean conf` = mean
score for the class over those detections. "Status" reflects what we
observed against the on-screen reality, not a benchmark metric.

### Pod-deck cards (the cards that gate EV quality on pod videos)

| Card | Match | n | Coverage | Mean conf | Status | Notes |
|------|-------|---|----------|-----------|--------|-------|
| `mega-knight` | 1, 2 | **0** | 0% | — | **fails** | Featured in player's hand frequently. **0 detections even at conf=0.05** in Match 1 — clean negative. Likely the worst single regression on pod footage. |
| `mini-pekka` | 1, 2, 3 | **0** | 0% | — | **fails** | In every match's deck. At conf=0.05 in Match 1: 1 lone sub-floor detection across 237 frames — effectively zero. |
| `mega-minion` | 1 | 4 | 1.7% | 0.49 | **flaky** | Slot #3 of pod's deck. Sparse but max conf only 0.57; the conf=0.05 rerun pulls in 21 detections (8.4% coverage) but median drops to 0.12 — model has a faint signal that mostly misses the floor. |
| `mega-minion` | 2 | 5 | 2.8% | 0.39 | **flaky** | Same pattern as Match 1; max conf 0.57. |
| `knight` (opponent / FP) | 1, 2, 3 | 2 / 22 / 5 | 0.8% / 12% / 2.2% | 0.32 / 0.48 / 0.29 | **n/a (not in pod deck)** | Pod owner confirmed they do not run knight in any of these matches. The `knight` detections here are either opponent-side units or HUD false-positives on card-tray icons. Match 2's higher count (max conf 0.94) is most plausibly an opponent's knight deployed in front of a tower. Detection quality data still useful for opponent-card EV inference, but not for pod-deck gating. |
| `tornado` | 1, 2 | **0** | 0% | — | **n/a (sampling)** | Spell visual ~0.3–1 s; 1 fps sampling almost guarantees misses. Cannot conclude detection failure from this alone. |
| `giant-snowball` | 1, 2 | **0** | 0% | — | **n/a (sampling)** | Same — sub-frame spell visual. |
| `bomber` | 1 | 1 | 0.4% | 0.33 | **flaky** | One detection above the default floor; at conf=0.05 the rerun shows 13 detections (5% coverage) but mean conf 0.14 — the model has a faint bomber-shaped signal that almost never crosses 0.25. Confounded by enemy-side bombers in opponent decks (bomber spawns from skeleton-barrel, etc.), so even the few default-conf hits aren't necessarily player-side. |
| `bomber` | 2 | 0 | 0% | — | **fails (this match)** | No detections at default conf in Match 2; no low-conf rerun was performed for this video. |
| `goblin-drill` | 1 | 10 | 4.2% | 0.36 | **good when above ground** | Drill spends most of its life buried; coverage is bound by how often it surfaces, not by the model. Max conf 0.65 — when visible, the model usually catches it. |
| `goblin-drill` | 2 | 2 | 1.1% | 0.50 | **good when above ground** | Same pattern — sparse but high-quality hits (max 0.61). |
| `wizard` | 3 | **0** | 0% | — | **fails** | Featured card in the deck. **0 detections even at conf=0.05** in Match 3 — clean negative. One stray `wizard` hit at 0.42 in Match 1 is unrelated (opponent had no wizard either; likely a HUD false positive). |
| `fireball` | 3 | 1 | 0.4% | 0.93 | **catches when sampled** | Single hit at very high confidence. Spell visual is ~0.3 s so absence elsewhere is sampling, not a detector failure. |
| `giant-skeleton` | 3 | **0** | 0% | — | **fires below floor** | Featured in deck. At conf=0.05 in Match 3: exactly 1 detection at conf 0.19 — the model has *some* weak signal but it never crosses the default 0.25 threshold. Different recommendation than mega-knight: a per-class threshold could partially recover this; a retrain probably can't be avoided either way. |
| `zappies` (`zappy`) | 1, 2, 3 | 109 / 121 / 73 | 21–34% | 0.54–0.55 | **good** | Most reliable pod-deck card across all matches. Detected even from the sparser HF distribution. |
| `mighty-miner` | 3 | 5 | 2.2% | 0.44 | **flaky** | Sparse but appears with reasonable confidence when seen. Low coverage may again be sampling, since the unit goes underground for parts of its life. |

### Cross-cutting observations (HUD / towers / opponent cards)

These aren't pod-deck cards, but they're load-bearing for the rest of
the pipeline (tower HP comes from the bar classes; emote and clock
classes anchor the HUD masking heuristics).

| Class | Behavior |
|-------|----------|
| `king-tower`, `queen-tower`, `cannoneer-tower`, `dagger-duchess-tower` | Detected in ~95–100% of frames, mean conf 0.5–0.85, max ~0.97. **Solid** — tower detection survives the distribution shift cleanly. |
| `tower-bar`, `bar`, `king-tower-bar` | 80–100% coverage, mean conf 0.6–0.9. **Solid** — feeds the OCR/HP-bar pipeline well. |
| `text`, `clock`, `emote` | Detected on every frame. Sometimes spurious (HUD card-tray icons get tagged as `clock` or `text` — cosmetic, not load-bearing). |
| `skeleton`, `goblin` | High counts (Match 1: 465 goblin detections, 95% coverage). Drilled into the persistent goblin cluster in Match 1 (frame 0 onward, box ≈ x∈[336,390], y∈[584,648]) — visually it sits on the **green decorative bush on top of the top king tower**. That single fixture explains most of the goblin count. Worth a follow-up if the EV path consumes raw goblin detections; the same king-tower bush will fire on every pod video taken in this arena. |
| `hog-rider`, `dark-prince`, `electro-wizard`, `bandit`, `witch`, `executioner`, `golem`, `sparky`, `royal-giant`, `lava-hound` | One- to two-shot detections from opponent decks. Confidences trend low (0.27–0.50) but plausibly real. Useful for the opponent half of EV inference if you treat anything below ~0.4 as noisy. |
| `tesla-evolution`, `goblin-drill`, `tombstone`, `cannon`, `inferno-tower` | Building detections fire intermittently; only `tesla-evolution` showed up across all 3 matches. Confidences modest. |

### False-positive surface to be aware of

While reviewing annotated frames the recurring artifacts were:

- **Bottom card-tray icons mis-detected as units.** The selected/next
  card icons get small bounding boxes labeled `knight`, `clock`,
  `queen-tower`, etc. The bottom ~10% of the frame is HUD; downstream
  consumers of detections should crop or mask it. The HF training
  distribution lacks the iPad-style `Next:` chip and the `Max / No
  Clan` banner, which is why these get hallucinated on.
- **Top hand chips similarly hallucinated.** The 4 enemy hand cards at
  the top get labeled (`queen-tower 0.33` on a card icon was visible
  in Match 1 frame 0).
- **King-tower decorative bush mis-classified as `goblin`.** Confirmed
  visually in Match 1: the green shrub painted on top of the top king
  tower fires a `goblin` detection at conf 0.25–0.50 in nearly every
  sampled frame. This single static fixture is the dominant source of
  the inflated goblin count.

---

## Sub-floor verification

To distinguish "model can't see this card" from "model sees it weakly
but the default `conf=0.25` filter erases it," each video that featured
a flagged big troop was re-run at `conf=0.05`. Result, restricted to
each video's deck-relevant cards:

| Card | Video re-run | Detections at conf=0.05 | Verdict |
|------|--------------|-------------------------|---------|
| `mega-knight` | Match 1 | 0 | Truly never fires; not a threshold issue. |
| `mini-pekka` | Match 1 | 1 (sub-floor) | Effectively never fires. |
| `wizard` | Match 3 | 0 | Truly never fires; not a threshold issue. |
| `giant-skeleton` | Match 3 | 1 at conf 0.19 | Fires below floor — the model has a faint signal that lowering the per-class threshold could partly recover. |
| `knight` | Match 1 | 11 (vs 2 at conf=0.25) | Most additions are sub-0.20 noise. The "raise the floor" recommendation below stands. |

Goblin counts in the same low-conf rerun (Match 3) jumped from 55 to
384 detections, mean conf 0.16, median 0.09 — i.e., the model
hallucinates faint goblin-shaped signals all over the screen at low
confidence. Lowering the global threshold is **not** a viable lever
for the missed big troops; it only adds noise.

---

## Risk to the EV pipeline (and recommendations)

Recommendations only — none implemented in this chunk per `CHUNK.md`.

1. **EV cannot price plays for cards the model never sees.** mega-knight,
   mini-pekka, wizard, and giant-skeleton are all marquee cards in this
   pod's two decks and produced **zero** detections in three matches
   sampled at 1 fps. If `analyze-video` runs on a pod video featuring
   these decks, the resulting `summary.json` will systematically
   under-report the player's offensive plays. Recommend gating EV
   output (or surfacing a warning) when `n_detections / video_duration`
   for player-side big troops sits at zero — the pipeline shouldn't
   silently produce confident EV deltas built only on swarm units.

2. **`bomber` needs the same per-class threshold treatment as `knight`.**
   In the player's deck for Match 1 and 2 the `bomber` class produces
   detections only at sub-floor confidence (median 0.10 in the
   conf=0.05 rerun, max 0.33 across 237 frames). Side-attribution is
   ambiguous because opponents may also run bomber-spawning cards. If
   the EV path consumes raw `bomber` detections, recommend either a
   raised per-class floor (~0.5) or a tracker-side filter that drops
   single-frame `bomber` hits with no corroborating motion.

3. **HUD region needs masking before detections feed downstream.** The
   bottom ~10% (card tray) and top ~12% (enemy hand chips + scoreboard)
   of pod frames produce false positives that look like real units to
   anything consuming the detection list. Recommend a viewer-aware
   mask in the video path (analogous to how `dataset/side.py` is split
   from `dataset/huggingface.py`) — or, cheaper, drop any detection
   whose centroid falls in `y < 0.13 * H` or `y > 0.87 * H` *before*
   tracking. This is a one-line guard, not a model change.

4. **Per-class confidence thresholds are the missing primitive.**
   Several classes show distinct, class-specific confidence profiles
   on pod footage (`mega-minion` and `bomber` produce mostly sub-floor
   hits; `goblin` produces high-coverage HUD false positives; `knight`
   detections, all opponent-side here, peak as high as 0.94 but mean
   sub-0.4). The cleanest fix is a per-class threshold dict consulted
   in `YoloDetector.infer` *before* it appends to its detection list,
   rather than the single global `conf=0.25`. **Recommendation only**;
   do not change the global default — that would silently drop other
   classes. This belongs in a separate PR with empirical per-class
   floors derived from a wider audit run.

5. **Spells are out of reach at 1 fps and likely 5 fps.** `fireball`
   produced one 0.93 detection; tornado and snowball produced none. The
   on-screen visuals are ~0.3–1 s. Either bump sampling fps for the
   spell-detection sub-step, or accept that the EV path will treat
   spells as invisible and rely on elixir-bar deltas to infer them.

6. **No critical, reproducible failure to escalate.** The model didn't
   crash, didn't swap two classes, and didn't degrade catastrophically
   on the *useful* parts of the frame (towers, swarms, HUD). The
   findings above are quality regressions and class-coverage gaps that
   warrant follow-up issues, not an emergency stop on this chunk.

---

## Reproducing the audit

```bash
# from repo root, with the project venv active (`uv sync` first if needed)
uv run python scripts/audit_pod_videos.py \
    --weights output/models/crpod_v1_best.pt \
    --videos data/pod-videos \
    --out /tmp/yolo_audit_run \
    --fps 1.0 \
    --conf 0.25 \
    --sample-every 30
```

Outputs per video:
- `<stem>__detections.csv` — one row per detection
- `<stem>__per_class.csv` — aggregated per-class stats
- `<stem>__frame_<NNNN>.jpg` — annotated samples (~every 30 s of game time)

The script is intentionally small (~140 lines) and does not import
anything from `src/crpod/` — it talks straight to ultralytics and cv2,
so it stays usable even if the pipeline package gets refactored.
