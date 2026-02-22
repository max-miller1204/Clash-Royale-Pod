# Team Workflow & Collaboration Guide

**Team Size:** 8 (1 pod leader + 7 members)

---

## Sub-Teams

Each sub-team owns a stage of the pipeline. Members should pair up within their team — one person drives, the other reviews.

| Sub-team | Members | Owns | Output |
| -------- | ------- | ---- | ------ |
| **Data & Detection** | 2–3 | Replay collection, annotation in Roboflow, YOLO training | COCO-format annotations + trained YOLO model weights |
| **Tracking & Feature Engineering** | 2–3 | ByteTrack integration, Tesseract OCR, event log construction | Per-match event log (CSV/Parquet) with timestamps, card plays, elixir counts, damage |
| **Modeling & Visualization** | 2 | LightGBM training, EV calculations, heatmaps, final report | Trained model, visualizations, written analysis |

**Pod Leader (Max):** Floats across all three teams — code review, architecture decisions, unblocking, and keeping interfaces consistent.

---

## Git Workflow

### Branch Strategy

```
main              ← stable, working code only
├── data/*        ← Data & Detection sub-team branches
├── tracking/*    ← Tracking & Feature Engineering branches
├── modeling/*    ← Modeling & Viz branches
└── docs/*        ← Documentation and writeup
```

### Rules

1. **Never push directly to `main`.** All changes go through pull requests.
2. **At least 1 review** on every PR before merging (ideally from someone on a different sub-team).
3. **Keep PRs small.** One feature or fix per PR. If a PR touches more than 5 files, consider splitting it.
4. **Write a short PR description** — what changed and why. Doesn't need to be long, just clear.
5. **Delete branches after merging.**

### Commit Messages

Use a simple prefix convention:

```
data: add annotation script for tower labels
tracking: fix ByteTrack ID assignment bug
model: train LightGBM v2 with placement features
viz: add elixir time-series plot
docs: update README with setup instructions
```

---

## Weekly Cadence

| Day | Activity | Format | Time |
| --- | -------- | ------ | ---- |
| **Monday** | Standup | Async (Discord/Slack) or 15-min call | Each person posts: what I did, what I'm doing, what's blocking me |
| **Wednesday** | Mid-week check-in | Async | Sub-team leads flag any blockers or interface issues |
| **Friday** | Weekly update | Async post to `#general` | Each sub-team posts a short summary of what shipped |

### Bi-Weekly (every other week)

- **Full team sync** — 30 min call to demo progress, discuss design decisions, and re-prioritize if needed.

---

## Communication

| Channel | Purpose |
| ------- | ------- |
| `#general` | Announcements, weekly updates, cross-team questions |
| `#data-detection` | Data & Detection sub-team discussion |
| `#tracking-features` | Tracking & Feature Engineering discussion |
| `#modeling-viz` | Modeling & Visualization discussion |
| `#help` | Stuck? Post here. Anyone on the team can answer. |

### Norms

- **Respond within 24 hours** on weekdays (even if it's "I'll look at this tomorrow").
- **Use threads** to keep channels readable.
- **Tag people directly** if you need something from them specifically.

---

## Interface Contracts

These are the agreements between sub-teams on what data formats they pass to each other. **Define these by the end of week 2.**

### Data & Detection → Tracking & Feature Engineering

```
output/
├── frames/                  # Extracted frames as PNGs
│   ├── match_001_frame_0001.png
│   └── ...
├── annotations/             # COCO-format JSON
│   └── annotations.json
└── models/
    └── yolo_best.pt         # Trained YOLO weights
```

### Tracking & Feature Engineering → Modeling & Visualization

```
output/
└── events/
    └── match_001_events.csv
```

Event log schema:

| Column | Type | Description |
| ------ | ---- | ----------- |
| `match_id` | string | Unique match identifier |
| `timestamp` | float | Seconds into the match |
| `player` | int | 1 (top) or 2 (bottom) |
| `card` | string | Card name (e.g., "hog_rider") |
| `elixir_cost` | int | Elixir spent |
| `placement_x` | float | X coordinate on the arena grid |
| `placement_y` | float | Y coordinate on the arena grid |
| `damage_dealt` | float | Damage dealt by this card play |
| `tower_damage` | float | Damage dealt to towers specifically |
| `elixir_bar` | int | Player's elixir at time of play |

---

## Task Tracking

Use **GitHub Issues** (or Trello if the team prefers a board view).

- Create one issue per task. Label by sub-team: `data`, `tracking`, `modeling`, `docs`.
- Assign issues to specific people — no unassigned work.
- Close issues via PR (include `Closes #123` in the PR description).

### Weekly Task Targets by Phase

| Weeks | Data & Detection | Tracking & Features | Modeling & Viz |
| ----- | ---------------- | ------------------- | -------------- |
| 1–3 | Collect replays, annotate 500+ frames, train YOLO v1 | Set up ByteTrack + Tesseract scaffolding, define event log schema | Set up repo structure, research LightGBM, draft viz mockups |
| 4–6 | Iterate on YOLO (fix bad detections, add training data) | Build full tracking + OCR pipeline, produce first event logs | Begin feature engineering from event logs, first EV estimates |
| 7–8 | Support tracking team with detection fixes | Polish event logs, handle edge cases | Train LightGBM, run statistical tests, build visualizations |
| 9–10 | Help with integration testing | Help with integration testing | Final report, dashboard, presentation prep |

---

## Dev Environment (Devcontainer)

The repo includes a devcontainer so everyone runs the same environment regardless of OS. This eliminates "works on my machine" issues with PyTorch, OpenCV, FFmpeg, and Tesseract.

### Prerequisites

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Install [VS Code](https://code.visualstudio.com/) with the **Dev Containers** extension

### Setup

1. Clone the repo
2. Open the repo folder in VS Code
3. When prompted, click **"Reopen in Container"** (or run `Dev Containers: Reopen in Container` from the command palette)
4. Wait for the container to build (~2–5 min the first time, cached after that)
5. You're done — Python, FFmpeg, Tesseract, and all pip dependencies are pre-installed

### What's included

| Tool | Version | Purpose |
| ---- | ------- | ------- |
| Python | 3.11 | Primary language |
| FFmpeg | system | Video/frame extraction |
| Tesseract | system | OCR for HUD reading |
| OpenCV | via pip | Image processing |
| PyTorch | via pip | Model training |
| Ultralytics YOLO | via pip | Object detection |

### Notes

- The `data/` folder is bind-mounted from your host machine so large files (replays, frames) stay local and don't bloat the container.
- This is a **CPU-only** setup for development and testing. For GPU training (YOLO, LightGBM), use Google Colab or a university GPU cluster.
- If you add a new pip dependency, add it to `requirements.txt` and rebuild the container (`Dev Containers: Rebuild Container`).

---

## Onboarding Checklist (Week 1)

Every team member should complete this by end of week 1:

- [ ] Install Docker Desktop and VS Code with the Dev Containers extension
- [ ] Clone the repo and open it in the devcontainer
- [ ] Verify the environment works: `python -c "import torch; import cv2; print('ready')"`
- [ ] Join all Discord/Slack channels
- [ ] Read `pod_summary.md` end-to-end
- [ ] Read this workflow doc
- [ ] Know which sub-team you're on and who your teammate(s) are
- [ ] Complete one small task (annotate 10 frames, write a utility function, etc.)
