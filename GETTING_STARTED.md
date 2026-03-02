# Getting Started — Week 1 Kickoff

This guide covers what each sub-team should do on day 1 and throughout week 1. Complete the shared onboarding first, then dive into your sub-team tasks.

---

## Onboarding (Everyone — Day 1)

Complete every item before starting sub-team work.

- [ ] Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and [VS Code](https://code.visualstudio.com/) with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
- [ ] Fork the repo on GitHub (click the **Fork** button)
- [ ] Clone **your fork** (not the main repo):
  ```bash
  git clone https://github.com/<your-username>/Clash-Royale-Pod.git
  cd Clash-Royale-Pod
  ```
- [ ] Add the main repo as upstream:
  ```bash
  git remote add upstream https://github.com/<main-repo-owner>/Clash-Royale-Pod.git
  ```
- [ ] Open in VS Code and click **"Reopen in Container"** when prompted (or run `Dev Containers: Reopen in Container` from the command palette)
- [ ] Wait for the container to build (~3–5 min the first time)
- [ ] Verify the environment:
  ```bash
  uv run python -c "import torch; import cv2; import ultralytics; print('ready')"
  bash .devcontainer/test_tools.sh
  ```
- [ ] Make sure your virtual environment is activated — your terminal prompt should start with `(workspace)`. If not:
  ```bash
  source .venv/bin/activate
  ```
- [ ] Authenticate at least one AI coding tool:
  ```bash
  claude    # Log in with your Claude account
  gemini    # Log in with your Google account
  codex     # Log in with your ChatGPT account
  ```
- [ ] Join all Discord/Slack channels (`#general`, `#data-detection`, `#tracking-features`, `#modeling-viz`, `#help`)
- [ ] Read [pod_summary.md](pod_summary.md) end-to-end
- [ ] Read [workflow.md](workflow.md) end-to-end
- [ ] Know which sub-team you're on and who your teammate(s) are

### Verify the Fork + PR Workflow

Every member should open one trivial PR to confirm the workflow works before doing real work.

1. Create a branch:
   ```bash
   git checkout -b docs/add-my-name
   ```
2. Make a small change (e.g., add your name to a contributors section or fix a typo)
3. Commit and push to your fork:
   ```bash
   git add <file>
   git commit -m "docs: add my name to contributors"
   git push -u origin docs/add-my-name
   ```
4. Open a Pull Request on GitHub from your fork's branch to the main repo's `main` branch
5. Confirm the PR shows up and CI checks run

---

## Data & Detection (2–3 people)

This team is on the **critical path** — the Tracking and Modeling teams cannot do real work until you produce labeled frames and a trained YOLO model. Start immediately.

### Day 1: Collect Replays and Set Up Roboflow

1. **Start collecting match replays**
   - Record matches directly from Clash Royale's in-game replay feature
   - Download replays from YouTube (use recent matches from the current season only — older data is unreliable due to balance patches)
   - Save raw video files locally in `data/replays/` (this folder is gitignored)
   - Target: **30+ full match replays by end of week 2**

2. **Set up a Roboflow project**
   - Create a Roboflow workspace for the team
   - Create a project with detection task type
   - Define label classes. Start with the most common cards and structures:
     - **Troops:** hog_rider, musketeer, knight, archers, minions, skeleton_army, valkyrie, wizard, etc.
     - **Spells:** fireball, arrows, zap, lightning, etc.
     - **Structures:** cannon, tesla, inferno_tower, etc.
     - **HUD elements:** elixir_bar, timer, tower_health, king_tower, princess_tower
   - Share the Roboflow project with all Data & Detection members
   - Prioritize the **15–20 most common cards** first — don't try to label everything at once

3. **Practice annotating**
   - Each team member annotates **10 frames** as practice
   - Use Roboflow's assisted labeling to speed things up
   - Review each other's annotations to ensure consistency (tight bounding boxes, correct labels)
   - Estimated time: ~2–3 minutes per frame

### Day 2–3: Frame Extraction Script

4. **Write a frame extraction script**
   - Create a Python script that takes a replay video and extracts frames at a configurable interval (e.g., every 0.5 seconds)
   - Use OpenCV + FFmpeg
   - Output frames as PNGs to `output/frames/` following the naming convention:
     ```
     output/frames/match_001_frame_0001.png
     output/frames/match_001_frame_0002.png
     ```
   - This is your first real PR: branch name `data/add-frame-extraction`

5. **Begin bulk annotation**
   - Upload extracted frames to Roboflow
   - Split work: each person owns a set of matches
   - Target: **500+ labeled frames by end of week 3**
   - Export annotations in **COCO format** to `output/annotations/annotations.json`

### Week 1 Deliverables

- [ ] 10+ match replays collected
- [ ] Roboflow project created with label classes defined
- [ ] Each member has annotated at least 10 practice frames
- [ ] Frame extraction script merged via PR
- [ ] Annotation pipeline established and bulk labeling underway

---

## Tracking & Feature Engineering (2–3 people)

You depend on the Data & Detection team's outputs (YOLO model + frames), but there is plenty of scaffolding and experimentation to do in week 1.

### Day 1: Scaffolding and Schema

1. **Finalize the event log schema**
   - Review the draft schema in [workflow.md](workflow.md) (Interface Contracts section):

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

   - Discuss with the Modeling team: do they need any additional columns?
   - Write a small utility that creates an empty event log DataFrame with the correct schema and types
   - PR branch: `tracking/define-event-schema`

2. **Set up skeleton modules**
   - Create placeholder Python modules for the tracking pipeline:
     - `tracking/tracker.py` — ByteTrack integration (empty class with method signatures)
     - `tracking/ocr.py` — Tesseract OCR for HUD reading (empty class with method signatures)
     - `tracking/event_builder.py` — Combines tracker + OCR output into the event log
   - This gives the team a structure to fill in and makes future PRs easier to review
   - PR branch: `tracking/add-pipeline-scaffolding`

### Day 2–3: Tesseract OCR Experiments

3. **Experiment with Tesseract on sample frames**
   - Ask the Data & Detection team for 5–10 sample frames (or screenshot them yourself)
   - Crop the HUD regions manually (elixir bar, timer, tower HP areas)
   - Test Tesseract OCR on these crops:
     ```python
     import pytesseract
     from PIL import Image

     img = Image.open("cropped_elixir.png")
     text = pytesseract.image_to_string(img, config="--psm 7 digits")
     print(text)
     ```
   - Document what works and what doesn't:
     - Does Tesseract read the elixir count reliably?
     - Does the stylized font cause problems?
     - What preprocessing (grayscale, threshold, invert) improves accuracy?
   - If Tesseract fails on HUD text, research fallback approaches:
     - Template matching for known digit images
     - Pixel-color heuristics for the elixir bar (count purple pixels)
   - PR branch: `tracking/tesseract-experiments`

4. **Read ByteTrack / supervision docs**
   - Read the [supervision documentation](https://supervision.roboflow.com/) for ByteTrack integration
   - Understand the input format ByteTrack expects (bounding boxes from YOLO detections)
   - Write notes on how to connect YOLO output → ByteTrack → tracked objects

### Week 1 Deliverables

- [ ] Event log schema finalized and utility function merged
- [ ] Pipeline skeleton modules created and merged
- [ ] Tesseract OCR tested on sample frames with results documented
- [ ] ByteTrack / supervision docs reviewed, integration plan written

---

## Modeling & Visualization (2 people)

You're furthest downstream in the pipeline, so week 1 is about preparation: learning the tools, designing outputs, and setting up structure so you're ready when event logs start flowing.

### Day 1: Research and Repo Structure

1. **Research LightGBM**
   - Read the [LightGBM documentation](https://lightgbm.readthedocs.io/)
   - Run a toy example to get familiar with the API:
     ```python
     import lightgbm as lgb
     import numpy as np

     # Create dummy data
     X_train = np.random.rand(100, 5)
     y_train = np.random.rand(100)

     dataset = lgb.Dataset(X_train, label=y_train)
     params = {"objective": "regression", "metric": "rmse", "verbose": -1}
     model = lgb.train(params, dataset, num_boost_round=100)

     predictions = model.predict(X_train[:5])
     print(predictions)
     ```
   - Understand: what features will you need? How do you evaluate the model? What hyperparameters matter most?

2. **Set up repo structure for modeling**
   - Create placeholder directories and modules:
     ```
     modeling/
     ├── __init__.py
     ├── features.py      # Feature engineering from event logs
     ├── train.py          # LightGBM training script
     └── evaluate.py       # Model evaluation and metrics
     visualization/
     ├── __init__.py
     ├── heatmaps.py       # Troop placement heatmaps
     ├── timeseries.py     # Elixir usage over time
     └── ev_breakdown.py   # Per-card EV charts
     ```
   - PR branch: `modeling/add-repo-structure`

### Day 2–3: Design Visualizations

3. **Draft visualization mockups**
   - Sketch (on paper, whiteboard, or in a notebook) what each output should look like:
     - **Troop placement heatmap:** Arena grid with color intensity showing where cards are played most often
     - **Elixir time-series:** Line chart showing both players' elixir over time, with card-play events marked
     - **Per-card EV breakdown:** Bar chart showing average EV per card, sorted by value
     - **Blunder detection:** Highlighted moments where a play was significantly below expected value
   - Share mockups with the team for feedback — these define what data the pipeline needs to produce

4. **Write a synthetic data generator**
   - Create a script that generates fake event logs matching the schema
   - This lets you build and test visualizations before real data is available:
     ```python
     import pandas as pd
     import numpy as np

     def generate_fake_events(n_events=100, match_id="fake_001"):
         cards = ["hog_rider", "musketeer", "fireball", "knight", "archers"]
         costs = {"hog_rider": 4, "musketeer": 4, "fireball": 4, "knight": 3, "archers": 3}

         data = {
             "match_id": [match_id] * n_events,
             "timestamp": sorted(np.random.uniform(0, 180, n_events)),
             "player": np.random.choice([1, 2], n_events),
             "card": np.random.choice(cards, n_events),
             # ... fill in remaining columns
         }
         data["elixir_cost"] = [costs[c] for c in data["card"]]
         return pd.DataFrame(data)
     ```
   - Use this to start building real visualization code immediately
   - PR branch: `modeling/add-synthetic-data-generator`

5. **Study the EV calculation methodology**
   - Review Option B in [pod_summary.md](pod_summary.md):
     - `E[value] = avg(damage dealt + elixir advantage − elixir cost)`
   - Think about edge cases: what if a card deals no damage but provides defensive value? How do you handle spells vs troops?
   - Write up initial thoughts and share with the team

### Week 1 Deliverables

- [ ] LightGBM toy example run successfully
- [ ] Repo structure for modeling and visualization created and merged
- [ ] Visualization mockups shared with the team
- [ ] Synthetic data generator merged, first draft visualizations started
- [ ] EV calculation methodology reviewed and initial questions documented

---

## Cross-Team Coordination

### Interface Contracts (Finalize by End of Week 2)

These are the agreements on what each team passes to the next. Get alignment early to avoid rework.

| From | To | What | Format |
| ---- | -- | ---- | ------ |
| Data & Detection | Tracking & Features | Extracted frames | PNGs in `output/frames/` |
| Data & Detection | Tracking & Features | Frame annotations | COCO JSON in `output/annotations/` |
| Data & Detection | Tracking & Features | Trained YOLO model | `output/models/yolo_best.pt` |
| Tracking & Features | Modeling & Viz | Match event logs | CSV in `output/events/match_XXX_events.csv` |

### Communication Rhythm

| Day | What | Where |
| --- | ---- | ----- |
| Monday | Standup: what I did, what I'm doing, what's blocking me | `#general` (async or 15-min call) |
| Wednesday | Mid-week check-in: sub-team leads flag blockers | `#general` (async) |
| Friday | Weekly update: each sub-team posts a short summary | `#general` (async) |

### Key Principle

**Data & Detection is on the critical path.** The entire pipeline depends on labeled frames and a working YOLO model. If you're on another team and have downtime, help Data & Detection annotate frames. The faster they deliver, the faster everyone else can do real work.
