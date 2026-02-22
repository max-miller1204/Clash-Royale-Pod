# Intro Pod Proposal Template

**Pod Leader:** Max Miller
**Topic Area:** Clash Royale Post-Game Analyzer

---

## RESEARCH QUESTION

**What specific question are you trying to answer? Why is this question interesting or relevant?**

I’m investigating how in-game decision-making efficiency and card interactions impact match outcomes in Clash Royale using computer vision and statistical modeling.

This question is interesting because it bridges machine learning, quantitative modeling, and game theory in a real-time strategic environment. Much like how chess engines evaluate moves and positions, this project aims to quantify the expected value (EV) of each action and card placement using data extracted from gameplay.

---

## DATA & METHODOLOGY

### What data will you use?

The primary data will come from Clash Royale match replays and game footage, which are publicly available through in-game replays and YouTube videos.

* **Target dataset size:** 50–100 full match replays, yielding ~500–1000 labeled frames for object detection and ~2000+ individual card-play events for EV modeling.
* **Annotation scope:** Each frame will be labeled with bounding boxes for troops, spells, structures, and HUD elements (elixir bar, tower health). Estimated annotation time is ~2–3 minutes per frame using Roboflow's assisted labeling.
* **Season scope:** Only matches from the current game season and forward, since balance changes, card reworks, and new evolutions make older data unreliable. Using recent data ensures the model reflects the current meta.

---

### What will you do with the data?

1. **Detect** — Train a YOLO object detection model to identify troops, spells, and tower states in each frame.
2. **Track** — Use ByteTrack to follow entities across frames, reconstructing troop movement, spell impact, and tower health over time.
3. **Extract** — Read HUD data (elixir count, timer, tower HP) via OCR to build a structured event log per match.
4. **Model** — Calculate the expected value (EV) of each card play based on resulting damage, elixir trade, and positional advantage.
5. **Evaluate** — Detect “blunders” by comparing a player’s move against the model’s predicted optimal play.
6. **Visualize** — Produce heatmaps of troop placements, time-series of elixir usage, and per-card EV breakdowns.

---

### What tools, methods, and languages will you use and why?

#### Languages & Environment

* **Python (primary):** Large ML/CV ecosystem and fast prototyping.

#### Core ML / CV Stack

* **PyTorch** for training custom models
* **Ultralytics YOLO (v8/11)** for object detection (troops/spells/structures) — fast and accurate
* **OpenCV + FFmpeg** for video/frame extraction, preprocessing, and augmentation

#### Tracking & HUD/OCR

* **ByteTrack** for multi-object tracking across frames
* **Tesseract** for reading on-screen numbers (elixir, timers); small regressors for health bars

#### EV / Outcome Modeling

* **LightGBM** to predict short-horizon outcome deltas (damage, win-probability proxy) from engineered features

#### Data & Statistics

* **NumPy / pandas** for feature engineering and data wrangling
* **SciPy / statsmodels** for statistical tests (e.g., Mann-Whitney for placement differences, permutation tests for EV shifts)

#### Annotation & Dataset Operations

* **Roboflow** for bounding box labeling and dataset management
* **COCO format** for annotations

---

## TEAM STRUCTURE

**Team Size:** 8

### Roles

| Role | Responsibility | People |
| ---- | -------------- | ------ |
| Pod Leader | Project management, architecture decisions, code review | 1 |
| Data & Detection | Replay collection, annotation, YOLO training, frame extraction | 2–3 |
| Tracking & Feature Engineering | ByteTrack integration, Tesseract OCR, event log construction | 2–3 |
| Modeling & Visualization | LightGBM training, EV calculations, heatmaps, final report | 2 |

### What will team members learn/do?

* Computer vision fundamentals (object detection, multi-object tracking)
* Data collection, annotation, and cleaning workflows
* Feature engineering and statistical analysis
* Building and evaluating predictive models
* Technical writing and documentation

**Estimated time commitment per week:**

* **Pod leader:** 14 hours
* **Team members:** 7–14 hours

---

## TIMELINE & DELIVERABLES

### Project Timeline

| Week | Phase | Deliverables |
| ---- | ----- | ------------ |
| 1 | Setup & Onboarding | Dev environment setup, team role assignments, collect initial match replays |
| 2 | Data Collection & Annotation | Record/download 30+ match replays, begin labeling frames in Roboflow |
| 3 | Data Collection & Annotation | Finish first annotation pass (~500+ labeled frames), establish COCO dataset |
| 4 | Model Training v1 | Train initial YOLO model on labeled data, evaluate detection accuracy |
| 5 | Tracking & OCR Pipeline | Integrate ByteTrack for troop tracking, Tesseract for elixir/timer reading |
| 6 | Feature Engineering | Extract per-play features (elixir trades, damage, placement), build tabular dataset |
| 7 | EV Modeling | Train LightGBM on engineered features, compute per-card EV estimates |
| 8 | Analysis & Visualization | Build heatmaps, elixir time-series, blunder detection; run statistical tests |
| 9 | Integration & Polish | End-to-end pipeline from replay video to analysis output, documentation |
| 10 | Final Presentation | Demo, writeup, retrospective |

---

## FEASIBILITY CHECK

### What could go wrong?

| Risk | Impact | Mitigation |
| ---- | ------ | ---------- |
| Balance patches invalidate trained models | Model accuracy drops mid-project | Pin data to a single season; retrain only if a major patch lands during weeks 4–8 |
| Annotation is too slow / tedious | Dataset too small for reliable detection | Use Roboflow's auto-label to bootstrap, then manually correct; prioritize the 15–20 most common cards first |
| YOLO struggles with overlapping troops | Missed or merged detections in crowded lanes | Increase training data for crowded scenes; tune NMS (non-max suppression) thresholds |
| OCR fails on stylized HUD text | Can't read elixir/timer accurately | Fall back to template matching or pixel-color heuristics for known HUD regions |
| Scope creep into Option A complexity | Team burns out, nothing ships | Start with Option B as the baseline; only add Option A components if B is working by week 7 |

### What resources/support do you need?

* Access to GPU resources (Google Colab Pro or university cluster) for YOLO training
* Advice on model evaluation metrics (mAP thresholds, statistical test selection)
* Clash Royale accounts at varying trophy ranges to capture diverse gameplay

### Success Criteria

**Minimum viable outcome (must hit):**
* End-to-end pipeline that takes a match replay video and outputs a structured event log with per-play elixir trade calculations
* YOLO model with mAP@0.5 ≥ 0.70 on the test set
* Written report with methodology, results, and limitations

**Stretch goals:**
* Per-card EV estimates with statistical significance tests
* Blunder detection that flags the top 3 worst plays per match
* Interactive dashboard for exploring match analysis results

---

# Algorithm Implementation

## Option A (More Complex)

### Goal

Find the expected value ( EV ) of each troop and how it changes based on other troops placed.

### Factors

* **Placement**
  Map out the grid and track troop placements. Analyze how placement impacts expected value.

* **Damage**
  Measure total damage dealt to troops and towers.

* **Elixir Trades**
  Compare elixir spent in a timeframe relative to the opponent.
  Example: Hog (-4), opponent counters with Tesla (+4) → Net = 0

* **Elixir Leaked**
  Measure how leaked elixir impacts total potential value over a game.

### Issues

* **Dependency**
  How do surrounding troops impact each card’s expected value?

---

### Mathematical Framework

| Component | Method | What It Does |
| --------- | ------ | ------------ |
| Overall Grade | Weighted z-score | Normalize each factor (damage, elixir, placement) to z-scores, take a weighted sum, and scale to a 0–100 grade |
| Per-Card EV | Expected value | For each card, compute E[value] = avg(damage dealt + elixir advantage − elixir cost) across all observed plays |
| Placement Multiplier | Linear scaling factor | Multiply card EV by a coefficient based on placement position (e.g., bridge vs. back) learned from data |
| Tempo | Running elixir differential | Track cumulative (your elixir spent − opponent elixir spent) over time; positive = you’re ahead on tempo |
| Elixir Leak Penalty | Sum of wasted elixir | Total elixir lost while at 10/10 bar; subtracted from overall grade as an inefficiency penalty |
| Card Synergy | Shapley values | Use cooperative game theory to attribute how much each card in a combo contributes to the total outcome |

---

## Option B (Simplified) — Recommended Starting Point

### Goal

Calculate elixir trades and accumulated damage per interaction. This is the minimum viable analysis and should be completed before attempting Option A.

### Factors

* **Damage**
  Total damage dealt to troops and towers per card play.

* **Elixir Trades**
  Net elixir advantage/disadvantage per interaction (e.g., Fireball costs 4 but kills a Musketeer worth 4 → net 0; Fireball kills Minion Horde worth 5 → net +1).

* **Elixir Leaked**
  Total elixir wasted at full bar (10/10), summed over the match.

### Limitation

* **No dependency modeling** — Option B treats each card play independently. It cannot answer "did this card perform well *because* of what was placed alongside it?" That requires Option A’s synergy analysis.

### Progression Path

Option B → validate pipeline works → add placement data (Option A factor 1) → add tempo tracking → add Shapley synergy analysis. Each step builds on the last.
