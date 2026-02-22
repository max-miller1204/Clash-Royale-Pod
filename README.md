# Clash Royale Post-Game Analyzer

A computer vision and statistical modeling pipeline that analyzes Clash Royale match replays to evaluate decision-making, calculate expected value (EV) of card plays, and detect suboptimal moves.

## Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension

### Setup

1. Clone the repo
   ```bash
   git clone <repo-url>
   cd Clash-Royale-Pod
   ```
2. Open in VS Code
   ```bash
   code .
   ```
3. When prompted, click **"Reopen in Container"** (or run `Dev Containers: Reopen in Container` from the command palette)
4. Wait for the container to build (~3-5 min the first time)
5. Verify everything works:
   ```bash
   uv run python -c "import torch; import cv2; import ultralytics; print('ready')"
   ```

That's it. Python, PyTorch, YOLO, OpenCV, FFmpeg, Tesseract, and all dependencies are pre-installed.

## Project Structure

```
Clash-Royale-Pod/
├── .devcontainer/          # Docker + devcontainer config
│   ├── Dockerfile
│   ├── devcontainer.json
│   ├── docker-compose.yml
│   └── test_tools.sh
├── .vscode/                # VS Code settings (auto-format, linting)
├── data/                   # Match replays, frames, annotations (gitignored)
├── output/                 # Pipeline outputs (gitignored)
│   ├── frames/             # Extracted frames
│   ├── annotations/        # COCO-format labels
│   ├── models/             # Trained model weights
│   └── events/             # Per-match event logs
├── Makefile                # Dev commands (lint, format, sync, etc.)
├── pyproject.toml          # Python dependencies (managed by uv)
├── uv.lock                 # Locked dependency versions
├── pod_summary.md          # Project proposal
└── workflow.md             # Team workflow and collaboration guide
```

## Common Commands

| Command | What it does |
| ------- | ------------ |
| `uv sync` | Install/sync all dependencies from lockfile |
| `uv add <package>` | Add a new dependency |
| `uv run python script.py` | Run a script in the project environment |
| `make lint` | Run ruff linter |
| `make format` | Auto-format code with ruff |
| `make type-check` | Run mypy type checker |

## Git Workflow

**Never push directly to `main`.** Use feature branches and pull requests.

```bash
# Create a branch for your work
git checkout -b data/add-annotation-script

# Make changes, then commit
git add <files>
git commit -m "data: add annotation script for tower labels"

# Push and open a PR
git push -u origin data/add-annotation-script
```

Branch prefixes by sub-team:
- `data/*` — Data & Detection
- `tracking/*` — Tracking & Feature Engineering
- `modeling/*` — Modeling & Visualization
- `docs/*` — Documentation

## Sub-Teams

| Team | Owns |
| ---- | ---- |
| **Data & Detection** | Replay collection, annotation in Roboflow, YOLO training |
| **Tracking & Feature Engineering** | ByteTrack (via supervision), Tesseract OCR, event log construction |
| **Modeling & Visualization** | LightGBM, EV calculations, heatmaps, statistical analysis |

## Key Docs

- **[pod_summary.md](pod_summary.md)** — Full project proposal, research question, algorithm design
- **[workflow.md](workflow.md)** — Team structure, communication, interface contracts, weekly cadence

## Tech Stack

| Tool | Purpose |
| ---- | ------- |
| Python 3.11 | Primary language |
| PyTorch + Ultralytics YOLO | Object detection (troops, spells, structures) |
| OpenCV + FFmpeg | Video/frame extraction and processing |
| supervision | Multi-object tracking (ByteTrack) |
| Tesseract | OCR for HUD reading (elixir, timers, tower HP) |
| LightGBM | EV modeling and outcome prediction |
| pandas / NumPy / SciPy | Data wrangling and statistical analysis |
| matplotlib / seaborn | Visualization |
| uv | Python package and environment management |
| Ruff | Linting and formatting |
