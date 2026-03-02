# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Clash Royale Post-Game Analyzer — a computer vision and statistical modeling pipeline that analyzes Clash Royale match replays. The pipeline: detects game entities (YOLO) → tracks them across frames (ByteTrack) → reads HUD data via OCR (Tesseract) → builds structured event logs → models expected value of card plays (LightGBM) → visualizes results.

This is a greenfield project (no source code yet). Python 3.11, managed with uv.

## Commands

```bash
make lint          # ruff check .
make format        # ruff format .
make type-check    # mypy .
uv sync            # install/sync all dependencies from lockfile
uv add <package>   # add a new dependency (updates pyproject.toml + uv.lock)
uv run python <script>  # run a script in the project environment
```

CI runs `ruff check`, `ruff format --check`, and `mypy` on all PRs to main.

## Architecture

Three-stage pipeline with defined interface contracts between stages:

1. **Data & Detection** → outputs frames (`output/frames/*.png`), COCO annotations (`output/annotations/annotations.json`), YOLO weights (`output/models/yolo_best.pt`)
2. **Tracking & Feature Engineering** → consumes YOLO detections, outputs per-match event logs (`output/events/match_XXX_events.csv`) with columns: match_id, timestamp, player, card, elixir_cost, placement_x, placement_y, damage_dealt, tower_damage, elixir_bar
3. **Modeling & Visualization** → consumes event logs, produces EV estimates, heatmaps, statistical analysis

## Git Workflow

- Shared repo + Pull Request model. Never push directly to `main`.
- `main` is branch-protected: all changes require a PR with at least one review.
- Branch prefixes map to subgroups: `data/*`, `tracking/*`, `modeling/*`, `docs/*`
- For multi-person features within a subgroup, branch off the shared feature branch (e.g. `tracking/bytetrack-tuning` off `tracking/bytetrack`) and PR back into it before merging to `main`.
- Commit message prefixes: `data:`, `tracking:`, `model:`, `viz:`, `docs:`
- `data/` and `output/` directories are gitignored (large files stay local).

## Dev Environment

Runs inside a devcontainer (Docker). System packages pre-installed: ffmpeg, tesseract-ocr, libgl1. The `.venv` should activate automatically (prompt shows `(workspace)`); if not, run `source .venv/bin/activate`.

## Key Dependencies

PyTorch + Ultralytics YOLO (detection), OpenCV + FFmpeg (video/frames), supervision (ByteTrack tracking), pytesseract (OCR), LightGBM (EV modeling), pandas/NumPy/SciPy (data), matplotlib/seaborn (viz).
