# Clash Royale Post-Game Analyzer

A computer vision and statistical modeling pipeline that analyzes Clash Royale match replays to evaluate decision-making, calculate expected value (EV) of card plays, and detect suboptimal moves.

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- System dependencies: `ffmpeg`, `tesseract`, and OpenCV native libs (see setup options below)

### Option A: Nix (recommended)

If you have [Nix](https://nixos.org/download) with flakes enabled (`experimental-features = nix-command flakes` in `~/.config/nix/nix.conf`):

```bash
git clone <repo-url>
cd Clash-Royale-Pod
nix develop           # first enter takes a few minutes — system deps + uv sync
uv run crpod --help
```

If you use direnv, `direnv allow` once and the shell auto-activates on `cd`.

The flake provides: Python 3.11, uv, ffmpeg, tesseract, ruff, plus the native libraries OpenCV/PyTorch need. Python packages (`torch`, `ultralytics`, `lightgbm`, …) are installed into `.venv` by `uv sync`.

#### NixOS users: extra step

`ultralytics` transitively pulls in `opencv-python` (the full GUI build), which `dlopen`s `libxcb.so.1` at import time. NixOS doesn't expose system libs on a standard search path, so the import crashes unless those libs are reachable via [`nix-ld`](https://github.com/Mic92/nix-ld). Add the following to your system config (the host module that sets `programs.nix-ld.enable`):

```nix
programs.nix-ld = {
  enable = true;
  libraries = with pkgs; [
    zlib
    xorg.libxcb
    xorg.libX11
    xorg.libXext
    xorg.libSM
    xorg.libICE
    libGL
    glib
    stdenv.cc.cc.lib  # libstdc++.so.6
  ];
};
```

Then `sudo nixos-rebuild switch` and open a new shell. Your shell must export `LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH` for pip-installed wheels to find these libs. Mac and Windows users don't need any of this — those platforms ship the libs by default.

### Option B: uv only (no Nix)

Install the system dependencies for your platform, then `uv sync`:

**macOS (Homebrew):**

```bash
brew install python@3.11 uv ffmpeg tesseract
```

**Ubuntu / Debian:**

```bash
sudo apt update
sudo apt install python3.11 ffmpeg tesseract-ocr libgl1 libglib2.0-0
# Install uv: https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and sync:

```bash
git clone <repo-url>
cd Clash-Royale-Pod
uv sync
uv run crpod --help
```

### Verify the environment

```bash
uv run python -c "import torch, cv2, ultralytics, lightgbm, pytesseract; print('ready')"
uv run pytest
```

### Running the pipeline

The HF TV-replay dataset (`chrisrca/clash-royale-tv-replays`) ships raw frame images, not pre-extracted card placements. YOLO detection is required to extract card placements from the frames. You'll need trained YOLO weights (see **Sub-Teams** — the Data & Detection team owns this).

```bash
# List available replays (no weights needed)
uv run crpod list-replays --arena arena_15

# Analyze one replay — requires trained YOLO weights
uv run crpod analyze arena_15 <replay_id> --weights output/models/yolo.pt

# Train an EV model on 50 replays
uv run crpod train --weights output/models/yolo.pt --out output/models/ev.joblib --max-replays 50
```

For custom video ingest (raw mp4 → YOLO → ByteTrack → OCR → pipeline), see `src/crpod/pipeline.py::analyze_video`. That path is stubbed until the Data & Detection sub-team trains YOLO weights.

## Project Structure

```
Clash-Royale-Pod/
├── flake.nix               # Nix dev environment (replaces devcontainer)
├── .envrc                  # direnv entrypoint (use flake)
├── src/crpod/              # Pipeline package
│   ├── dataset/            # HF replay loader + raw video iterator
│   ├── detection/          # YOLO wrapper (custom-replay path)
│   ├── tracking/           # ByteTrack wrapper (custom-replay path)
│   ├── ocr/                # Tesseract HUD reader
│   ├── features/           # Elixir ledger, interactions, placement zones
│   ├── modeling/           # LightGBM EV model
│   ├── visualization/      # Heatmaps, tempo plots, EV breakdowns
│   ├── pipeline.py         # End-to-end orchestrator
│   └── __main__.py         # `crpod` CLI
├── tests/                  # pytest — pure-logic feature tests
├── output/                 # Pipeline artifacts (gitignored)
├── Makefile                # Dev shortcuts (sync, lint, format, test)
├── pyproject.toml          # Python dependencies (uv)
└── uv.lock                 # Locked dependency versions
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

We use the **Shared Repo + Pull Request** workflow. Everyone works on the same repo — no forks. No one pushes directly to `main` — all changes go through PRs that are reviewed and merged by the maintainer (@max).

### First-time setup (do this once)

1. **Get added as a collaborator** — ask @max to add you to the repo on GitHub (Settings → Collaborators).
2. **Clone the repo:**
   ```bash
   git clone https://github.com/<repo-owner>/Clash-Royale-Pod.git
   cd Clash-Royale-Pod
   ```
3. Set up the environment using either Option A (Nix) or Option B (uv only) above.

That's it. No forks, no upstream remotes. Everyone pushes to the same repo.

### Already have a fork?

If you previously forked the repo, you can switch to the shared model. Delete your fork on GitHub, then update your local clone to point at the main repo:

```bash
# Check your current remote
git remote -v

# If origin points to your fork, update it to the main repo
git remote set-url origin https://github.com/<repo-owner>/Clash-Royale-Pod.git

# Verify
git remote -v
```

### The workflow for every task

1. **Pull the latest main** before starting new work:
   ```bash
   git checkout main
   git pull
   ```
2. **Create a branch** for your specific task:
   ```bash
   git checkout -b data/add-annotation-script
   ```
3. **Do your work and commit** with descriptive messages:
   ```bash
   git add <files>
   git commit -m "data: add annotation script for tower labels"
   ```
4. **Push your branch:**
   ```bash
   git push -u origin data/add-annotation-script
   ```
5. **Open a Pull Request** on GitHub from your branch to `main`.
6. **Review & merge** — automated checks run, the maintainer reviews the code, and once approved it gets merged.

### Collaborating within a subgroup (multi-person features)

When two or more people in the same subgroup need to work on the same feature, use a shared feature branch with sub-branches:

1. **Person A creates the shared feature branch:**
   ```bash
   git checkout main
   git pull
   git checkout -b tracking/bytetrack
   git push -u origin tracking/bytetrack
   ```

2. **Person B pulls the shared branch:**
   ```bash
   git fetch
   git checkout tracking/bytetrack
   ```

3. **Each person creates a sub-branch off the shared branch:**
   ```bash
   # Person A
   git checkout tracking/bytetrack
   git checkout -b tracking/bytetrack-detection

   # Person B
   git checkout tracking/bytetrack
   git checkout -b tracking/bytetrack-scoring
   ```

4. **Each person pushes their sub-branch and PRs into the shared branch (not `main`):**
   ```bash
   # Person A
   git checkout tracking/bytetrack-detection
   git push -u origin tracking/bytetrack-detection
   ```
   Then on GitHub: open a PR and **change the base branch** from `main` to `tracking/bytetrack` (click the base dropdown).

   Person B does the same:
   ```bash
   git checkout tracking/bytetrack-scoring
   git push -u origin tracking/bytetrack-scoring
   ```
   Same thing on GitHub: PR into `tracking/bytetrack`.

5. **After both sub-PRs are reviewed and merged**, PR the shared branch into `main`:
   ```bash
   git checkout tracking/bytetrack
   git pull
   ```
   Then on GitHub: open a PR from `tracking/bytetrack` → `main`.

6. **Clean up after the PR is merged:**
   ```bash
   git checkout main
   git pull
   git branch -d tracking/bytetrack
   git branch -d tracking/bytetrack-detection
   git branch -d tracking/bytetrack-scoring
   ```
   GitHub will also show a "Delete branch" button on each merged PR to clean up the remote branches.

### Updating an open PR

When the maintainer reviews your PR and requests changes, you don't need to open a new PR. Just push more commits to the same branch — the PR updates automatically:

```bash
# You're still on your branch
git checkout data/add-annotation-script

# Make the requested fixes
git add <files>
git commit -m "data: fix edge case in annotation parser"

# Push to the same branch — the PR updates automatically
git push
```

### Branch naming

Use a prefix matching your sub-team:
- `data/*` — Data & Detection
- `tracking/*` — Tracking & Feature Engineering
- `modeling/*` — Modeling & Visualization
- `docs/*` — Documentation

### Keep branches short-lived

A branch should be created for a specific task, worked on for a few days, PR'd, merged, and then deleted. Don't let branches sit for weeks — the longer a branch lives, the more `main` changes underneath it, and the more likely you are to hit merge conflicts. Finish the task, open the PR, get it merged, move on.

### Handling merge conflicts

Conflicts happen when two people edit the same lines in the same file. If your PR shows a conflict:

```bash
# Update your local main
git checkout main
git pull

# Merge main into your branch
git checkout data/my-branch
git merge main
```

Git will mark conflicts in the file. Open it, pick the correct version, remove the `<<<<<<<` / `=======` / `>>>>>>>` markers, then:

```bash
git add <conflicted-file>
git commit -m "data: resolve merge conflict"
git push
```

**To minimize conflicts:** pull often, keep branches short-lived, and try to split work by file when possible (if you own `annotate.py` and your teammate owns `train.py`, you'll almost never conflict).

## Project Management

### Branch protection

The `main` branch is protected. No one can push to it directly — all changes must go through a Pull Request and be approved by the maintainer.

### Issue tracker

Use GitHub Issues to report bugs, propose features, and organize tasks. Reference issues in your commits and PRs (e.g., `fixes #12`).

### Maintainer

@max holds merge privileges and reviews all PRs. If your PR is ready for review, request a review from @max on GitHub.

## Managing Dependencies Across the Team

Since we use **uv + `pyproject.toml`**, adding packages on feature branches merges cleanly:

1. **Add a package** on your branch: `uv add <package>` — this updates `pyproject.toml` and `uv.lock`.
2. **Include both files** in your PR commit.
3. **On merge**, Git auto-merges `pyproject.toml` as long as different people edited different lines. If there's a conflict in `uv.lock`, resolve `pyproject.toml` first, then run `uv lock` to regenerate the lockfile.

### When main is updated

After pulling the latest changes from main, check what files changed and follow the appropriate steps:

| What changed | What to do |
| ------------ | ---------- |
| `pyproject.toml` / `uv.lock` (new Python packages) | Run `uv sync` — no rebuild needed |
| `flake.nix` / `flake.lock` (system tools or nixpkgs pin) | Nix users: re-enter the shell (`exit` then `nix develop`). Non-Nix users: no action needed unless new system deps were added — check the commit message |
| `Makefile` | No action needed |

`uv sync` is always safe to run.

## Sub-Teams

| Team | Owns |
| ---- | ---- |
| **Data & Detection** | Replay collection, annotation in Roboflow, YOLO training |
| **Tracking & Feature Engineering** | ByteTrack (via supervision), Tesseract OCR, event log construction |
| **Modeling & Visualization** | LightGBM, EV calculations, heatmaps, statistical analysis |

## Key Docs

- **[pod_summary.md](./pod_summary.md)** — Full project proposal, research question, algorithm design

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
| Claude Code / Gemini CLI / Codex CLI | AI coding assistants |
