<!--
SYNC IMPACT REPORT
Version change: 0.0.0 (template) → 1.0.0
Bump rationale: Initial ratification — every principle and section is newly defined,
which is a MAJOR by semver-for-governance (no prior baseline existed to be compatible with).

Modified principles:
  - [PRINCIPLE_1_NAME] → I. Reproducible Environments
  - [PRINCIPLE_2_NAME] → II. Pipeline Modularity
  - [PRINCIPLE_3_NAME] → III. CLI as Public Interface
  - [PRINCIPLE_4_NAME] → IV. Pragmatic Testing & Quality Gates
  - [PRINCIPLE_5_NAME] → V. MVP-First Scope Discipline

Added sections:
  - Tech Stack & Operational Constraints
  - Development Workflow & Review Process
  - Governance

Removed sections: none.

Templates requiring updates:
  - .specify/templates/plan-template.md ✅ aligned (generic "Constitution Check" gate
    already references this file; no principle names to rewrite)
  - .specify/templates/spec-template.md ✅ no constitution references; nothing to change
  - .specify/templates/tasks-template.md ✅ no constitution references; nothing to change
  - .specify/templates/checklist-template.md ✅ generic; no change
  - README.md ✅ aligned with Tech Stack & Workflow sections; no edits needed
  - CLAUDE.md ✅ no principle references; no edits needed

Follow-up TODOs: none.
-->

# Clash Royale Post-Game Analyzer Constitution

## Core Principles

### I. Reproducible Environments

The development environment MUST be reproducible from the locked manifests in this
repository. Python dependencies are pinned in `uv.lock` (resolved from
`pyproject.toml`); system dependencies (Python 3.11, ffmpeg, tesseract, OpenCV/PyTorch
native libs) are pinned in `flake.lock` (resolved from `flake.nix`). CI MUST run inside
the same `nix develop` shell that local contributors use, so a green CI run implies a
green local run on a clean checkout. Adding a Python package requires `uv add` (which
updates both `pyproject.toml` and `uv.lock`); adding a system tool requires editing
`flake.nix` and refreshing `flake.lock`. Ad-hoc `pip install` or unpinned global tools
are prohibited.

**Rationale:** This is a multi-person student pod with mixed OSes (macOS, Linux,
NixOS). "Works on my machine" failures cost more than the friction of pinning, and the
CV/ML stack (PyTorch, Ultralytics, OpenCV) is unforgiving about driver, glibc, and
shared-library mismatches.

### II. Pipeline Modularity

The analyzer is a sequence of independently substitutable stages: dataset →
detection → tracking → ocr → features → modeling → visualization. Each stage lives in
its own subpackage under `src/crpod/` and communicates with adjacent stages only via
the dataclasses in `src/crpod/types.py` (`CardPlay`, `HudState`, `Interaction`, etc.).
A new stage implementation (e.g., a different detector or a different EV model) MUST
be drop-in replaceable without modifying any other subpackage. Cross-stage shortcuts
that bypass `types.py` are prohibited.

**Rationale:** Sub-teams work in parallel on different stages, and individual stages
get rewritten as research progresses (KataCR-trained YOLO replaced the Roboflow plan;
the elixir-trade EV proxy will be replaced by a real damage signal). Stable contracts
between stages are what let those swaps happen without coordinated rewrites.

### III. CLI as Public Interface

All user-facing entry points to the pipeline MUST be exposed as `crpod`
subcommands defined in `src/crpod/__main__.py` (e.g., `crpod list-replays`,
`crpod analyze`, `crpod train`). Subcommands MUST validate required artifacts
(weights, datasets, output paths) before performing expensive work such as HF
downloads or GPU inference. Diagnostic output goes to stdout in human-readable form;
errors go to stderr with a non-zero exit code. Pipeline functionality MUST NOT be
reachable only through ad-hoc scripts or notebooks.

**Rationale:** The CLI is the contract that integration tests, CI, and downstream
consumers (live mode, dashboards) all depend on. Pre-validation of inputs prevents
wasted GPU minutes and confusing late failures — exactly the failure mode that bit us
during the YOLO weights smoke test.

### IV. Pragmatic Testing & Quality Gates

Pure-logic features (elixir ledger, interaction windows, placement zones, OCR
parsing) MUST have pytest coverage exercising representative inputs. Stages that
depend on heavy artifacts (YOLO weights, HF dataset shards, GPUs) are exempt from unit
tests but MUST be covered by smoke runs documented in `docs/TODO.md` or a feature
spec. Every PR to `main` MUST pass the four CI gates: `ruff check`, `ruff format
--check`, `mypy src`, and `pytest`. Bypassing CI (`--no-verify`, force-push to a
protected branch, merging a red PR) is prohibited.

This project does NOT mandate test-first development. Tests are required for
shipped pure-logic features but may be written alongside or after the implementation.

**Rationale:** Strict TDD slows research-style code where the right contract is
discovered by experiment. But the four CI gates catch the regressions that actually
hurt this codebase (lint drift, type errors in shared dataclasses, broken
elixir/interaction math) without forcing premature test design.

### V. MVP-First Scope Discipline

Work MUST follow the Option B → Option A progression documented in
`pod_summary.md`. Ship the simplest end-to-end pipeline (elixir trades, accumulated
damage, leaked elixir) before adding placement multipliers, tempo tracking, or
Shapley synergy. A new feature spec MUST identify which Option-B baseline it
depends on, and MUST NOT block on or be blocked by another in-flight expansion.
Real-time mode, blunder detection, dashboards, and other items past the 10-week pod
timeline are explicitly stretch goals — they MAY be specced and partially
implemented, but they MUST NOT delay the offline MVP.

**Rationale:** A 10-week pod with 8 part-time students cannot ship Option A from a
cold start. The feasibility check in `pod_summary.md` flags scope creep into Option A
as the highest-impact risk. This principle makes that risk a structural rule, not a
hope.

## Tech Stack & Operational Constraints

The following stack is fixed. Substitutions require an amendment to this
constitution.

- **Language**: Python 3.11.
- **Package manager**: `uv` (lockfile-driven; `pip` and `poetry` are not used).
- **Environment manager**: Nix flake (`flake.nix`) is the canonical dev shell;
  `uv sync` without Nix is supported as a fallback for macOS/Ubuntu only.
- **Detection**: PyTorch + Ultralytics YOLO (currently YOLOv8s trained on the
  KataCR public dataset). Weights live under `output/models/` and are not committed
  to git.
- **Tracking**: `supervision.ByteTrack`.
- **OCR**: Tesseract via `pytesseract`, with empirical HUD region rects in
  `src/crpod/ocr/`.
- **Modeling**: LightGBM for EV regression; joblib artifacts under
  `output/models/`.
- **Visualization**: matplotlib / seaborn.
- **Lint/format/types**: `ruff` (line length 100, py311 target) and `mypy`
  (`mypy_path = src`, `ignore_missing_imports = true`).

Operational expectations:

- GPU work (YOLO training, large-batch inference) runs on rented Brev/A4000-class
  hardware. A full training run MUST stay under ~$0.50 of compute; runs exceeding
  that need a written justification in the PR description.
- All pipeline outputs (frames, detections, models, analyses) live under `output/`
  and are gitignored. The repository contains code and small fixtures only.
- The dataset target is the HuggingFace `chrisrca/clash-royale-tv-replays` corpus
  pinned to a single Clash Royale season — older seasons MUST NOT be mixed in
  without an explicit research justification.

## Development Workflow & Review Process

- **No direct pushes to `main`.** `main` is a protected branch; every change
  lands via Pull Request reviewed and merged by the maintainer (`@max`).
- **Branch naming.** Use the sub-team prefix: `data/*`, `tracking/*`,
  `modeling/*`, or `docs/*`. Spec-Kit feature branches (e.g., `001-feature-name`)
  are also accepted.
- **Branch lifetime.** Branches are short-lived (days, not weeks). When a branch
  outlives its task, rebase or close it.
- **Multi-person features.** Use a shared base branch with sub-branches PRing
  into it (per the README "Collaborating within a subgroup" workflow); only the
  shared branch PRs into `main`.
- **Commit hygiene.** Commit messages start with the sub-team prefix
  (`data: …`, `tracking: …`, `modeling: …`, `docs: …`, `feat: …`, `fix: …`) and
  describe the why, not just the what.
- **CI gate.** GitHub Actions runs `ruff check`, `ruff format --check`,
  `mypy src`, and `pytest` inside `nix develop` on every PR. All four MUST be green
  before merge.
- **Spec-Kit workflow.** Non-trivial features SHOULD go through
  `/speckit-specify` → `/speckit-plan` → `/speckit-tasks` → `/speckit-implement`,
  with the resulting `specs/<branch>/` artifacts committed alongside the code.
- **Review focus.** Reviewers MUST verify (1) constitution compliance, especially
  Principle II (no cross-stage shortcuts) and Principle V (MVP scope), (2) that any
  new dependency is pinned via `uv add` or `flake.nix`, and (3) that public
  behavior is reachable through a `crpod` subcommand.

## Governance

This constitution supersedes informal team conventions. Where this document and
`README.md`, `pod_summary.md`, or any in-line comment disagree, this document
controls and the other source MUST be updated.

**Amendment procedure.** Amendments are proposed via PR that edits this file and
runs through `/speckit-constitution` to refresh the Sync Impact Report. The PR MUST
identify the version bump (MAJOR / MINOR / PATCH), call out which dependent
templates and docs were re-checked, and be approved by the maintainer (`@max`)
before merge.

**Versioning policy.** Constitution version follows semantic versioning:

- **MAJOR**: removing or fundamentally redefining a principle, or removing a
  governance rule.
- **MINOR**: adding a new principle/section, or materially expanding the scope of
  an existing one (new MUST/SHOULD requirements).
- **PATCH**: clarifications, wording fixes, typo repairs, or rationale edits that
  do not change the rules.

**Compliance review.** Every Pull Request description MUST include a
"Constitution Check" line confirming the change complies with this document, or
listing the specific deviation with justification (which the maintainer evaluates
under the amendment procedure). Recurring deviations are a signal that the
constitution should be amended, not that the rule should be quietly ignored.

**Runtime guidance.** For day-to-day "how do I do X" questions (commands, branch
flow, dependency updates), `README.md` is the canonical reference. This document
defines the rules; `README.md` defines the mechanics.

**Version**: 1.0.0 | **Ratified**: 2026-04-30 | **Last Amended**: 2026-04-30
