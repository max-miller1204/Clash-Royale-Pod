## Context

`src/crpod/__main__.py` currently has three subcommand handlers —
`_cmd_analyze`, `_cmd_analyze_video`, `_cmd_train`. Only `_cmd_analyze_video`
follows the project convention of validating every argument before importing
the heavy `crpod.pipeline` module (see lines 70–88, where `analyze_video` is
imported only after all path/numeric checks pass). The convention exists in
code but is not yet enforced uniformly.

Two unrelated issues use the same exit pattern: `_cmd_analyze` and `_cmd_train`
use `sys.exit(f"...")` (which prints to stderr without prefix) instead of the
explicit `print(f"error: ...", file=sys.stderr); sys.exit(1)` pattern that
`_cmd_analyze_video` uses. We unify on the latter so the spec's "stable error
prefix" requirement holds across all subcommands.

## Goals / Non-Goals

**Goals:**

- All three subcommands validate filesystem and numeric arguments before any
  heavy import.
- All argument errors share the format: `error: <message>` to stderr, exit code 1.
- Add a `tests/test_cli.py` with subprocess-style tests that exercise each
  fail-fast path locally — no GPU, no Brev, no HuggingFace network access.

**Non-Goals:**

- Refactoring argparse types (e.g., promoting `_positive_float` to a
  module-level helper for `--max-replays`). We could, but the diff is bigger
  than it looks because `argparse` runs type validators *before* `func` is
  called, and that's a different layer. Keep the validation inline.
- Changing the exit-code convention for non-argument errors (FileNotFoundError
  raised inside `analyze_video`, etc.). Those already use exit 1.
- Touching `_cmd_analyze_video` — it already complies.

## Decisions

### Decision 1: Inline validation, not a shared helper

`_cmd_analyze_video` validates inline (one block of `if ...: print/exit` per
argument). We mirror that style in `_cmd_analyze` and `_cmd_train` rather than
extracting a `_require_path(...)` helper. Reasons:

- Three subcommands × ~3 checks each = 9 lines of repetition. Not enough mass
  to justify abstraction.
- Inline keeps the validation visible at the start of each handler — important
  because the *ordering* (validation before import) is the contract.
- Matches `openspec/config.yaml`'s "three similar lines is better than a
  premature abstraction" guidance.

### Decision 2: Validate `Path(args.out).parent` for train, not the full `--out` path

`--out` is the file we'll *write*, so it shouldn't exist yet. What must exist
is its parent directory. We use `args.out.parent.exists()` (or treat a missing
parent as the failure). We do not auto-create it, because in `_cmd_train` the
`--out` is the final model artifact — silently creating a parent directory
could mask typos like `--out output/modls/foo.joblib`.

Contrast with `_cmd_analyze_video`, which uses `out_dir.mkdir(parents=True,
exist_ok=True)` because that's a directory of result artifacts where
convenience wins.

### Decision 3: `--max-replays` validated as part of subcommand handler, not as `argparse` type

Adding `type=_positive_int` to the argparse declaration would surface the error
via argparse's own format (exit code 2, not 1). To keep all argument errors on
the spec-mandated `error: ... / exit 1` surface, we validate inside `_cmd_train`
at the same layer as the path checks.

### Decision 4: CLI tests use `subprocess.run(["python", "-m", "crpod", ...])`

We invoke the CLI as a subprocess, capturing stdout/stderr/exit code. Reasons:

- Lets us assert that heavy modules were *not* imported.
- Avoids monkey-patching `sys.exit` inside pytest, which is fragile.
- Matches how a user actually runs the CLI.
- Tests stay local and cheap; no fixtures or HF network calls.
