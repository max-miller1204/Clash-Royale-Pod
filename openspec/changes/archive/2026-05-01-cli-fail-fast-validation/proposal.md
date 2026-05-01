## Why

`crpod analyze-video` already validates all path arguments before importing the
heavy `pipeline` module, giving users a fast failure with a clear message. The
sibling commands `crpod analyze` and `crpod train` skip this discipline:
`--model` is never checked in `analyze`, and `train` neither validates the
`--out` parent directory nor enforces `--max-replays > 0`. Users who mistype a
path wait through HuggingFace dataset reads or YOLO weight loads before getting
a far less specific error. This change closes the gap so all three CLI
commands fail fast with consistent error messages.

## What Changes

- `crpod analyze` validates `--model` exists (when provided) before any pipeline
  import.
- `crpod train` validates that the parent directory of `--out` exists and that
  `--max-replays > 0` before any HuggingFace loader work.
- All three commands share the same exit conventions: stderr message prefixed
  `error: `, exit code `1`, and validation runs *before* heavy imports.
- Add a CLI test module (`tests/test_cli.py`) covering each fail-fast path —
  none exist today.

## Capabilities

### New Capabilities

- `cli`: Behavior contract for `crpod`'s argparse-driven entry points — argument
  validation, exit codes, and ordering of validation versus heavy imports.

### Modified Capabilities

_(none — `openspec/specs/` is currently empty; this change introduces the first
capability spec.)_

## Impact

- `src/crpod/__main__.py`: add validation guards in `_cmd_analyze` and
  `_cmd_train`; reuse the same `print(..., file=sys.stderr); sys.exit(1)`
  pattern already in `_cmd_analyze_video`.
- `tests/test_cli.py`: new file with subprocess-style tests covering each
  fail-fast path. Pure local pytest — no Brev / no GPU.
- No new system deps in `flake.nix`, no new Python deps in `pyproject.toml`.
- No fixtures or HuggingFace dataset reads required.
