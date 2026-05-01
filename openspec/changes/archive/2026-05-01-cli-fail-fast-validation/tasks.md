## 1. `src/crpod/__main__.py` — `_cmd_analyze`

- [x] 1.1 Replace `sys.exit(f"weights file not found: {args.weights}")` with the `print(f"error: weights file not found: {args.weights}", file=sys.stderr); sys.exit(1)` pattern (mirroring `_cmd_analyze_video`).
- [x] 1.2 Add a guard for `--model`: if `args.model is not None` and `not args.model.exists()`, emit `error: EV model file not found: {args.model}` to stderr and exit 1.
- [x] 1.3 Confirm both guards run before `EvModel.load(args.model)` and `analyze_hf_replay(...)` are called.

## 2. `src/crpod/__main__.py` — `_cmd_train`

- [x] 2.1 Replace `sys.exit(f"weights file not found: {args.weights}")` with the standard `error: ... / exit 1` pattern.
- [x] 2.2 Add a guard: if `args.out.parent` does not exist, emit `error: output directory does not exist: {args.out.parent}` to stderr and exit 1.
- [x] 2.3 Add a guard: if `args.max_replays <= 0`, emit `error: --max-replays must be > 0` to stderr and exit 1.
- [x] 2.4 Confirm all guards run before `HFReplayLoader(...)` is constructed.

## 3. `tests/test_cli.py` — new file

- [x] 3.1 Create `tests/test_cli.py` with a `_run(args)` helper that subprocess-invokes `python -m crpod ...` from the repo root and returns `(returncode, stdout, stderr)`.
- [x] 3.2 Test: `analyze` with a missing `--weights` exits 1, stderr starts with `error:` and names the missing path.
- [x] 3.3 Test: `analyze` with valid `--weights` (use `tmp_path` to create an empty stub file) and a missing `--model` exits 1 and names the missing model path.
- [x] 3.4 Test: `train` with a missing `--weights` exits 1 with the expected error format.
- [x] 3.5 Test: `train` with valid `--weights` and `--out` whose parent directory does not exist exits 1 with the unreachable-output message.
- [x] 3.6 Test: `train` with valid `--weights`, valid `--out` parent, and `--max-replays 0` exits 1 with the positive-value message.

## 4. Verify

- [x] 4.1 Run `nix develop --command uv run pytest tests/test_cli.py -v` and confirm all new tests pass.
- [x] 4.2 Run `nix develop --command uv run pytest` (full suite) and confirm no regressions.
- [x] 4.3 Run `nix develop --command uv run ruff check src tests` and `nix develop --command uv run mypy src` — confirm clean.
- [x] 4.4 Run `openspec validate cli-fail-fast-validation` and confirm the change validates against its schema.
