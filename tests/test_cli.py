"""Argument-validation and fail-fast contract tests for the `crpod` CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "crpod", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_analyze_missing_weights(tmp_path: Path) -> None:
    missing = tmp_path / "no-weights.pt"
    result = _run(["analyze", "arena_15", "abc123", "--weights", str(missing)])

    assert result.returncode == 1
    assert result.stderr.startswith("error: ")
    assert "weights file not found" in result.stderr
    assert str(missing) in result.stderr


def test_analyze_missing_model(tmp_path: Path) -> None:
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"")
    missing_model = tmp_path / "no-model.joblib"

    result = _run(
        [
            "analyze",
            "arena_15",
            "abc123",
            "--weights",
            str(weights),
            "--model",
            str(missing_model),
        ]
    )

    assert result.returncode == 1
    assert result.stderr.startswith("error: ")
    assert "EV model file not found" in result.stderr
    assert str(missing_model) in result.stderr


def test_train_missing_weights(tmp_path: Path) -> None:
    missing = tmp_path / "no-weights.pt"
    out = tmp_path / "model.joblib"

    result = _run(["train", "--weights", str(missing), "--out", str(out)])

    assert result.returncode == 1
    assert result.stderr.startswith("error: ")
    assert "weights file not found" in result.stderr
    assert str(missing) in result.stderr


def test_train_unreachable_out_parent(tmp_path: Path) -> None:
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"")
    out = tmp_path / "missing-dir" / "model.joblib"

    result = _run(["train", "--weights", str(weights), "--out", str(out)])

    assert result.returncode == 1
    assert result.stderr.startswith("error: ")
    assert "output directory does not exist" in result.stderr
    assert str(out.parent) in result.stderr


def test_train_non_positive_max_replays(tmp_path: Path) -> None:
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"")
    out = tmp_path / "model.joblib"

    result = _run(
        [
            "train",
            "--weights",
            str(weights),
            "--out",
            str(out),
            "--max-replays",
            "0",
        ]
    )

    assert result.returncode == 1
    assert result.stderr.startswith("error: ")
    assert "--max-replays must be > 0" in result.stderr
