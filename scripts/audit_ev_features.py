"""Print LightGBM feature importances for an EvModel artifact.

Used by wave 2J to decide which features qualify for the audit drop rule
(`gain < 1% × max_gain` AND `split < 5`). The default path points at the
wave-2I model in the main worktree (`output/` is gitignored).

Usage:
    uv run python scripts/audit_ev_features.py [path/to/ev_model.joblib]
"""

from __future__ import annotations

import sys
from pathlib import Path

from crpod.modeling.ev import EvModel

DEFAULT_PATH = Path("/Users/max/Clash-Royale-Pod/output/models/ev_wave2i.joblib")


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PATH
    ev = EvModel.load(path)
    booster = ev.model.booster_

    names = booster.feature_name()
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")

    max_gain = max(gain) if len(gain) else 0.0
    rows = sorted(
        zip(names, gain, split, strict=True),
        key=lambda r: r[1],
        reverse=True,
    )

    print(f"# Feature importance for {path.name}")
    print(f"# max_gain = {max_gain:.4f}")
    print(f"{'feature':<35} {'gain':>14} {'split':>8} {'gain/max':>10} {'drops?':>8}")
    for name, g, s in rows:
        ratio = (g / max_gain) if max_gain else 0.0
        drops = "DROP" if (ratio < 0.01 and s < 5) else ""
        print(f"{name:<35} {g:>14.4f} {s:>8d} {ratio:>10.4%} {drops:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
