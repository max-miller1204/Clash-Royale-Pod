"""Dump the trained YOLO model's class names to a checked-in snapshot.

Run once on a machine that has output/models/crpod_v1_best.pt. The
output file (src/crpod/detection/katacr_classes.txt) is the canonical
list of class strings the deployed weights can emit. Tests assert
that KATACR_TO_CARD ∪ KATACR_NON_CARD covers it exactly.

Re-run when the model is retrained.

Usage:
    uv run python scripts/dump_katacr_classes.py
    uv run python scripts/dump_katacr_classes.py --weights other.pt --out other.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_WEIGHTS = Path("output/models/crpod_v1_best.pt")
DEFAULT_OUT = Path("src/crpod/detection/katacr_classes.txt")


def dump(weights: Path, out: Path) -> int:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = model.names  # dict[int, str] — keyed by class id, in order
    if not isinstance(names, dict):
        raise RuntimeError(f"expected model.names to be a dict, got {type(names)!r}")

    ordered = [names[i] for i in sorted(names.keys())]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ordered) + "\n")
    return len(ordered)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    if not args.weights.exists():
        raise SystemExit(
            f"weights not found: {args.weights}\n"
            "Run this script on a machine that has the trained .pt "
            "(e.g. the brev instance from scripts/brev_train.sh)."
        )

    n = dump(args.weights, args.out)
    print(f"wrote {n} class names to {args.out}")


if __name__ == "__main__":
    main()
