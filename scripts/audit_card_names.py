"""Audit YOLO weights' class names against `CARD_COSTS`.

Loads a `.pt` checkpoint via `ultralytics.YOLO`, normalizes each class label
through `crpod.constants.normalize_card_name`, and prints a coverage report:
which classes resolve to a real entry in `CARD_COSTS` and which fall back to
the default. Use the unmapped list to drive expansions of `CARD_COSTS` or
`_KATACR_ALIASES`.

Usage:
    uv run python scripts/audit_card_names.py output/models/crpod_v1_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crpod.constants import CARD_COSTS, normalize_card_name


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("weights", type=Path, help="path to YOLO .pt checkpoint")
    args = p.parse_args()

    weights: Path = args.weights
    if not weights.exists():
        print(f"weights file not found: {weights}", file=sys.stderr)
        return 1

    from ultralytics import YOLO  # heavy import; defer until weights confirmed

    model = YOLO(str(weights))
    names: dict[int, str] = dict(model.model.names)

    mapped: list[tuple[str, str, int]] = []
    unmapped: list[tuple[str, str]] = []
    for _, raw in sorted(names.items()):
        key = normalize_card_name(raw)
        if key in CARD_COSTS:
            mapped.append((raw, key, CARD_COSTS[key]))
        else:
            unmapped.append((raw, key))

    print("=== Mapped ===")
    for raw, key, cost in mapped:
        print(f"  {raw:35s} -> {key:25s} ({cost} elixir)")

    print()
    print("=== Unmapped ===")
    for raw, key in unmapped:
        print(f"  {raw:35s} -> {key}")

    print()
    print(
        f"summary: {len(mapped)}/{len(names)} classes mapped to CARD_COSTS "
        f"({len(unmapped)} unmapped)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
