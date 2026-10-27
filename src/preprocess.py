"""Create reproducible, group-aware train/validation/test feature splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def _assign_label_groups(label_frame: pd.DataFrame, seed: int) -> dict[str, str]:
    """Assign whole source groups approximately 70/15/15 within one class."""
    groups = list(label_frame.groupby("source_group", sort=True))
    if len(groups) < 3:
        raise ValueError(f"Class {label_frame['label'].iloc[0]!r} has only {len(groups)} source groups; need at least three.")
    random.Random(seed).shuffle(groups)
    totals = {name: 0 for name in ("train", "validation", "test")}
    targets = {"train": len(label_frame) * 0.70, "validation": len(label_frame) * 0.15, "test": len(label_frame) * 0.15}
    assignment: dict[str, str] = {}
    for group_name, group in groups:
        split = min(totals, key=lambda name: totals[name] / max(targets[name], 1))
        assignment[str(group_name)] = split
        totals[split] += len(group)
    return assignment


def make_splits(features_csv: Path, output_dir: Path, seed: int) -> dict[str, int]:
    frame = pd.read_csv(features_csv)
    required = {"label", "source_path", "source_group"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Feature CSV is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Feature CSV contains no rows")
    assignments: dict[str, str] = {}
    for index, (_, label_frame) in enumerate(frame.groupby("label", sort=True)):
        assignments.update(_assign_label_groups(label_frame, seed + index))
    frame["split"] = frame["source_group"].map(assignments)
    if frame["split"].isna().any():
        raise RuntimeError("Some rows were not assigned to a split")
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        subset = frame.loc[frame["split"] == split].drop(columns="split")
        subset.to_csv(output_dir / f"{split}.csv", index=False)
        counts[split] = len(subset)
    classes = sorted(frame["label"].unique().tolist())
    (output_dir / "classes.json").write_text(json.dumps(classes, indent=2) + "\n", encoding="utf-8")
    (output_dir / "split_summary.json").write_text(json.dumps({"seed": seed, "splits": counts, "class_counts": frame.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(), "source_groups": frame.groupby("split")["source_group"].nunique().to_dict()}, indent=2) + "\n", encoding="utf-8")
    if frame.groupby("source_group")["split"].nunique().gt(1).any():
        raise RuntimeError("Source-group leakage detected")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("dataset/processed/features.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print("Created group-aware splits:", make_splits(args.features, args.output_dir, args.seed))


if __name__ == "__main__":
    main()
