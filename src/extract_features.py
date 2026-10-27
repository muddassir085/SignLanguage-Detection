"""Extract compact MediaPipe Holistic landmark features from labelled images."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import pandas as pd

from landmarks import create_holistic, extract_features_from_rgb, feature_names

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def source_group(path: Path, raw_root: Path, group_size: int) -> str:
    """Create a conservative capture group from a nested source or contiguous image ID."""
    relative_parent = path.parent.relative_to(raw_root)
    nested_source = relative_parent.parts[1:]
    if nested_source:
        return f"{relative_parent.parts[0]}:{'/'.join(nested_source)}"
    match = re.search(r"(\d+)$", path.stem)
    bucket = int(match.group(1)) // group_size if match else path.stem
    return f"{relative_parent.parts[0]}:capture_{bucket}"


def iter_images(raw_root: Path):
    for class_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield class_dir.name, image_path


def extract_dataset(raw_root: Path, output_csv: Path, group_size: int) -> int:
    rows: list[dict[str, object]] = []
    columns = feature_names()
    unreadable = 0
    with create_holistic(static_image_mode=True) as holistic:
        for index, (label, image_path) in enumerate(iter_images(raw_root), start=1):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Skipping unreadable image: {image_path}", file=sys.stderr)
                unreadable += 1
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            extraction = extract_features_from_rgb(rgb, holistic)
            row = dict(zip(columns, extraction.features.tolist(), strict=True))
            row.update(
                label=label,
                source_path=image_path.relative_to(raw_root).as_posix(),
                source_group=source_group(image_path, raw_root, group_size),
            )
            rows.append(row)
            if index % 100 == 0:
                print(f"Extracted {index} images")

    if not rows:
        raise RuntimeError(f"No readable images found under {raw_root}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Wrote {len(rows)} feature rows to {output_csv} ({unreadable} unreadable images skipped)")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--output", type=Path, default=Path("dataset/processed/features.csv"))
    parser.add_argument("--group-size", type=int, default=15, help="Adjacent numbered frames per source group.")
    args = parser.parse_args()
    if args.group_size < 1:
        parser.error("--group-size must be positive")
    extract_dataset(args.raw_dir, args.output, args.group_size)


if __name__ == "__main__":
    main()

