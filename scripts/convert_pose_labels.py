#!/usr/bin/env python3
"""Convert six-field survivor labels to standard five-field YOLO detection labels.

The source dataset stores ``class x y width height pose_state``. This utility
preserves the source labels in a sibling backup directory and removes only the
pose-state field from the active train/val/test labels used by YOLO detection.

Usage:
    python scripts/convert_pose_labels.py \
        --dataset data/datasets/new_dataset3
"""

import argparse
import shutil
from pathlib import Path
from typing import Iterable


POSE_NAMES = {
    0: "bent",
    1: "kneeling",
    2: "lying",
    3: "sitting",
    4: "upright",
}


def convert_labels(dataset: Path, backup_name: str = "labels_with_pose") -> tuple[int, int]:
    """Back up and convert split labels in a prepared YOLO dataset.

    Args:
        dataset: Dataset root containing ``train``, ``val``, and ``test`` folders.
        backup_name: Name used for each split's original-label backup folder.

    Returns:
        Tuple of ``(converted_files, converted_rows)``.

    Raises:
        FileNotFoundError: If a required split or labels directory is missing.
        ValueError: If a label row is not six-field pose-aware YOLO data.
    """
    converted_files = 0
    converted_rows = 0
    for split in ("train", "val", "test"):
        labels_dir = dataset / split / "labels"
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"Missing labels directory: {labels_dir}")
        backup_dir = dataset / split / backup_name
        if not backup_dir.exists():
            shutil.copytree(labels_dir, backup_dir)

        for label_path in labels_dir.glob("*.txt"):
            output_rows = []
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 6:
                    raise ValueError(
                        f"{label_path}:{line_number}: expected 6 fields, got {len(fields)}"
                    )
                pose_id = int(float(fields[5]))
                if pose_id not in POSE_NAMES:
                    raise ValueError(
                        f"{label_path}:{line_number}: unknown pose ID {pose_id}"
                    )
                output_rows.append(" ".join(fields[:5]))
                converted_rows += 1
            label_path.write_text("\n".join(output_rows) + "\n", encoding="utf-8")
            converted_files += 1
    return converted_files, converted_rows


def main() -> int:
    """Parse CLI arguments and convert the dataset labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    files, rows = convert_labels(args.dataset)
    print(f"Converted {rows} label rows across {files} files.")
    print("Original six-field labels were preserved in split labels_with_pose folders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
