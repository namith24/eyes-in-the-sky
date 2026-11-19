"""
Creates a reproducible subset of new_dataset3 for fast training.
Samples 1000 train / 300 val / 300 test images with seed=42.
"""
import os
import random
import shutil
from pathlib import Path

SEED = 42
SPLITS = {
    "train": 1000,
    "val": 300,
    "test": 300,
}

SRC_BASE = Path("d:/CAPSTONE SOFTWARE/data/datasets/new_dataset3")
DST_BASE = Path("d:/CAPSTONE SOFTWARE/data/datasets/new_dataset3_subset")

random.seed(SEED)

for split, n in SPLITS.items():
    src_img_dir = SRC_BASE / split / "images"
    src_lbl_dir = SRC_BASE / split / "labels"
    dst_img_dir = DST_BASE / split / "images"
    dst_lbl_dir = DST_BASE / split / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted([f for f in src_img_dir.iterdir() if f.suffix == ".png"])
    n = min(n, len(all_images))
    sampled = random.sample(all_images, n)

    copied = 0
    for img_path in sampled:
        lbl_path = src_lbl_dir / (img_path.stem + ".txt")
        shutil.copy2(img_path, dst_img_dir / img_path.name)
        if lbl_path.exists():
            shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)
        else:
            # Write empty label file if missing
            (dst_lbl_dir / (img_path.stem + ".txt")).write_text("")
        copied += 1

    print(f"{split}: copied {copied} images to {dst_img_dir}")

print("Subset creation complete.")
