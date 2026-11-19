"""Build a replay video from test images, used as the backup "drone" when no drone is connected.

Every image is resized to one frame size (the writer silently drops frames of a different size)
and held for --hold seconds, so the clip looks like a slow survey flight.

    python scripts/build_test_clip.py              # 30 images, 1 s each -> demo_output/test_clip.avi
    python scripts/build_test_clip.py --n 60 --hold 0.5
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent
img_dir = ROOT / "data/datasets/new_dataset3_subset/test/images"
out_path = ROOT / "demo_output/test_clip.avi"

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30, help="number of test images")
ap.add_argument("--hold", type=float, default=1.0, help="seconds each image stays on screen")
ap.add_argument("--fps", type=int, default=10)
ap.add_argument("--width", type=int, default=960)
ap.add_argument("--height", type=int, default=540)
args = ap.parse_args()

imgs = sorted(p for p in img_dir.glob("*.*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))[: args.n]
if not imgs:
    print(f"ERROR: no images found in {img_dir}")
    sys.exit(1)

size = (args.width, args.height)
writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"XVID"), args.fps, size)
repeat = max(1, round(args.hold * args.fps))
written = 0
for p in imgs:
    frame = cv2.imread(str(p))
    if frame is None:
        continue
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    for _ in range(repeat):
        writer.write(frame)
    written += 1
writer.release()
print(f"Written {written} images x {repeat} frames -> {out_path} ({size[0]}x{size[1]} @ {args.fps} fps)")
