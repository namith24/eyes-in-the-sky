"""
Demo inference script — Task 1 & 2.

Runs the best trained model on 12 selected test images (mix of easy/hard,
small-object frames included), saves annotated images to demo_output/,
generates a demo GIF from any sequential frames found, and prints
parameter count + GFLOPs for both baseline and GAM+CBAM enhanced models.

Usage:
    python scripts/demo_inference.py
"""

import sys
import os
import time
import json
import random
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
TEST_DIR = ROOT / "data/datasets/new_dataset3_subset/test/images"
DEMO_OUT = ROOT / "demo_output"
DEMO_OUT.mkdir(exist_ok=True)

BASELINE_CKPT  = ROOT / "runs/new_dataset3/train/weights/best.pt"
ENHANCED_CKPT  = ROOT / "runs/new_dataset3_enhanced/train/weights/best.pt"
PRETRAINED_PT  = ROOT / "yolo11n.pt"          # original YOLO11n weights

# Use the best trained checkpoint as the "original" model; fall back to raw
# pretrained weights only if the trained checkpoint doesn't exist.
ORIGINAL_CKPT = BASELINE_CKPT if BASELINE_CKPT.exists() else PRETRAINED_PT

CONF_THRESH = 0.25
IOU_THRESH  = 0.45
IMG_SIZE    = 640   # full resolution for demo

# ── Helper: pick 12 diverse test images ────────────────────────────────────
def pick_demo_images(test_dir: Path, n: int = 12, seed: int = 42) -> list:
    """Select n images spread across scene indices for variety.

    Images with low frame indices tend to have more/larger survivors (easy).
    High frame indices are harder / more cluttered.  We mix both buckets,
    plus explicitly include images whose label files contain small objects.
    """
    all_imgs = sorted(test_dir.glob("*.png"))
    if len(all_imgs) == 0:
        raise FileNotFoundError(f"No PNG images found in {test_dir}")

    rng = random.Random(seed)

    # Split into thirds: easy (first third), medium, hard (last third)
    third = max(1, len(all_imgs) // 3)
    easy   = all_imgs[:third]
    medium = all_imgs[third: 2 * third]
    hard   = all_imgs[2 * third:]

    picks = []
    picks += rng.sample(easy,   min(4, len(easy)))
    picks += rng.sample(medium, min(4, len(medium)))
    picks += rng.sample(hard,   min(4, len(hard)))

    # Try to swap in images that have small-object ground-truth labels
    label_dir = test_dir.parent.parent / "test" / "labels"   # new_dataset3/test/labels
    if not label_dir.exists():
        # fall back to sibling labels/ directory
        label_dir = test_dir.parent / "labels"

    small_imgs = []
    if label_dir.exists():
        for img in all_imgs:
            lbl = label_dir / (img.stem + ".txt")
            if not lbl.exists():
                continue
            for line in lbl.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # YOLO format: class cx cy w h (normalised)
                w_norm = float(parts[3])
                h_norm = float(parts[4])
                # Estimate pixel size at 640 px
                pw = w_norm * 640
                ph = h_norm * 640
                if pw < 32 and ph < 32:
                    small_imgs.append(img)
                    break

    if small_imgs:
        # Replace up to 4 picks with small-object images
        small_sample = rng.sample(small_imgs, min(4, len(small_imgs)))
        picks = picks[: max(0, n - len(small_sample))] + small_sample

    # Trim / pad to exactly n
    picks = picks[:n]
    while len(picks) < n:
        picks.append(rng.choice(all_imgs))

    return picks


# ── Helper: draw detections on a frame ─────────────────────────────────────
COLOURS = [
    (0, 255, 0),    # green  – survivor
    (0, 128, 255),  # orange
    (255, 0, 128),  # pink
]

def annotate_frame(img: np.ndarray, result, model_label: str) -> np.ndarray:
    """Draw bounding boxes + confidence scores onto img and return a copy."""
    out = img.copy()
    boxes = result.boxes

    n_det = 0
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = f"{result.names[cls]} {conf:.2f}"
            colour = COLOURS[cls % len(COLOURS)]

            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            n_det += 1

    # Overlay: model name + detection count + timestamp
    ts = time.strftime("%H:%M:%S UTC")
    info = f"{model_label} | Detections: {n_det} | {ts}"
    cv2.rectangle(out, (0, 0), (len(info) * 9 + 10, 24), (0, 0, 0), -1)
    cv2.putText(out, info, (5, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ── Task 2 helper: model complexity ────────────────────────────────────────
def print_complexity(label: str, ckpt: Path, imgsz: int = 640):
    """Load model and print parameter count + GFLOPs using thop / ultralytics."""
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"  Checkpoint: {ckpt}")
    print(f"{'='*55}")

    if not ckpt.exists():
        print(f"  [SKIP] checkpoint not found: {ckpt}")
        return

    model = YOLO(str(ckpt))

    # ultralytics summary gives params + GFLOPs
    try:
        # model.info() returns (layers, params, gradients, gflops) in recent versions
        info = model.info(verbose=False, imgsz=imgsz)
        if isinstance(info, (list, tuple)) and len(info) >= 4:
            layers, params, grads, gflops = info[0], info[1], info[2], info[3]
            print(f"  Layers     : {layers}")
            print(f"  Parameters : {params:,}")
            print(f"  GFLOPs     : {gflops:.2f}  (imgsz={imgsz})")
        else:
            # Fallback: use thop
            import torch
            from thop import profile
            dummy = torch.zeros(1, 3, imgsz, imgsz)
            macs, params = profile(model.model, inputs=(dummy,), verbose=False)
            gflops = macs * 2 / 1e9
            print(f"  Parameters : {int(params):,}")
            print(f"  GFLOPs     : {gflops:.2f}  (imgsz={imgsz})")
    except Exception as e:
        # Last-resort: count parameters manually
        params = sum(p.numel() for p in model.model.parameters())
        print(f"  Parameters : {params:,}  (GFLOPs unavailable: {e})")

    print()


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  UAV SURVIVOR DETECTION — DEMO INFERENCE")
    print("="*60)

    # ── Task 2: complexity for both models ──
    print("\n[TASK 2] Model complexity (parameters + GFLOPs)")
    print_complexity("BASELINE  model (best.pt)", ORIGINAL_CKPT, imgsz=IMG_SIZE)
    print_complexity("ENHANCED  model (GAM+CBAM best.pt)", ENHANCED_CKPT, imgsz=IMG_SIZE)

    # ── Task 1: demo inference ──
    print("\n[TASK 1] Running inference on 12 demo images")
    print(f"  Checkpoint : {ORIGINAL_CKPT}")
    print(f"  Output dir : {DEMO_OUT}")

    model = YOLO(str(ORIGINAL_CKPT))
    demo_imgs = pick_demo_images(TEST_DIR, n=12)
    print(f"  Selected   : {len(demo_imgs)} images")

    gif_frames = []
    results_log = []

    for i, img_path in enumerate(demo_imgs, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [WARN] Could not read {img_path.name}, skipping")
            continue

        preds = model.predict(
            source=str(img_path),
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            imgsz=IMG_SIZE,
            device="cpu",
            verbose=False,
        )

        result  = preds[0]
        n_det   = len(result.boxes) if result.boxes is not None else 0
        annotated = annotate_frame(img_bgr, result, "YOLO11n-baseline")

        # Save annotated image
        out_name = f"{i:02d}_{img_path.stem}_det{n_det}.jpg"
        out_path = DEMO_OUT / out_name
        cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

        results_log.append({
            "index": i,
            "filename": img_path.name,
            "detections": n_det,
            "output": out_name,
        })
        print(f"  [{i:02d}] {img_path.name:<45} → {n_det} detection(s)  → {out_name}")

        # Collect small version for GIF
        h, w = annotated.shape[:2]
        scale = min(1.0, 480 / max(h, w))
        small = cv2.resize(annotated, (int(w * scale), int(h * scale)))
        gif_frames.append(small)

    # ── Build GIF ──
    gif_path = DEMO_OUT / "demo_detections.gif"
    try:
        from PIL import Image as PILImage
        pil_frames = []
        for f in gif_frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_frames.append(PILImage.fromarray(rgb))

        if pil_frames:
            pil_frames[0].save(
                str(gif_path),
                save_all=True,
                append_images=pil_frames[1:],
                duration=700,   # ms per frame
                loop=0,
            )
            print(f"\n  GIF saved  : {gif_path}")
        else:
            print("\n  [WARN] No frames for GIF")
    except ImportError:
        print("\n  [INFO] Pillow not installed — GIF skipped. Install with: pip install Pillow")

    # ── Save results log ──
    log_path = DEMO_OUT / "demo_results.json"
    log_path.write_text(json.dumps(results_log, indent=2))
    print(f"  JSON log   : {log_path}")

    total_det = sum(r["detections"] for r in results_log)
    print(f"\n  Total detections across {len(results_log)} images: {total_det}")
    print("\n[DONE] All demo outputs saved to demo_output/")


if __name__ == "__main__":
    main()
