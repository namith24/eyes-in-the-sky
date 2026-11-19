#!/usr/bin/env python3
"""Evaluate all trained model checkpoints and produce a consolidated report."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultralytics import YOLO

DATASET_YAML = Path("D:/CAPSTONE SOFTWARE/configs/new_dataset3_subset.yaml")
# Match the 640px training resolution so small survivors are evaluated consistently.
IMG_SIZE = 640
DEVICE = "cpu"
CONF = 0.25
IOU = 0.5

MODELS = {
    "baseline_main":  Path("D:/CAPSTONE SOFTWARE/runs/new_dataset3/train/weights/best.pt"),
    "enhanced_main":  Path("D:/CAPSTONE SOFTWARE/runs/new_dataset3_enhanced/train/weights/best.pt"),
    "ablation_baseline": Path("D:/CAPSTONE SOFTWARE/results/ablation/baseline/train/weights/best.pt"),
    "ablation_gam":      Path("D:/CAPSTONE SOFTWARE/results/ablation/gam-only/train/weights/best.pt"),
    "ablation_cbam":     Path("D:/CAPSTONE SOFTWARE/results/ablation/cbam-only/train/weights/best.pt"),
    "ablation_gam_cbam": Path("D:/CAPSTONE SOFTWARE/results/ablation/gam-cbam/train/weights/best.pt"),
}

results = {}

for name, ckpt in MODELS.items():
    if not ckpt.exists():
        print(f"[SKIP] {name} — checkpoint not found: {ckpt}")
        continue
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"Checkpoint: {ckpt}")
    print(f"{'='*60}")
    try:
        model = YOLO(str(ckpt))
        val_results = model.val(
            data=str(DATASET_YAML),
            imgsz=IMG_SIZE,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            verbose=True,
            split="test",
        )
        metrics = {
            "map50":    round(float(val_results.box.map50), 4),
            "map50_95": round(float(val_results.box.map),   4),
            "precision":round(float(val_results.box.mp),    4),
            "recall":   round(float(val_results.box.mr),    4),
        }
        results[name] = metrics
        print(f"  mAP@50:    {metrics['map50']}")
        print(f"  mAP@50:95: {metrics['map50_95']}")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall:    {metrics['recall']}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results[name] = {"error": str(e)}

# Save JSON
out_path = Path("D:/CAPSTONE SOFTWARE/results/evaluation_summary.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

# Print markdown table
print("\n\n" + "="*60)
print("FINAL EVALUATION SUMMARY")
print("="*60)
print(f"{'Model':<25} {'mAP@50':>8} {'mAP@50:95':>10} {'Precision':>10} {'Recall':>8}")
print("-"*65)
for name, m in results.items():
    if "error" not in m:
        print(f"{name:<25} {m['map50']:>8.4f} {m['map50_95']:>10.4f} {m['precision']:>10.4f} {m['recall']:>8.4f}")
    else:
        print(f"{name:<25}  ERROR: {m['error'][:40]}")

print(f"\nResults saved to: {out_path}")
