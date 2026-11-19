"""Apply the pre-registered selection rule on VALIDATION, then score the chosen
configuration (plus ablation rows) once on TEST and render qualitative images.

Rule: best validation mAP@50 among non-baseline variants with latency <= 100 ms
(>= 10 FPS) on a 640x480 frame on the project laptop CPU.
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "final_comparison"
BASE = "baseline:full@480"

val = json.loads((RES / "inference_variants_val.json").read_text())
ok = {k: v for k, v in val.items() if k != BASE and v["latency_ms_640x480"] <= 100}
proposed = max(ok, key=lambda k: ok[k]["all"]["AP50"])
variant = proposed.split(":")[1]
both = f"enhanced:{variant}"
selection = {
    "rule": "max val mAP@50, latency_ms_640x480 <= 100, excluding baseline:full@480",
    "candidates": {k: {"val_mAP50": v["all"]["AP50"], "latency_ms": v["latency_ms_640x480"]} for k, v in val.items()},
    "proposed": proposed,
    "both": both,
}
(RES / "selection.json").write_text(json.dumps(selection, indent=2))
print("Selected on validation:", proposed, flush=True)

runs = list(dict.fromkeys([BASE, "enhanced:full@480", proposed, f"baseline:{variant}", both]))
subprocess.run([sys.executable, str(ROOT / "scripts" / "inference_variants_eval.py"), "--split", "test",
                "--variants", *runs], check=True)

# ---- qualitative images (one held-out test image, conf >= 0.25)
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from uav_survivor_detection.models import custom_yolo  # noqa: E402,F401
from ultralytics import YOLO  # noqa: E402
import inference_variants_eval as ive  # noqa: E402
import size_stratified_eval as sse  # noqa: E402

assets = ROOT / "presentation" / "assets"
test_dir = ROOT / "data" / "datasets" / "new_dataset3_subset" / "test"
img_path = test_dir / "images" / "fire_image0227_0.png"
img = cv2.imread(str(img_path))
gt = cv2.imread(str(assets / "problem_gt.png"))
cv2.imwrite(str(assets / "qual_gt.png"), gt)
counts = {"gt": sum(1 for line in (test_dir / "labels" / f"{img_path.stem}.txt").read_text().split("\n") if line.strip())}
for name, key in [("baseline", BASE), ("proposed", proposed)]:
    m, v = key.split(":")
    model = YOLO(str(sse.MODELS[m]))
    boxes, scores = ive.run_variant(model, img, v, conf=0.25)
    out = img.copy()
    for x1, y1, x2, y2 in boxes.numpy().astype(int):
        cv2.rectangle(out, (x1, y1), (x2, y2), (12, 89, 232), 2)
    cv2.imwrite(str(assets / f"qual_{name}.png"), out)
    counts[name] = int(len(boxes))
(assets / "qual_counts.json").write_text(json.dumps(counts))
print("qualitative counts:", counts)
