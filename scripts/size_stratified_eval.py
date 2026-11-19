"""Size-stratified evaluation (research paper Section X "Small-object AP", Graphs 3 & 8).

Runs each model on the held-out TEST split and scores it with COCO-style AP,
separately for each person size (side length = sqrt(box area) in original-image pixels):
    tiny    : < 10 px     (C2A paper: 47% of instances)
    small   : 10 - 32 px
    medium+ : >= 32 px
Writes results/final_comparison/size_stratified.json and .md.

Usage:
    python scripts/size_stratified_eval.py --imgsz 480 800
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from uav_survivor_detection.models import custom_yolo  # noqa: E402,F401
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402
from ultralytics import YOLO  # noqa: E402

TEST = ROOT / "data" / "datasets" / "new_dataset3_subset" / "test"
MODELS = {
    "baseline": ROOT / "runs" / "final_comparison" / "baseline" / "weights" / "best.pt",
    "enhanced": ROOT / "runs" / "final_comparison" / "enhanced" / "weights" / "best.pt",
}
BUCKETS = {"all": [0, 1e10], "tiny (<10px)": [0, 10 ** 2], "small (10-32px)": [10 ** 2, 32 ** 2],
           "medium+ (>=32px)": [32 ** 2, 1e10]}


def build_gt():
    images, anns, sizes = [], [], {}
    for i, img_path in enumerate(sorted((TEST / "images").glob("*.*"))):
        h, w = cv2.imread(str(img_path)).shape[:2]
        images.append({"id": i, "file_name": img_path.name, "width": w, "height": h})
        sizes[img_path.name] = (i, w, h)
        lbl = TEST / "labels" / f"{img_path.stem}.txt"
        if not lbl.exists():
            continue
        for line in lbl.read_text().split("\n"):
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            bw, bh = bw * w, bh * h
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": 1, "iscrowd": 0,
                         "bbox": [cx * w - bw / 2, cy * h - bh / 2, bw, bh], "area": bw * bh})
    return {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "survivor"}]}, sizes


def predict(model_path, sizes, imgsz):
    model = YOLO(str(model_path))
    dets = []
    for name, (img_id, _, _) in sizes.items():
        r = model.predict(str(TEST / "images" / name), imgsz=imgsz, conf=0.001, max_det=1000,
                          device="cpu", verbose=False)[0]
        for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            dets.append({"image_id": img_id, "category_id": 1, "score": float(s),
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]})
    return dets


def score(gt_coco, dets):
    out = score_at(gt_coco, dets)
    # Recall at the operating threshold used in the live demo (conf >= 0.25)
    op = score_at(gt_coco, [d for d in dets if d["score"] >= 0.25])
    for b in out:
        out[b]["recall50_conf0.25"] = op[b]["recall50"]
    return out


def score_at(gt_coco, dets):
    out = {}
    if not dets:  # no detections at all: everything is zero
        return {name: {"AP50": 0.0, "AP50_95": 0.0, "recall50": 0.0} for name in BUCKETS}
    dt = gt_coco.loadRes(dets)
    for name, rng in BUCKETS.items():
        ev = COCOeval(gt_coco, dt, "bbox")
        ev.params.areaRng, ev.params.areaRngLbl = [rng], [name]
        ev.params.maxDets = [1000]
        with contextlib.redirect_stdout(io.StringIO()):
            ev.evaluate()
            ev.accumulate()
        p = ev.eval["precision"][:, :, 0, 0, 0]  # [iou, recall]
        r = ev.eval["recall"][:, 0, 0, 0]        # [iou]
        valid = lambda a: float(np.mean(a[a > -1])) if np.any(a > -1) else float("nan")  # noqa: E731
        out[name] = {"AP50": round(valid(p[0]), 4), "AP50_95": round(valid(p), 4),
                     "recall50": round(float(r[0]), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, nargs="+", default=[480, 800])
    args = ap.parse_args()

    gt, sizes = build_gt()
    gt_coco = COCO()
    gt_coco.dataset = gt
    with contextlib.redirect_stdout(io.StringIO()):
        gt_coco.createIndex()
    counts = {}
    for name, (lo, hi) in BUCKETS.items():
        counts[name] = sum(lo <= a["area"] < hi for a in gt["annotations"])
    print("GT instances per bucket:", counts, flush=True)

    results = {"instances": counts, "runs": {}}
    for imgsz in args.imgsz:
        for model_name, path in MODELS.items():
            key = f"{model_name}@{imgsz}"
            cache = ROOT / "runs" / "final_comparison" / f"test_dets_{key.replace('@', '_')}.json"
            if cache.exists():
                dets = json.loads(cache.read_text())
            else:
                dets = predict(path, sizes, imgsz)
                cache.write_text(json.dumps(dets))
            results["runs"][key] = score(gt_coco, dets)
            print(key, json.dumps(results["runs"][key]), flush=True)

    out_dir = ROOT / "results" / "final_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "size_stratified.json").write_text(json.dumps(results, indent=2))
    lines = ["# Size-stratified results (test split, COCO-style, recall/AP at IoU 0.5)", "",
             "Instances: " + ", ".join(f"{k}: {v}" for k, v in counts.items()), "",
             "recall = max recall over all detections; recall@0.25 = recall at the demo's conf>=0.25 threshold", "",
             "| Run | " + " | ".join(f"{b} AP50 | {b} recall | {b} recall@0.25" for b in BUCKETS) + " |",
             "|---|" + "---|---|---|" * len(BUCKETS)]
    for key, res in results["runs"].items():
        lines.append(f"| {key} | " + " | ".join(
            f"{res[b]['AP50']:.3f} | {res[b]['recall50']:.3f} | {res[b]['recall50_conf0.25']:.3f}"
            for b in BUCKETS) + " |")
    (out_dir / "size_stratified.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
