"""Small-object-aware INFERENCE variants vs. standard YOLO11n inference.

Selection protocol (fixed before looking at test results):
  1. Score every variant on the VALIDATION split.
  2. Pick the variant with the best mAP@50 whose CPU latency on a 640x480 frame
     is <= 100 ms (the project's >= 10 FPS requirement).
  3. Report that single variant once on the TEST split, against the standard baseline.

Variants (same trained weights, only inference changes):
  full@N      : standard single pass, image resized to N px            (baseline = full@480)
  tta@N       : Ultralytics test-time augmentation (flip + multi-scale)
  tiled@N     : full@N pass + 2x2 overlapping tiles (each at N px), merged with NMS
                (slicing-aided inference, after SAHI: Akyon et al., ICIP 2022)

Usage:
    python scripts/inference_variants_eval.py --split val
    python scripts/inference_variants_eval.py --split test --variants baseline:full@480 baseline:tiled@640
"""

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from uav_survivor_detection.models import custom_yolo  # noqa: E402,F401
from pycocotools.coco import COCO  # noqa: E402
from ultralytics import YOLO  # noqa: E402
import size_stratified_eval as sse  # noqa: E402

DEFAULT_VARIANTS = [f"{m}:{v}" for m in ("baseline", "enhanced")
                    for v in ("full@480", "full@640", "full@800", "full@960", "tta@640", "tiled@480", "tiled@640")]


def run_variant(model, img, variant, conf=0.001):
    kind, size = variant.split("@")
    size = int(size)
    if kind == "full":
        r = model.predict(img, imgsz=size, conf=conf, max_det=1000, verbose=False)[0]
        return r.boxes.xyxy.cpu(), r.boxes.conf.cpu()
    if kind == "tta":
        r = model.predict(img, imgsz=size, conf=conf, max_det=1000, augment=True, verbose=False)[0]
        return r.boxes.xyxy.cpu(), r.boxes.conf.cpu()
    if kind == "tiled":
        h, w = img.shape[:2]
        boxes, scores = [], []
        r = model.predict(img, imgsz=size, conf=conf, max_det=1000, verbose=False)[0]
        boxes.append(r.boxes.xyxy.cpu()); scores.append(r.boxes.conf.cpu())
        th, tw = int(h * 0.6), int(w * 0.6)  # 2x2 tiles with 20% overlap
        for y0 in (0, h - th):
            for x0 in (0, w - tw):
                tile = img[y0:y0 + th, x0:x0 + tw]
                r = model.predict(tile, imgsz=size, conf=conf, max_det=1000, verbose=False)[0]
                b = r.boxes.xyxy.cpu().clone()
                b[:, [0, 2]] += x0
                b[:, [1, 3]] += y0
                boxes.append(b); scores.append(r.boxes.conf.cpu())
        b, s = torch.cat(boxes), torch.cat(scores)
        keep = torchvision.ops.nms(b, s, 0.5)[:1000]
        return b[keep], s[keep]
    raise ValueError(variant)


def latency_ms(model, variant, n=15):
    frame = cv2.resize(cv2.imread(str(next((sse.TEST / "images").glob("*.*")))), (640, 480))
    for _ in range(3):
        run_variant(model, frame, variant, conf=0.25)
    t = time.perf_counter()
    for _ in range(n):
        run_variant(model, frame, variant, conf=0.25)
    return (time.perf_counter() - t) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    args = ap.parse_args()

    sse.TEST = ROOT / "data" / "datasets" / "new_dataset3_subset" / args.split
    gt, sizes = sse.build_gt()
    gt_coco = COCO()
    gt_coco.dataset = gt
    with contextlib.redirect_stdout(io.StringIO()):
        gt_coco.createIndex()

    out_file = ROOT / "results" / "final_comparison" / f"inference_variants_{args.split}.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else {}
    models = {}
    for spec in args.variants:
        mname, variant = spec.split(":")
        if spec in results:
            print(spec, "cached", flush=True)
            continue
        model = models.setdefault(mname, YOLO(str(sse.MODELS[mname])))
        dets = []
        for name, (img_id, _, _) in sizes.items():
            b, s = run_variant(model, cv2.imread(str(sse.TEST / "images" / name)), variant)
            for (x1, y1, x2, y2), sc in zip(b.numpy(), s.numpy()):
                dets.append({"image_id": img_id, "category_id": 1, "score": float(sc),
                             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]})
        res = sse.score(gt_coco, dets)
        res["latency_ms_640x480"] = round(latency_ms(model, variant), 1)
        results[spec] = res
        out_file.write_text(json.dumps(results, indent=2))
        a, t = res["all"], res["tiny (<10px)"]
        print(f"{spec:22s} mAP50={a['AP50']:.4f} mAP50-95={a['AP50_95']:.4f} tinyAP50={t['AP50']:.4f} "
              f"R@0.25={a['recall50_conf0.25']:.3f} tinyR@0.25={t['recall50_conf0.25']:.3f} "
              f"latency={res['latency_ms_640x480']}ms", flush=True)


if __name__ == "__main__":
    main()
