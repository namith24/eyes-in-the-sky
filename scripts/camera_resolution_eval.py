"""Does the 960-px gain survive a low-resolution camera? (report Section 6.6)

The C2A test images are re-captured the way the ESP32-CAM would deliver them: each image is shrunk to fit the
camera frame (320 x 240 = the drone firmware, 640 x 480 = our alternative firmware; never enlarged, a camera
cannot add pixels) and JPEG-compressed. The baseline (480 px) and the proposed setting (960 px) are then run on
these degraded frames. Detections are mapped back to original-image coordinates and scored with the same
COCO-style evaluator and the same ground truth as Tables 6-7, so the "original" row must reproduce them.

    python scripts/camera_resolution_eval.py
Writes results/final_comparison/camera_resolution.json and .md.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from size_stratified_eval import MODELS, TEST, build_gt, score  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from ultralytics import YOLO  # noqa: E402

CAMERAS = {"original (C2A file)": None, "ESP32 640 x 480": (640, 480), "ESP32 320 x 240": (320, 240)}
JPEG_QUALITY = 70          # the ESP32 sketch uses jpeg_quality 12 on its 0-63 scale (lower = better)
CONFIGS = {"baseline @480": 480, "proposed @960": 960}


def capture(img, frame):
    """Simulate the camera: fit the scene into the frame (downscale only) and JPEG-compress it."""
    if frame is None:
        return img, 1.0
    h, w = img.shape[:2]
    s = min(frame[0] / w, frame[1] / h, 1.0)
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), s


def main():
    gt, sizes = build_gt()
    gt_coco = COCO()
    gt_coco.dataset = gt
    with contextlib.redirect_stdout(io.StringIO()):
        gt_coco.createIndex()
    model = YOLO(str(MODELS["baseline"]))
    results = {"jpeg_quality": JPEG_QUALITY, "runs": {}, "scale": {}}
    for cam, frame in CAMERAS.items():
        frames = {}
        for name, (img_id, _, _) in sizes.items():
            frames[img_id] = capture(cv2.imread(str(TEST / "images" / name)), frame)
        scales = [s for _, s in frames.values()]
        results["scale"][cam] = {"median": float(np.median(scales)), "min": float(min(scales))}
        for cfg, imgsz in CONFIGS.items():
            dets = []
            for img_id, (img, s) in frames.items():
                r = model.predict(img, imgsz=imgsz, conf=0.001, max_det=1000, device="cpu", verbose=False)[0]
                for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy() / s, r.boxes.conf.cpu().numpy()):
                    dets.append({"image_id": img_id, "category_id": 1, "score": float(c),
                                 "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]})
            res = score(gt_coco, dets)
            results["runs"][f"{cam} | {cfg}"] = res
            print(f"{cam:22s} {cfg:14s} mAP50={res['all']['AP50']:.3f}  tinyAP50={res['tiny (<10px)']['AP50']:.3f}  "
                  f"R@.25={res['all']['recall50_conf0.25']:.3f}  tinyR@.25={res['tiny (<10px)']['recall50_conf0.25']:.3f}",
                  flush=True)
    out = ROOT / "results" / "final_comparison"
    (out / "camera_resolution.json").write_text(json.dumps(results, indent=2))
    lines = ["# 960-px gain under simulated ESP32 capture (test split, COCO-style, IoU 0.5)", "",
             f"JPEG quality {JPEG_QUALITY}; downscale only. Median image scale: " +
             ", ".join(f"{k}: {v['median']:.2f}" for k, v in results["scale"].items()), "",
             "| Camera | Config | mAP@50 | AP@50 <10px | Recall@0.25 | Recall@0.25 <10px |", "|---|---|---|---|---|---|"]
    for key, r in results["runs"].items():
        cam, cfg = key.split(" | ")
        lines.append(f"| {cam} | {cfg} | {r['all']['AP50']:.3f} | {r['tiny (<10px)']['AP50']:.3f} | "
                     f"{r['all']['recall50_conf0.25']:.3f} | {r['tiny (<10px)']['recall50_conf0.25']:.3f} |")
    (out / "camera_resolution.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
