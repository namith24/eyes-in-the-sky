"""Fair baseline vs. enhanced comparison (same data, epochs, image size, seed).

Trains:
  1. Baseline  : stock YOLO11n (COCO-pretrained)
  2. Enhanced  : YOLO11n-P2-CBAM (same pretrained weights where layers match)
then evaluates both on the held-out TEST split and writes
results/final_comparison/comparison.{json,md}.

Usage:
    python scripts/train_compare.py                       # full run
    python scripts/train_compare.py --epochs 1 --fraction 0.03 --tag smoke
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_survivor_detection.models.custom_yolo import ENHANCED_YAML, register_custom_modules  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DATA = ROOT / "configs" / "new_dataset3_subset.yaml"


def n_params(model):
    return sum(p.numel() for p in model.model.parameters())


def cbam_count(model):
    return sum(type(m).__name__ == "CBAM" for m in model.model.modules())


def train(name, model, args, out):
    t0 = time.time()
    model.train(
        data=str(DATA), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device="cpu", workers=args.workers, seed=42, deterministic=True,
        project=str(out), name=name, exist_ok=True, fraction=args.fraction,
        patience=100, plots=True, verbose=False, cache=args.cache or False,
    )
    return out / name / "weights" / "best.pt", time.time() - t0


def evaluate(ckpt, args, out, name):
    m = YOLO(str(ckpt))
    r = m.val(data=str(DATA), split="test", imgsz=args.imgsz, batch=args.batch,
              device="cpu", project=str(out), name=f"test_{name}", exist_ok=True,
              plots=True, verbose=False)
    speed = r.speed  # ms per image
    return {
        "checkpoint": str(ckpt.relative_to(ROOT)),
        "params": n_params(m),
        "cbam_layers": cbam_count(m),
        "precision": round(float(r.box.mp), 4),
        "recall": round(float(r.box.mr), 4),
        "mAP50": round(float(r.box.map50), 4),
        "mAP50_95": round(float(r.box.map), 4),
        "inference_ms": round(speed["inference"], 1),
        "fps_model_only": round(1000 / max(speed["inference"], 1e-6), 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=480)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--tag", default="final_comparison")
    p.add_argument("--cache", choices=["ram", "disk"], default=None,
                   help="cache images (ram is fastest but needs several GB free)")
    p.add_argument("--only", choices=["baseline", "enhanced"], default=None)
    args = p.parse_args()

    out = ROOT / "runs" / args.tag
    res_dir = ROOT / "results" / args.tag
    res_dir.mkdir(parents=True, exist_ok=True)
    results_file = res_dir / "comparison.json"
    results = json.loads(results_file.read_text()) if results_file.exists() else {}
    results["settings"] = {k: v for k, v in vars(args).items() if k != "only"} | {"seed": 42, "data": str(DATA.name)}

    register_custom_modules()
    jobs = {
        "baseline": lambda: YOLO("yolo11n.pt"),
        "enhanced": lambda: YOLO(str(ENHANCED_YAML)).load(str(ROOT / "yolo11n.pt")),
    }
    for name, build in jobs.items():
        if args.only and name != args.only:
            continue
        print(f"\n===== TRAINING {name.upper()} =====", flush=True)
        best, secs = train(name, build(), args, out)
        metrics = evaluate(best, args, out, name)
        metrics["train_minutes"] = round(secs / 60, 1)
        results[name] = metrics
        results_file.write_text(json.dumps(results, indent=2))
        print(f"{name}: {metrics}", flush=True)

    if "baseline" in results and "enhanced" in results:
        b, e = results["baseline"], results["enhanced"]
        rows = ["| Metric | Baseline YOLO11n | Enhanced YOLO11n-P2-CBAM | Change |",
                "|---|---|---|---|"]
        for k in ["precision", "recall", "mAP50", "mAP50_95"]:
            ch = (e[k] - b[k]) / b[k] * 100 if b[k] else 0
            rows.append(f"| {k} | {b[k]:.4f} | {e[k]:.4f} | {ch:+.1f}% |")
        for k in ["params", "cbam_layers", "inference_ms", "fps_model_only", "train_minutes"]:
            rows.append(f"| {k} | {b[k]} | {e[k]} | |")
        s = results["settings"]
        md = (f"# Baseline vs Enhanced (test split)\n\nSame data, epochs={s['epochs']}, "
              f"imgsz={s['imgsz']}, batch={s['batch']}, seed=42, CPU.\n\n" + "\n".join(rows) + "\n")
        (res_dir / "comparison.md").write_text(md, encoding="utf-8")
        print("\n" + md, flush=True)


if __name__ == "__main__":
    main()
