#!/usr/bin/env python3
"""Run real-time UAV survivor detection from a YAML configuration.

Usage:
    python scripts/inference.py --config configs/inference_config.yaml
    python scripts/inference.py --config configs/inference_config.yaml --source 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_survivor_detection.config.manager import ConfigurationManager
from uav_survivor_detection.inference.pipeline import InferencePipeline


def parse_args() -> argparse.Namespace:
    """Parse inference command-line options."""
    parser = argparse.ArgumentParser(description="Run real-time UAV detection")
    parser.add_argument("--config", type=Path, required=True, help="Inference YAML path")
    parser.add_argument("--source", type=str, default=None, help="Override video source")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override model checkpoint")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--conf-threshold", type=float, default=None)
    parser.add_argument("--iou-threshold", type=float, default=None)
    parser.add_argument("--no-display", action="store_true", help="Disable the OpenCV window")
    parser.add_argument("--no-logging", action="store_true", help="Disable CSV/JSON logging")
    return parser.parse_args()


def main() -> int:
    """Load configuration and run the inference pipeline."""
    args = parse_args()
    system_config = ConfigurationManager(args.config).load_config()
    if system_config.inference is None:
        raise ValueError("The configuration must contain an inference section")

    config = system_config.inference
    if args.source is not None:
        config.video_source = args.source
    if args.checkpoint is not None:
        config.model_checkpoint = args.checkpoint
    if args.device is not None:
        config.device = args.device
    if args.conf_threshold is not None:
        config.conf_threshold = args.conf_threshold
    if args.iou_threshold is not None:
        config.iou_threshold = args.iou_threshold
    if args.no_display:
        config.display_enabled = False
    if args.no_logging:
        config.logging_enabled = False

    print(f"Connecting to video source: {config.video_source}")
    print(f"Model checkpoint: {config.model_checkpoint}")
    print(f"Device: {config.device}")
    pipeline = InferencePipeline(config)
    try:
        pipeline.run()
    except KeyboardInterrupt:
        pipeline.shutdown()
    except Exception as error:
        pipeline.shutdown()
        print(f"Inference failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())