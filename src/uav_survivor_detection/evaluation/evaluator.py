"""Evaluator module for comprehensive model evaluation.

This module provides evaluation capabilities for YOLO models on UAV survivor detection tasks.
It computes detection metrics (mAP@50, mAP@50:95, Small_Object_AP), efficiency metrics (FPS, latency),
and model complexity metrics (parameter count, GFLOPs).

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.6**
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
import json
from datetime import datetime

import numpy as np
import torch
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2

from ..config.models import EvaluatorConfig


@dataclass
class DetectionMetrics:
    """Detection performance metrics.
    
    Attributes:
        precision: Overall precision score
        recall: Overall recall score
        map50: Mean Average Precision at IoU threshold 0.50
        map50_95: Mean Average Precision averaged over IoU thresholds 0.50-0.95
        small_object_ap: Average Precision for small objects (<32x32 pixels)
        per_class_ap: Average Precision per class
    """
    precision: float
    recall: float
    map50: float
    map50_95: float
    small_object_ap: float
    per_class_ap: Dict[str, float] = field(default_factory=dict)


@dataclass
class EfficiencyMetrics:
    """Inference efficiency metrics.
    
    Attributes:
        fps_mean: Mean frames per second
        fps_median: Median frames per second
        latency_ms_mean: Mean latency in milliseconds
        latency_ms_median: Median latency in milliseconds
        latency_ms_p95: 95th percentile latency in milliseconds
    """
    fps_mean: float
    fps_median: float
    latency_ms_mean: float
    latency_ms_median: float
    latency_ms_p95: float


@dataclass
class ModelComplexity:
    """Model complexity metrics.
    
    Attributes:
        parameters: Total number of model parameters
        gflops: Giga floating-point operations per forward pass
    """
    parameters: int
    gflops: float


@dataclass
class EvaluationResult:
    """Complete evaluation result.
    
    Attributes:
        detection_metrics: Detection performance metrics
        efficiency_metrics: Inference efficiency metrics
        complexity_metrics: Model complexity metrics
        total_images: Total number of images evaluated
        total_detections: Total number of detections made
        evaluation_duration_seconds: Total evaluation duration
        timestamp: Evaluation timestamp (ISO format)
    """
    detection_metrics: DetectionMetrics
    efficiency_metrics: EfficiencyMetrics
    complexity_metrics: ModelComplexity
    total_images: int
    total_detections: int
    evaluation_duration_seconds: float
    timestamp: str


class Evaluator:
    """Comprehensive model evaluator for UAV survivor detection.
    
    This class provides methods to evaluate YOLO models on test datasets,
    computing detection metrics, efficiency metrics, and model complexity.
    
    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.6**
    """
    
    def __init__(self, config: EvaluatorConfig) -> None:
        """Initialize evaluator with configuration.
        
        Args:
            config: EvaluatorConfig object with checkpoint path, test dataset path, and thresholds
        """
        if not isinstance(config, EvaluatorConfig):
            raise TypeError(f"Expected EvaluatorConfig, got {type(config)}")
        
        self.config = config
        self.model: Optional[YOLO] = None
        self.latency_measurements: List[float] = []
        
    def load_model(self) -> None:
        """Load YOLO model from checkpoint.
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        if self.config.checkpoint_path is None:
            raise ValueError("checkpoint_path must be specified in config")
        
        if not Path(self.config.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.config.checkpoint_path}")
        
        self.model = YOLO(self.config.checkpoint_path)
        
        # Set device
        if self.config.device == "cpu":
            self.model.to("cpu")
        
    def evaluate(self) -> EvaluationResult:
        """Run complete evaluation on test dataset.
        
        Returns:
            EvaluationResult with all metrics
            
        Raises:
            ValueError: If model not loaded or test dataset path not specified
            
        **Validates: Requirement 5.6** - Complete test set evaluation
        """
        if self.model is None:
            self.load_model()
        
        if self.config.test_dataset_path is None:
            raise ValueError("test_dataset_path must be specified in config")
        
        start_time = time.time()
        
        # Compute all metrics
        detection_metrics = self.compute_detection_metrics()
        efficiency_metrics = self.compute_efficiency_metrics()
        complexity_metrics = self.compute_complexity_metrics()
        
        # Count total images and detections
        total_images = self._count_test_images()
        total_detections = self._count_total_detections()
        
        evaluation_duration = time.time() - start_time
        
        result = EvaluationResult(
            detection_metrics=detection_metrics,
            efficiency_metrics=efficiency_metrics,
            complexity_metrics=complexity_metrics,
            total_images=total_images,
            total_detections=total_detections,
            evaluation_duration_seconds=evaluation_duration,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        return result
    
    def compute_detection_metrics(self) -> DetectionMetrics:
        """Compute detection performance metrics.
        
        Uses ultralytics model.val() for mAP@50 and mAP@50:95, then computes
        Small_Object_AP using pycocotools on filtered ground-truth boxes.
        
        Returns:
            DetectionMetrics with precision, recall, mAP scores
            
        **Validates: Requirement 5.1** - Compute precision, recall, mAP@50, mAP@50:95, Small_Object_AP
        """
        if self.model is None:
            raise ValueError("Model must be loaded before computing metrics")
        
        # Run ultralytics validation to get mAP@50 and mAP@50:95
        results = self.model.val(
            data=str(self.config.test_dataset_path),
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.img_size,
            device=self.config.device,
            verbose=False
        )
        
        # Extract metrics from results
        # ultralytics results object has metrics attribute
        precision = float(results.box.mp) if hasattr(results.box, 'mp') else 0.0
        recall = float(results.box.mr) if hasattr(results.box, 'mr') else 0.0
        map50 = float(results.box.map50) if hasattr(results.box, 'map50') else 0.0
        map50_95 = float(results.box.map) if hasattr(results.box, 'map') else 0.0
        
        # Extract per-class AP
        per_class_ap = {}
        if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap'):
            class_names = self.model.names
            for idx, ap_value in zip(results.box.ap_class_index, results.box.ap):
                class_name = class_names.get(int(idx), f"class_{idx}")
                per_class_ap[class_name] = float(ap_value)
        
        # Compute Small_Object_AP
        small_object_ap = self._compute_small_object_ap()
        
        return DetectionMetrics(
            precision=precision,
            recall=recall,
            map50=map50,
            map50_95=map50_95,
            small_object_ap=small_object_ap,
            per_class_ap=per_class_ap
        )
    
    def _compute_small_object_ap(self) -> float:
        """Compute Average Precision for small objects (<32x32 pixels).
        
        Filters ground-truth boxes where area < 32*32 and computes AP on this subset
        using pycocotools.
        
        Returns:
            Small object AP score
            
        **Validates: Requirement 5.1** - Small_Object_AP computation
        """
        try:
            # Get test image paths and convert YOLO annotations to COCO format
            test_images = self._get_test_image_paths()
            if not test_images:
                print("Warning: No test images found for Small_Object_AP computation")
                return 0.0
            
            # Create COCO-format ground truth and predictions
            gt_coco_data = self._convert_yolo_to_coco_format(test_images, is_ground_truth=True)
            pred_coco_data = self._convert_yolo_to_coco_format(test_images, is_ground_truth=False)
            
            if not gt_coco_data or not pred_coco_data:
                print("Warning: Could not convert annotations to COCO format")
                return 0.0
            
            # Filter ground truth for small objects only (area < 32*32 = 1024 pixels)
            filtered_gt_annotations = []
            for ann in gt_coco_data.get('annotations', []):
                bbox = ann['bbox']  # COCO format: [x, y, width, height]
                area = bbox[2] * bbox[3]
                if area < 1024:  # 32x32 pixels
                    filtered_gt_annotations.append(ann)
            
            if not filtered_gt_annotations:
                print("Warning: No small objects (<32x32) found in ground truth")
                return 0.0
            
            # Create filtered COCO ground truth
            filtered_gt_coco = gt_coco_data.copy()
            filtered_gt_coco['annotations'] = filtered_gt_annotations
            
            # Get image IDs with small objects
            small_obj_image_ids = set(ann['image_id'] for ann in filtered_gt_annotations)
            
            # Filter predictions to only include images with small objects
            filtered_pred_annotations = [
                pred for pred in pred_coco_data.get('annotations', [])
                if pred['image_id'] in small_obj_image_ids
            ]
            
            # Write temporary COCO JSON files
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_file:
                json.dump(filtered_gt_coco, gt_file)
                gt_file_path = gt_file.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as pred_file:
                json.dump(filtered_pred_annotations, pred_file)
                pred_file_path = pred_file.name
            
            # Use pycocotools to compute AP
            coco_gt = COCO(gt_file_path)
            coco_dt = coco_gt.loadRes(pred_file_path)
            
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Get AP (first metric in stats is AP @ IoU=0.50:0.95)
            small_object_ap = float(coco_eval.stats[0])
            
            # Clean up temporary files
            Path(gt_file_path).unlink(missing_ok=True)
            Path(pred_file_path).unlink(missing_ok=True)
            
            return small_object_ap
            
        except Exception as e:
            # If computation fails, return 0.0 and log warning
            print(f"Warning: Could not compute Small_Object_AP: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def _convert_yolo_to_coco_format(self, image_paths: List[Path], is_ground_truth: bool = True) -> dict:
        """Convert YOLO annotations to COCO format.
        
        Args:
            image_paths: List of image file paths
            is_ground_truth: If True, load from label files; if False, run inference
            
        Returns:
            COCO-format dictionary with images, annotations, and categories
        """
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Get class names from model
        if self.model and hasattr(self.model, 'names'):
            class_names = self.model.names
        else:
            class_names = {0: "person"}  # Default for survivor detection
        
        # Create categories
        for class_id, class_name in class_names.items():
            coco_data["categories"].append({
                "id": int(class_id),
                "name": class_name,
                "supercategory": "object"
            })
        
        annotation_id = 1
        
        for image_id, img_path in enumerate(image_paths, start=1):
            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Add image entry
            coco_data["images"].append({
                "id": image_id,
                "file_name": img_path.name,
                "width": img_width,
                "height": img_height
            })
            
            if is_ground_truth:
                # Load YOLO format labels
                label_path = self._get_label_path(img_path)
                if not label_path.exists():
                    continue
                
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * img_width
                        y_center = float(parts[2]) * img_height
                        width = float(parts[3]) * img_width
                        height = float(parts[4]) * img_height
                        
                        # Convert to COCO format (x_min, y_min, width, height)
                        x_min = x_center - width / 2
                        y_min = y_center - height / 2
                        
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": [x_min, y_min, width, height],
                            "area": width * height,
                            "iscrowd": 0
                        })
                        annotation_id += 1
            else:
                # Run inference to get predictions
                results = self.model.predict(
                    str(img_path),
                    conf=self.config.conf_threshold,
                    iou=self.config.iou_threshold,
                    imgsz=self.config.img_size,
                    device=self.config.device,
                    verbose=False
                )
                
                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        for i in range(len(boxes)):
                            # Get box coordinates (xyxy format)
                            xyxy = boxes.xyxy[i].cpu().numpy()
                            conf = float(boxes.conf[i].cpu().numpy())
                            cls = int(boxes.cls[i].cpu().numpy())
                            
                            x_min, y_min, x_max, y_max = xyxy
                            width = x_max - x_min
                            height = y_max - y_min
                            
                            coco_data["annotations"].append({
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": cls,
                                "bbox": [float(x_min), float(y_min), float(width), float(height)],
                                "area": float(width * height),
                                "score": conf,
                                "iscrowd": 0
                            })
                            annotation_id += 1
        
        return coco_data
    
    def _get_label_path(self, image_path: Path) -> Path:
        """Get corresponding label file path for an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Path to label file (replaces 'images' with 'labels' and extension with '.txt')
        """
        # Replace 'images' directory with 'labels'
        label_path = Path(str(image_path).replace('/images/', '/labels/').replace('\\images\\', '\\labels\\'))
        # Change extension to .txt
        label_path = label_path.with_suffix('.txt')
        return label_path
    
    def compute_efficiency_metrics(self) -> EfficiencyMetrics:
        """Compute inference efficiency metrics.
        
        Measures FPS and latency over test dataset using time.perf_counter().
        
        Returns:
            EfficiencyMetrics with FPS and latency statistics
            
        **Validates: Requirement 5.2** - Measure inference FPS and latency
        """
        if self.model is None:
            raise ValueError("Model must be loaded before computing metrics")
        
        self.latency_measurements = []
        
        # Get test images
        test_images = self._get_test_image_paths()
        
        # Handle empty test images
        if not test_images:
            print("Warning: No test images found for efficiency metrics computation")
            return EfficiencyMetrics(
                fps_mean=0.0,
                fps_median=0.0,
                latency_ms_mean=0.0,
                latency_ms_median=0.0,
                latency_ms_p95=0.0
            )
            
        # Warm up model (run a few iterations to stabilize timing)
        warmup_image = test_images[0]
        for _ in range(min(5, len(test_images))):
            _ = self.model.predict(
                warmup_image,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.img_size,
                device=self.config.device,
                verbose=False
            )
        
        # Measure latency for each image (sample at most 100 images if test set is very large)
        sample_images = test_images[:100] if len(test_images) > 100 else test_images
        for img_path in sample_images:
            start_time = time.perf_counter()
            
            _ = self.model.predict(
                img_path,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.img_size,
                device=self.config.device,
                verbose=False
            )
            
            end_time = time.perf_counter()
            latency_seconds = end_time - start_time
            self.latency_measurements.append(latency_seconds)
        
        # Convert to milliseconds
        latency_ms = np.array(self.latency_measurements) * 1000
        
        # Compute FPS (inverse of latency)
        fps_values = 1.0 / np.array(self.latency_measurements)
        
        return EfficiencyMetrics(
            fps_mean=float(np.mean(fps_values)),
            fps_median=float(np.median(fps_values)),
            latency_ms_mean=float(np.mean(latency_ms)),
            latency_ms_median=float(np.median(latency_ms)),
            latency_ms_p95=float(np.percentile(latency_ms, 95))
        )
    
    def compute_complexity_metrics(self) -> ModelComplexity:
        """Compute model complexity metrics.
        
        Computes parameter count and GFLOPs for the loaded model.
        
        Returns:
            ModelComplexity with parameter count and GFLOPs
            
        **Validates: Requirement 5.3** - Report model parameter count and GFLOPs
        """
        if self.model is None:
            raise ValueError("Model must be loaded before computing metrics")
        
        # Get model
        pytorch_model = self.model.model
        
        # Count parameters
        total_params = sum(p.numel() for p in pytorch_model.parameters())
        
        # Compute GFLOPs using thop library
        try:
            from thop import profile
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, self.config.img_size, self.config.img_size)
            if self.config.device == "cuda" and torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
                pytorch_model = pytorch_model.cuda()
            
            # Profile model
            flops, _ = profile(pytorch_model, inputs=(dummy_input,), verbose=False)
            gflops = flops / 1e9  # Convert to GFLOPs
            
        except Exception as e:
            print(f"Warning: Could not compute GFLOPs using thop: {e}")
            gflops = 0.0
        
        return ModelComplexity(
            parameters=total_params,
            gflops=gflops
        )
    
    def _get_test_image_paths(self) -> List[Path]:
        """Get list of test image paths from dataset.
        
        Returns:
            List of image file paths
        """
        # Parse the dataset YAML or directory to get test images
        test_path = Path(self.config.test_dataset_path)
        
        if test_path.is_file():
            # It's a YAML file, parse it
            import yaml
            with open(test_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Get test images directory
            if 'test' in data:
                test_dir = Path(data['test'])
            elif 'val' in data:
                test_dir = Path(data['val'])  # Fall back to validation
            else:
                raise ValueError(f"No test or val path in dataset YAML: {test_path}")
            
            # Handle relative paths
            if 'path' in data and not test_dir.is_absolute():
                base_path = Path(data['path'])
                test_dir = base_path / test_dir
            elif not test_dir.is_absolute():
                test_dir = test_path.parent / test_dir
            
        elif test_path.is_dir():
            # It's a directory
            test_dir = test_path
        else:
            raise ValueError(f"Invalid test dataset path: {test_path}")
        
        # Find images directory
        if (test_dir / "images").exists():
            images_dir = test_dir / "images"
        else:
            images_dir = test_dir
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(images_dir.glob(f"*{ext}"))
            image_paths.extend(images_dir.glob(f"*{ext.upper()}"))
        
        return sorted(image_paths)
    
    def _count_test_images(self) -> int:
        """Count total number of test images.
        
        Returns:
            Number of test images
        """
        return len(self._get_test_image_paths())
    
    def _count_total_detections(self) -> int:
        """Count total detections made by the model across all test images.

        Runs the model in predict mode over all test images and sums the
        detection count from each frame.  This gives an accurate total rather
        than the placeholder zero that was here before.

        Returns:
            Total detection count across the test set
        """
        if self.model is None:
            return 0

        image_paths = self._get_test_image_paths()
        if not image_paths:
            return 0

        total = 0
        for img_path in image_paths:
            try:
                preds = self.model.predict(
                    source=str(img_path),
                    conf=self.config.conf_threshold,
                    iou=self.config.iou_threshold,
                    imgsz=self.config.img_size,
                    device=self.config.device,
                    verbose=False,
                )
                for result in preds:
                    total += len(result.boxes) if result.boxes is not None else 0
            except Exception:
                continue

        return total
    
    def to_dict(self, result: EvaluationResult) -> dict:
        """Convert EvaluationResult to dictionary for JSON serialization.
        
        Args:
            result: EvaluationResult object
            
        Returns:
            Dictionary representation
        """
        return {
            "detection_metrics": {
                "precision": result.detection_metrics.precision,
                "recall": result.detection_metrics.recall,
                "map50": result.detection_metrics.map50,
                "map50_95": result.detection_metrics.map50_95,
                "small_object_ap": result.detection_metrics.small_object_ap,
                "per_class_ap": result.detection_metrics.per_class_ap
            },
            "efficiency_metrics": {
                "fps_mean": result.efficiency_metrics.fps_mean,
                "fps_median": result.efficiency_metrics.fps_median,
                "latency_ms_mean": result.efficiency_metrics.latency_ms_mean,
                "latency_ms_median": result.efficiency_metrics.latency_ms_median,
                "latency_ms_p95": result.efficiency_metrics.latency_ms_p95
            },
            "complexity_metrics": {
                "parameters": result.complexity_metrics.parameters,
                "gflops": result.complexity_metrics.gflops
            },
            "total_images": result.total_images,
            "total_detections": result.total_detections,
            "evaluation_duration_seconds": result.evaluation_duration_seconds,
            "timestamp": result.timestamp
        }
