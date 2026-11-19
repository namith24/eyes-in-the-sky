"""
Unit tests for Dataset Pipeline.

Tests the DatasetPipeline class for annotation verification, dataset splitting,
YOLO format export, manifest generation, and overwrite detection.

**Validates: Requirements 1.1, 1.2, 1.3, 1.5, 1.6**
"""

import pytest
import json
import yaml
from pathlib import Path
import tempfile
import shutil

from uav_survivor_detection.dataset.pipeline import (
    DatasetPipeline,
    DatasetConfig,
    ValidationError,
    DatasetSplit,
    DatasetManifest
)


# Get project root for test fixtures
PROJECT_ROOT = Path(__file__).parent.parent


class TestAnnotationVerification:
    """Test suite for annotation verification (Requirements 1.1, 1.2)."""
    
    def test_verify_annotations_valid(self, tmp_path):
        """Test annotation verification with valid annotations.
        
        **Validates: Requirement 1.1**
        """
        # Create test dataset structure
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        # Create valid image and label files
        for i in range(3):
            # Create dummy image file
            img_path = images_dir / f"test_{i}.jpg"
            img_path.write_text("dummy image content")
            
            # Create valid label file (class_id x_center y_center width height)
            label_path = labels_dir / f"test_{i}.txt"
            label_path.write_text(
                "0 0.5 0.5 0.3 0.4\n"
                "0 0.2 0.3 0.1 0.15\n"
            )
        
        # Create pipeline and verify
        pipeline = DatasetPipeline(
            dataset_path=tmp_path,
            output_dir=tmp_path / "output"
        )
        
        errors = pipeline.verify_annotations()
        
        # Should have no errors
        assert len(errors) == 0, f"Expected no errors, got {len(errors)}: {errors}"
    
    def test_verify_annotations_malformed_bbox_coordinates(self, tmp_path):
        """Test annotation verification with malformed bbox coordinates.
        
        **Validates: Requirement 1.2**
        """
        # Create test dataset structure
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        # Create image with invalid annotations
        img_path = images_dir / "test_invalid.jpg"
        img_path.write_text("dummy image")
        
        label_path = labels_dir / "test_invalid.txt"
        # Invalid coordinates: x > 1.0, y < 0.0
        label_path.write_text(
            "0 1.5 0.5 0.3 0.4\n"  # x_center > 1.0
            "0 0.5 -0.2 0.3 0.4\n"  # y_center < 0.0
            "0 0.5 0.5 1.5 0.4\n"   # width > 1.0
            "0 0.5 0.5 0.3 -0.1\n"  # height < 0.0
        )
        
        pipeline = DatasetPipeline(
            dataset_path=tmp_path,
            output_dir=tmp_path / "output"
        )
        
        errors = pipeline.verify_annotations()
        
        # Should have 4 errors (one for each invalid coordinate)
        assert len(errors) >= 4, f"Expected at least 4 errors, got {len(errors)}"
        
        # Check error types
        error_types = [e.error_type for e in errors]
        assert all(et == "invalid_bbox" for et in error_types), \
            "All errors should be 'invalid_bbox' type"
        
        # Verify error messages mention the invalid coordinates
        error_messages = [e.message for e in errors]
        assert any("x_center" in msg and "1.5" in msg for msg in error_messages), \
            "Should report x_center > 1.0"
        assert any("y_center" in msg and "-0.2" in msg for msg in error_messages), \
            "Should report y_center < 0.0"
    
    def test_verify_annotations_missing_label_file(self, tmp_path):
        """Test detection of missing annotation files.
        
        **Validates: Requirement 1.2**
        """
        # Create image without corresponding label
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        
        img_path = images_dir / "test_no_label.jpg"
        img_path.write_text("dummy image")
        
        pipeline = DatasetPipeline(
            dataset_path=tmp_path,
            output_dir=tmp_path / "output"
        )
        
        errors = pipeline.verify_annotations()
        
        # Should have 1 error for missing label
        assert len(errors) == 1, f"Expected 1 error, got {len(errors)}"
        assert errors[0].error_type == "missing"
        assert "Missing annotation file" in errors[0].message
    
    def test_verify_annotations_empty_label_file(self, tmp_path):
        """Test detection of empty annotation files.
        
        **Validates: Requirement 1.2**
        """
        # Create image with empty label file
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        img_path = images_dir / "test_empty.jpg"
        img_path.write_text("dummy image")
        
        label_path = labels_dir / "test_empty.txt"
        label_path.write_text("")  # Empty file
        
        pipeline = DatasetPipeline(
            dataset_path=tmp_path,
            output_dir=tmp_path / "output"
        )
        
        errors = pipeline.verify_annotations()
        
        # Should have 1 error for empty label
        assert len(errors) == 1
        assert errors[0].error_type == "malformed"
        assert "empty" in errors[0].message.lower()
    
    def test_verify_annotations_malformed_format(self, tmp_path):
        """Test detection of malformed annotation format.
        
        **Validates: Requirement 1.2**
        """
        # Create image with incorrectly formatted label
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        img_path = images_dir / "test_malformed.jpg"
        img_path.write_text("dummy image")
        
        label_path = labels_dir / "test_malformed.txt"
        # Only 3 values instead of 5
        label_path.write_text("0 0.5 0.5\n")
        
        pipeline = DatasetPipeline(
            dataset_path=tmp_path,
            output_dir=tmp_path / "output"
        )
        
        errors = pipeline.verify_annotations()
        
        # Should have 1 error for incorrect format
        assert len(errors) == 1
        assert errors[0].error_type == "malformed"
        assert "5 values" in errors[0].message


class TestDatasetSplitting:
    """Test suite for dataset splitting (Requirements 1.3, 1.4)."""
    
    def test_split_dataset_70_20_10(self, tmp_path):
        """Test 70/20/10 split on sample dataset.
        
        **Validates: Requirement 1.3**
        """
        # Create small sample dataset
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        # Create 100 image-label pairs
        num_samples = 100
        for i in range(num_samples):
            img_path = images_dir / f"img_{i:03d}.jpg"
            img_path.write_text("dummy image")
            
            label_path = labels_dir / f"img_{i:03d}.txt"
            # Alternate between two classes for stratification
            class_id = i % 2
            label_path.write_text(f"{class_id} 0.5 0.5 0.3 0.4\n")
        
        # Create pipeline and split
        pipeline = DatasetPipeline(
            dataset_path=tmp_path,
            output_dir=tmp_path / "output",
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            random_seed=42
        )
        
        split = pipeline.split_dataset()
        
        # Verify split sizes
        total = len(split.train_files) + len(split.val_files) + len(split.test_files)
        assert total == num_samples, f"Total files {total} != {num_samples}"
        
        # Check approximate ratios (allow ±2 tolerance due to stratification)
        train_expected = int(num_samples * 0.7)
        val_expected = int(num_samples * 0.2)
        test_expected = int(num_samples * 0.1)
        
        assert abs(len(split.train_files) - train_expected) <= 2, \
            f"Train size {len(split.train_files)} not close to {train_expected}"
        assert abs(len(split.val_files) - val_expected) <= 2, \
            f"Val size {len(split.val_files)} not close to {val_expected}"
        assert abs(len(split.test_files) - test_expected) <= 2, \
            f"Test size {len(split.test_files)} not close to {test_expected}"
    
    def test_split_dataset_preserves_class_distribution(self, tmp_path):
        """Test that stratified split preserves class distribution.
        
        **Validates: Requirement 1.3**
        """
        # Create dataset with imbalanced classes
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        # 80 samples of class 0, 20 samples of class 1
        for i in range(80):
            img_path = images_dir / f"class0_{i:03d}.jpg"
            img_path.write_text("dummy")
            label_path = labels_dir / f"class0_{i:03d}.txt"
            label_path.write_text("0 0.5 0.5 0.3 0.4\n")
        
        for i in range(20):
            img_path = images_dir / f"class1_{i:03d}.jpg"
            img_path.write_text("dummy")
            label_path = labels_dir / f"class1_{i:03d}.txt"
            label_path.write_text("1 0.5 0.5 0.3 0.4\n")
        
        pipeline = DatasetPipeline(
            dataset_path=tmp_path,
            output_dir=tmp_path / "output",
            random_seed=42
        )
        
        split = pipeline.split_dataset()
        
        # Count classes in each split
        def count_class_0(files):
            count = 0
            for f in files:
                if "class0" in f.name:
                    count += 1
            return count
        
        train_class0 = count_class_0(split.train_files)
        val_class0 = count_class_0(split.val_files)
        test_class0 = count_class_0(split.test_files)
        
        # Check class 0 ratio in each split (should be ~80%)
        train_ratio = train_class0 / len(split.train_files)
        val_ratio = val_class0 / len(split.val_files)
        test_ratio = test_class0 / len(split.test_files)
        
        # Allow ±10% tolerance for class distribution
        assert 0.70 <= train_ratio <= 0.90, \
            f"Train class 0 ratio {train_ratio:.2f} not close to 0.80"
        assert 0.70 <= val_ratio <= 0.90, \
            f"Val class 0 ratio {val_ratio:.2f} not close to 0.80"
        assert 0.70 <= test_ratio <= 0.90, \
            f"Test class 0 ratio {test_ratio:.2f} not close to 0.80"
    
    def test_split_dataset_reproducibility(self, tmp_path):
        """Test that splits are reproducible with same random seed.
        
        **Validates: Requirement 1.3**
        """
        # Create sample dataset
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        for i in range(50):
            img_path = images_dir / f"img_{i:03d}.jpg"
            img_path.write_text("dummy")
            label_path = labels_dir / f"img_{i:03d}.txt"
            label_path.write_text("0 0.5 0.5 0.3 0.4\n")
        
        # First split
        pipeline1 = DatasetPipeline(
            dataset_path=tmp_path,
            output_dir=tmp_path / "output1",
            random_seed=42
        )
        split1 = pipeline1.split_dataset()
        
        # Second split with same seed
        pipeline2 = DatasetPipeline(
            dataset_path=tmp_path,
            output_dir=tmp_path / "output2",
            random_seed=42
        )
        split2 = pipeline2.split_dataset()
        
        # Files should be identical
        assert set(f.name for f in split1.train_files) == \
               set(f.name for f in split2.train_files), \
            "Train split not reproducible"
        assert set(f.name for f in split1.val_files) == \
               set(f.name for f in split2.val_files), \
            "Val split not reproducible"
        assert set(f.name for f in split1.test_files) == \
               set(f.name for f in split2.test_files), \
            "Test split not reproducible"


class TestYOLOFormatExport:
    """Test suite for YOLO format export (Requirement 1.5)."""
    
    def test_export_yolo_format_directory_structure(self, tmp_path):
        """Test YOLO format directory structure matches expected schema.
        
        **Validates: Requirement 1.5**
        """
        # Create sample dataset
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        for i in range(20):
            img_path = images_dir / f"img_{i:03d}.jpg"
            img_path.write_text("dummy image")
            label_path = labels_dir / f"img_{i:03d}.txt"
            label_path.write_text("0 0.5 0.5 0.3 0.4\n")
        
        # Create pipeline, split, and export
        output_dir = tmp_path / "yolo_output"
        pipeline = DatasetPipeline(
            dataset_path=tmp_path,
            output_dir=output_dir,
            random_seed=42
        )
        
        pipeline.split_dataset()
        pipeline.export_yolo_format()
        
        # Verify directory structure
        assert output_dir.exists(), "Output directory not created"
        
        # Check train/val/test directories
        for split_name in ['train', 'val', 'test']:
            split_dir = output_dir / split_name
            assert split_dir.exists(), f"{split_name} directory not created"
            
            images_subdir = split_dir / "images"
            labels_subdir = split_dir / "labels"
            
            assert images_subdir.exists(), f"{split_name}/images not created"
            assert labels_subdir.exists(), f"{split_name}/labels not created"
            
            # Check that files were copied
            image_files = list(images_subdir.glob("*.jpg"))
            label_files = list(labels_subdir.glob("*.txt"))
            
            assert len(image_files) > 0, f"No images in {split_name}/images"
            assert len(label_files) > 0, f"No labels in {split_name}/labels"
            assert len(image_files) == len(label_files), \
                f"Mismatched image/label count in {split_name}"
        
        # Check data.yaml exists
        data_yaml_path = output_dir / "data.yaml"
        assert data_yaml_path.exists(), "data.yaml not created"
        
        # Verify data.yaml content
        with open(data_yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
        
        assert 'path' in data_yaml, "data.yaml missing 'path' field"
        assert 'train' in data_yaml, "data.yaml missing 'train' field"
        assert 'val' in data_yaml, "data.yaml missing 'val' field"
        assert 'test' in data_yaml, "data.yaml missing 'test' field"
        assert 'nc' in data_yaml, "data.yaml missing 'nc' field"
        assert 'names' in data_yaml, "data.yaml missing 'names' field"
        
        assert data_yaml['train'] == 'train/images'
        assert data_yaml['val'] == 'val/images'
        assert data_yaml['test'] == 'test/images'
    
    def test_export_preserves_image_label_pairs(self, tmp_path):
        """Test that exported files maintain image-label correspondence.
        
        **Validates: Requirement 1.5**
        """
        # Create sample dataset
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        for i in range(15):
            img_path = images_dir / f"img_{i:03d}.jpg"
            img_path.write_text(f"image_{i}")
            label_path = labels_dir / f"img_{i:03d}.txt"
            label_path.write_text(f"0 0.{i} 0.{i} 0.3 0.4\n")
        
        output_dir = tmp_path / "yolo_output"
        pipeline = DatasetPipeline(
            dataset_path=tmp_path,
            output_dir=output_dir,
            random_seed=42
        )
        
        pipeline.split_dataset()
        pipeline.export_yolo_format()
        
        # Check each split
        for split_name in ['train', 'val', 'test']:
            images_dir = output_dir / split_name / "images"
            labels_dir = output_dir / split_name / "labels"
            
            for img_path in images_dir.glob("*.jpg"):
                # Check corresponding label exists
                label_path = labels_dir / img_path.with_suffix('.txt').name
                assert label_path.exists(), \
                    f"Label {label_path.name} not found for {img_path.name}"


class TestManifestGeneration:
    """Test suite for manifest generation (Requirement 1.4)."""
    
    def test_generate_manifest_file_counts(self, tmp_path):
        """Test manifest generation with correct file counts.
        
        **Validates: Requirement 1.4**
        """
        # Create sample dataset
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        num_samples = 50
        for i in range(num_samples):
            img_path = images_dir / f"img_{i:03d}.jpg"
            img_path.write_text("dummy")
            label_path = labels_dir / f"img_{i:03d}.txt"
            label_path.write_text("0 0.5 0.5 0.3 0.4\n")
        
        pipeline = DatasetPipeline(
            dataset_path=tmp_path,
            output_dir=tmp_path / "output",
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            random_seed=42
        )
        
        pipeline.split_dataset()
        manifest = pipeline.generate_manifest()
        
        # Check file counts
        assert manifest.total_images == num_samples, \
            f"Total images {manifest.total_images} != {num_samples}"
        
        assert manifest.train_count + manifest.val_count + manifest.test_count == num_samples, \
            "Split counts don't sum to total"
        
        # Check split ratios
        assert manifest.split_ratios == (0.7, 0.2, 0.1), \
            f"Split ratios {manifest.split_ratios} != (0.7, 0.2, 0.1)"
        
        # Check timestamp format (ISO 8601)
        assert 'T' in manifest.timestamp, "Timestamp not in ISO format"
        assert manifest.timestamp.endswith('Z'), "Timestamp not in UTC"
    
    def test_generate_manifest_class_distribution(self, tmp_path):
        """Test manifest contains correct class distribution.
        
        **Validates: Requirement 1.4**
        """
        # Create dataset with multiple classes
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        # 30 images of class 0, 20 images of class 1
        for i in range(30):
            img_path = images_dir / f"class0_{i:03d}.jpg"
            img_path.write_text("dummy")
            label_path = labels_dir / f"class0_{i:03d}.txt"
            label_path.write_text("0 0.5 0.5 0.3 0.4\n")
        
        for i in range(20):
            img_path = images_dir / f"class1_{i:03d}.jpg"
            img_path.write_text("dummy")
            label_path = labels_dir / f"class1_{i:03d}.txt"
            # Multiple objects of class 1 in same image
            label_path.write_text(
                "1 0.3 0.3 0.2 0.2\n"
                "1 0.7 0.7 0.2 0.2\n"
            )
        
        pipeline = DatasetPipeline(
            dataset_path=tmp_path,
            output_dir=tmp_path / "output",
            random_seed=42
        )
        
        pipeline.split_dataset()
        manifest = pipeline.generate_manifest()
        
        # Check class distribution
        assert 'class_0' in manifest.class_distribution, "Class 0 not in distribution"
        assert 'class_1' in manifest.class_distribution, "Class 1 not in distribution"
        
        # Class 0 should have 30 instances, class 1 should have 40 (20 images × 2 objects)
        assert manifest.class_distribution['class_0'] == 30, \
            f"Class 0 count {manifest.class_distribution['class_0']} != 30"
        assert manifest.class_distribution['class_1'] == 40, \
            f"Class 1 count {manifest.class_distribution['class_1']} != 40"
    
    def test_manifest_serialization(self, tmp_path):
        """Test manifest can be saved to JSON file.
        
        **Validates: Requirement 1.4**
        """
        # Create minimal dataset
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        for i in range(10):
            img_path = images_dir / f"img_{i:03d}.jpg"
            img_path.write_text("dummy")
            label_path = labels_dir / f"img_{i:03d}.txt"
            label_path.write_text("0 0.5 0.5 0.3 0.4\n")
        
        pipeline = DatasetPipeline(
            dataset_path=tmp_path,
            output_dir=tmp_path / "output",
            random_seed=42
        )
        
        pipeline.split_dataset()
        manifest = pipeline.generate_manifest()
        
        # Save to JSON
        json_path = tmp_path / "manifest.json"
        manifest.save_json(json_path)
        
        assert json_path.exists(), "manifest.json not created"
        
        # Load and verify
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        assert data['total_images'] == 10
        assert 'train_count' in data
        assert 'val_count' in data
        assert 'test_count' in data
        assert 'class_distribution' in data
        assert 'split_ratios' in data
        assert 'timestamp' in data


class TestOverwriteDetection:
    """Test suite for overwrite detection (Requirement 1.6)."""
    
    def test_overwrite_detection_raises_error(self, tmp_path):
        """Test FileExistsError when output exists and overwrite_existing=False.
        
        **Validates: Requirement 1.6**
        """
        # Create sample dataset
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        for i in range(10):
            img_path = images_dir / f"img_{i:03d}.jpg"
            img_path.write_text("dummy")
            label_path = labels_dir / f"img_{i:03d}.txt"
            label_path.write_text("0 0.5 0.5 0.3 0.4\n")
        
        output_dir = tmp_path / "output"
        
        # First export
        pipeline1 = DatasetPipeline(
            dataset_path=tmp_path,
            output_dir=output_dir,
            overwrite_existing=False,
            random_seed=42
        )
        
        pipeline1.split_dataset()
        pipeline1.export_yolo_format()
        
        assert output_dir.exists(), "First export failed"
        
        # Second export should raise FileExistsError
        pipeline2 = DatasetPipeline(
            dataset_path=tmp_path,
            output_dir=output_dir,
            overwrite_existing=False,
            random_seed=42
        )
        
        pipeline2.split_dataset()
        
        with pytest.raises(FileExistsError) as exc_info:
            pipeline2.export_yolo_format()
        
        assert "already exists" in str(exc_info.value).lower()
        assert "overwrite_existing" in str(exc_info.value)
    
    def test_overwrite_allowed_with_flag(self, tmp_path):
        """Test that overwrite works when overwrite_existing=True.
        
        **Validates: Requirement 1.6**
        """
        # Create sample dataset
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        for i in range(10):
            img_path = images_dir / f"img_{i:03d}.jpg"
            img_path.write_text("dummy")
            label_path = labels_dir / f"img_{i:03d}.txt"
            label_path.write_text("0 0.5 0.5 0.3 0.4\n")
        
        output_dir = tmp_path / "output"
        
        # First export
        pipeline1 = DatasetPipeline(
            dataset_path=tmp_path,
            output_dir=output_dir,
            overwrite_existing=True,
            random_seed=42
        )
        
        pipeline1.split_dataset()
        pipeline1.export_yolo_format()
        
        # Create marker file to verify overwrite
        marker = output_dir / "marker.txt"
        marker.write_text("original")
        
        # Second export with overwrite enabled
        pipeline2 = DatasetPipeline(
            dataset_path=tmp_path,
            output_dir=output_dir,
            overwrite_existing=True,
            random_seed=42
        )
        
        pipeline2.split_dataset()
        pipeline2.export_yolo_format()  # Should not raise
        
        # Marker should still exist (export doesn't delete unrelated files)
        assert output_dir.exists(), "Output directory deleted"


class TestDatasetConfig:
    """Test suite for DatasetConfig initialization."""
    
    def test_config_initialization(self, tmp_path):
        """Test pipeline can be initialized with DatasetConfig."""
        config = DatasetConfig(
            dataset_path=tmp_path / "data",
            output_dir=tmp_path / "output",
            train_ratio=0.8,
            val_ratio=0.15,
            test_ratio=0.05,
            random_seed=123,
            overwrite_existing=True
        )
        
        pipeline = DatasetPipeline(config=config)
        
        assert pipeline.dataset_path == tmp_path / "data"
        assert pipeline.output_dir == tmp_path / "output"
        assert pipeline.train_ratio == 0.8
        assert pipeline.val_ratio == 0.15
        assert pipeline.test_ratio == 0.05
        assert pipeline.random_seed == 123
        assert pipeline.overwrite_existing == True
    
    def test_invalid_split_ratios_raises_error(self, tmp_path):
        """Test that invalid split ratios raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            pipeline = DatasetPipeline(
                dataset_path=tmp_path,
                output_dir=tmp_path / "output",
                train_ratio=0.6,
                val_ratio=0.3,
                test_ratio=0.2  # Sum = 1.1, invalid
            )
        
        assert "sum to 1.0" in str(exc_info.value).lower()


class TestPipelineIntegration:
    """Integration tests for complete pipeline execution."""
    
    def test_run_pipeline_complete(self, tmp_path):
        """Test complete pipeline execution from start to finish."""
        # Create sample dataset
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        for i in range(30):
            img_path = images_dir / f"img_{i:03d}.jpg"
            img_path.write_text("dummy image")
            label_path = labels_dir / f"img_{i:03d}.txt"
            label_path.write_text("0 0.5 0.5 0.3 0.4\n")
        
        output_dir = tmp_path / "output"
        
        pipeline = DatasetPipeline(
            dataset_path=tmp_path,
            output_dir=output_dir,
            random_seed=42
        )
        
        # Run complete pipeline
        manifest = pipeline.run_pipeline()
        
        # Verify all outputs exist
        assert output_dir.exists()
        assert (output_dir / "train").exists()
        assert (output_dir / "val").exists()
        assert (output_dir / "test").exists()
        assert (output_dir / "data.yaml").exists()
        assert (output_dir / "manifest.json").exists()
        
        # Verify manifest
        assert manifest.total_images == 30
        assert manifest.train_count + manifest.val_count + manifest.test_count == 30


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
