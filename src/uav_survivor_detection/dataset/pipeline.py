"""Dataset pipeline for preparing C2A dataset with reproducible splits and YOLO format export.

This module implements the Dataset Preparation Pipeline (Requirement 1) for the UAV Survivor
Detection System. It verifies annotations, performs stratified train/val/test splitting, exports
to YOLO format, and generates a dataset manifest.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6**
"""

import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml

from sklearn.model_selection import train_test_split


@dataclass
class DatasetConfig:
    """Configuration for dataset pipeline.
    
    Attributes:
        dataset_path: Path to raw C2A dataset
        output_dir: Directory for processed YOLO-format output
        train_ratio: Fraction of data for training (default 0.7)
        val_ratio: Fraction of data for validation (default 0.2)
        test_ratio: Fraction of data for testing (default 0.1)
        random_seed: Random seed for reproducible splits (default 42)
        overwrite_existing: Whether to overwrite existing output directory (default False)
    """
    dataset_path: Path
    output_dir: Path
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    random_seed: int = 42
    overwrite_existing: bool = False


@dataclass
class ValidationError:
    """Represents a validation error found in the dataset.
    
    Attributes:
        file_path: Path to the file with validation error
        error_type: Type of error (e.g., 'missing', 'malformed', 'invalid_bbox')
        message: Detailed error message
    """
    file_path: Path
    error_type: str
    message: str
    
    def __str__(self) -> str:
        return f"{self.error_type}: {self.file_path} - {self.message}"


@dataclass
class DatasetSplit:
    """Represents a dataset split into train/val/test subsets.
    
    Attributes:
        train_files: List of image file paths for training subset
        val_files: List of image file paths for validation subset
        test_files: List of image file paths for test subset
    """
    train_files: List[Path]
    val_files: List[Path]
    test_files: List[Path]


@dataclass
class DatasetManifest:
    """Dataset manifest with metadata and class distribution.
    
    Attributes:
        total_images: Total number of images in dataset
        train_count: Number of images in training subset
        val_count: Number of images in validation subset
        test_count: Number of images in test subset
        class_distribution: Mapping of class name to total count across all subsets
        split_ratios: Tuple of (train, val, test) ratios
        timestamp: ISO 8601 timestamp of manifest generation
    """
    total_images: int
    train_count: int
    val_count: int
    test_count: int
    class_distribution: Dict[str, int]
    split_ratios: Tuple[float, float, float]
    timestamp: str
    
    def to_dict(self) -> dict:
        """Convert manifest to dictionary for JSON serialization."""
        return asdict(self)
    
    def save_json(self, output_path: Path) -> None:
        """Save manifest to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class DatasetPipeline:
    """Dataset preparation pipeline for C2A dataset.
    
    This class handles:
    - Annotation verification (Requirement 1.1, 1.2)
    - Stratified dataset splitting (Requirement 1.3, 1.4)
    - YOLO format export (Requirement 1.5)
    - Manifest generation (Requirement 1.4)
    - Overwrite protection (Requirement 1.6)
    
    Attributes:
        dataset_path: Path to raw C2A dataset
        output_dir: Directory for processed YOLO-format output
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        random_seed: Random seed for reproducible splits
        overwrite_existing: Whether to overwrite existing output directory
    """
    
    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        overwrite_existing: bool = False,
        config: Optional['DatasetConfig'] = None
    ):
        """Initialize dataset pipeline.
        
        Can be initialized with either individual parameters or a DatasetConfig object.
        If config is provided, it takes precedence over individual parameters.
        
        Args:
            dataset_path: Path to raw C2A dataset
            output_dir: Directory for processed output
            train_ratio: Fraction for training (default 0.7)
            val_ratio: Fraction for validation (default 0.2)
            test_ratio: Fraction for testing (default 0.1)
            random_seed: Random seed for reproducibility (default 42)
            overwrite_existing: Overwrite existing output (default False)
            config: DatasetConfig object (if provided, overrides other parameters)
            
        Raises:
            ValueError: If split ratios don't sum to 1.0 or if neither dataset_path nor config is provided
        """
        # Use config if provided, otherwise use individual parameters
        if config is not None:
            self.dataset_path = Path(config.dataset_path)
            self.output_dir = Path(config.output_dir)
            self.train_ratio = config.train_ratio
            self.val_ratio = config.val_ratio
            self.test_ratio = config.test_ratio
            self.random_seed = config.random_seed
            self.overwrite_existing = config.overwrite_existing
        else:
            if dataset_path is None or output_dir is None:
                raise ValueError("Must provide either config or both dataset_path and output_dir")
            self.dataset_path = Path(dataset_path)
            self.output_dir = Path(output_dir)
            self.train_ratio = train_ratio
            self.val_ratio = val_ratio
            self.test_ratio = test_ratio
            self.random_seed = random_seed
            self.overwrite_existing = overwrite_existing
        
        # Validate split ratios
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= ratio_sum <= 1.01):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {ratio_sum:.3f} "
                f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )
        
        # Internal state
        self._dataset_split: Optional[DatasetSplit] = None
        self._class_distribution: Dict[str, int] = {}
    
    def verify_annotations(self, dataset_path: Optional[Path] = None) -> List[ValidationError]:
        """Verify all annotation files are present and well-formed.
        
        **Validates: Requirements 1.1, 1.2**
        
        Checks:
        - All image files have corresponding label files
        - Label files are non-empty
        - Bounding box coordinates are in valid [0, 1] range
        - Each line has correct format: class_id x_center y_center width height
        
        Args:
            dataset_path: Path to dataset (uses self.dataset_path if None)
            
        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []
        path = Path(dataset_path) if dataset_path else self.dataset_path
        
        if not path.exists():
            errors.append(ValidationError(
                file_path=path,
                error_type="missing",
                message="Dataset path does not exist"
            ))
            return errors
        
        # Find all image files (common formats)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(path.rglob(f'*{ext}'))
            image_files.extend(path.rglob(f'*{ext.upper()}'))
        
        # Deduplicate files (case-insensitive filesystems may return duplicates)
        image_files = list(dict.fromkeys(image_files))
        
        if not image_files:
            errors.append(ValidationError(
                file_path=path,
                error_type="missing",
                message="No image files found in dataset"
            ))
            return errors
        
        # Verify each image has a corresponding label file
        for img_path in image_files:
            # YOLO format: label files are in same directory or parallel labels/ directory
            # Try same directory first
            label_path = img_path.with_suffix('.txt')
            
            # Try labels/ directory
            if not label_path.exists():
                # Check for labels/ subdirectory at same level as images/
                parent = img_path.parent
                if parent.name == 'images':
                    labels_dir = parent.parent / 'labels'
                    label_path = labels_dir / img_path.name
                    label_path = label_path.with_suffix('.txt')
            
            # Check if label file exists
            if not label_path.exists():
                errors.append(ValidationError(
                    file_path=label_path,
                    error_type="missing",
                    message=f"Missing annotation file for image {img_path.name}"
                ))
                continue
            
            # Check if label file is empty
            if label_path.stat().st_size == 0:
                errors.append(ValidationError(
                    file_path=label_path,
                    error_type="malformed",
                    message="Annotation file is empty"
                ))
                continue
            
            # Parse and validate label file
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        errors.append(ValidationError(
                            file_path=label_path,
                            error_type="malformed",
                            message=f"Line {line_num}: Expected 5 values (class_id x y w h), got {len(parts)}"
                        ))
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Validate bounding box coordinates are in [0, 1] range
                        if not (0.0 <= x_center <= 1.0):
                            errors.append(ValidationError(
                                file_path=label_path,
                                error_type="invalid_bbox",
                                message=f"Line {line_num}: x_center {x_center} not in [0, 1]"
                            ))
                        
                        if not (0.0 <= y_center <= 1.0):
                            errors.append(ValidationError(
                                file_path=label_path,
                                error_type="invalid_bbox",
                                message=f"Line {line_num}: y_center {y_center} not in [0, 1]"
                            ))
                        
                        if not (0.0 < width <= 1.0):
                            errors.append(ValidationError(
                                file_path=label_path,
                                error_type="invalid_bbox",
                                message=f"Line {line_num}: width {width} not in (0, 1]"
                            ))
                        
                        if not (0.0 < height <= 1.0):
                            errors.append(ValidationError(
                                file_path=label_path,
                                error_type="invalid_bbox",
                                message=f"Line {line_num}: height {height} not in (0, 1]"
                            ))
                    
                    except ValueError as e:
                        errors.append(ValidationError(
                            file_path=label_path,
                            error_type="malformed",
                            message=f"Line {line_num}: Invalid numeric values - {str(e)}"
                        ))
            
            except Exception as e:
                errors.append(ValidationError(
                    file_path=label_path,
                    error_type="malformed",
                    message=f"Failed to read annotation file: {str(e)}"
                ))
        
        return errors
    
    def split_dataset(
        self,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        random_seed: Optional[int] = None
    ) -> DatasetSplit:
        """Split dataset with stratified class distribution.
        
        **Validates: Requirements 1.3, 1.4**
        
        Uses sklearn.train_test_split with stratification to ensure balanced
        class distribution across train/val/test subsets.
        
        Args:
            train_ratio: Training fraction (uses instance default if None)
            val_ratio: Validation fraction (uses instance default if None)
            test_ratio: Test fraction (uses instance default if None)
            random_seed: Random seed (uses instance default if None)
            
        Returns:
            DatasetSplit containing file paths for each subset
            
        Raises:
            ValueError: If no valid images found or ratios invalid
        """
        # Use instance defaults if not provided
        train_r = train_ratio if train_ratio is not None else self.train_ratio
        val_r = val_ratio if val_ratio is not None else self.val_ratio
        test_r = test_ratio if test_ratio is not None else self.test_ratio
        seed = random_seed if random_seed is not None else self.random_seed
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.dataset_path.rglob(f'*{ext}'))
            image_files.extend(self.dataset_path.rglob(f'*{ext.upper()}'))
        
        # Deduplicate files (case-insensitive filesystems may return duplicates)
        image_files = list(dict.fromkeys(image_files))
        
        if not image_files:
            raise ValueError(f"No image files found in {self.dataset_path}")
        
        # Extract class labels for stratification
        # Read each image's label file and collect all class IDs
        image_labels = []  # List of primary class ID for each image
        valid_images = []  # Only include images with valid labels
        
        for img_path in image_files:
            # Find corresponding label file
            label_path = img_path.with_suffix('.txt')
            
            if not label_path.exists():
                # Try labels/ directory
                parent = img_path.parent
                if parent.name == 'images':
                    labels_dir = parent.parent / 'labels'
                    label_path = labels_dir / img_path.name
                    label_path = label_path.with_suffix('.txt')
            
            if not label_path.exists():
                continue  # Skip images without labels
            
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                # Use first class ID as stratification label
                # (for multi-class images, this provides reasonable stratification)
                if lines:
                    first_line = lines[0].strip()
                    if first_line:
                        class_id = int(first_line.split()[0])
                        valid_images.append(img_path)
                        image_labels.append(class_id)
            except (ValueError, IndexError):
                continue  # Skip malformed labels
        
        if not valid_images:
            raise ValueError("No valid image-label pairs found in dataset")
        
        # Perform stratified split: first split into train and temp (val + test)
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            valid_images,
            image_labels,
            train_size=train_r,
            random_state=seed,
            stratify=image_labels
        )
        
        # Split temp into val and test
        # Calculate relative ratio for val within temp
        relative_val_ratio = val_r / (val_r + test_r)
        
        val_files, test_files, _, _ = train_test_split(
            temp_files,
            temp_labels,
            train_size=relative_val_ratio,
            random_state=seed,
            stratify=temp_labels
        )
        
        # Store split for later use
        self._dataset_split = DatasetSplit(
            train_files=train_files,
            val_files=val_files,
            test_files=test_files
        )
        
        # Compute class distribution
        self._compute_class_distribution()
        
        return self._dataset_split
    
    def _compute_class_distribution(self) -> None:
        """Compute class distribution across all subsets.
        
        Updates self._class_distribution with counts per class.
        """
        if not self._dataset_split:
            return
        
        class_counts: Dict[int, int] = {}
        
        all_files = (
            self._dataset_split.train_files +
            self._dataset_split.val_files +
            self._dataset_split.test_files
        )
        
        for img_path in all_files:
            # Find label file
            label_path = img_path.with_suffix('.txt')
            
            if not label_path.exists():
                parent = img_path.parent
                if parent.name == 'images':
                    labels_dir = parent.parent / 'labels'
                    label_path = labels_dir / img_path.name
                    label_path = label_path.with_suffix('.txt')
            
            if not label_path.exists():
                continue
            
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            class_id = int(line.split()[0])
                            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            except (ValueError, IndexError):
                continue
        
        # Convert to string keys (class names or IDs)
        # For C2A dataset, typically class 0 is "person"
        self._class_distribution = {
            f"class_{class_id}": count
            for class_id, count in sorted(class_counts.items())
        }
    
    def export_yolo_format(self, output_dir: Optional[Path] = None) -> None:
        """Export dataset in YOLO format directory structure.
        
        **Validates: Requirements 1.5, 1.6**
        
        Creates directory structure:
        output_dir/
          train/
            images/
            labels/
          val/
            images/
            labels/
          test/
            images/
            labels/
          data.yaml
        
        Args:
            output_dir: Output directory (uses instance default if None)
            
        Raises:
            ValueError: If split_dataset() has not been called
            FileExistsError: If output exists and overwrite_existing is False
        """
        if not self._dataset_split:
            raise ValueError("Must call split_dataset() before export_yolo_format()")
        
        out_dir = Path(output_dir) if output_dir else self.output_dir
        
        # Check for existing output directory (Requirement 1.6)
        if out_dir.exists() and not self.overwrite_existing:
            raise FileExistsError(
                f"Output directory {out_dir} already exists. "
                "Set overwrite_existing=True to overwrite."
            )
        
        # Create output directory structure
        for split_name in ['train', 'val', 'test']:
            (out_dir / split_name / 'images').mkdir(parents=True, exist_ok=True)
            (out_dir / split_name / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Copy files for each split
        splits = {
            'train': self._dataset_split.train_files,
            'val': self._dataset_split.val_files,
            'test': self._dataset_split.test_files
        }
        
        for split_name, files in splits.items():
            images_dir = out_dir / split_name / 'images'
            labels_dir = out_dir / split_name / 'labels'
            
            for img_path in files:
                # Copy image file
                dst_img = images_dir / img_path.name
                # Only copy if source and destination are different
                if img_path.resolve() != dst_img.resolve():
                    shutil.copy2(img_path, dst_img)
                
                # Find and copy label file
                label_path = img_path.with_suffix('.txt')
                
                if not label_path.exists():
                    parent = img_path.parent
                    if parent.name == 'images':
                        labels_dir_src = parent.parent / 'labels'
                        label_path = labels_dir_src / img_path.name
                        label_path = label_path.with_suffix('.txt')
                
                if label_path.exists():
                    dst_label = labels_dir / label_path.name
                    # Only copy if source and destination are different
                    if label_path.resolve() != dst_label.resolve():
                        shutil.copy2(label_path, dst_label)
        
        # Generate data.yaml for ultralytics
        self._generate_data_yaml(out_dir)
    
    def _generate_data_yaml(self, output_dir: Path) -> None:
        """Generate data.yaml file for ultralytics YOLO.
        
        Args:
            output_dir: Output directory containing train/val/test splits
        """
        # Determine class names
        # For C2A dataset, typically just "person" (class 0)
        # Extract from class distribution
        class_ids = sorted([
            int(k.replace('class_', ''))
            for k in self._class_distribution.keys()
        ])
        
        # Map class IDs to names (default to "person" for class 0)
        class_names = []
        for class_id in class_ids:
            if class_id == 0:
                class_names.append('person')
            else:
                class_names.append(f'class_{class_id}')
        
        data_yaml = {
            'path': str(output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    def generate_manifest(self) -> DatasetManifest:
        """Generate dataset manifest with metadata and class distribution.
        
        **Validates: Requirement 1.4**
        
        Returns:
            DatasetManifest object with counts and metadata
            
        Raises:
            ValueError: If split_dataset() has not been called
        """
        if not self._dataset_split:
            raise ValueError("Must call split_dataset() before generate_manifest()")
        
        train_count = len(self._dataset_split.train_files)
        val_count = len(self._dataset_split.val_files)
        test_count = len(self._dataset_split.test_files)
        total = train_count + val_count + test_count
        
        manifest = DatasetManifest(
            total_images=total,
            train_count=train_count,
            val_count=val_count,
            test_count=test_count,
            class_distribution=self._class_distribution,
            split_ratios=(self.train_ratio, self.val_ratio, self.test_ratio),
            timestamp=datetime.utcnow().isoformat() + 'Z'
        )
        
        return manifest
    
    def run_pipeline(self, skip_verification: bool = False) -> DatasetManifest:
        """Run complete dataset preparation pipeline.
        
        This is a convenience method that executes the full pipeline:
        1. Verify annotations (unless skipped)
        2. Split dataset with stratification
        3. Export to YOLO format
        4. Generate manifest
        
        Args:
            skip_verification: Skip annotation verification step
            
        Returns:
            DatasetManifest with final statistics
            
        Raises:
            ValueError: If validation errors found and not skipped
            FileExistsError: If output exists and overwrite disabled
        """
        # Step 1: Verify annotations
        if not skip_verification:
            errors = self.verify_annotations()
            if errors:
                error_msg = f"Found {len(errors)} validation errors:\n"
                error_msg += "\n".join(str(e) for e in errors[:10])
                if len(errors) > 10:
                    error_msg += f"\n... and {len(errors) - 10} more"
                raise ValueError(error_msg)
        
        # Step 2: Split dataset
        self.split_dataset()
        
        # Step 3: Export to YOLO format
        self.export_yolo_format()
        
        # Step 4: Generate manifest
        manifest = self.generate_manifest()
        
        # Save manifest to output directory
        manifest.save_json(self.output_dir / 'manifest.json')
        
        return manifest
