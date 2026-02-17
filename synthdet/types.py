"""Core data types for SynthDet.

Every module in the library produces/consumes these types. The design supports:
- Phase 1: Dataset analysis and gap-based synthesis strategy
- Phase 2+: Active learning feedback from trained models
- Phase 4+: SPC quality monitoring on activation distributions
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AnnotationSource(str, enum.Enum):
    """Tracks how an annotation was produced for quality filtering."""

    human = "human"
    compositor = "compositor"
    inpainting = "inpainting"
    grounding_dino = "grounding_dino"
    owl_vit = "owl_vit"
    sam_refined = "sam_refined"
    unknown = "unknown"


class BBoxSizeBucket(str, enum.Enum):
    """Size category based on normalized bbox area.

    Thresholds (area = width * height, both normalized 0-1):
        tiny:   area < 0.005
        small:  0.005 <= area < 0.02
        medium: 0.02 <= area < 0.08
        large:  area >= 0.08
    """

    tiny = "tiny"
    small = "small"
    medium = "medium"
    large = "large"


# Area thresholds for size bucketing
SIZE_BUCKET_THRESHOLDS: dict[BBoxSizeBucket, tuple[float, float]] = {
    BBoxSizeBucket.tiny: (0.0, 0.005),
    BBoxSizeBucket.small: (0.005, 0.02),
    BBoxSizeBucket.medium: (0.02, 0.08),
    BBoxSizeBucket.large: (0.08, 1.0),
}


class SpatialRegion(str, enum.Enum):
    """3x3 spatial grid region based on bbox center position."""

    top_left = "top_left"
    top_center = "top_center"
    top_right = "top_right"
    middle_left = "middle_left"
    middle_center = "middle_center"
    middle_right = "middle_right"
    bottom_left = "bottom_left"
    bottom_center = "bottom_center"
    bottom_right = "bottom_right"


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BBox:
    """A single bounding box in YOLO normalized format.

    All coordinates are normalized to [0, 1].
    """

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float = 1.0
    source: AnnotationSource = AnnotationSource.unknown

    @property
    def area(self) -> float:
        """Normalized area (width * height)."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Width / height ratio. Returns inf if height is 0."""
        if self.height == 0:
            return float("inf")
        return self.width / self.height

    @property
    def size_bucket(self) -> BBoxSizeBucket:
        """Categorize bbox by area."""
        a = self.area
        for bucket, (lo, hi) in SIZE_BUCKET_THRESHOLDS.items():
            if lo <= a < hi:
                return bucket
        return BBoxSizeBucket.large

    @property
    def spatial_region(self) -> SpatialRegion:
        """Determine which 3x3 grid cell the bbox center falls in."""
        col = min(int(self.x_center * 3), 2)
        row = min(int(self.y_center * 3), 2)
        idx = row * 3 + col
        return list(SpatialRegion)[idx]

    def to_yolo_line(self) -> str:
        """Serialize to a YOLO label line: 'class_id x y w h'."""
        return (
            f"{self.class_id} "
            f"{self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )

    @classmethod
    def from_yolo_line(
        cls,
        line: str,
        source: AnnotationSource = AnnotationSource.unknown,
    ) -> BBox:
        """Parse a YOLO label line.

        Accepts both 5-field (class x y w h) and 6-field (class x y w h conf) formats.
        """
        parts = line.strip().split()
        if len(parts) < 5:
            raise ValueError(f"Expected at least 5 fields, got {len(parts)}: {line!r}")
        class_id = int(parts[0])
        x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        conf = float(parts[5]) if len(parts) > 5 else 1.0
        return cls(
            class_id=class_id,
            x_center=x,
            y_center=y,
            width=w,
            height=h,
            confidence=conf,
            source=source,
        )


@dataclass
class ImageRecord:
    """An image with its bounding box annotations and metadata.

    This is the universal interchange type — every module produces/consumes
    ImageRecord objects.
    """

    image_path: Path
    bboxes: list[BBox]
    image: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_negative(self) -> bool:
        """True if this image has no annotations (negative example)."""
        return len(self.bboxes) == 0

    def load_image(self) -> np.ndarray:
        """Load image from disk (BGR, as OpenCV reads it). Caches in self.image."""
        if self.image is not None:
            return self.image
        import cv2

        img = cv2.imread(str(self.image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {self.image_path}")
        self.image = img
        return img


@dataclass
class Dataset:
    """A complete YOLO-format dataset with train/valid/test splits."""

    root: Path
    class_names: list[str]
    train: list[ImageRecord]
    valid: list[ImageRecord]
    test: list[ImageRecord]

    @property
    def all_records(self) -> list[ImageRecord]:
        """All records across all splits."""
        return self.train + self.valid + self.test

    def all_bboxes(self, split: str | None = None) -> list[BBox]:
        """All bounding boxes, optionally filtered by split name."""
        if split is None:
            records = self.all_records
        elif split == "train":
            records = self.train
        elif split in ("valid", "val"):
            records = self.valid
        elif split == "test":
            records = self.test
        else:
            raise ValueError(f"Unknown split: {split!r}")
        return [bbox for rec in records for bbox in rec.bboxes]


# ---------------------------------------------------------------------------
# Statistics types
# ---------------------------------------------------------------------------


@dataclass
class BBoxSizeStats:
    """Descriptive statistics for bounding box dimensions."""

    count: int
    width_mean: float
    width_std: float
    width_min: float
    width_max: float
    height_mean: float
    height_std: float
    height_min: float
    height_max: float
    area_mean: float
    area_std: float
    area_min: float
    area_max: float
    aspect_ratio_mean: float
    aspect_ratio_std: float


@dataclass
class ClassDistribution:
    """Per-class annotation statistics."""

    class_id: int
    class_name: str
    count: int
    fraction: float
    size_stats: BBoxSizeStats | None
    bucket_counts: dict[BBoxSizeBucket, int]
    region_counts: dict[SpatialRegion, int]


@dataclass
class DatasetStatistics:
    """Complete statistical fingerprint of a dataset."""

    # Totals
    total_images: int
    total_annotations: int
    negative_images: int
    negative_ratio: float
    unique_source_images: int

    # Per-split counts
    split_image_counts: dict[str, int]
    split_annotation_counts: dict[str, int]

    # Per-class
    class_distributions: list[ClassDistribution]

    # Overall bbox stats
    overall_size_stats: BBoxSizeStats | None
    overall_bucket_counts: dict[BBoxSizeBucket, int]
    overall_region_counts: dict[SpatialRegion, int]

    # Uniformity scores (normalized entropy, 0=skewed, 1=uniform)
    bucket_uniformity: float
    region_uniformity: float

    # Annotations per image
    annotations_per_image_mean: float
    annotations_per_image_std: float
    annotations_per_image_max: int
    annotations_per_image_histogram: dict[int, int]  # count -> num_images

    # Aspect ratio bins (log-spaced)
    aspect_ratio_bins: list[tuple[float, float, int]]  # (lo, hi, count)

    # Image dimensions
    image_width: int | None
    image_height: int | None


# ---------------------------------------------------------------------------
# Quality monitoring types (SPC) — defined now, implemented in future phases
# ---------------------------------------------------------------------------


@dataclass
class ActivationDistributionSnapshot:
    """Per-layer activation statistics from a detection backbone.

    Captured by running images through the model and recording layer outputs.
    Used as baseline for Shewhart control charts.
    """

    layer_name: str
    mean: float
    std: float
    percentiles: dict[int, float]  # e.g. {5: val, 25: val, 50: val, 75: val, 95: val}
    num_images: int
    timestamp: str


@dataclass
class QualityControlChart:
    """Shewhart X-bar control chart for monitoring distribution drift.

    Western Electric rules:
    - 1 point beyond 3-sigma (UCL/LCL)
    - 2 of 3 consecutive points beyond 2-sigma
    - 7+ consecutive points above or below center line
    """

    metric_name: str
    center_line: float
    ucl: float  # upper control limit (center + 3*sigma)
    lcl: float  # lower control limit (center - 3*sigma)
    sigma: float
    values: list[float]
    out_of_control_indices: list[int]


@dataclass
class QualityMetrics:
    """Per-generation-batch quality assessment."""

    batch_id: str
    activation_snapshots: list[ActivationDistributionSnapshot]
    control_charts: list[QualityControlChart]
    alerts: list[str]


# ---------------------------------------------------------------------------
# Active learning types
# ---------------------------------------------------------------------------


@dataclass
class ModelPerformanceProfile:
    """Where the detection model struggles — used to drive targeted generation.

    Populated by evaluating a trained YOLO model on validation data, then
    breaking down metrics by bbox size bucket and spatial region.
    """

    overall_map50: float
    overall_map50_95: float
    per_class_map: dict[int, float]
    per_bucket_map: dict[BBoxSizeBucket, float]
    per_region_map: dict[SpatialRegion, float]
    false_negative_buckets: dict[BBoxSizeBucket, int]
    false_negative_regions: dict[SpatialRegion, int]
    confusion_pairs: list[tuple[int, int, int]]  # (true_class, pred_class, count)


@dataclass
class ActiveLearningSignal:
    """A signal indicating what to generate next, derived from model feedback or SPC."""

    target_classes: list[int]
    target_size_buckets: list[BBoxSizeBucket]
    target_regions: list[SpatialRegion]
    priority: float  # 0.0 to 1.0
    rationale: str
    source: str  # "dataset_gap" | "model_feedback" | "spc_alert"


@dataclass
class GenerationTask:
    """A concrete, prioritized generation task with targeting parameters."""

    task_id: str
    priority: float  # 0.0 to 1.0
    num_images: int
    target_classes: list[int]
    target_size_buckets: list[BBoxSizeBucket]
    target_regions: list[SpatialRegion]
    suggested_prompts: list[str]
    rationale: str
    method: str  # "compositor" | "inpainting" | "diffusion" | "augmentation"


# ---------------------------------------------------------------------------
# Synthesis strategy
# ---------------------------------------------------------------------------


@dataclass
class SynthesisStrategy:
    """What synthetic data to generate and why.

    In Phase 1, this is purely gap-driven (exploration).
    In later phases, active_learning_signals from model feedback drive
    exploitation of known weaknesses.
    """

    # Core targets
    target_total_images: int
    target_class_counts: dict[int, int]
    negative_ratio: float

    # Gap analysis
    size_bucket_gaps: dict[BBoxSizeBucket, int]
    spatial_gaps: dict[SpatialRegion, int]
    aspect_ratio_gaps: list[tuple[float, float, int]]  # (lo, hi, deficit)

    # Prioritized tasks
    generation_tasks: list[GenerationTask]

    # Active learning (empty in Phase 1, populated when model feedback available)
    active_learning_signals: list[ActiveLearningSignal] = field(default_factory=list)

    # Quality baselines (None until first generation cycle)
    quality_baselines: QualityMetrics | None = None

    @property
    def total_synthetic_images(self) -> int:
        return sum(t.num_images for t in self.generation_tasks)

    @property
    def has_model_feedback(self) -> bool:
        return any(s.source == "model_feedback" for s in self.active_learning_signals)


# ---------------------------------------------------------------------------
# Utility functions for types
# ---------------------------------------------------------------------------


def compute_uniformity(counts: dict[Any, int]) -> float:
    """Normalized entropy of a count distribution. 0=maximally skewed, 1=uniform."""
    n = len(counts)  # total bins (including zeros)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    values = [v for v in counts.values() if v > 0]
    total = sum(values)
    if total == 0:
        return 0.0
    entropy = -sum((v / total) * math.log(v / total) for v in values if v > 0)
    max_entropy = math.log(n)
    if max_entropy == 0:
        return 0.0
    return entropy / max_entropy
