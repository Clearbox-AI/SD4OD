"""Dataset statistical analysis and gap identification."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from synthdet.config import AnalysisConfig
from synthdet.types import (
    BBox,
    BBoxSizeBucket,
    BBoxSizeStats,
    ClassDistribution,
    Dataset,
    DatasetStatistics,
    SpatialRegion,
    compute_uniformity,
)
from synthdet.utils.image import get_image_dimensions, group_augmentation_variants


# ---------------------------------------------------------------------------
# Bbox statistics
# ---------------------------------------------------------------------------


def compute_bbox_size_stats(bboxes: list[BBox]) -> BBoxSizeStats | None:
    """Compute descriptive statistics for a list of bboxes."""
    if not bboxes:
        return None
    widths = np.array([b.width for b in bboxes])
    heights = np.array([b.height for b in bboxes])
    areas = np.array([b.area for b in bboxes])
    ars = np.array([b.aspect_ratio for b in bboxes])
    # Filter out inf aspect ratios for stats
    finite_ars = ars[np.isfinite(ars)]
    if len(finite_ars) == 0:
        finite_ars = np.array([0.0])

    return BBoxSizeStats(
        count=len(bboxes),
        width_mean=float(np.mean(widths)),
        width_std=float(np.std(widths)),
        width_min=float(np.min(widths)),
        width_max=float(np.max(widths)),
        height_mean=float(np.mean(heights)),
        height_std=float(np.std(heights)),
        height_min=float(np.min(heights)),
        height_max=float(np.max(heights)),
        area_mean=float(np.mean(areas)),
        area_std=float(np.std(areas)),
        area_min=float(np.min(areas)),
        area_max=float(np.max(areas)),
        aspect_ratio_mean=float(np.mean(finite_ars)),
        aspect_ratio_std=float(np.std(finite_ars)),
    )


def compute_size_bucket_counts(bboxes: list[BBox]) -> dict[BBoxSizeBucket, int]:
    """Count annotations per size bucket."""
    counts: dict[BBoxSizeBucket, int] = {b: 0 for b in BBoxSizeBucket}
    for bbox in bboxes:
        counts[bbox.size_bucket] += 1
    return counts


def compute_spatial_distribution(bboxes: list[BBox]) -> dict[SpatialRegion, int]:
    """Count annotations per spatial region."""
    counts: dict[SpatialRegion, int] = {r: 0 for r in SpatialRegion}
    for bbox in bboxes:
        counts[bbox.spatial_region] += 1
    return counts


def compute_aspect_ratio_histogram(
    bboxes: list[BBox], num_bins: int = 8
) -> list[tuple[float, float, int]]:
    """Compute a log-spaced aspect ratio histogram.

    Returns list of (bin_low, bin_high, count).
    """
    if not bboxes:
        return []
    ars = [b.aspect_ratio for b in bboxes if math.isfinite(b.aspect_ratio) and b.aspect_ratio > 0]
    if not ars:
        return []

    log_ars = np.log10(ars)
    lo, hi = float(np.min(log_ars)), float(np.max(log_ars))
    if lo == hi:
        return [(10**lo, 10**hi, len(ars))]

    edges = np.linspace(lo, hi, num_bins + 1)
    bins: list[tuple[float, float, int]] = []
    for i in range(num_bins):
        low = 10 ** edges[i]
        high = 10 ** edges[i + 1]
        if i < num_bins - 1:
            count = sum(1 for a in ars if edges[i] <= np.log10(a) < edges[i + 1])
        else:
            # Last bin is inclusive on both sides
            count = sum(1 for a in ars if edges[i] <= np.log10(a) <= edges[i + 1])
        bins.append((float(low), float(high), count))
    return bins


def compute_class_distribution(
    dataset: Dataset,
) -> list[ClassDistribution]:
    """Compute per-class distribution statistics."""
    all_bboxes = dataset.all_bboxes()
    total = len(all_bboxes)
    distributions: list[ClassDistribution] = []

    for class_id, class_name in enumerate(dataset.class_names):
        class_bboxes = [b for b in all_bboxes if b.class_id == class_id]
        count = len(class_bboxes)
        distributions.append(ClassDistribution(
            class_id=class_id,
            class_name=class_name,
            count=count,
            fraction=count / total if total > 0 else 0.0,
            size_stats=compute_bbox_size_stats(class_bboxes),
            bucket_counts=compute_size_bucket_counts(class_bboxes),
            region_counts=compute_spatial_distribution(class_bboxes),
        ))

    return distributions


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def compute_dataset_statistics(
    dataset: Dataset,
    config: AnalysisConfig | None = None,
) -> DatasetStatistics:
    """Compute the complete statistical fingerprint of a dataset."""
    if config is None:
        config = AnalysisConfig()

    all_bboxes = dataset.all_bboxes()
    all_records = dataset.all_records

    # Totals
    total_images = len(all_records)
    total_annotations = len(all_bboxes)
    negative_images = sum(1 for r in all_records if r.is_negative)

    # Unique source images (group Roboflow augmentation variants)
    all_paths = [r.image_path for r in dataset.train]
    groups = group_augmentation_variants(all_paths)
    unique_source_images = len(groups)

    # Per-split
    split_image_counts = {
        "train": len(dataset.train),
        "valid": len(dataset.valid),
        "test": len(dataset.test),
    }
    split_annotation_counts = {
        "train": len(dataset.all_bboxes("train")),
        "valid": len(dataset.all_bboxes("valid")),
        "test": len(dataset.all_bboxes("test")),
    }

    # Overall bbox stats
    overall_size_stats = compute_bbox_size_stats(all_bboxes)
    overall_bucket_counts = compute_size_bucket_counts(all_bboxes)
    overall_region_counts = compute_spatial_distribution(all_bboxes)

    # Uniformity
    bucket_uniformity = compute_uniformity(overall_bucket_counts)
    region_uniformity = compute_uniformity(overall_region_counts)

    # Annotations per image
    ann_counts = [len(r.bboxes) for r in all_records]
    ann_arr = np.array(ann_counts) if ann_counts else np.array([0])
    ann_histogram = dict(Counter(ann_counts))

    # Aspect ratio histogram
    aspect_ratio_bins = compute_aspect_ratio_histogram(all_bboxes, config.aspect_ratio_num_bins)

    # Image dimensions (sample first image)
    img_w, img_h = None, None
    for rec in all_records:
        if rec.image_path.is_file():
            try:
                img_w, img_h = get_image_dimensions(rec.image_path)
            except Exception:
                pass
            break

    return DatasetStatistics(
        total_images=total_images,
        total_annotations=total_annotations,
        negative_images=negative_images,
        negative_ratio=negative_images / total_images if total_images > 0 else 0.0,
        unique_source_images=unique_source_images,
        split_image_counts=split_image_counts,
        split_annotation_counts=split_annotation_counts,
        class_distributions=compute_class_distribution(dataset),
        overall_size_stats=overall_size_stats,
        overall_bucket_counts=overall_bucket_counts,
        overall_region_counts=overall_region_counts,
        bucket_uniformity=bucket_uniformity,
        region_uniformity=region_uniformity,
        annotations_per_image_mean=float(np.mean(ann_arr)),
        annotations_per_image_std=float(np.std(ann_arr)),
        annotations_per_image_max=int(np.max(ann_arr)),
        annotations_per_image_histogram=ann_histogram,
        aspect_ratio_bins=aspect_ratio_bins,
        image_width=img_w,
        image_height=img_h,
    )


# ---------------------------------------------------------------------------
# Gap identification (consumed by strategy module)
# ---------------------------------------------------------------------------


def identify_size_bucket_gaps(
    stats: DatasetStatistics, min_per_bucket: int = 50
) -> dict[BBoxSizeBucket, int]:
    """Identify size buckets with fewer annotations than the minimum.

    Returns dict mapping bucket to deficit (how many more are needed).
    """
    gaps: dict[BBoxSizeBucket, int] = {}
    for bucket, count in stats.overall_bucket_counts.items():
        deficit = min_per_bucket - count
        if deficit > 0:
            gaps[bucket] = deficit
    return gaps


def identify_spatial_gaps(
    stats: DatasetStatistics, min_per_region: int = 30
) -> dict[SpatialRegion, int]:
    """Identify spatial regions with fewer annotations than the minimum."""
    gaps: dict[SpatialRegion, int] = {}
    for region, count in stats.overall_region_counts.items():
        deficit = min_per_region - count
        if deficit > 0:
            gaps[region] = deficit
    return gaps


def identify_aspect_ratio_gaps(
    stats: DatasetStatistics, min_per_bin: int = 20
) -> list[tuple[float, float, int]]:
    """Identify aspect ratio bins with fewer annotations than the minimum.

    Returns list of (bin_low, bin_high, deficit).
    """
    gaps: list[tuple[float, float, int]] = []
    for lo, hi, count in stats.aspect_ratio_bins:
        deficit = min_per_bin - count
        if deficit > 0:
            gaps.append((lo, hi, deficit))
    return gaps
