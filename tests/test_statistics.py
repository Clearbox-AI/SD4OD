"""Tests for synthdet.analysis.statistics."""

from __future__ import annotations

import pytest

from synthdet.analysis.statistics import (
    compute_aspect_ratio_histogram,
    compute_bbox_size_stats,
    compute_class_distribution,
    compute_dataset_statistics,
    compute_size_bucket_counts,
    compute_spatial_distribution,
    identify_aspect_ratio_gaps,
    identify_size_bucket_gaps,
    identify_spatial_gaps,
)
from synthdet.config import AnalysisConfig
from synthdet.types import BBox, BBoxSizeBucket, SpatialRegion


class TestComputeBboxSizeStats:
    def test_basic(self, sample_bboxes):
        stats = compute_bbox_size_stats(sample_bboxes)
        assert stats is not None
        assert stats.count == 4
        assert stats.width_mean > 0
        assert stats.area_min > 0

    def test_empty(self):
        assert compute_bbox_size_stats([]) is None


class TestComputeSizeBucketCounts:
    def test_all_buckets_present(self, sample_bboxes):
        counts = compute_size_bucket_counts(sample_bboxes)
        assert set(counts.keys()) == set(BBoxSizeBucket)
        assert sum(counts.values()) == len(sample_bboxes)

    def test_empty(self):
        counts = compute_size_bucket_counts([])
        assert all(v == 0 for v in counts.values())


class TestComputeSpatialDistribution:
    def test_all_regions_present(self, sample_bboxes):
        counts = compute_spatial_distribution(sample_bboxes)
        assert set(counts.keys()) == set(SpatialRegion)
        assert sum(counts.values()) == len(sample_bboxes)


class TestComputeAspectRatioHistogram:
    def test_varied_ratios(self):
        bboxes = [
            BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.1, height=0.4),   # 0.25
            BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.2),   # 1.0
            BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.4, height=0.1),   # 4.0
            BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.3, height=0.15),  # 2.0
        ]
        bins = compute_aspect_ratio_histogram(bboxes, num_bins=4)
        assert len(bins) == 4
        total = sum(count for _, _, count in bins)
        assert total == len(bboxes)

    def test_all_same_ratio(self, sample_bboxes):
        # sample_bboxes all have aspect_ratio=1.0, produces single bin
        bins = compute_aspect_ratio_histogram(sample_bboxes, num_bins=4)
        assert len(bins) == 1
        assert bins[0][2] == len(sample_bboxes)

    def test_empty(self):
        assert compute_aspect_ratio_histogram([]) == []


class TestComputeDatasetStatistics:
    def test_mini_dataset(self, mini_dataset):
        stats = compute_dataset_statistics(mini_dataset)
        assert stats.total_images == 3
        assert stats.total_annotations == 4
        assert stats.negative_images == 0
        assert stats.split_image_counts["train"] == 2
        assert stats.split_annotation_counts["train"] == 3
        assert stats.unique_source_images == 1  # both train imgs share "laptop_a" source

    def test_uniformity_scores(self, mini_dataset):
        stats = compute_dataset_statistics(mini_dataset)
        assert 0.0 <= stats.bucket_uniformity <= 1.0
        assert 0.0 <= stats.region_uniformity <= 1.0

    def test_annotations_per_image(self, mini_dataset):
        stats = compute_dataset_statistics(mini_dataset)
        assert stats.annotations_per_image_max >= 1


class TestIdentifySizeBucketGaps:
    def test_finds_gaps(self, mini_dataset):
        stats = compute_dataset_statistics(mini_dataset)
        gaps = identify_size_bucket_gaps(stats, min_per_bucket=10)
        # With only 4 annotations, most buckets should have gaps
        assert len(gaps) > 0
        for bucket, deficit in gaps.items():
            assert deficit > 0

    def test_no_gaps_with_low_threshold(self, mini_dataset):
        stats = compute_dataset_statistics(mini_dataset)
        gaps = identify_size_bucket_gaps(stats, min_per_bucket=0)
        assert len(gaps) == 0


class TestIdentifySpatialGaps:
    def test_finds_gaps(self, mini_dataset):
        stats = compute_dataset_statistics(mini_dataset)
        gaps = identify_spatial_gaps(stats, min_per_region=5)
        assert len(gaps) > 0


class TestIdentifyAspectRatioGaps:
    def test_finds_gaps(self, mini_dataset):
        stats = compute_dataset_statistics(mini_dataset)
        gaps = identify_aspect_ratio_gaps(stats, min_per_bin=5)
        assert len(gaps) > 0
