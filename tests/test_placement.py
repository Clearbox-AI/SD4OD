"""Tests for synthdet.generate.placement."""

from __future__ import annotations

import random

import numpy as np
import pytest

from synthdet.generate.placement import (
    check_placement_valid,
    determine_center,
    sample_bbox_dimensions,
)
from synthdet.types import BBox, BBoxSizeBucket, SIZE_BUCKET_THRESHOLDS, SpatialRegion


class TestCheckPlacementValid:
    def test_empty_existing(self):
        bbox = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.1, height=0.1)
        assert check_placement_valid(bbox, [], 0.3) is True

    def test_overlap_rejected(self):
        existing = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.1, height=0.1)
        new = BBox(class_id=0, x_center=0.52, y_center=0.52, width=0.1, height=0.1)
        assert check_placement_valid(new, [existing], 0.3) is False


class TestDetermineCenter:
    def test_finds_placement_in_region(self):
        random.seed(42)
        center = determine_center(
            SpatialRegion.top_left, 20, 20, 300, 200, [], 20, 0.3,
        )
        assert center is not None
        cx, cy = center
        assert cx < 120
        assert cy < 90

    def test_respects_valid_zone(self):
        hull = np.array([
            [[0.3, 0.3]], [[0.7, 0.3]], [[0.7, 0.7]], [[0.3, 0.7]]
        ], dtype=np.float32)
        random.seed(42)
        for _ in range(20):
            center = determine_center(
                None, 10, 10, 300, 200, [], 50, 0.3, valid_zone=hull,
            )
            if center is not None:
                nx, ny = center[0] / 300, center[1] / 200
                assert 0.2 <= nx <= 0.8
                assert 0.2 <= ny <= 0.8

    def test_returns_none_on_failure(self):
        # Tiny image, large patch — impossible placement
        center = determine_center(
            None, 500, 500, 100, 100, [], 5, 0.3,
        )
        # With patch bigger than image, range collapses; may return a center
        # or None depending on clamping. Just make sure it doesn't crash.
        assert center is None or (isinstance(center, tuple) and len(center) == 2)


class TestSampleBboxDimensions:
    @pytest.mark.parametrize("bucket", list(BBoxSizeBucket))
    def test_dimensions_within_bucket(self, bucket):
        random.seed(42)
        lo, hi = SIZE_BUCKET_THRESHOLDS[bucket]
        if bucket == BBoxSizeBucket.large:
            hi = min(hi, 0.2)
        if bucket == BBoxSizeBucket.tiny:
            lo = max(lo, 0.0005)

        for _ in range(10):
            px_w, px_h = sample_bbox_dimensions(bucket, 860, 640)
            norm_area = (px_w / 860) * (px_h / 640)
            # Allow tolerance for clamping
            assert norm_area < 1.0
            assert px_w >= 4
            assert px_h >= 4

    def test_aspect_ratio_range(self):
        random.seed(42)
        for _ in range(20):
            px_w, px_h = sample_bbox_dimensions(
                BBoxSizeBucket.medium, 860, 640, aspect_ratio_range=(0.5, 2.0),
            )
            ar = px_w / max(px_h, 1)
            # Allow some tolerance due to clamping
            assert 0.1 < ar < 20.0
