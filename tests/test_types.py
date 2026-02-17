"""Tests for synthdet.types."""

from __future__ import annotations

import math

import pytest

from synthdet.types import (
    AnnotationSource,
    BBox,
    BBoxSizeBucket,
    SpatialRegion,
    compute_uniformity,
)


class TestBBox:
    def test_area(self, sample_bbox):
        assert sample_bbox.area == pytest.approx(0.01)

    def test_aspect_ratio(self, sample_bbox):
        assert sample_bbox.aspect_ratio == pytest.approx(1.0)

    def test_aspect_ratio_wide(self):
        bbox = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.4, height=0.1)
        assert bbox.aspect_ratio == pytest.approx(4.0)

    def test_aspect_ratio_zero_height(self):
        bbox = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.1, height=0.0)
        assert bbox.aspect_ratio == float("inf")

    def test_size_bucket_tiny(self):
        bbox = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.01, height=0.01)
        assert bbox.size_bucket == BBoxSizeBucket.tiny  # area = 0.0001

    def test_size_bucket_small(self):
        bbox = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.1, height=0.1)
        assert bbox.size_bucket == BBoxSizeBucket.small  # area = 0.01

    def test_size_bucket_medium(self):
        bbox = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.2)
        assert bbox.size_bucket == BBoxSizeBucket.medium  # area = 0.04

    def test_size_bucket_large(self):
        bbox = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.3, height=0.3)
        assert bbox.size_bucket == BBoxSizeBucket.large  # area = 0.09

    def test_spatial_region_top_left(self):
        bbox = BBox(class_id=0, x_center=0.1, y_center=0.1, width=0.05, height=0.05)
        assert bbox.spatial_region == SpatialRegion.top_left

    def test_spatial_region_bottom_right(self):
        bbox = BBox(class_id=0, x_center=0.9, y_center=0.9, width=0.05, height=0.05)
        assert bbox.spatial_region == SpatialRegion.bottom_right

    def test_spatial_region_middle_center(self):
        bbox = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.05, height=0.05)
        assert bbox.spatial_region == SpatialRegion.middle_center

    def test_spatial_region_edge_case_1_0(self):
        # x_center=1.0 should map to right column (idx 2)
        bbox = BBox(class_id=0, x_center=1.0, y_center=1.0, width=0.01, height=0.01)
        assert bbox.spatial_region == SpatialRegion.bottom_right

    def test_to_yolo_line(self, sample_bbox):
        line = sample_bbox.to_yolo_line()
        assert line.startswith("0 ")
        parts = line.split()
        assert len(parts) == 5
        assert int(parts[0]) == 0
        assert float(parts[1]) == pytest.approx(0.5)

    def test_from_yolo_line(self):
        line = "0 0.5 0.5 0.1 0.1"
        bbox = BBox.from_yolo_line(line, source=AnnotationSource.human)
        assert bbox.class_id == 0
        assert bbox.x_center == pytest.approx(0.5)
        assert bbox.width == pytest.approx(0.1)
        assert bbox.source == AnnotationSource.human

    def test_from_yolo_line_with_confidence(self):
        line = "1 0.3 0.4 0.2 0.15 0.95"
        bbox = BBox.from_yolo_line(line)
        assert bbox.class_id == 1
        assert bbox.confidence == pytest.approx(0.95)

    def test_from_yolo_line_roundtrip(self, sample_bbox):
        line = sample_bbox.to_yolo_line()
        restored = BBox.from_yolo_line(line)
        assert restored.class_id == sample_bbox.class_id
        assert restored.x_center == pytest.approx(sample_bbox.x_center, abs=1e-5)
        assert restored.width == pytest.approx(sample_bbox.width, abs=1e-5)

    def test_from_yolo_line_too_few_fields(self):
        with pytest.raises(ValueError, match="at least 5 fields"):
            BBox.from_yolo_line("0 0.5 0.5")

    def test_frozen(self, sample_bbox):
        with pytest.raises(AttributeError):
            sample_bbox.class_id = 1  # type: ignore[misc]


class TestImageRecord:
    def test_is_negative_true(self, tmp_path):
        from synthdet.types import ImageRecord
        rec = ImageRecord(image_path=tmp_path / "test.jpg", bboxes=[])
        assert rec.is_negative is True

    def test_is_negative_false(self, tmp_path, sample_bbox):
        from synthdet.types import ImageRecord
        rec = ImageRecord(image_path=tmp_path / "test.jpg", bboxes=[sample_bbox])
        assert rec.is_negative is False


class TestDataset:
    def test_all_records(self, mini_dataset):
        assert len(mini_dataset.all_records) == 3  # 2 train + 1 valid

    def test_all_bboxes(self, mini_dataset):
        all_bboxes = mini_dataset.all_bboxes()
        assert len(all_bboxes) == 4  # 2+1+1

    def test_all_bboxes_by_split(self, mini_dataset):
        train_bboxes = mini_dataset.all_bboxes("train")
        valid_bboxes = mini_dataset.all_bboxes("valid")
        assert len(train_bboxes) == 3
        assert len(valid_bboxes) == 1

    def test_all_bboxes_invalid_split(self, mini_dataset):
        with pytest.raises(ValueError, match="Unknown split"):
            mini_dataset.all_bboxes("nonexistent")


class TestComputeUniformity:
    def test_uniform(self):
        counts = {"a": 10, "b": 10, "c": 10}
        assert compute_uniformity(counts) == pytest.approx(1.0)

    def test_maximally_skewed(self):
        counts = {"a": 100, "b": 0, "c": 0}
        assert compute_uniformity(counts) == pytest.approx(0.0)

    def test_partial_skew(self):
        counts = {"a": 90, "b": 5, "c": 5}
        u = compute_uniformity(counts)
        assert 0.0 < u < 1.0

    def test_single_bin(self):
        counts = {"a": 100}
        assert compute_uniformity(counts) == 1.0

    def test_empty(self):
        counts: dict[str, int] = {}
        assert compute_uniformity(counts) == 0.0
