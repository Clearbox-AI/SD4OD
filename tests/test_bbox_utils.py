"""Tests for synthdet.utils.bbox."""

from __future__ import annotations

from pathlib import Path

import pytest

from synthdet.types import AnnotationSource, BBox
from synthdet.utils.bbox import (
    bbox_iou,
    clip_bbox,
    coco_to_yolo,
    parse_yolo_label_file,
    pascal_voc_to_yolo,
    validate_bbox,
    yolo_to_coco,
    yolo_to_pascal_voc,
)


class TestYoloPascalVocConversion:
    def test_round_trip(self):
        bbox = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.3)
        x1, y1, x2, y2 = yolo_to_pascal_voc(bbox, 100, 200)
        restored = pascal_voc_to_yolo(x1, y1, x2, y2, 100, 200, class_id=0)
        assert restored.x_center == pytest.approx(bbox.x_center, abs=0.02)
        assert restored.y_center == pytest.approx(bbox.y_center, abs=0.02)
        assert restored.width == pytest.approx(bbox.width, abs=0.02)
        assert restored.height == pytest.approx(bbox.height, abs=0.02)

    def test_known_values(self):
        bbox = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.5, height=0.5)
        x1, y1, x2, y2 = yolo_to_pascal_voc(bbox, 100, 100)
        assert x1 == 25
        assert y1 == 25
        assert x2 == 75
        assert y2 == 75


class TestYoloCocoConversion:
    def test_round_trip(self):
        bbox = BBox(class_id=1, x_center=0.4, y_center=0.6, width=0.3, height=0.2)
        x, y, w, h = yolo_to_coco(bbox, 200, 300)
        restored = coco_to_yolo(x, y, w, h, 200, 300, class_id=1)
        assert restored.x_center == pytest.approx(bbox.x_center, abs=0.02)
        assert restored.width == pytest.approx(bbox.width, abs=0.02)


class TestClipBbox:
    def test_no_change_needed(self):
        bbox = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.2)
        clipped = clip_bbox(bbox)
        assert clipped.x_center == pytest.approx(0.5)
        assert clipped.width == pytest.approx(0.2)

    def test_clips_overflow(self):
        bbox = BBox(class_id=0, x_center=0.95, y_center=0.95, width=0.2, height=0.2)
        clipped = clip_bbox(bbox)
        assert clipped.x_center + clipped.width / 2 <= 1.0
        assert clipped.y_center + clipped.height / 2 <= 1.0


class TestBboxIou:
    def test_identical(self):
        bbox = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.2)
        assert bbox_iou(bbox, bbox) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = BBox(class_id=0, x_center=0.2, y_center=0.2, width=0.1, height=0.1)
        b = BBox(class_id=0, x_center=0.8, y_center=0.8, width=0.1, height=0.1)
        assert bbox_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.4, height=0.4)
        b = BBox(class_id=0, x_center=0.6, y_center=0.6, width=0.4, height=0.4)
        iou = bbox_iou(a, b)
        assert 0.0 < iou < 1.0


class TestValidateBbox:
    def test_valid(self):
        bbox = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.2)
        assert validate_bbox(bbox) == []

    def test_negative_class_id(self):
        bbox = BBox(class_id=-1, x_center=0.5, y_center=0.5, width=0.2, height=0.2)
        errors = validate_bbox(bbox)
        assert any("Negative class_id" in e for e in errors)

    def test_class_id_out_of_range(self):
        bbox = BBox(class_id=5, x_center=0.5, y_center=0.5, width=0.2, height=0.2)
        errors = validate_bbox(bbox, num_classes=3)
        assert any("class_id 5 >= num_classes 3" in e for e in errors)

    def test_zero_width(self):
        bbox = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.0, height=0.2)
        errors = validate_bbox(bbox)
        assert any("Non-positive" in e for e in errors)


class TestParseYoloLabelFile:
    def test_normal_file(self, tmp_path):
        label = tmp_path / "test.txt"
        label.write_text("0 0.5 0.5 0.1 0.1\n1 0.3 0.7 0.2 0.2\n")
        bboxes = parse_yolo_label_file(label)
        assert len(bboxes) == 2
        assert bboxes[0].class_id == 0
        assert bboxes[1].class_id == 1

    def test_empty_file(self, tmp_path):
        label = tmp_path / "empty.txt"
        label.write_text("")
        bboxes = parse_yolo_label_file(label)
        assert bboxes == []

    def test_no_trailing_newline(self, tmp_path):
        label = tmp_path / "no_newline.txt"
        label.write_text("0 0.5 0.5 0.1 0.1")  # no trailing \n
        bboxes = parse_yolo_label_file(label)
        assert len(bboxes) == 1

    def test_blank_lines(self, tmp_path):
        label = tmp_path / "blanks.txt"
        label.write_text("0 0.5 0.5 0.1 0.1\n\n\n1 0.3 0.7 0.2 0.2\n\n")
        bboxes = parse_yolo_label_file(label)
        assert len(bboxes) == 2

    def test_whitespace_only(self, tmp_path):
        label = tmp_path / "spaces.txt"
        label.write_text("   \n  \n")
        bboxes = parse_yolo_label_file(label)
        assert bboxes == []

    def test_custom_source(self, tmp_path):
        label = tmp_path / "test.txt"
        label.write_text("0 0.5 0.5 0.1 0.1\n")
        bboxes = parse_yolo_label_file(label, source=AnnotationSource.compositor)
        assert bboxes[0].source == AnnotationSource.compositor
