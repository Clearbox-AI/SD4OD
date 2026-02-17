"""Tests for the YOLO dataset writer."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from synthdet.annotate.yolo_writer import (
    write_data_yaml,
    write_yolo_dataset,
    write_yolo_label,
    write_yolo_split,
)
from synthdet.types import AnnotationSource, BBox, ImageRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_records() -> list[ImageRecord]:
    """ImageRecords with in-memory images."""
    records = []
    for i in range(3):
        img = np.full((200, 300, 3), 100 + i * 30, dtype=np.uint8)
        bboxes = [
            BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.1, height=0.1,
                 source=AnnotationSource.compositor),
        ]
        if i == 2:
            # Third image is negative
            bboxes = []
        records.append(ImageRecord(
            image_path=Path(f"synth_{i:04d}.jpg"),
            bboxes=bboxes,
            image=img,
        ))
    return records


# ---------------------------------------------------------------------------
# write_yolo_label tests
# ---------------------------------------------------------------------------


class TestWriteYoloLabel:
    def test_write_single_label(self, tmp_path):
        label_path = tmp_path / "test.txt"
        bboxes = [
            BBox(class_id=0, x_center=0.5, y_center=0.4, width=0.1, height=0.2),
            BBox(class_id=1, x_center=0.3, y_center=0.7, width=0.05, height=0.08),
        ]
        write_yolo_label(label_path, bboxes)

        content = label_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("0 ")
        assert lines[1].startswith("1 ")

        # Check format: class_id x y w h
        parts = lines[0].split()
        assert len(parts) == 5
        assert float(parts[1]) == pytest.approx(0.5, abs=1e-4)

    def test_write_empty_label(self, tmp_path):
        label_path = tmp_path / "empty.txt"
        write_yolo_label(label_path, [])
        content = label_path.read_text()
        assert content == ""

    def test_creates_parent_dirs(self, tmp_path):
        label_path = tmp_path / "nested" / "dir" / "test.txt"
        write_yolo_label(label_path, [])
        assert label_path.is_file()


# ---------------------------------------------------------------------------
# write_yolo_split tests
# ---------------------------------------------------------------------------


class TestWriteYoloSplit:
    def test_write_split(self, tmp_path, sample_records):
        write_yolo_split(sample_records, tmp_path, "train")

        images_dir = tmp_path / "train" / "images"
        labels_dir = tmp_path / "train" / "labels"
        assert images_dir.is_dir()
        assert labels_dir.is_dir()

        image_files = list(images_dir.glob("*.jpg"))
        label_files = list(labels_dir.glob("*.txt"))
        assert len(image_files) == 3
        assert len(label_files) == 3

    def test_negative_label_is_empty(self, tmp_path, sample_records):
        write_yolo_split(sample_records, tmp_path, "train")

        # The third record is negative
        label_files = sorted((tmp_path / "train" / "labels").glob("*.txt"))
        # Find the empty one
        empty_count = sum(1 for f in label_files if f.read_text().strip() == "")
        assert empty_count == 1

    def test_images_are_valid(self, tmp_path, sample_records):
        write_yolo_split(sample_records, tmp_path, "valid")

        for img_path in (tmp_path / "valid" / "images").glob("*.jpg"):
            img = cv2.imread(str(img_path))
            assert img is not None
            assert img.shape == (200, 300, 3)


# ---------------------------------------------------------------------------
# write_data_yaml tests
# ---------------------------------------------------------------------------


class TestWriteDataYaml:
    def test_write_data_yaml(self, tmp_path):
        yaml_path = write_data_yaml(
            tmp_path, ["Scratch", "Stain"], ["train", "valid"]
        )
        assert yaml_path.is_file()

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        assert data["nc"] == 2
        assert data["names"] == ["Scratch", "Stain"]
        assert data["train"] == "train/images"
        assert data["val"] == "valid/images"

    def test_single_class(self, tmp_path):
        yaml_path = write_data_yaml(tmp_path, ["Scratch"], ["train"])
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        assert data["nc"] == 1


# ---------------------------------------------------------------------------
# write_yolo_dataset tests
# ---------------------------------------------------------------------------


class TestWriteYoloDataset:
    def test_write_complete_dataset(self, tmp_path, sample_records):
        records_by_split = {
            "train": sample_records[:2],
            "valid": sample_records[2:],
        }
        yaml_path = write_yolo_dataset(
            records_by_split, tmp_path, ["Scratch"]
        )

        assert yaml_path.is_file()
        assert (tmp_path / "train" / "images").is_dir()
        assert (tmp_path / "train" / "labels").is_dir()
        assert (tmp_path / "valid" / "images").is_dir()
        assert (tmp_path / "valid" / "labels").is_dir()

    def test_round_trip_write_read(self, tmp_path, sample_records):
        """Write dataset, then read it back and verify bboxes match."""
        records_by_split = {
            "train": sample_records[:2],
            "valid": sample_records[2:],
        }
        yaml_path = write_yolo_dataset(
            records_by_split, tmp_path, ["Scratch"]
        )

        # Read back
        from synthdet.analysis.loader import load_yolo_dataset
        loaded = load_yolo_dataset(yaml_path)

        assert len(loaded.train) == 2
        assert len(loaded.valid) == 1

        # Check train bboxes
        for original, loaded_rec in zip(sample_records[:2], sorted(loaded.train, key=lambda r: r.image_path)):
            assert len(loaded_rec.bboxes) == len(original.bboxes)
            for ob, lb in zip(original.bboxes, loaded_rec.bboxes):
                assert ob.class_id == lb.class_id
                assert ob.x_center == pytest.approx(lb.x_center, abs=1e-4)
                assert ob.y_center == pytest.approx(lb.y_center, abs=1e-4)
                assert ob.width == pytest.approx(lb.width, abs=1e-4)
                assert ob.height == pytest.approx(lb.height, abs=1e-4)

        # Valid record should be negative
        assert loaded.valid[0].is_negative
