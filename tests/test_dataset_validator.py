"""Tests for the YOLO dataset validator."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from synthdet.pipeline.validator import (
    ValidationIssue,
    ValidationReport,
    validate_bbox_line,
    validate_dataset,
    validate_split_balance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_dataset(tmp_path) -> Path:
    """Create a minimal valid YOLO dataset on disk."""
    ds = tmp_path / "dataset"
    for split in ("train", "valid"):
        (ds / split / "images").mkdir(parents=True)
        (ds / split / "labels").mkdir(parents=True)

    # Train: 3 images with labels
    for i in range(3):
        img = np.full((100, 150, 3), 180, dtype=np.uint8)
        cv2.imwrite(str(ds / "train" / "images" / f"img_{i}.jpg"), img)
        (ds / "train" / "labels" / f"img_{i}.txt").write_text(
            f"0 0.5 0.5 0.{i + 1} 0.{i + 1}\n"
        )

    # Valid: 1 image with label
    img = np.full((100, 150, 3), 180, dtype=np.uint8)
    cv2.imwrite(str(ds / "valid" / "images" / "val_0.jpg"), img)
    (ds / "valid" / "labels" / "val_0.txt").write_text("0 0.5 0.5 0.2 0.3\n")

    # data.yaml
    (ds / "data.yaml").write_text(yaml.dump({
        "names": ["Scratch"],
        "nc": 1,
        "train": "train/images",
        "val": "valid/images",
    }))

    return ds


# ---------------------------------------------------------------------------
# validate_bbox_line
# ---------------------------------------------------------------------------


class TestValidateBBoxLine:
    def test_valid_line(self):
        issues = validate_bbox_line("0 0.5 0.5 0.1 0.1", "test.txt")
        assert issues == []

    def test_empty_line(self):
        assert validate_bbox_line("", "test.txt") == []
        assert validate_bbox_line("  ", "test.txt") == []

    def test_wrong_field_count(self):
        issues = validate_bbox_line("0 0.5 0.5", "test.txt")
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert "5 fields" in issues[0].message

    def test_non_integer_class_id(self):
        issues = validate_bbox_line("abc 0.5 0.5 0.1 0.1", "test.txt")
        assert len(issues) == 1
        assert "Non-integer" in issues[0].message

    def test_negative_class_id(self):
        issues = validate_bbox_line("-1 0.5 0.5 0.1 0.1", "test.txt")
        assert any("Negative class_id" in i.message for i in issues)

    def test_coord_out_of_range(self):
        issues = validate_bbox_line("0 1.5 0.5 0.1 0.1", "test.txt")
        assert any("outside [0, 1]" in i.message for i in issues)

    def test_negative_width(self):
        issues = validate_bbox_line("0 0.5 0.5 -0.1 0.1", "test.txt")
        assert any("outside [0, 1]" in i.message for i in issues)

    def test_zero_width(self):
        issues = validate_bbox_line("0 0.5 0.5 0.0 0.1", "test.txt")
        assert any("must be positive" in i.message for i in issues)

    def test_non_numeric_coord(self):
        issues = validate_bbox_line("0 abc 0.5 0.1 0.1", "test.txt")
        assert any("Non-numeric" in i.message for i in issues)


# ---------------------------------------------------------------------------
# validate_split_balance
# ---------------------------------------------------------------------------


class TestValidateSplitBalance:
    def test_balanced_splits(self):
        issues = validate_split_balance({"train": 85, "valid": 15})
        assert issues == []

    def test_empty_dataset(self):
        issues = validate_split_balance({"train": 0, "valid": 0})
        assert len(issues) == 1
        assert "empty" in issues[0].message.lower()

    def test_very_small_split(self):
        issues = validate_split_balance({"train": 99, "valid": 1})
        # valid is 1%, train is 99%
        assert any("valid" in i.message for i in issues)

    def test_single_split_no_imbalance_warning(self):
        # A single split should not trigger the "very imbalanced" warning
        issues = validate_split_balance({"train": 100})
        assert not any("imbalanced" in i.message.lower() for i in issues)


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


class TestValidationReport:
    def test_is_valid_no_issues(self):
        report = ValidationReport(total_images=10, total_labels=10)
        assert report.is_valid
        assert report.errors == []
        assert report.warnings == []

    def test_is_valid_with_warnings(self):
        report = ValidationReport(issues=[
            ValidationIssue("warning", "balance", None, "test warning"),
        ])
        assert report.is_valid

    def test_is_invalid_with_errors(self):
        report = ValidationReport(issues=[
            ValidationIssue("error", "bbox", "f.txt", "bad bbox"),
        ])
        assert not report.is_valid
        assert len(report.errors) == 1

    def test_summary(self):
        report = ValidationReport(total_images=5, total_labels=5)
        s = report.summary()
        assert "VALID" in s
        assert "Images: 5" in s


# ---------------------------------------------------------------------------
# validate_dataset (integration)
# ---------------------------------------------------------------------------


class TestValidateDataset:
    def test_valid_dataset(self, valid_dataset):
        report = validate_dataset(valid_dataset)
        assert report.is_valid
        assert report.total_images == 4
        assert report.total_labels == 4

    def test_missing_data_yaml(self, tmp_path):
        report = validate_dataset(tmp_path)
        assert not report.is_valid
        assert any("data.yaml" in i.message for i in report.errors)

    def test_missing_label(self, valid_dataset):
        # Remove one label file
        (valid_dataset / "train" / "labels" / "img_0.txt").unlink()
        report = validate_dataset(valid_dataset)
        assert any(
            "no corresponding label" in i.message
            for i in report.warnings
        )

    def test_orphan_label(self, valid_dataset):
        # Add a label with no image
        (valid_dataset / "train" / "labels" / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        report = validate_dataset(valid_dataset)
        assert any(
            "no corresponding image" in i.message
            for i in report.warnings
        )

    def test_invalid_bbox_in_label(self, valid_dataset):
        (valid_dataset / "train" / "labels" / "img_0.txt").write_text("0 1.5 0.5 0.1 0.1\n")
        report = validate_dataset(valid_dataset)
        assert not report.is_valid
        assert any("outside [0, 1]" in i.message for i in report.errors)

    def test_empty_label_is_ok(self, valid_dataset):
        """Empty label files represent negative examples — not an error."""
        (valid_dataset / "train" / "labels" / "img_0.txt").write_text("")
        report = validate_dataset(valid_dataset)
        # Should not have bbox errors from the empty file
        bbox_errors = [i for i in report.errors if i.category == "bbox"]
        assert len(bbox_errors) == 0

    def test_check_images(self, valid_dataset):
        report = validate_dataset(valid_dataset, check_images=True)
        assert report.is_valid

    def test_invalid_yaml(self, tmp_path):
        (tmp_path / "data.yaml").write_text("names: [Scratch\n  - broken yaml")
        report = validate_dataset(tmp_path)
        assert not report.is_valid

    def test_missing_names_in_yaml(self, tmp_path):
        (tmp_path / "data.yaml").write_text(yaml.dump({"nc": 1}))
        report = validate_dataset(tmp_path)
        assert not report.is_valid
        assert any("names" in i.message for i in report.errors)
