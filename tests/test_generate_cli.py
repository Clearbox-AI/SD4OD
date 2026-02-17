"""Tests for the generation CLI."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from synthdet.generate.__main__ import main
from synthdet.types import AnnotationSource, BBox, ImageRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mini_gen_dataset(tmp_path) -> Path:
    """Create a minimal dataset for CLI testing."""
    ds_dir = tmp_path / "dataset"
    train_imgs = ds_dir / "train" / "images"
    train_lbls = ds_dir / "train" / "labels"
    valid_imgs = ds_dir / "valid" / "images"
    valid_lbls = ds_dir / "valid" / "labels"

    train_imgs.mkdir(parents=True)
    train_lbls.mkdir(parents=True)
    valid_imgs.mkdir(parents=True)
    valid_lbls.mkdir(parents=True)

    # Create 4 train images (so unique source count is reasonable)
    for i in range(4):
        img = np.full((100, 150, 3), 150 + i * 10, dtype=np.uint8)
        cv2.rectangle(img, (30 + i * 5, 20 + i * 3), (80 + i * 5, 60 + i * 3), (0, 0, 200), -1)
        cv2.imwrite(str(train_imgs / f"train_{i}.jpg"), img)
        bbox = BBox(class_id=0, x_center=0.4, y_center=0.4, width=0.3, height=0.3)
        (train_lbls / f"train_{i}.txt").write_text(bbox.to_yolo_line() + "\n")

    # Create 1 valid image
    img = np.full((100, 150, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (50, 30), (100, 70), (200, 0, 0), -1)
    cv2.imwrite(str(valid_imgs / "val_0.jpg"), img)
    bbox = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.3, height=0.4)
    (valid_lbls / "val_0.txt").write_text(bbox.to_yolo_line() + "\n")

    # Write data.yaml
    data_yaml = ds_dir / "data.yaml"
    data_yaml.write_text(
        "train: train/images\n"
        "val: valid/images\n"
        "nc: 1\n"
        "names: ['Scratch']\n"
    )

    return data_yaml


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateCLI:
    def test_cli_runs_on_mini_dataset(self, mini_gen_dataset, tmp_path):
        output_dir = tmp_path / "output"
        result = main([
            str(mini_gen_dataset),
            "--output", str(output_dir),
            "--seed", "42",
        ])
        assert result == 0

    def test_cli_creates_output_structure(self, mini_gen_dataset, tmp_path):
        output_dir = tmp_path / "output"
        main([
            str(mini_gen_dataset),
            "--output", str(output_dir),
            "--seed", "42",
        ])

        assert (output_dir / "data.yaml").is_file()
        assert (output_dir / "train" / "images").is_dir()
        assert (output_dir / "train" / "labels").is_dir()

        # Check data.yaml is valid
        with open(output_dir / "data.yaml") as f:
            data = yaml.safe_load(f)
        assert data["nc"] == 1
        assert "names" in data

        # Should have generated images
        train_images = list((output_dir / "train" / "images").glob("*"))
        assert len(train_images) > 0

    def test_cli_json_output(self, mini_gen_dataset, tmp_path, capsys):
        output_dir = tmp_path / "output_json"
        result = main([
            str(mini_gen_dataset),
            "--output", str(output_dir),
            "--seed", "42",
            "--json",
        ])
        assert result == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "total_images" in data
        assert "output_dir" in data
        assert "tasks" in data
        assert data["total_images"] > 0

    def test_cli_missing_data_yaml(self, tmp_path):
        result = main([
            str(tmp_path / "nonexistent.yaml"),
            "--output", str(tmp_path / "out"),
        ])
        assert result == 1

    def test_cli_with_augmentation(self, mini_gen_dataset, tmp_path):
        output_dir = tmp_path / "output_aug"
        result = main([
            str(mini_gen_dataset),
            "--output", str(output_dir),
            "--seed", "42",
            "--augment",
        ])
        assert result == 0

        # Should have more images due to augmentation
        train_images = list((output_dir / "train" / "images").glob("*"))
        assert len(train_images) > 0
