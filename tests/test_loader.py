"""Tests for synthdet.analysis.loader."""

from __future__ import annotations

import pytest

from synthdet.analysis.loader import load_yolo_dataset, parse_data_yaml
from tests.conftest import has_real_dataset


class TestParseDataYaml:
    def test_parses_mini(self, mini_dataset_path):
        data = parse_data_yaml(mini_dataset_path)
        assert data["nc"] == 2
        assert data["names"] == ["Scratch", "Stain"]
        # Paths should be resolved to absolute
        assert "/" in data["train"]

    def test_resolves_relative_paths(self, mini_dataset_path):
        data = parse_data_yaml(mini_dataset_path)
        from pathlib import Path
        assert Path(data["train"]).is_dir()


class TestLoadYoloDataset:
    def test_mini_dataset(self, mini_dataset):
        assert len(mini_dataset.train) == 2
        assert len(mini_dataset.valid) == 1
        assert len(mini_dataset.class_names) == 2
        assert mini_dataset.class_names[0] == "Scratch"
        assert mini_dataset.class_names[1] == "Stain"

    def test_mini_annotations(self, mini_dataset):
        # train image 1: 2 bboxes, train image 2: 1 bbox, valid image: 1 bbox
        total = len(mini_dataset.all_bboxes())
        assert total == 4

    def test_mini_bbox_values(self, mini_dataset):
        # First train image should have bbox at center
        train_bboxes = mini_dataset.all_bboxes("train")
        assert any(
            b.x_center == pytest.approx(0.5) and b.y_center == pytest.approx(0.5)
            for b in train_bboxes
        )

    @pytest.mark.integration
    @pytest.mark.skipif(not has_real_dataset(), reason="Real dataset not available")
    def test_real_dataset(self, real_dataset_path):
        dataset = load_yolo_dataset(real_dataset_path)
        assert len(dataset.train) == 104
        assert len(dataset.valid) == 7
        assert len(dataset.test) == 5
        assert dataset.class_names == ["Scratch"]
        total_annotations = len(dataset.all_bboxes())
        assert total_annotations == 192
