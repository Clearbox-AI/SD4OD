"""Tests for synthdet.analysis.__main__ CLI."""

from __future__ import annotations

import json

import pytest

from synthdet.analysis.__main__ import main
from tests.conftest import has_real_dataset


class TestCLI:
    def test_mini_dataset_rich(self, mini_dataset_path):
        """CLI runs and exits 0 with rich output."""
        ret = main([str(mini_dataset_path)])
        assert ret == 0

    def test_mini_dataset_json(self, mini_dataset_path, capsys):
        """CLI produces valid JSON output."""
        ret = main([str(mini_dataset_path), "--json"])
        assert ret == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "dataset" in data
        assert "strategy" in data
        assert data["dataset"]["total_images"] >= 2
        assert data["dataset"]["total_annotations"] >= 3

    def test_nonexistent_file(self, tmp_path):
        ret = main([str(tmp_path / "nope.yaml")])
        assert ret == 1

    def test_with_config(self, mini_dataset_path, tmp_path):
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("analysis:\n  multiplier: 3.0\n")
        ret = main([str(mini_dataset_path), "--config", str(config_path)])
        assert ret == 0

    @pytest.mark.integration
    @pytest.mark.skipif(not has_real_dataset(), reason="Real dataset not available")
    def test_real_dataset_json(self, real_dataset_path, capsys):
        ret = main([str(real_dataset_path), "--json"])
        assert ret == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["dataset"]["total_images"] == 116
        assert data["dataset"]["total_annotations"] == 192
        assert data["dataset"]["negative_images"] == 0
