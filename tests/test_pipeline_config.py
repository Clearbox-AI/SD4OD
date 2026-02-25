"""Tests for pipeline configuration schema."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from synthdet.config import SynthDetConfig
from synthdet.pipeline.config_schema import VALID_METHODS, PipelineConfig


class TestPipelineConfig:
    def test_default_construction(self):
        cfg = PipelineConfig()
        assert cfg.methods == ["compositor"]
        assert cfg.train_split_ratio == 0.85
        assert cfg.validate_output is True
        assert cfg.augment is False
        assert cfg.dry_run is False
        assert isinstance(cfg.synthdet, SynthDetConfig)

    def test_custom_methods(self):
        cfg = PipelineConfig(methods=["compositor", "inpainting"])
        assert cfg.methods == ["compositor", "inpainting"]

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            PipelineConfig(methods=["nonexistent"])

    def test_all_valid_methods_accepted(self):
        cfg = PipelineConfig(methods=list(VALID_METHODS))
        assert set(cfg.methods) == VALID_METHODS

    def test_invalid_split_ratio(self):
        with pytest.raises(ValueError, match="train_split_ratio"):
            PipelineConfig(train_split_ratio=0.0)
        with pytest.raises(ValueError, match="train_split_ratio"):
            PipelineConfig(train_split_ratio=1.0)

    def test_from_yaml(self, tmp_path):
        cfg_path = tmp_path / "pipeline.yaml"
        cfg_path.write_text(yaml.dump({
            "methods": ["compositor", "inpainting"],
            "train_split_ratio": 0.9,
            "augment": True,
            "analysis": {"multiplier": 3.0},
        }))

        cfg = PipelineConfig.from_yaml(cfg_path)
        assert cfg.methods == ["compositor", "inpainting"]
        assert cfg.train_split_ratio == 0.9
        assert cfg.augment is True
        assert cfg.synthdet.analysis.multiplier == 3.0

    def test_from_yaml_empty(self, tmp_path):
        cfg_path = tmp_path / "empty.yaml"
        cfg_path.write_text("")
        cfg = PipelineConfig.from_yaml(cfg_path)
        assert cfg.methods == ["compositor"]

    def test_from_yaml_synthdet_only(self, tmp_path):
        cfg_path = tmp_path / "synthdet_only.yaml"
        cfg_path.write_text(yaml.dump({
            "compositor": {"max_defects_per_image": 2},
        }))
        cfg = PipelineConfig.from_yaml(cfg_path)
        assert cfg.synthdet.compositor.max_defects_per_image == 2
        assert cfg.methods == ["compositor"]
