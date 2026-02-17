"""Tests for synthdet.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from synthdet.config import AnalysisConfig, QualityMonitoringConfig, SynthDetConfig


class TestAnalysisConfig:
    def test_defaults(self):
        cfg = AnalysisConfig()
        assert cfg.negative_ratio == 0.15
        assert cfg.multiplier == 5.0
        assert cfg.min_per_bucket == 50
        assert cfg.min_per_region == 30

    def test_custom(self):
        cfg = AnalysisConfig(multiplier=10.0, min_per_bucket=100)
        assert cfg.multiplier == 10.0
        assert cfg.min_per_bucket == 100


class TestQualityMonitoringConfig:
    def test_defaults(self):
        cfg = QualityMonitoringConfig()
        assert cfg.control_limit_sigma == 3.0
        assert len(cfg.activation_layers) > 0
        assert cfg.trend_window == 7


class TestSynthDetConfig:
    def test_default(self):
        cfg = SynthDetConfig.default()
        assert isinstance(cfg.analysis, AnalysisConfig)
        assert isinstance(cfg.quality_monitoring, QualityMonitoringConfig)

    def test_from_yaml(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            "analysis:\n  multiplier: 10.0\n  min_per_bucket: 100\n"
        )
        cfg = SynthDetConfig.from_yaml(yaml_path)
        assert cfg.analysis.multiplier == 10.0
        assert cfg.analysis.min_per_bucket == 100
        # Other fields keep defaults
        assert cfg.analysis.negative_ratio == 0.15

    def test_from_empty_yaml(self, tmp_path):
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")
        cfg = SynthDetConfig.from_yaml(yaml_path)
        assert cfg.analysis.multiplier == 5.0
