"""Tests for synthdet.quality.monitor — mock ActivationCapturer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from synthdet.config import QualityMonitoringConfig
from synthdet.quality.monitor import QualityMonitor
from synthdet.types import (
    ActivationDistributionSnapshot,
    QualityMetrics,
)


def _make_snapshot(
    layer_name: str,
    mean: float = 1.0,
    std: float = 0.1,
    num_images: int = 10,
) -> ActivationDistributionSnapshot:
    """Helper to create a test snapshot."""
    return ActivationDistributionSnapshot(
        layer_name=layer_name,
        mean=mean,
        std=std,
        percentiles={5: mean - 0.2, 25: mean - 0.1, 50: mean, 75: mean + 0.1, 95: mean + 0.2},
        num_images=num_images,
        timestamp="2026-01-01T00:00:00Z",
    )


class TestHasBaseline:
    def test_false_initially(self):
        mon = QualityMonitor(device="cpu")
        assert mon.has_baseline is False

    def test_true_after_establish(self):
        mon = QualityMonitor(device="cpu")
        snapshots = [_make_snapshot("backbone.stage3")]
        with patch.object(mon._capturer, "capture", return_value=snapshots):
            mon.establish_baseline([np.zeros((64, 64, 3))])
        assert mon.has_baseline is True


class TestEstablishBaseline:
    def test_stores_baseline_from_snapshots(self):
        mon = QualityMonitor(device="cpu")
        snapshots = [
            _make_snapshot("layer_a", mean=1.0),
            _make_snapshot("layer_b", mean=2.0),
        ]
        with patch.object(mon._capturer, "capture", return_value=snapshots):
            result = mon.establish_baseline([np.zeros((64, 64, 3))])
        assert len(result) == 2
        assert mon.has_baseline
        assert "layer_a" in mon._baseline
        assert "layer_b" in mon._baseline


class TestMonitorBatch:
    def test_raises_without_baseline(self):
        mon = QualityMonitor(device="cpu")
        with pytest.raises(RuntimeError, match="No baseline"):
            mon.monitor_batch([np.zeros((64, 64, 3))], "batch_001")

    def test_no_alerts_when_matching(self):
        mon = QualityMonitor(device="cpu")
        # Baseline with wide spread so new values easily fit
        mon._establish_baseline_from_means({
            "layer_a": [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 0.6],
        })

        # New snapshot whose percentile values all fall within baseline's 3σ
        new_snap = ActivationDistributionSnapshot(
            layer_name="layer_a",
            mean=1.0,
            std=0.1,
            percentiles={5: 0.9, 25: 0.95, 50: 1.0, 75: 1.05, 95: 1.1},
            num_images=10,
            timestamp="2026-01-01T00:00:00Z",
        )
        with patch.object(mon._capturer, "capture", return_value=[new_snap]):
            metrics = mon.monitor_batch([np.zeros((64, 64, 3))], "batch_001")

        assert isinstance(metrics, QualityMetrics)
        assert metrics.batch_id == "batch_001"
        assert metrics.alerts == []

    def test_alerts_when_drifted(self):
        mon = QualityMonitor(device="cpu")
        # Tight baseline around 1.0
        mon._establish_baseline_from_means({
            "layer_a": [1.0, 1.01, 0.99, 1.0, 1.005, 0.995, 1.0, 1.002, 0.998, 1.0],
        })

        # New snapshot way off baseline
        drifted_snap = _make_snapshot("layer_a", mean=50.0)
        # Override percentiles to be far from baseline
        drifted_snap = ActivationDistributionSnapshot(
            layer_name="layer_a",
            mean=50.0,
            std=5.0,
            percentiles={5: 45.0, 25: 47.0, 50: 50.0, 75: 53.0, 95: 55.0},
            num_images=10,
            timestamp="2026-01-01T00:00:00Z",
        )
        with patch.object(mon._capturer, "capture", return_value=[drifted_snap]):
            metrics = mon.monitor_batch([np.zeros((64, 64, 3))], "batch_002")

        assert len(metrics.alerts) > 0
        assert "layer_a" in metrics.alerts[0]

    def test_control_charts_populated(self):
        mon = QualityMonitor(device="cpu")
        mon._establish_baseline_from_means({
            "layer_a": [1.0, 1.1, 0.9],
        })
        new_snap = _make_snapshot("layer_a", mean=1.0)
        with patch.object(mon._capturer, "capture", return_value=[new_snap]):
            metrics = mon.monitor_batch([np.zeros((64, 64, 3))], "b1")

        assert len(metrics.control_charts) == 1
        chart = metrics.control_charts[0]
        assert chart.metric_name == "layer_a_mean"
        assert chart.center_line is not None
        assert chart.ucl > chart.lcl


class TestSaveLoadBaseline:
    def test_roundtrip(self):
        mon = QualityMonitor(device="cpu")
        mon._establish_baseline_from_means({
            "layer_a": [1.0, 1.1, 0.9],
            "layer_b": [2.0, 2.1, 1.9],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline.json"
            mon.save_baseline(path)
            assert path.exists()

            mon2 = QualityMonitor(device="cpu")
            assert not mon2.has_baseline
            mon2.load_baseline(path)
            assert mon2.has_baseline
            assert mon2._baseline["layer_a"] == [1.0, 1.1, 0.9]
            assert mon2._baseline["layer_b"] == [2.0, 2.1, 1.9]

    def test_save_without_baseline_raises(self):
        mon = QualityMonitor(device="cpu")
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="No baseline"):
                mon.save_baseline(Path(tmpdir) / "baseline.json")


class TestStrategyIntegration:
    """Verify that QualityMetrics with alerts → ActiveLearningSignal(source='spc_alert')."""

    def test_spc_alert_signal_created(self):
        from synthdet.analysis.strategy import generate_synthesis_strategy
        from synthdet.types import (
            BBoxSizeBucket,
            ClassDistribution,
            Dataset,
            DatasetStatistics,
            ImageRecord,
            QualityControlChart,
            SpatialRegion,
        )

        # Minimal dataset and stats
        dataset = Dataset(
            root=Path("/fake"),
            class_names=["Scratch"],
            train=[ImageRecord(Path("/fake/img.jpg"), [])],
            valid=[],
            test=[],
        )
        stats = DatasetStatistics(
            total_images=1,
            total_annotations=0,
            negative_images=1,
            negative_ratio=1.0,
            unique_source_images=1,
            split_image_counts={"train": 1, "valid": 0, "test": 0},
            split_annotation_counts={"train": 0, "valid": 0, "test": 0},
            class_distributions=[
                ClassDistribution(0, "Scratch", 0, 0.0, None, {}, {}),
            ],
            overall_size_stats=None,
            overall_bucket_counts={b: 0 for b in BBoxSizeBucket},
            overall_region_counts={r: 0 for r in SpatialRegion},
            bucket_uniformity=0.0,
            region_uniformity=0.0,
            annotations_per_image_mean=0.0,
            annotations_per_image_std=0.0,
            annotations_per_image_max=0,
            annotations_per_image_histogram={0: 1},
            aspect_ratio_bins=[],
            image_width=640,
            image_height=480,
        )

        # QualityMetrics with alerts
        qm = QualityMetrics(
            batch_id="test",
            activation_snapshots=[],
            control_charts=[
                QualityControlChart(
                    metric_name="test",
                    center_line=1.0,
                    ucl=2.0,
                    lcl=0.0,
                    sigma=0.33,
                    values=[3.0],
                    out_of_control_indices=[0],
                ),
            ],
            alerts=["test_layer: 1 out-of-control point(s)"],
        )

        strategy = generate_synthesis_strategy(
            dataset, stats, quality_metrics=qm
        )

        spc_signals = [s for s in strategy.active_learning_signals if s.source == "spc_alert"]
        assert len(spc_signals) == 1
        assert spc_signals[0].priority == 0.95
        assert "SPC alert" in spc_signals[0].rationale
