"""Tests for synthdet.training.trainer."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from synthdet.config import TrainingConfig
from synthdet.training.trainer import TrainingResult, YOLOTrainer, _parse_metrics_csv


class TestResolveDevice:
    def test_auto_returns_none(self):
        trainer = YOLOTrainer(TrainingConfig(device="auto"))
        assert trainer._resolve_device() is None

    def test_explicit_device(self):
        trainer = YOLOTrainer(TrainingConfig(device="cuda:0"))
        assert trainer._resolve_device() == "cuda:0"

    def test_cpu(self):
        trainer = YOLOTrainer(TrainingConfig(device="cpu"))
        assert trainer._resolve_device() == "cpu"


class TestEnsureModel:
    def test_raises_without_ultralytics(self):
        import sys
        saved = sys.modules.pop("ultralytics", None)
        try:
            with patch.dict("sys.modules", {"ultralytics": None}):
                trainer = YOLOTrainer()
                with pytest.raises(ImportError, match="ultralytics"):
                    trainer._ensure_model()
        finally:
            if saved is not None:
                sys.modules["ultralytics"] = saved

    def test_loads_model(self):
        mock_yolo_cls = MagicMock()
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        mock_module = MagicMock()
        mock_module.YOLO = mock_yolo_cls

        with patch.dict("sys.modules", {"ultralytics": mock_module}):
            trainer = YOLOTrainer(TrainingConfig(model_arch="yolov8s.pt"))
            result = trainer._ensure_model()
            mock_yolo_cls.assert_called_once_with("yolov8s.pt")
            assert result is mock_model

    def test_custom_weights(self):
        mock_yolo_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.YOLO = mock_yolo_cls

        with patch.dict("sys.modules", {"ultralytics": mock_module}):
            trainer = YOLOTrainer()
            trainer._ensure_model(weights="/tmp/custom.pt")
            mock_yolo_cls.assert_called_once_with("/tmp/custom.pt")


class TestParseMetricsCsv:
    def test_empty_file(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        csv_path.write_text("")
        assert _parse_metrics_csv(csv_path) == []

    def test_missing_file(self, tmp_path):
        csv_path = tmp_path / "nonexistent.csv"
        assert _parse_metrics_csv(csv_path) == []

    def test_valid_csv(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)"])
            writer.writerow(["1", "0.45", "0.30"])
            writer.writerow(["2", "0.55", "0.40"])

        rows = _parse_metrics_csv(csv_path)
        assert len(rows) == 2
        assert rows[0]["epoch"] == 1.0
        assert rows[1]["metrics/mAP50(B)"] == pytest.approx(0.55)


class TestTrain:
    def test_train_calls_model(self, tmp_path):
        """train() calls model.train() with correct params and returns TrainingResult."""
        mock_yolo_cls = MagicMock()
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        # Mock training results
        mock_results = MagicMock()
        mock_results.box.map50 = 0.65
        mock_results.box.map = 0.45
        mock_model.train.return_value = mock_results

        mock_module = MagicMock()
        mock_module.YOLO = mock_yolo_cls

        config = TrainingConfig(
            epochs=10,
            batch_size=8,
            project=str(tmp_path / "runs"),
            name="test_run",
        )

        # Create expected output structure
        run_dir = tmp_path / "runs" / "test_run"
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").touch()
        (weights_dir / "last.pt").touch()

        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 1\nnames: ['scratch']")

        with patch.dict("sys.modules", {"ultralytics": mock_module}):
            trainer = YOLOTrainer(config)
            result = trainer.train(data_yaml, iteration=0)

        assert isinstance(result, TrainingResult)
        assert result.best_map50 == 0.65
        assert result.best_map50_95 == 0.45
        mock_model.train.assert_called_once()

        # Check the kwargs passed to model.train
        call_kwargs = mock_model.train.call_args.kwargs
        assert call_kwargs["epochs"] == 10
        assert call_kwargs["batch"] == 8
        assert call_kwargs["data"] == str(data_yaml)

    def test_iteration_naming(self, tmp_path):
        """Iteration > 0 appends _iterN to the experiment name."""
        mock_yolo_cls = MagicMock()
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        mock_results = MagicMock()
        mock_results.box.map50 = 0.5
        mock_results.box.map = 0.3
        mock_model.train.return_value = mock_results

        mock_module = MagicMock()
        mock_module.YOLO = mock_yolo_cls

        config = TrainingConfig(
            project=str(tmp_path / "runs"),
            name="experiment",
        )
        run_dir = tmp_path / "runs" / "experiment_iter2"
        (run_dir / "weights").mkdir(parents=True)

        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 1")

        with patch.dict("sys.modules", {"ultralytics": mock_module}):
            trainer = YOLOTrainer(config)
            trainer.train(data_yaml, iteration=2)

        call_kwargs = mock_model.train.call_args.kwargs
        assert call_kwargs["name"] == "experiment_iter2"
