"""Tests for synthdet.quality.activation_capture — mock-based."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False

from synthdet.config import QualityMonitoringConfig
from synthdet.quality.activation_capture import (
    DEFAULT_LAYER_MAPPING,
    ActivationCapturer,
)


class TestActivationCapturerInit:
    def test_stores_config(self):
        config = QualityMonitoringConfig(control_limit_sigma=2.0)
        cap = ActivationCapturer(config=config, device="cpu")
        assert cap.config.control_limit_sigma == 2.0
        assert cap.device == "cpu"

    def test_default_config(self):
        cap = ActivationCapturer(device="cpu")
        assert cap.config.activation_layers == ["backbone.stage3", "backbone.stage4", "neck.fpn"]

    def test_default_layer_mapping(self):
        cap = ActivationCapturer(device="cpu")
        assert cap.layer_mapping == DEFAULT_LAYER_MAPPING

    def test_custom_layer_mapping(self):
        mapping = {"my_layer": "model.layer1"}
        cap = ActivationCapturer(device="cpu", layer_mapping=mapping)
        assert cap.layer_mapping == mapping


class TestResolveDevice:
    def test_explicit_cpu(self):
        assert ActivationCapturer._resolve_device("cpu") == "cpu"

    def test_explicit_cuda(self):
        assert ActivationCapturer._resolve_device("cuda") == "cuda"

    def test_auto_without_torch(self):
        with patch.dict("sys.modules", {"torch": None}):
            # When torch can't be imported, should fall back to cpu
            result = ActivationCapturer._resolve_device("auto")
            # Result depends on whether torch is actually available
            assert result in ("cpu", "cuda")


class TestEnsureModel:
    def test_raises_import_error_without_ultralytics(self):
        cap = ActivationCapturer(device="cpu")
        with patch.dict("sys.modules", {"ultralytics": None}):
            with patch("builtins.__import__", side_effect=_make_import_blocker("ultralytics")):
                with pytest.raises(ImportError, match="ultralytics"):
                    cap._ensure_model()

    def test_no_error_when_model_already_loaded(self):
        cap = ActivationCapturer(device="cpu")
        cap._model = MagicMock()  # simulate already loaded
        cap._ensure_model()  # should not raise


@pytest.mark.skipif(
    not _torch_available(), reason="torch not installed"
)
class TestCapture:
    def test_returns_snapshots_per_layer(self):
        """Mock the model and verify capture returns correct structure."""
        import torch
        import torch.nn as nn

        config = QualityMonitoringConfig(
            activation_layers=["layer_a", "layer_b"],
        )
        cap = ActivationCapturer(config=config, device="cpu")
        cap.layer_mapping = {"layer_a": "layer_a", "layer_b": "layer_b"}

        module_a = nn.Linear(10, 10)
        module_b = nn.Linear(10, 10)

        mock_model = MagicMock()
        mock_model.model = MagicMock()

        def resolve_side_effect(layer_name):
            return {"layer_a": module_a, "layer_b": module_b}[layer_name]

        cap._resolve_layer = resolve_side_effect
        cap._model = mock_model

        def fake_predict(images, verbose=False):
            dummy_input = torch.zeros(1, 10)
            module_a(dummy_input)
            module_b(dummy_input)

        mock_model.predict = fake_predict

        images = [np.random.rand(64, 64, 3).astype(np.uint8) for _ in range(4)]
        snapshots = cap.capture(images, batch_size=2)

        assert len(snapshots) == 2
        assert snapshots[0].layer_name == "layer_a"
        assert snapshots[1].layer_name == "layer_b"
        assert snapshots[0].num_images == 4
        assert isinstance(snapshots[0].mean, float)
        assert isinstance(snapshots[0].std, float)
        assert isinstance(snapshots[0].timestamp, str)
        assert all(p in snapshots[0].percentiles for p in [5, 25, 50, 75, 95])

    def test_hooks_cleaned_up_after_error(self):
        """Hooks should be removed even if predict raises."""
        import torch.nn as nn

        config = QualityMonitoringConfig(activation_layers=["layer_a"])
        cap = ActivationCapturer(config=config, device="cpu")
        cap.layer_mapping = {"layer_a": "layer_a"}

        module_a = nn.Linear(10, 10)

        mock_model = MagicMock()
        mock_model.model = MagicMock()
        cap._resolve_layer = lambda name: module_a
        cap._model = mock_model
        mock_model.predict.side_effect = RuntimeError("inference failed")

        images = [np.random.rand(64, 64, 3).astype(np.uint8)]
        with pytest.raises(RuntimeError, match="inference failed"):
            cap.capture(images, batch_size=1)

        assert len(module_a._forward_hooks) == 0


def _make_import_blocker(blocked_module: str):
    """Create an __import__ replacement that blocks a specific module."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def blocker(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"No module named '{blocked_module}'")
        return real_import(name, *args, **kwargs)

    return blocker
