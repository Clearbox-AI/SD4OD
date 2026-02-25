"""Activation distribution capture via PyTorch forward hooks.

Requires ``ultralytics`` (optional ``quality`` extra). The YOLO model is
lazy-loaded on first use so importing this module is always safe.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from synthdet.config import QualityMonitoringConfig
from synthdet.types import ActivationDistributionSnapshot

# Default mapping from human-readable layer names to ultralytics YOLO layer paths.
DEFAULT_LAYER_MAPPING: dict[str, str] = {
    "backbone.stage3": "model.model.4",
    "backbone.stage4": "model.model.6",
    "neck.fpn": "model.model.12",
}


class ActivationCapturer:
    """Captures activation distributions from a YOLO detection backbone.

    Processes images in sub-batches and records per-sub-batch mean activation
    statistics for each monitored layer. These means form the distribution
    that Shewhart control charts monitor for drift.
    """

    def __init__(
        self,
        model_path: str | None = None,
        config: QualityMonitoringConfig | None = None,
        device: str = "auto",
        layer_mapping: dict[str, str] | None = None,
    ) -> None:
        self.model_path = model_path
        self.config = config or QualityMonitoringConfig()
        self.device = self._resolve_device(device)
        self.layer_mapping = layer_mapping or dict(DEFAULT_LAYER_MAPPING)
        self._model: Any = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_model(self) -> None:
        """Lazy-load the YOLO model from ``ultralytics``."""
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for activation capture. "
                "Install it with: pip install 'synthdet[quality]'"
            ) from exc
        path = self.model_path or "yolov8n.pt"
        self._model = YOLO(path)
        self._model.to(self.device)

    def _resolve_layer(self, layer_name: str) -> Any:
        """Walk dotted path to find the ``nn.Module`` inside the YOLO model."""
        import torch.nn as nn

        path = self.layer_mapping.get(layer_name, layer_name)
        module: nn.Module = self._model.model
        for attr in path.split("."):
            if attr.isdigit():
                module = module[int(attr)]  # type: ignore[index]
            else:
                module = getattr(module, attr)
        return module

    def _register_hooks(self, storage: dict[str, list]) -> list:
        """Register forward hooks and return handles for cleanup."""
        handles = []
        for layer_name in self.config.activation_layers:
            module = self._resolve_layer(layer_name)

            def _make_hook(name: str):
                def hook(_mod, _inp, output):
                    if hasattr(output, "detach"):
                        storage[name].append(output.detach().cpu().numpy())
                return hook

            h = module.register_forward_hook(_make_hook(layer_name))
            handles.append(h)
        return handles

    def capture(
        self,
        images: list[np.ndarray],
        batch_size: int = 8,
    ) -> list[ActivationDistributionSnapshot]:
        """Run images through the model and capture activation snapshots.

        Each snapshot records per-layer statistics over one sub-batch. Multiple
        sub-batches produce a distribution of means for the control chart.

        Returns one ``ActivationDistributionSnapshot`` per monitored layer,
        with ``mean`` and ``std`` computed across all sub-batch means.
        """
        self._ensure_model()

        # Storage: layer_name -> list of per-sub-batch mean values
        per_layer_means: dict[str, list[float]] = {
            name: [] for name in self.config.activation_layers
        }

        for start in range(0, len(images), batch_size):
            sub_batch = images[start : start + batch_size]
            hook_storage: dict[str, list] = {
                name: [] for name in self.config.activation_layers
            }
            handles = self._register_hooks(hook_storage)
            try:
                # Run inference (suppress output)
                self._model.predict(sub_batch, verbose=False)
            finally:
                for h in handles:
                    h.remove()

            # Compute sub-batch mean for each layer
            for layer_name in self.config.activation_layers:
                activations = hook_storage[layer_name]
                if activations:
                    batch_mean = float(np.mean([np.mean(a) for a in activations]))
                    per_layer_means[layer_name].append(batch_mean)

        # Build snapshots
        snapshots: list[ActivationDistributionSnapshot] = []
        timestamp = datetime.now(timezone.utc).isoformat()
        for layer_name in self.config.activation_layers:
            snapshots.append(
                self._compute_snapshot(
                    layer_name, per_layer_means[layer_name], len(images), timestamp
                )
            )
        return snapshots

    def _compute_snapshot(
        self,
        layer_name: str,
        sub_batch_means: list[float],
        num_images: int,
        timestamp: str,
    ) -> ActivationDistributionSnapshot:
        """Build a snapshot from sub-batch mean activations."""
        arr = np.array(sub_batch_means) if sub_batch_means else np.array([0.0])
        percentiles = {
            p: float(np.percentile(arr, p)) for p in self.config.snapshot_percentiles
        }
        return ActivationDistributionSnapshot(
            layer_name=layer_name,
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            percentiles=percentiles,
            num_images=num_images,
            timestamp=timestamp,
        )
