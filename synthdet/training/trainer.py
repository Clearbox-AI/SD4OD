"""YOLO training wrapper with lazy ultralytics import.

Wraps ``ultralytics.YOLO.train()`` with SynthDet configuration,
result parsing, and iteration-aware experiment naming.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from synthdet.config import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Outcome of a YOLO training run."""

    best_weights: Path
    last_weights: Path
    epochs_completed: int
    best_map50: float
    best_map50_95: float
    training_time_seconds: float
    metrics_history: list[dict[str, float]]
    project_dir: Path


def _parse_metrics_csv(csv_path: Path) -> list[dict[str, float]]:
    """Parse ultralytics results.csv into a list of per-epoch metric dicts."""
    if not csv_path.is_file():
        return []
    rows: list[dict[str, float]] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, float] = {}
            for k, v in row.items():
                k = k.strip()
                try:
                    parsed[k] = float(v.strip())
                except (ValueError, AttributeError):
                    continue
            if parsed:
                rows.append(parsed)
    return rows


class YOLOTrainer:
    """Wraps ultralytics YOLO training with SynthDet configuration."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self._model = None

    def _resolve_device(self) -> str | None:
        """Resolve 'auto' device to None (let ultralytics decide)."""
        if self.config.device == "auto":
            return None
        return self.config.device

    def _ensure_model(self, weights: str | None = None) -> object:
        """Lazy-load the YOLO model.

        Args:
            weights: Path to weights. Falls back to ``config.model_arch``.

        Returns:
            The loaded YOLO model instance.

        Raises:
            ImportError: If ultralytics is not installed.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for training. "
                "Install with: pip install ultralytics"
            ) from exc

        model_path = weights or self.config.model_arch
        self._model = YOLO(model_path)
        return self._model

    def train(
        self,
        data_yaml: Path,
        iteration: int = 0,
        weights: str | None = None,
    ) -> TrainingResult:
        """Train a YOLO model on the given dataset.

        Args:
            data_yaml: Path to YOLO ``data.yaml``.
            iteration: Active learning iteration number (affects experiment name).
            weights: Optional weights path for warm-start. Defaults to config.model_arch.

        Returns:
            TrainingResult with paths to weights and training metrics.
        """
        model = self._ensure_model(weights)

        name = f"{self.config.name}_iter{iteration}" if iteration > 0 else self.config.name
        device = self._resolve_device()

        train_kwargs: dict = {
            "data": str(data_yaml),
            "epochs": self.config.epochs,
            "batch": self.config.batch_size,
            "imgsz": self.config.imgsz,
            "patience": self.config.patience,
            "lr0": self.config.lr0,
            "optimizer": self.config.optimizer,
            "workers": self.config.workers,
            "project": self.config.project,
            "name": name,
            "verbose": self.config.verbose,
            "pretrained": self.config.pretrained,
            "exist_ok": True,
        }
        if device is not None:
            train_kwargs["device"] = device
        if self.config.freeze_layers > 0:
            train_kwargs["freeze"] = self.config.freeze_layers
        if self.config.resume and weights:
            train_kwargs["resume"] = True

        logger.info("Starting YOLO training: %s (iteration %d)", name, iteration)
        t0 = time.monotonic()
        results = model.train(**train_kwargs)  # type: ignore[attr-defined]
        elapsed = time.monotonic() - t0

        # Locate output directory
        project_dir = Path(self.config.project) / name
        best_weights = project_dir / "weights" / "best.pt"
        last_weights = project_dir / "weights" / "last.pt"
        csv_path = project_dir / "results.csv"

        metrics_history = _parse_metrics_csv(csv_path)
        epochs_completed = len(metrics_history)

        # Extract best mAP from results or metrics history
        best_map50 = 0.0
        best_map50_95 = 0.0
        if hasattr(results, "box"):
            best_map50 = float(getattr(results.box, "map50", 0.0))
            best_map50_95 = float(getattr(results.box, "map", 0.0))
        elif metrics_history:
            # Fallback: scan CSV for best metrics/mAP50(B)
            for row in metrics_history:
                m50 = row.get("metrics/mAP50(B)", 0.0)
                m50_95 = row.get("metrics/mAP50-95(B)", 0.0)
                if m50 > best_map50:
                    best_map50 = m50
                    best_map50_95 = m50_95

        logger.info(
            "Training complete: %d epochs, mAP50=%.3f in %.1fs",
            epochs_completed, best_map50, elapsed,
        )

        return TrainingResult(
            best_weights=best_weights,
            last_weights=last_weights,
            epochs_completed=epochs_completed,
            best_map50=best_map50,
            best_map50_95=best_map50_95,
            training_time_seconds=elapsed,
            metrics_history=metrics_history,
            project_dir=project_dir,
        )
