"""Active learning loop: generate → train → evaluate → refine strategy.

Coordinates N iterations of synthetic data generation with model-feedback-driven
targeting. Each iteration trains from the previous iteration's best weights
(warm start) and feeds the ``ModelPerformanceProfile`` back into the strategy
generator.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from synthdet.config import ActiveLearningConfig, SynthDetConfig, TrainingConfig
from synthdet.pipeline.config_schema import PipelineConfig
from synthdet.pipeline.orchestrator import PipelineResult, run_pipeline
from synthdet.training.evaluator import ModelEvaluator
from synthdet.training.trainer import TrainingResult, YOLOTrainer
from synthdet.types import ModelPerformanceProfile, QualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class IterationResult:
    """Outcome of a single active learning iteration."""

    iteration: int
    pipeline_result: PipelineResult
    training_result: TrainingResult
    profile: ModelPerformanceProfile
    quality_metrics: QualityMetrics | None
    map50: float
    map50_improvement: float


@dataclass
class ActiveLearningResult:
    """Outcome of the full active learning loop."""

    iterations: list[IterationResult]
    final_profile: ModelPerformanceProfile
    final_weights: Path
    final_map50: float
    total_training_time_seconds: float
    stopped_reason: str  # "max_iterations" | "convergence" | "no_records"
    total_cost_usd: float


class ActiveLearningLoop:
    """N-iteration coordinator: generate → merge → train → evaluate → refine.

    Args:
        data_yaml: Path to the original YOLO dataset ``data.yaml``.
        output_dir: Root directory for all outputs.
        pipeline_config: Pipeline configuration for generation.
        training_config: YOLO training configuration.
        al_config: Active learning loop configuration.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        data_yaml: Path,
        output_dir: Path,
        pipeline_config: PipelineConfig | None = None,
        training_config: TrainingConfig | None = None,
        al_config: ActiveLearningConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.data_yaml = data_yaml
        self.output_dir = output_dir
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.training_config = training_config or TrainingConfig()
        self.al_config = al_config or ActiveLearningConfig()
        self.seed = seed

        self._trainer = YOLOTrainer(self.training_config)
        self._evaluator = ModelEvaluator(self.al_config)

    def run(self) -> ActiveLearningResult:
        """Execute the active learning loop.

        Returns:
            ActiveLearningResult with all iteration details.
        """
        iterations: list[IterationResult] = []
        model_profile: ModelPerformanceProfile | None = None
        quality_metrics: QualityMetrics | None = None
        current_weights: str | None = None
        prev_map50 = 0.0
        total_cost = 0.0
        total_train_time = 0.0
        consecutive_low_improvement = 0
        stopped_reason = "max_iterations"

        # Track dataset yamls for merging
        all_data_yamls: list[Path] = [self.data_yaml]

        for i in range(self.al_config.max_iterations):
            iter_seed = (self.seed + i) if self.seed is not None else None
            if iter_seed is not None:
                random.seed(iter_seed)
                np.random.seed(iter_seed)

            logger.info("=== Active Learning Iteration %d/%d ===", i + 1, self.al_config.max_iterations)

            # Step 1: Generate synthetic data
            iter_output = self.output_dir / f"iter_{i}"
            pipeline_result = self._run_generation(
                iter_output, model_profile, quality_metrics, iter_seed,
            )

            if pipeline_result.total_records == 0:
                logger.warning("No records generated in iteration %d, stopping.", i)
                stopped_reason = "no_records"
                break

            total_cost += pipeline_result.total_cost_usd

            # Step 2: Merge datasets
            iter_data_yaml = iter_output / "data.yaml"
            if iter_data_yaml.is_file():
                all_data_yamls.append(iter_data_yaml)

            if self.al_config.accumulate_data:
                merged_yaml = self._merge_multiple_datasets(all_data_yamls, self.output_dir / "merged")
            else:
                merged_yaml = self._merge_two_datasets(
                    self.data_yaml, iter_data_yaml, self.output_dir / "merged"
                )

            # Step 3: Train
            training_result = self._trainer.train(
                merged_yaml, iteration=i, weights=current_weights,
            )
            total_train_time += training_result.training_time_seconds
            current_weights = str(training_result.best_weights)

            # Step 4: Evaluate
            from synthdet.analysis.loader import load_yolo_dataset
            merged_dataset = load_yolo_dataset(merged_yaml)
            model_profile = self._evaluator.evaluate(
                training_result.best_weights, merged_yaml, merged_dataset,
            )

            # Step 5: Optional quality monitoring
            quality_metrics = self._run_quality_monitoring(
                merged_dataset, training_result.best_weights, i,
            )

            # Record iteration
            map50 = training_result.best_map50
            improvement = map50 - prev_map50

            iter_result = IterationResult(
                iteration=i,
                pipeline_result=pipeline_result,
                training_result=training_result,
                profile=model_profile,
                quality_metrics=quality_metrics,
                map50=map50,
                map50_improvement=improvement,
            )
            iterations.append(iter_result)

            logger.info(
                "Iteration %d: mAP50=%.3f (improvement=%.3f)",
                i, map50, improvement,
            )

            # Check convergence
            if i > 0 and improvement < self.al_config.convergence_threshold:
                consecutive_low_improvement += 1
                if consecutive_low_improvement >= 2:
                    logger.info("Convergence reached after %d iterations.", i + 1)
                    stopped_reason = "convergence"
                    break
            else:
                consecutive_low_improvement = 0

            prev_map50 = map50

        return self._build_result(iterations, stopped_reason, total_cost, total_train_time)

    def _run_generation(
        self,
        output_dir: Path,
        model_profile: ModelPerformanceProfile | None,
        quality_metrics: QualityMetrics | None,
        seed: int | None,
    ) -> PipelineResult:
        """Run the generation pipeline for one iteration."""
        return run_pipeline(
            self.data_yaml,
            output_dir,
            config=self.pipeline_config,
            seed=seed,
            model_profile=model_profile,
            quality_metrics=quality_metrics,
        )

    def _merge_two_datasets(
        self,
        base_yaml: Path,
        new_yaml: Path,
        output_dir: Path,
    ) -> Path:
        """Merge two YOLO datasets using list-of-paths in data.yaml."""
        return self._merge_multiple_datasets([base_yaml, new_yaml], output_dir)

    def _merge_multiple_datasets(
        self,
        yaml_paths: list[Path],
        output_dir: Path,
    ) -> Path:
        """Merge N YOLO datasets by writing a data.yaml with absolute paths.

        Uses ultralytics' support for lists of paths in train/val fields.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Read the first yaml for class info
        with open(yaml_paths[0]) as f:
            base_data = yaml.safe_load(f) or {}

        train_paths: list[str] = []
        val_paths: list[str] = []

        for yp in yaml_paths:
            with open(yp) as f:
                data = yaml.safe_load(f) or {}
            parent = yp.parent

            # Resolve train paths
            train_val = data.get("train", "")
            if train_val:
                if isinstance(train_val, list):
                    for p in train_val:
                        train_paths.append(str((parent / p).resolve()))
                else:
                    train_paths.append(str((parent / train_val).resolve()))

            # Resolve val paths
            val_val = data.get("val", "")
            if val_val:
                if isinstance(val_val, list):
                    for p in val_val:
                        val_paths.append(str((parent / p).resolve()))
                else:
                    val_paths.append(str((parent / val_val).resolve()))

        merged = {
            "nc": base_data.get("nc", 1),
            "names": base_data.get("names", []),
            "train": train_paths if len(train_paths) > 1 else (train_paths[0] if train_paths else ""),
            "val": val_paths if len(val_paths) > 1 else (val_paths[0] if val_paths else ""),
        }

        merged_yaml = output_dir / "data.yaml"
        with open(merged_yaml, "w") as f:
            yaml.dump(merged, f, default_flow_style=False, sort_keys=False)

        logger.info("Merged %d datasets into %s", len(yaml_paths), merged_yaml)
        return merged_yaml

    def _run_quality_monitoring(
        self,
        dataset: object,
        weights_path: Path,
        iteration: int,
    ) -> QualityMetrics | None:
        """Run SPC quality monitoring if enabled."""
        if not self.al_config.enable_quality_monitoring:
            return None

        try:
            from synthdet.quality.monitor import QualityMonitor

            monitor = QualityMonitor(model_path=str(weights_path))

            if self.al_config.quality_baseline_path:
                monitor.load_baseline(Path(self.al_config.quality_baseline_path))
            elif not monitor.has_baseline:
                # Establish baseline from original dataset val images
                import numpy as np
                val_images = []
                for rec in dataset.valid:  # type: ignore[attr-defined]
                    try:
                        val_images.append(rec.load_image())
                    except Exception:
                        continue
                if val_images:
                    monitor.establish_baseline(val_images)
                else:
                    return None

            # Monitor synthetic batch
            synth_images = []
            for rec in dataset.train[:50]:  # type: ignore[attr-defined]
                try:
                    synth_images.append(rec.load_image())
                except Exception:
                    continue

            if synth_images:
                return monitor.monitor_batch(synth_images, batch_id=f"iter_{iteration}")
        except Exception as exc:
            logger.warning("Quality monitoring failed: %s", exc)

        return None

    def _build_result(
        self,
        iterations: list[IterationResult],
        stopped_reason: str,
        total_cost: float,
        total_train_time: float,
    ) -> ActiveLearningResult:
        """Build the final ActiveLearningResult."""
        if not iterations:
            # Edge case: no iterations completed (e.g., first gen produced 0 records)
            from synthdet.types import BBoxSizeBucket, SpatialRegion
            empty_profile = ModelPerformanceProfile(
                overall_map50=0.0,
                overall_map50_95=0.0,
                per_class_map={},
                per_bucket_map={b: 0.0 for b in BBoxSizeBucket},
                per_region_map={r: 0.0 for r in SpatialRegion},
                false_negative_buckets={b: 0 for b in BBoxSizeBucket},
                false_negative_regions={r: 0 for r in SpatialRegion},
                confusion_pairs=[],
            )
            return ActiveLearningResult(
                iterations=[],
                final_profile=empty_profile,
                final_weights=Path(""),
                final_map50=0.0,
                total_training_time_seconds=0.0,
                stopped_reason=stopped_reason,
                total_cost_usd=total_cost,
            )

        last = iterations[-1]
        return ActiveLearningResult(
            iterations=iterations,
            final_profile=last.profile,
            final_weights=last.training_result.best_weights,
            final_map50=last.map50,
            total_training_time_seconds=total_train_time,
            stopped_reason=stopped_reason,
            total_cost_usd=total_cost,
        )
