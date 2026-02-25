"""Tests for synthdet.training.loop."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from synthdet.config import ActiveLearningConfig, TrainingConfig
from synthdet.pipeline.config_schema import PipelineConfig
from synthdet.pipeline.orchestrator import PipelineResult
from synthdet.pipeline.validator import ValidationReport
from synthdet.training.loop import ActiveLearningLoop, ActiveLearningResult, IterationResult
from synthdet.training.trainer import TrainingResult
from synthdet.types import (
    BBoxSizeBucket,
    Dataset,
    ModelPerformanceProfile,
    QualityMetrics,
    SpatialRegion,
)


def _make_profile(map50: float = 0.5) -> ModelPerformanceProfile:
    return ModelPerformanceProfile(
        overall_map50=map50,
        overall_map50_95=map50 * 0.7,
        per_class_map={0: map50},
        per_bucket_map={b: map50 for b in BBoxSizeBucket},
        per_region_map={r: map50 for r in SpatialRegion},
        false_negative_buckets={b: 0 for b in BBoxSizeBucket},
        false_negative_regions={r: 0 for r in SpatialRegion},
        confusion_pairs=[],
    )


def _make_pipeline_result(output_dir: Path, num_records: int = 10) -> PipelineResult:
    return PipelineResult(
        output_dataset=Dataset(
            root=output_dir, class_names=["scratch"], train=[], valid=[], test=[],
        ),
        output_dir=output_dir,
        methods_used=["compositor"],
        records_per_method={"compositor": num_records},
        total_records=num_records,
        train_count=num_records,
        valid_count=0,
        total_cost_usd=0.0,
        cost_per_method={"compositor": 0.0},
        validation_report=None,
        dry_run=False,
    )


def _make_training_result(project_dir: Path, map50: float = 0.5) -> TrainingResult:
    return TrainingResult(
        best_weights=project_dir / "weights" / "best.pt",
        last_weights=project_dir / "weights" / "last.pt",
        epochs_completed=10,
        best_map50=map50,
        best_map50_95=map50 * 0.7,
        training_time_seconds=60.0,
        metrics_history=[],
        project_dir=project_dir,
    )


def _setup_data_yaml(tmp_path: Path) -> Path:
    """Create a minimal data.yaml for testing."""
    data_yaml = tmp_path / "data.yaml"
    data = {
        "nc": 1,
        "names": ["scratch"],
        "train": "train/images",
        "val": "valid/images",
    }
    with open(data_yaml, "w") as f:
        yaml.dump(data, f)
    # Create dirs so resolve works
    (tmp_path / "train" / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "valid" / "images").mkdir(parents=True, exist_ok=True)
    return data_yaml


class TestSingleIteration:
    @patch("synthdet.analysis.loader.load_yolo_dataset")
    @patch.object(ActiveLearningLoop, "_run_quality_monitoring", return_value=None)
    def test_max_iterations_one(self, mock_qm, mock_load, tmp_path):
        data_yaml = _setup_data_yaml(tmp_path)
        mock_load.return_value = Dataset(
            root=tmp_path, class_names=["scratch"], train=[], valid=[], test=[],
        )

        al_config = ActiveLearningConfig(max_iterations=1)
        loop = ActiveLearningLoop(
            data_yaml=data_yaml,
            output_dir=tmp_path / "output",
            al_config=al_config,
            seed=42,
        )

        # Create iter output data.yaml
        iter_dir = tmp_path / "output" / "iter_0"
        iter_dir.mkdir(parents=True, exist_ok=True)

        profile = _make_profile(0.6)

        with patch.object(loop, "_run_generation") as mock_gen, \
             patch.object(loop._trainer, "train") as mock_train, \
             patch.object(loop._evaluator, "evaluate", return_value=profile):

            # Setup generation to create a data.yaml
            def gen_side_effect(output_dir, *args, **kwargs):
                output_dir.mkdir(parents=True, exist_ok=True)
                iter_yaml = output_dir / "data.yaml"
                with open(iter_yaml, "w") as f:
                    yaml.dump({"nc": 1, "names": ["scratch"], "train": "train/images", "val": "valid/images"}, f)
                (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
                (output_dir / "valid" / "images").mkdir(parents=True, exist_ok=True)
                return _make_pipeline_result(output_dir)

            mock_gen.side_effect = gen_side_effect
            mock_train.return_value = _make_training_result(tmp_path / "runs" / "train", 0.6)

            result = loop.run()

        assert isinstance(result, ActiveLearningResult)
        assert len(result.iterations) == 1
        assert result.stopped_reason == "max_iterations"
        assert result.final_map50 == 0.6


class TestConvergenceStop:
    @patch("synthdet.analysis.loader.load_yolo_dataset")
    @patch.object(ActiveLearningLoop, "_run_quality_monitoring", return_value=None)
    def test_convergence_after_flat_map(self, mock_qm, mock_load, tmp_path):
        data_yaml = _setup_data_yaml(tmp_path)
        mock_load.return_value = Dataset(
            root=tmp_path, class_names=["scratch"], train=[], valid=[], test=[],
        )

        al_config = ActiveLearningConfig(
            max_iterations=10,
            convergence_threshold=0.01,
        )
        loop = ActiveLearningLoop(
            data_yaml=data_yaml,
            output_dir=tmp_path / "output",
            al_config=al_config,
            seed=42,
        )

        # mAP values: 0.5, 0.5, 0.5 → improvement = 0 for 2 consecutive → stop
        map_values = [0.5, 0.5, 0.5, 0.5, 0.5]
        call_count = [0]

        with patch.object(loop, "_run_generation") as mock_gen, \
             patch.object(loop._trainer, "train") as mock_train, \
             patch.object(loop._evaluator, "evaluate") as mock_eval:

            def gen_side_effect(output_dir, *args, **kwargs):
                output_dir.mkdir(parents=True, exist_ok=True)
                iter_yaml = output_dir / "data.yaml"
                with open(iter_yaml, "w") as f:
                    yaml.dump({"nc": 1, "names": ["scratch"], "train": "train/images", "val": "valid/images"}, f)
                (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
                (output_dir / "valid" / "images").mkdir(parents=True, exist_ok=True)
                return _make_pipeline_result(output_dir)

            def train_side_effect(data_yaml, iteration=0, weights=None):
                idx = min(call_count[0], len(map_values) - 1)
                call_count[0] += 1
                return _make_training_result(
                    tmp_path / "runs" / f"train_iter{iteration}",
                    map50=map_values[idx],
                )

            mock_gen.side_effect = gen_side_effect
            mock_train.side_effect = train_side_effect
            mock_eval.side_effect = lambda *a, **k: _make_profile(map_values[min(call_count[0] - 1, len(map_values) - 1)])

            result = loop.run()

        # Should stop after 3 iterations (iter 1 and 2 both have improvement < threshold)
        assert result.stopped_reason == "convergence"
        assert len(result.iterations) <= 4  # at most 4 before convergence detected


class TestNoRecordsStop:
    @patch("synthdet.analysis.loader.load_yolo_dataset")
    @patch.object(ActiveLearningLoop, "_run_quality_monitoring", return_value=None)
    def test_stops_on_zero_records(self, mock_qm, mock_load, tmp_path):
        data_yaml = _setup_data_yaml(tmp_path)

        loop = ActiveLearningLoop(
            data_yaml=data_yaml,
            output_dir=tmp_path / "output",
            al_config=ActiveLearningConfig(max_iterations=3),
        )

        with patch.object(loop, "_run_generation") as mock_gen:
            mock_gen.return_value = _make_pipeline_result(tmp_path, num_records=0)
            result = loop.run()

        assert result.stopped_reason == "no_records"
        assert len(result.iterations) == 0


class TestAccumulateData:
    @patch("synthdet.analysis.loader.load_yolo_dataset")
    @patch.object(ActiveLearningLoop, "_run_quality_monitoring", return_value=None)
    def test_accumulate_true_merges_all(self, mock_qm, mock_load, tmp_path):
        data_yaml = _setup_data_yaml(tmp_path)
        mock_load.return_value = Dataset(
            root=tmp_path, class_names=["scratch"], train=[], valid=[], test=[],
        )

        al_config = ActiveLearningConfig(max_iterations=2, accumulate_data=True)
        loop = ActiveLearningLoop(
            data_yaml=data_yaml,
            output_dir=tmp_path / "output",
            al_config=al_config,
            seed=0,
        )

        with patch.object(loop, "_run_generation") as mock_gen, \
             patch.object(loop._trainer, "train") as mock_train, \
             patch.object(loop._evaluator, "evaluate", return_value=_make_profile(0.7)):

            def gen_side_effect(output_dir, *args, **kwargs):
                output_dir.mkdir(parents=True, exist_ok=True)
                iter_yaml = output_dir / "data.yaml"
                with open(iter_yaml, "w") as f:
                    yaml.dump({"nc": 1, "names": ["scratch"], "train": "train/images", "val": "valid/images"}, f)
                (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
                (output_dir / "valid" / "images").mkdir(parents=True, exist_ok=True)
                return _make_pipeline_result(output_dir)

            mock_gen.side_effect = gen_side_effect
            mock_train.return_value = _make_training_result(tmp_path / "runs" / "t", 0.7)

            result = loop.run()

        # Check that merged data.yaml was written with multiple paths
        merged_yaml = tmp_path / "output" / "merged" / "data.yaml"
        assert merged_yaml.is_file()
        with open(merged_yaml) as f:
            merged_data = yaml.safe_load(f)
        # With accumulate=True and 2 iterations, train should have 3 paths
        # (original + iter_0 + iter_1)
        train_val = merged_data.get("train", "")
        if isinstance(train_val, list):
            assert len(train_val) == 3

    @patch("synthdet.analysis.loader.load_yolo_dataset")
    @patch.object(ActiveLearningLoop, "_run_quality_monitoring", return_value=None)
    def test_accumulate_false_only_original_plus_current(self, mock_qm, mock_load, tmp_path):
        data_yaml = _setup_data_yaml(tmp_path)
        mock_load.return_value = Dataset(
            root=tmp_path, class_names=["scratch"], train=[], valid=[], test=[],
        )

        al_config = ActiveLearningConfig(max_iterations=1, accumulate_data=False)
        loop = ActiveLearningLoop(
            data_yaml=data_yaml,
            output_dir=tmp_path / "output",
            al_config=al_config,
            seed=0,
        )

        with patch.object(loop, "_run_generation") as mock_gen, \
             patch.object(loop._trainer, "train") as mock_train, \
             patch.object(loop._evaluator, "evaluate", return_value=_make_profile()):

            def gen_side_effect(output_dir, *args, **kwargs):
                output_dir.mkdir(parents=True, exist_ok=True)
                iter_yaml = output_dir / "data.yaml"
                with open(iter_yaml, "w") as f:
                    yaml.dump({"nc": 1, "names": ["scratch"], "train": "train/images", "val": "valid/images"}, f)
                (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
                (output_dir / "valid" / "images").mkdir(parents=True, exist_ok=True)
                return _make_pipeline_result(output_dir)

            mock_gen.side_effect = gen_side_effect
            mock_train.return_value = _make_training_result(tmp_path / "runs" / "t")

            result = loop.run()

        merged_yaml = tmp_path / "output" / "merged" / "data.yaml"
        assert merged_yaml.is_file()
        with open(merged_yaml) as f:
            merged_data = yaml.safe_load(f)
        train_val = merged_data.get("train", "")
        # With accumulate=False: only original + current = 2 paths
        if isinstance(train_val, list):
            assert len(train_val) == 2


class TestSeedIncrementing:
    @patch("synthdet.analysis.loader.load_yolo_dataset")
    @patch.object(ActiveLearningLoop, "_run_quality_monitoring", return_value=None)
    def test_seed_per_iteration(self, mock_qm, mock_load, tmp_path):
        data_yaml = _setup_data_yaml(tmp_path)
        mock_load.return_value = Dataset(
            root=tmp_path, class_names=["scratch"], train=[], valid=[], test=[],
        )

        loop = ActiveLearningLoop(
            data_yaml=data_yaml,
            output_dir=tmp_path / "output",
            al_config=ActiveLearningConfig(max_iterations=2),
            seed=100,
        )

        seeds_used = []

        with patch.object(loop, "_run_generation") as mock_gen, \
             patch.object(loop._trainer, "train") as mock_train, \
             patch.object(loop._evaluator, "evaluate", return_value=_make_profile()):

            def gen_side_effect(output_dir, model_profile, quality_metrics, seed):
                seeds_used.append(seed)
                output_dir.mkdir(parents=True, exist_ok=True)
                iter_yaml = output_dir / "data.yaml"
                with open(iter_yaml, "w") as f:
                    yaml.dump({"nc": 1, "names": ["scratch"], "train": "train/images", "val": "valid/images"}, f)
                (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
                (output_dir / "valid" / "images").mkdir(parents=True, exist_ok=True)
                return _make_pipeline_result(output_dir)

            mock_gen.side_effect = gen_side_effect
            mock_train.return_value = _make_training_result(tmp_path / "runs" / "t")

            loop.run()

        assert seeds_used == [100, 101]


class TestWarmStart:
    @patch("synthdet.analysis.loader.load_yolo_dataset")
    @patch.object(ActiveLearningLoop, "_run_quality_monitoring", return_value=None)
    def test_iteration1_uses_iter0_weights(self, mock_qm, mock_load, tmp_path):
        data_yaml = _setup_data_yaml(tmp_path)
        mock_load.return_value = Dataset(
            root=tmp_path, class_names=["scratch"], train=[], valid=[], test=[],
        )

        loop = ActiveLearningLoop(
            data_yaml=data_yaml,
            output_dir=tmp_path / "output",
            al_config=ActiveLearningConfig(max_iterations=2),
        )

        weights_used = []

        with patch.object(loop, "_run_generation") as mock_gen, \
             patch.object(loop._trainer, "train") as mock_train, \
             patch.object(loop._evaluator, "evaluate", return_value=_make_profile()):

            def gen_side_effect(output_dir, *args, **kwargs):
                output_dir.mkdir(parents=True, exist_ok=True)
                iter_yaml = output_dir / "data.yaml"
                with open(iter_yaml, "w") as f:
                    yaml.dump({"nc": 1, "names": ["scratch"], "train": "train/images", "val": "valid/images"}, f)
                (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
                (output_dir / "valid" / "images").mkdir(parents=True, exist_ok=True)
                return _make_pipeline_result(output_dir)

            def train_side_effect(data_yaml, iteration=0, weights=None):
                weights_used.append(weights)
                best_path = tmp_path / f"runs/train_iter{iteration}/weights/best.pt"
                best_path.parent.mkdir(parents=True, exist_ok=True)
                best_path.touch()
                return _make_training_result(
                    tmp_path / f"runs/train_iter{iteration}", 0.5 + iteration * 0.1,
                )

            mock_gen.side_effect = gen_side_effect
            mock_train.side_effect = train_side_effect

            loop.run()

        # First iteration: no prior weights
        assert weights_used[0] is None
        # Second iteration: uses first iteration's best weights
        assert weights_used[1] is not None
        assert "best.pt" in str(weights_used[1])


class TestModelProfilePassedToStrategy:
    @patch("synthdet.analysis.loader.load_yolo_dataset")
    @patch.object(ActiveLearningLoop, "_run_quality_monitoring", return_value=None)
    def test_profile_fed_to_next_iteration(self, mock_qm, mock_load, tmp_path):
        data_yaml = _setup_data_yaml(tmp_path)
        mock_load.return_value = Dataset(
            root=tmp_path, class_names=["scratch"], train=[], valid=[], test=[],
        )

        loop = ActiveLearningLoop(
            data_yaml=data_yaml,
            output_dir=tmp_path / "output",
            al_config=ActiveLearningConfig(max_iterations=2),
        )

        profiles_passed = []

        with patch.object(loop, "_run_generation") as mock_gen, \
             patch.object(loop._trainer, "train") as mock_train, \
             patch.object(loop._evaluator, "evaluate", return_value=_make_profile()):

            def gen_side_effect(output_dir, model_profile, quality_metrics, seed):
                profiles_passed.append(model_profile)
                output_dir.mkdir(parents=True, exist_ok=True)
                iter_yaml = output_dir / "data.yaml"
                with open(iter_yaml, "w") as f:
                    yaml.dump({"nc": 1, "names": ["scratch"], "train": "train/images", "val": "valid/images"}, f)
                (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
                (output_dir / "valid" / "images").mkdir(parents=True, exist_ok=True)
                return _make_pipeline_result(output_dir)

            mock_gen.side_effect = gen_side_effect
            mock_train.return_value = _make_training_result(tmp_path / "runs" / "t")

            loop.run()

        # First iteration: no profile yet
        assert profiles_passed[0] is None
        # Second iteration: profile from first evaluation
        assert profiles_passed[1] is not None
        assert isinstance(profiles_passed[1], ModelPerformanceProfile)


class TestQualityMonitoringDisabled:
    @patch("synthdet.analysis.loader.load_yolo_dataset")
    def test_disabled_returns_none(self, mock_load, tmp_path):
        data_yaml = _setup_data_yaml(tmp_path)
        mock_load.return_value = Dataset(
            root=tmp_path, class_names=["scratch"], train=[], valid=[], test=[],
        )

        al_config = ActiveLearningConfig(
            max_iterations=1,
            enable_quality_monitoring=False,
        )
        loop = ActiveLearningLoop(
            data_yaml=data_yaml,
            output_dir=tmp_path / "output",
            al_config=al_config,
        )

        with patch.object(loop, "_run_generation") as mock_gen, \
             patch.object(loop._trainer, "train") as mock_train, \
             patch.object(loop._evaluator, "evaluate", return_value=_make_profile()):

            def gen_side_effect(output_dir, *args, **kwargs):
                output_dir.mkdir(parents=True, exist_ok=True)
                iter_yaml = output_dir / "data.yaml"
                with open(iter_yaml, "w") as f:
                    yaml.dump({"nc": 1, "names": ["scratch"], "train": "train/images", "val": "valid/images"}, f)
                (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
                (output_dir / "valid" / "images").mkdir(parents=True, exist_ok=True)
                return _make_pipeline_result(output_dir)

            mock_gen.side_effect = gen_side_effect
            mock_train.return_value = _make_training_result(tmp_path / "runs" / "t")

            result = loop.run()

        assert result.iterations[0].quality_metrics is None
