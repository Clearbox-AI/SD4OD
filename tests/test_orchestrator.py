"""Tests for the pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from synthdet.pipeline.config_schema import PipelineConfig
from synthdet.pipeline.orchestrator import (
    PipelineResult,
    _scale_strategy,
    run_pipeline,
)
from synthdet.types import (
    BBox,
    BBoxSizeBucket,
    GenerationTask,
    SpatialRegion,
    SynthesisStrategy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mini_pipeline_dataset(tmp_path) -> Path:
    """Create a minimal dataset suitable for pipeline testing."""
    ds_dir = tmp_path / "dataset"
    train_imgs = ds_dir / "train" / "images"
    train_lbls = ds_dir / "train" / "labels"
    valid_imgs = ds_dir / "valid" / "images"
    valid_lbls = ds_dir / "valid" / "labels"

    for d in (train_imgs, train_lbls, valid_imgs, valid_lbls):
        d.mkdir(parents=True)

    # 4 train images with defects at varied positions
    for i in range(4):
        img = np.full((100, 150, 3), 150 + i * 10, dtype=np.uint8)
        x1, y1 = 30 + i * 10, 20 + i * 5
        x2, y2 = x1 + 40, y1 + 30
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 200), -1)
        cv2.imwrite(str(train_imgs / f"train_{i}.jpg"), img)

        cx = (x1 + x2) / 2 / 150
        cy = (y1 + y2) / 2 / 100
        w = (x2 - x1) / 150
        h = (y2 - y1) / 100
        (train_lbls / f"train_{i}.txt").write_text(
            f"0 {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n"
        )

    # 1 valid image
    img = np.full((100, 150, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (50, 30), (100, 70), (200, 0, 0), -1)
    cv2.imwrite(str(valid_imgs / "val_0.jpg"), img)
    (valid_lbls / "val_0.txt").write_text("0 0.5000 0.5000 0.3333 0.4000\n")

    (ds_dir / "data.yaml").write_text(
        "train: train/images\nval: valid/images\nnc: 1\nnames: ['Scratch']\n"
    )
    return ds_dir / "data.yaml"


# ---------------------------------------------------------------------------
# _scale_strategy
# ---------------------------------------------------------------------------


class TestScaleStrategy:
    def test_scale_divides_images(self):
        strategy = SynthesisStrategy(
            target_total_images=100,
            target_class_counts={0: 100},
            negative_ratio=0.15,
            size_bucket_gaps={},
            spatial_gaps={},
            aspect_ratio_gaps=[],
            generation_tasks=[
                GenerationTask(
                    task_id="t1", priority=0.7, num_images=20,
                    target_classes=[0],
                    target_size_buckets=[BBoxSizeBucket.small],
                    target_regions=list(SpatialRegion),
                    suggested_prompts=[], rationale="Test", method="compositor",
                ),
            ],
        )
        scaled = _scale_strategy(strategy, 4)
        assert scaled.generation_tasks[0].num_images == 5
        # Original unchanged
        assert strategy.generation_tasks[0].num_images == 20

    def test_scale_minimum_one(self):
        strategy = SynthesisStrategy(
            target_total_images=10,
            target_class_counts={0: 10},
            negative_ratio=0.15,
            size_bucket_gaps={},
            spatial_gaps={},
            aspect_ratio_gaps=[],
            generation_tasks=[
                GenerationTask(
                    task_id="t1", priority=0.7, num_images=1,
                    target_classes=[0],
                    target_size_buckets=[BBoxSizeBucket.small],
                    target_regions=list(SpatialRegion),
                    suggested_prompts=[], rationale="Test", method="compositor",
                ),
            ],
        )
        scaled = _scale_strategy(strategy, 3)
        assert scaled.generation_tasks[0].num_images >= 1

    def test_no_scale_single_method(self):
        strategy = SynthesisStrategy(
            target_total_images=10,
            target_class_counts={0: 10},
            negative_ratio=0.15,
            size_bucket_gaps={},
            spatial_gaps={},
            aspect_ratio_gaps=[],
            generation_tasks=[
                GenerationTask(
                    task_id="t1", priority=0.7, num_images=10,
                    target_classes=[0],
                    target_size_buckets=[BBoxSizeBucket.small],
                    target_regions=list(SpatialRegion),
                    suggested_prompts=[], rationale="Test", method="compositor",
                ),
            ],
        )
        scaled = _scale_strategy(strategy, 1)
        assert scaled.generation_tasks[0].num_images == 10


# ---------------------------------------------------------------------------
# run_pipeline — compositor only (no API)
# ---------------------------------------------------------------------------


class TestRunPipelineCompositor:
    def test_compositor_produces_output(self, mini_pipeline_dataset, tmp_path):
        output_dir = tmp_path / "output"
        cfg = PipelineConfig(
            methods=["compositor"],
            validate_output=True,
        )
        result = run_pipeline(mini_pipeline_dataset, output_dir, config=cfg, seed=42)

        assert isinstance(result, PipelineResult)
        assert result.total_records > 0
        assert result.train_count > 0
        assert result.dry_run is False
        assert result.methods_used == ["compositor"]
        assert "compositor" in result.records_per_method
        assert result.total_cost_usd == 0.0

        # Output files exist
        assert (output_dir / "data.yaml").is_file()
        assert (output_dir / "train" / "images").is_dir()
        assert (output_dir / "train" / "labels").is_dir()

    def test_validation_report_populated(self, mini_pipeline_dataset, tmp_path):
        output_dir = tmp_path / "output"
        cfg = PipelineConfig(methods=["compositor"], validate_output=True)
        result = run_pipeline(mini_pipeline_dataset, output_dir, config=cfg, seed=42)

        assert result.validation_report is not None
        assert result.validation_report.is_valid
        assert result.validation_report.total_images > 0

    def test_validation_disabled(self, mini_pipeline_dataset, tmp_path):
        output_dir = tmp_path / "output"
        cfg = PipelineConfig(methods=["compositor"], validate_output=False)
        result = run_pipeline(mini_pipeline_dataset, output_dir, config=cfg, seed=42)

        assert result.validation_report is None

    def test_augmentation_increases_count(self, mini_pipeline_dataset, tmp_path):
        output_no_aug = tmp_path / "no_aug"
        cfg_no = PipelineConfig(methods=["compositor"], augment=False, validate_output=False)
        result_no = run_pipeline(mini_pipeline_dataset, output_no_aug, config=cfg_no, seed=42)

        output_aug = tmp_path / "with_aug"
        cfg_aug = PipelineConfig(methods=["compositor"], augment=True, validate_output=False)
        result_aug = run_pipeline(mini_pipeline_dataset, output_aug, config=cfg_aug, seed=42)

        assert result_aug.total_records > result_no.total_records

    def test_seed_reproducibility(self, mini_pipeline_dataset, tmp_path):
        cfg = PipelineConfig(methods=["compositor"], validate_output=False)

        out1 = tmp_path / "out1"
        r1 = run_pipeline(mini_pipeline_dataset, out1, config=cfg, seed=42)

        out2 = tmp_path / "out2"
        r2 = run_pipeline(mini_pipeline_dataset, out2, config=cfg, seed=42)

        assert r1.total_records == r2.total_records


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_returns_zero_records(self, mini_pipeline_dataset, tmp_path):
        output_dir = tmp_path / "dry"
        cfg = PipelineConfig(
            methods=["inpainting"],
            dry_run=True,
            validate_output=False,
        )
        result = run_pipeline(mini_pipeline_dataset, output_dir, config=cfg, seed=42)

        assert result.dry_run is True
        assert result.total_records == 0
        assert result.train_count == 0
        assert result.valid_count == 0
        assert result.total_cost_usd >= 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unknown_method_in_run_pipeline(self, mini_pipeline_dataset, tmp_path):
        """PipelineConfig validates methods, but test the orchestrator path too."""
        cfg = PipelineConfig(methods=["compositor"])
        # Bypass validation by directly mutating
        cfg.methods = ["nonexistent"]
        with pytest.raises(ValueError, match="Unknown method"):
            run_pipeline(mini_pipeline_dataset, tmp_path / "out", config=cfg, seed=42)
