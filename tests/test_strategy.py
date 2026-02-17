"""Tests for synthdet.analysis.strategy."""

from __future__ import annotations

import pytest

from synthdet.analysis.statistics import compute_dataset_statistics
from synthdet.analysis.strategy import generate_synthesis_strategy
from synthdet.config import AnalysisConfig
from synthdet.types import BBoxSizeBucket, SpatialRegion


class TestGenerateSynthesisStrategy:
    def test_basic(self, mini_dataset):
        stats = compute_dataset_statistics(mini_dataset)
        strategy = generate_synthesis_strategy(mini_dataset, stats)
        assert strategy.target_total_images > 0
        assert len(strategy.generation_tasks) > 0

    def test_tasks_sorted_by_priority(self, mini_dataset):
        stats = compute_dataset_statistics(mini_dataset)
        strategy = generate_synthesis_strategy(mini_dataset, stats)
        priorities = [t.priority for t in strategy.generation_tasks]
        assert priorities == sorted(priorities, reverse=True)

    def test_negative_example_task_present(self, mini_dataset):
        stats = compute_dataset_statistics(mini_dataset)
        strategy = generate_synthesis_strategy(mini_dataset, stats)
        # mini_dataset has 0 negative examples, so a task should be created
        negative_tasks = [t for t in strategy.generation_tasks if "negative" in t.task_id.lower()]
        assert len(negative_tasks) >= 1

    def test_no_model_feedback(self, mini_dataset):
        stats = compute_dataset_statistics(mini_dataset)
        strategy = generate_synthesis_strategy(mini_dataset, stats)
        assert not strategy.has_model_feedback
        assert strategy.active_learning_signals == []

    def test_custom_config(self, mini_dataset):
        stats = compute_dataset_statistics(mini_dataset)
        config = AnalysisConfig(multiplier=2.0, min_per_bucket=5)
        strategy = generate_synthesis_strategy(mini_dataset, stats, config=config)
        assert strategy.target_total_images == stats.total_images * 2

    def test_size_bucket_gaps_reflected(self, mini_dataset):
        stats = compute_dataset_statistics(mini_dataset)
        strategy = generate_synthesis_strategy(mini_dataset, stats)
        assert len(strategy.size_bucket_gaps) > 0

    def test_spatial_gaps_reflected(self, mini_dataset):
        stats = compute_dataset_statistics(mini_dataset)
        strategy = generate_synthesis_strategy(mini_dataset, stats)
        assert len(strategy.spatial_gaps) > 0

    def test_total_synthetic_images(self, mini_dataset):
        stats = compute_dataset_statistics(mini_dataset)
        strategy = generate_synthesis_strategy(mini_dataset, stats)
        assert strategy.total_synthetic_images > 0
