"""Tests for synthdet.analysis.diversity — mock EmbeddingComputer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from synthdet.analysis.diversity import DiversityAnalyzer
from synthdet.types import (
    BBox,
    Dataset,
    DiversityReport,
    ImageRecord,
)


def _make_dataset(
    n_images: int = 5,
    n_classes: int = 1,
    bboxes_per_image: int = 1,
    n_negatives: int = 0,
) -> Dataset:
    """Create a minimal test dataset."""
    records = []
    for i in range(n_images):
        bboxes = []
        if i >= n_images - n_negatives:
            pass  # negative example
        else:
            for _ in range(bboxes_per_image):
                bboxes.append(BBox(
                    class_id=i % n_classes,
                    x_center=0.5, y_center=0.5,
                    width=0.1, height=0.1,
                ))
        records.append(ImageRecord(Path(f"/fake/img_{i}.jpg"), bboxes))
    return Dataset(
        root=Path("/fake"),
        class_names=[f"class_{i}" for i in range(n_classes)],
        train=records,
        valid=[],
        test=[],
    )


def _mock_computer(embedding_fn):
    """Create a mock EmbeddingComputer with custom embedding function."""
    computer = MagicMock()
    computer.compute_from_paths = MagicMock(side_effect=embedding_fn)
    computer.embedding_dim = 64
    return computer


class TestPairwiseCosineDistance:
    def test_identical_embeddings_zero_distance(self):
        emb = np.ones((5, 64), dtype=np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        dist = DiversityAnalyzer._compute_pairwise_cosine_distance(emb)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_embeddings_distance_one(self):
        # Create orthogonal unit vectors using identity-like matrix
        emb = np.eye(4, dtype=np.float32)  # 4 orthogonal vectors in 4D
        dist = DiversityAnalyzer._compute_pairwise_cosine_distance(emb)
        assert dist == pytest.approx(1.0, abs=1e-6)

    def test_single_image_returns_zero(self):
        emb = np.random.randn(1, 64).astype(np.float32)
        dist = DiversityAnalyzer._compute_pairwise_cosine_distance(emb)
        assert dist == 0.0

    def test_empty_returns_zero(self):
        emb = np.zeros((0, 64), dtype=np.float32)
        dist = DiversityAnalyzer._compute_pairwise_cosine_distance(emb)
        assert dist == 0.0


class TestPerClassDiversity:
    def test_grouping_by_class(self):
        records = [
            ImageRecord(Path(f"/fake/{i}.jpg"), [BBox(0, 0.5, 0.5, 0.1, 0.1)])
            for i in range(3)
        ] + [
            ImageRecord(Path(f"/fake/{i+3}.jpg"), [BBox(1, 0.5, 0.5, 0.1, 0.1)])
            for i in range(3)
        ]
        # Embeddings: first 3 are similar, last 3 are different
        emb = np.random.randn(6, 64).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        # Make class_0 embeddings identical
        emb[:3] = emb[0]

        result = DiversityAnalyzer._compute_per_class_diversity(
            emb, records, ["Scratch", "Stain"]
        )
        assert "Scratch" in result
        assert "Stain" in result
        assert result["Scratch"] == pytest.approx(0.0, abs=1e-6)  # identical
        assert result["Stain"] > 0  # random vectors have some distance


class TestCoverageRatio:
    def test_identical_sets_full_coverage(self):
        emb = np.random.randn(5, 64).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        coverage = DiversityAnalyzer._compute_coverage_ratio(emb, emb, threshold=0.3)
        assert coverage == pytest.approx(1.0)

    def test_disjoint_sets_low_coverage(self):
        # Target: positive quadrant; Reference: negative quadrant
        target = np.eye(4, dtype=np.float32)
        reference = -np.eye(4, dtype=np.float32)
        coverage = DiversityAnalyzer._compute_coverage_ratio(target, reference, threshold=0.3)
        # Cosine similarity between e_i and -e_j is 0 (i!=j) or -1 (i==j)
        # Max similarity per reference = 0 (for off-diagonal) or -1 (for diagonal)
        # Neither exceeds 1 - 0.3 = 0.7
        assert coverage == 0.0

    def test_empty_target(self):
        target = np.zeros((0, 64), dtype=np.float32)
        reference = np.random.randn(5, 64).astype(np.float32)
        coverage = DiversityAnalyzer._compute_coverage_ratio(target, reference)
        assert coverage == 0.0

    def test_empty_reference(self):
        target = np.random.randn(5, 64).astype(np.float32)
        reference = np.zeros((0, 64), dtype=np.float32)
        coverage = DiversityAnalyzer._compute_coverage_ratio(target, reference)
        assert coverage == 0.0


class TestOutlierDetection:
    def test_flags_known_outlier(self):
        # 9 similar vectors + 1 very different
        emb = np.ones((10, 64), dtype=np.float32) * 0.1
        emb = emb + np.random.randn(10, 64).astype(np.float32) * 0.01
        emb[9] = np.random.randn(64).astype(np.float32) * 10  # outlier
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

        records = [ImageRecord(Path(f"/fake/{i}.jpg"), []) for i in range(10)]
        indices, z_scores = DiversityAnalyzer._detect_outliers(emb, records, z_threshold=2.0)
        assert 9 in indices
        assert len(z_scores) == len(indices)

    def test_no_outliers_in_uniform(self):
        # All identical
        emb = np.ones((10, 64), dtype=np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        records = [ImageRecord(Path(f"/fake/{i}.jpg"), []) for i in range(10)]
        indices, _ = DiversityAnalyzer._detect_outliers(emb, records)
        assert len(indices) == 0

    def test_too_few_images(self):
        emb = np.random.randn(2, 64).astype(np.float32)
        records = [ImageRecord(Path(f"/fake/{i}.jpg"), []) for i in range(2)]
        indices, z_scores = DiversityAnalyzer._detect_outliers(emb, records)
        assert indices == []
        assert z_scores == []


class TestDiversityAnalyzerAnalyze:
    def test_full_analysis(self):
        dataset = _make_dataset(n_images=5, n_classes=1)

        def mock_embed(paths, **kwargs):
            n = len(paths)
            emb = np.random.randn(n, 64).astype(np.float32)
            return emb / np.linalg.norm(emb, axis=1, keepdims=True)

        computer = _mock_computer(mock_embed)
        analyzer = DiversityAnalyzer(embedding_computer=computer)
        report = analyzer.analyze(dataset, split="train")

        assert isinstance(report, DiversityReport)
        assert report.num_images == 5
        assert report.embedding_dim == 64
        assert 0.0 <= report.mean_pairwise_cosine_distance <= 2.0
        assert "class_0" in report.per_class_diversity
        assert report.coverage_ratio is None  # no reference

    def test_with_reference_dataset(self):
        dataset = _make_dataset(n_images=5)
        reference = _make_dataset(n_images=5)

        def mock_embed(paths, **kwargs):
            n = len(paths)
            emb = np.random.randn(n, 64).astype(np.float32)
            return emb / np.linalg.norm(emb, axis=1, keepdims=True)

        computer = _mock_computer(mock_embed)
        analyzer = DiversityAnalyzer(embedding_computer=computer)
        report = analyzer.analyze(dataset, reference_dataset=reference)

        assert report.coverage_ratio is not None
        assert 0.0 <= report.coverage_ratio <= 1.0

    def test_empty_split(self):
        dataset = _make_dataset(n_images=3)
        # valid split is empty

        computer = _mock_computer(lambda paths, **kw: np.zeros((0, 64), dtype=np.float32))
        computer.embedding_dim = 64
        analyzer = DiversityAnalyzer(embedding_computer=computer)
        report = analyzer.analyze(dataset, split="valid")

        assert report.num_images == 0
        assert report.mean_pairwise_cosine_distance == 0.0

    def test_only_negatives(self):
        dataset = _make_dataset(n_images=3, n_negatives=3)

        def mock_embed(paths, **kwargs):
            n = len(paths)
            emb = np.random.randn(n, 64).astype(np.float32)
            return emb / np.linalg.norm(emb, axis=1, keepdims=True)

        computer = _mock_computer(mock_embed)
        analyzer = DiversityAnalyzer(embedding_computer=computer)
        report = analyzer.analyze(dataset)

        assert report.num_images == 3
        assert "_negative" in report.per_class_diversity
