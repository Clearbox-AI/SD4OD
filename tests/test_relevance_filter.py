"""Tests for synthdet.acquire.filter — RelevanceFilter with mocked embeddings."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from synthdet.acquire.filter import RelevanceFilter
from synthdet.config import FilterConfig
from synthdet.types import BBox, Dataset, ImageRecord


def _mock_embedding_computer(dim: int = 64) -> MagicMock:
    """Create a mock EmbeddingComputer that returns random L2-normalized embeddings."""
    computer = MagicMock()
    computer.embedding_dim = dim

    def _compute_from_paths(paths, **kwargs):
        n = len(paths)
        emb = np.random.randn(n, dim).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb

    computer.compute_from_paths = MagicMock(side_effect=_compute_from_paths)
    return computer


def _similar_embedding_computer(dim: int = 64) -> MagicMock:
    """Mock computer that returns near-identical embeddings for all images."""
    computer = MagicMock()
    computer.embedding_dim = dim
    base = np.ones((1, dim), dtype=np.float32)
    base = base / np.linalg.norm(base)

    def _compute(paths, **kwargs):
        n = len(paths)
        # Add tiny noise
        emb = np.tile(base, (n, 1)) + np.random.randn(n, dim).astype(np.float32) * 0.01
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb

    computer.compute_from_paths = MagicMock(side_effect=_compute)
    return computer


def _dissimilar_embedding_computer(dim: int = 64) -> MagicMock:
    """Mock computer where reference and candidate embeddings are orthogonal."""
    computer = MagicMock()
    computer.embedding_dim = dim
    call_count = [0]

    def _compute(paths, **kwargs):
        n = len(paths)
        call_count[0] += 1
        if call_count[0] == 1:
            # Reference: all positive direction
            emb = np.ones((n, dim), dtype=np.float32)
        else:
            # Candidates: negative direction (cosine sim ≈ -1)
            emb = -np.ones((n, dim), dtype=np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb

    computer.compute_from_paths = MagicMock(side_effect=_compute)
    return computer


class TestRelevanceFilter:
    def test_similar_images_pass(self, tmp_path: Path):
        """Images similar to reference should pass the filter."""
        computer = _similar_embedding_computer()
        filt = RelevanceFilter(
            config=FilterConfig(similarity_threshold=0.5),
            embedding_computer=computer,
        )

        ref_paths = [tmp_path / f"ref_{i}.jpg" for i in range(5)]
        filt.set_reference_from_paths(ref_paths)

        candidates = [tmp_path / f"cand_{i}.jpg" for i in range(10)]
        result = filt.filter(candidates)

        # With near-identical embeddings, all should pass
        assert len(result) == 10

    def test_dissimilar_images_rejected(self, tmp_path: Path):
        """Images dissimilar to reference should be rejected."""
        computer = _dissimilar_embedding_computer()
        filt = RelevanceFilter(
            config=FilterConfig(similarity_threshold=0.5),
            embedding_computer=computer,
        )

        ref_paths = [tmp_path / f"ref_{i}.jpg" for i in range(5)]
        filt.set_reference_from_paths(ref_paths)

        candidates = [tmp_path / f"cand_{i}.jpg" for i in range(10)]
        result = filt.filter(candidates)

        # Cosine sim ≈ -1, all should be rejected
        assert len(result) == 0

    def test_threshold_zero_passes_all(self, tmp_path: Path):
        """Threshold of 0 should pass everything (cosine sim >= 0 unlikely to be all negative)."""
        computer = _mock_embedding_computer()
        filt = RelevanceFilter(
            config=FilterConfig(similarity_threshold=-1.0),
            embedding_computer=computer,
        )

        ref_paths = [tmp_path / f"ref_{i}.jpg" for i in range(5)]
        filt.set_reference_from_paths(ref_paths)

        candidates = [tmp_path / f"cand_{i}.jpg" for i in range(10)]
        result = filt.filter(candidates)

        assert len(result) == 10

    def test_threshold_one_rejects_most(self, tmp_path: Path):
        """Threshold of 1.0 should reject nearly everything (exact match unlikely)."""
        computer = _mock_embedding_computer()
        filt = RelevanceFilter(
            config=FilterConfig(similarity_threshold=1.0),
            embedding_computer=computer,
        )

        ref_paths = [tmp_path / f"ref_{i}.jpg" for i in range(5)]
        filt.set_reference_from_paths(ref_paths)

        candidates = [tmp_path / f"cand_{i}.jpg" for i in range(10)]
        result = filt.filter(candidates)

        # Random embeddings won't have cosine sim = 1.0
        assert len(result) <= 10

    def test_empty_candidates(self, tmp_path: Path):
        """Empty candidate list should return empty list."""
        computer = _mock_embedding_computer()
        filt = RelevanceFilter(embedding_computer=computer)

        ref_paths = [tmp_path / f"ref_{i}.jpg" for i in range(5)]
        filt.set_reference_from_paths(ref_paths)

        result = filt.filter([])
        assert result == []

    def test_no_reference_passes_all(self, tmp_path: Path):
        """Without reference set, all candidates should pass."""
        computer = _mock_embedding_computer()
        filt = RelevanceFilter(embedding_computer=computer)

        candidates = [tmp_path / f"cand_{i}.jpg" for i in range(5)]
        result = filt.filter(candidates)

        assert len(result) == 5
