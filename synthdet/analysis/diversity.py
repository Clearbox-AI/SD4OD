"""Embedding-based diversity analysis of YOLO datasets.

Requires ``open-clip-torch`` (optional ``embeddings`` extra) via
``EmbeddingComputer``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from synthdet.types import Dataset, DiversityReport, ImageRecord
from synthdet.utils.embeddings import EmbeddingComputer


class DiversityAnalyzer:
    """Analyze visual diversity of a dataset using CLIP/OpenCLIP embeddings.

    Computes:
    - Mean pairwise cosine distance (higher = more diverse)
    - Per-class diversity scores
    - Coverage ratio vs. a reference dataset
    - Outlier detection based on embedding distance
    """

    def __init__(
        self,
        embedding_computer: EmbeddingComputer | None = None,
        **kwargs,
    ) -> None:
        self._computer = embedding_computer or EmbeddingComputer(**kwargs)

    def analyze(
        self,
        dataset: Dataset,
        split: str = "train",
        reference_dataset: Dataset | None = None,
        reference_split: str = "train",
    ) -> DiversityReport:
        """Run diversity analysis on a dataset split.

        Args:
            dataset: Target dataset to analyze.
            split: Which split to analyze ("train", "valid", "test").
            reference_dataset: Optional reference dataset for coverage ratio.
            reference_split: Which split of the reference dataset to use.

        Returns:
            DiversityReport with diversity metrics.
        """
        records = self._get_records(dataset, split)
        if not records:
            return DiversityReport(
                num_images=0,
                embedding_dim=self._computer.embedding_dim,
                mean_pairwise_cosine_distance=0.0,
                per_class_diversity={},
                coverage_ratio=None,
                outlier_indices=[],
                outlier_z_scores=[],
            )

        paths = [str(r.image_path) for r in records]
        embeddings = self._computer.compute_from_paths(paths)

        mean_dist = self._compute_pairwise_cosine_distance(embeddings)
        per_class = self._compute_per_class_diversity(
            embeddings, records, dataset.class_names
        )

        coverage = None
        if reference_dataset is not None:
            ref_records = self._get_records(reference_dataset, reference_split)
            if ref_records:
                ref_paths = [str(r.image_path) for r in ref_records]
                ref_embeddings = self._computer.compute_from_paths(ref_paths)
                coverage = self._compute_coverage_ratio(embeddings, ref_embeddings)

        outlier_indices, outlier_z_scores = self._detect_outliers(
            embeddings, records
        )

        return DiversityReport(
            num_images=len(records),
            embedding_dim=embeddings.shape[1] if embeddings.ndim == 2 else 0,
            mean_pairwise_cosine_distance=mean_dist,
            per_class_diversity=per_class,
            coverage_ratio=coverage,
            outlier_indices=outlier_indices,
            outlier_z_scores=outlier_z_scores,
        )

    @staticmethod
    def _get_records(dataset: Dataset, split: str) -> list[ImageRecord]:
        if split == "train":
            return dataset.train
        elif split in ("valid", "val"):
            return dataset.valid
        elif split == "test":
            return dataset.test
        else:
            raise ValueError(f"Unknown split: {split!r}")

    @staticmethod
    def _compute_pairwise_cosine_distance(embeddings: np.ndarray) -> float:
        """Mean pairwise cosine distance. Assumes L2-normalized embeddings."""
        n = len(embeddings)
        if n < 2:
            return 0.0
        # Cosine similarity matrix (embeddings are L2-normalized)
        sim_matrix = embeddings @ embeddings.T
        # Extract upper triangle (excluding diagonal)
        upper_indices = np.triu_indices(n, k=1)
        similarities = sim_matrix[upper_indices]
        # Cosine distance = 1 - cosine similarity
        return float(np.mean(1.0 - similarities))

    @staticmethod
    def _compute_per_class_diversity(
        embeddings: np.ndarray,
        records: list[ImageRecord],
        class_names: list[str],
    ) -> dict[str, float]:
        """Compute mean pairwise cosine distance per class."""
        # Group record indices by class
        class_indices: dict[int, list[int]] = defaultdict(list)
        for i, rec in enumerate(records):
            for bbox in rec.bboxes:
                class_indices[bbox.class_id].append(i)
            if rec.is_negative:
                # Negatives grouped under a special key
                class_indices[-1].append(i)

        result: dict[str, float] = {}
        for class_id, indices in class_indices.items():
            unique_indices = sorted(set(indices))
            if class_id == -1:
                name = "_negative"
            elif 0 <= class_id < len(class_names):
                name = class_names[class_id]
            else:
                name = f"class_{class_id}"

            if len(unique_indices) < 2:
                result[name] = 0.0
                continue

            class_emb = embeddings[unique_indices]
            sim_matrix = class_emb @ class_emb.T
            upper = np.triu_indices(len(class_emb), k=1)
            result[name] = float(np.mean(1.0 - sim_matrix[upper]))

        return result

    @staticmethod
    def _compute_coverage_ratio(
        target: np.ndarray,
        reference: np.ndarray,
        threshold: float = 0.3,
    ) -> float:
        """Fraction of reference images "covered" by the target dataset.

        A reference image is covered if at least one target image has
        cosine distance < ``threshold`` to it.
        """
        if len(reference) == 0 or len(target) == 0:
            return 0.0
        # Similarity: (n_ref, n_target)
        sim = reference @ target.T
        # Max similarity per reference image
        max_sim = np.max(sim, axis=1)
        # Covered if distance < threshold, i.e., similarity > 1 - threshold
        covered = np.sum(max_sim > (1.0 - threshold))
        return float(covered / len(reference))

    @staticmethod
    def _detect_outliers(
        embeddings: np.ndarray,
        records: list[ImageRecord],
        z_threshold: float = 2.5,
    ) -> tuple[list[int], list[float]]:
        """Detect outlier images based on mean distance to all other images.

        Returns (indices, z_scores) of outlier images.
        """
        n = len(embeddings)
        if n < 3:
            return [], []

        # Compute mean cosine distance from each image to all others
        sim_matrix = embeddings @ embeddings.T
        np.fill_diagonal(sim_matrix, 0.0)
        mean_distances = 1.0 - sim_matrix.sum(axis=1) / (n - 1)

        mu = np.mean(mean_distances)
        sigma = np.std(mean_distances)
        if sigma < 1e-10:
            return [], []

        z_scores = (mean_distances - mu) / sigma
        outlier_mask = z_scores > z_threshold
        indices = np.where(outlier_mask)[0].tolist()
        scores = z_scores[outlier_mask].tolist()

        return indices, scores
