"""CLIP-based relevance filtering for acquired images.

Compares candidate images against a reference set (from the real dataset)
using cosine similarity in CLIP embedding space. Keeps candidates that are
sufficiently similar to the reference distribution.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from synthdet.config import FilterConfig
from synthdet.types import Dataset
from synthdet.utils.embeddings import EmbeddingComputer

logger = logging.getLogger(__name__)


class RelevanceFilter:
    """Filter images by CLIP embedding similarity to a reference set.

    Keeps candidates whose max cosine similarity to *any* reference image
    meets or exceeds ``similarity_threshold``.
    """

    def __init__(
        self,
        config: FilterConfig | None = None,
        embedding_computer: EmbeddingComputer | None = None,
    ) -> None:
        self.config = config or FilterConfig()
        self._embedding_computer = embedding_computer
        self._reference_embeddings: np.ndarray | None = None

    @property
    def embedding_computer(self) -> EmbeddingComputer:
        if self._embedding_computer is None:
            self._embedding_computer = EmbeddingComputer(
                model_name=self.config.model_name,
                pretrained=self.config.pretrained,
                device=self.config.device,
                batch_size=self.config.batch_size,
            )
        return self._embedding_computer

    def set_reference_from_dataset(
        self, dataset: Dataset, split: str = "train"
    ) -> None:
        """Set reference embeddings by sampling images from a dataset split.

        Args:
            dataset: Source dataset.
            split: Which split to sample from.
        """
        if split == "train":
            records = dataset.train
        elif split in ("valid", "val"):
            records = dataset.valid
        elif split == "test":
            records = dataset.test
        else:
            raise ValueError(f"Unknown split: {split!r}")

        # Sample up to reference_count images
        import random
        sampled = random.sample(records, min(self.config.reference_count, len(records)))
        paths = [r.image_path for r in sampled]
        self.set_reference_from_paths(paths)

    def set_reference_from_paths(self, paths: list[Path]) -> None:
        """Set reference embeddings from a list of image paths."""
        if not paths:
            self._reference_embeddings = None
            return
        self._reference_embeddings = self.embedding_computer.compute_from_paths(paths)
        logger.info("Set %d reference embeddings", len(paths))

    def filter(self, candidate_paths: list[Path]) -> list[Path]:
        """Filter candidates by similarity to the reference set.

        Args:
            candidate_paths: Paths to candidate images.

        Returns:
            Paths that pass the similarity threshold.
        """
        if not candidate_paths:
            return []

        # No reference = pass all through
        if self._reference_embeddings is None or len(self._reference_embeddings) == 0:
            return list(candidate_paths)

        # Compute candidate embeddings
        candidate_embeddings = self.embedding_computer.compute_from_paths(candidate_paths)

        # Cosine similarity matrix: (num_candidates, num_reference)
        # Embeddings are already L2-normalized, so dot product = cosine sim
        sim_matrix = candidate_embeddings @ self._reference_embeddings.T

        # Max similarity per candidate
        max_sims = sim_matrix.max(axis=1)

        # Keep candidates above threshold
        kept: list[Path] = []
        for i, path in enumerate(candidate_paths):
            if max_sims[i] >= self.config.similarity_threshold:
                kept.append(path)

        logger.info(
            "Relevance filter: %d / %d candidates passed (threshold=%.2f)",
            len(kept), len(candidate_paths), self.config.similarity_threshold,
        )
        return kept
