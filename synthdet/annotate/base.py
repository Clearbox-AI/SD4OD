"""Protocols for annotation and annotation verification."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from synthdet.types import BBox


@runtime_checkable
class Annotator(Protocol):
    """Protocol for automatic annotation.

    Implementations: GroundingDINO (Phase 4), OWL-ViT (Phase 4).
    """

    def annotate(self, image: np.ndarray, class_names: list[str]) -> list[BBox]:
        """Annotate an image with bounding boxes.

        Args:
            image: BGR image array.
            class_names: List of class names to detect.

        Returns:
            List of detected BBox objects.
        """
        ...


@runtime_checkable
class AnnotationVerifier(Protocol):
    """Protocol for annotation quality verification.

    Implementations: VLM-based verifier (Phase 4).
    """

    def verify(self, image: np.ndarray, bboxes: list[BBox]) -> list[tuple[BBox, float]]:
        """Verify annotation quality.

        Args:
            image: BGR image array.
            bboxes: Bounding boxes to verify.

        Returns:
            List of (bbox, confidence_score) tuples.
        """
        ...
