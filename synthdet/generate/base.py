"""Protocol for synthetic image generators."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from synthdet.types import GenerationTask, ImageRecord


@runtime_checkable
class ImageGenerator(Protocol):
    """Protocol for synthetic image generators.

    Implementations: DefectCompositor, Inpainting (Phase 3),
    DiffusionLocal (Phase 3), DiffusionAPI (Phase 3).
    """

    def generate(self, task: GenerationTask, **kwargs: object) -> list[ImageRecord]:
        """Generate synthetic images for a given task.

        Args:
            task: A prioritized generation task with targeting parameters.
            **kwargs: Generator-specific arguments.

        Returns:
            List of ImageRecord objects with synthetic images and annotations.
        """
        ...
