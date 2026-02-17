"""Protocol for image acquisition from external sources."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ImageAcquirer(Protocol):
    """Protocol for acquiring images from external sources.

    Implementations: WebScraper (Phase 6), API search (Phase 6).
    """

    def acquire(self, query: str, num_images: int, output_dir: Path) -> list[Path]:
        """Acquire images matching a query.

        Args:
            query: Search query describing desired images.
            num_images: Number of images to acquire.
            output_dir: Directory to save downloaded images.

        Returns:
            List of paths to acquired images.
        """
        ...
