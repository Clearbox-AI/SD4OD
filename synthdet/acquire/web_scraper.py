"""Web image scraper — acquires background images from search engines.

Satisfies the ``ImageAcquirer`` protocol from ``acquire/base.py``.
Requires the ``icrawler`` package (optional ``scraping`` extra).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from synthdet.config import WebScraperConfig
from synthdet.utils.image import compute_perceptual_hash, get_image_dimensions

logger = logging.getLogger(__name__)

_SUPPORTED_ENGINES = {"google", "bing"}


class WebScraper:
    """Acquire images from web search engines.

    Uses ``icrawler`` for the actual crawling, then applies post-filters
    for resolution, file type, and perceptual deduplication.
    """

    def __init__(self, config: WebScraperConfig | None = None) -> None:
        self.config = config or WebScraperConfig()
        if self.config.search_engine not in _SUPPORTED_ENGINES:
            raise ValueError(
                f"Unsupported search engine: {self.config.search_engine!r}. "
                f"Supported: {_SUPPORTED_ENGINES}"
            )

    def acquire(self, query: str, num_images: int, output_dir: Path) -> list[Path]:
        """Acquire images matching a query from the web.

        Args:
            query: Search query describing desired images.
            num_images: Target number of images to return.
            output_dir: Directory to save downloaded images.

        Returns:
            List of paths to acquired (and filtered) images.
        """
        try:
            import icrawler
            from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
        except ImportError as exc:
            raise ImportError(
                "icrawler is required for web scraping. "
                "Install it with: pip install 'synthdet[scraping]'"
            ) from exc

        output_dir.mkdir(parents=True, exist_ok=True)

        # Over-fetch to compensate for filtering
        fetch_count = num_images * 2

        if self.config.search_engine == "google":
            crawler = GoogleImageCrawler(
                storage={"root_dir": str(output_dir)},
                log_level=logging.WARNING,
            )
        else:
            crawler = BingImageCrawler(
                storage={"root_dir": str(output_dir)},
                log_level=logging.WARNING,
            )

        logger.info("Crawling %d images for query: %r", fetch_count, query)
        crawler.crawl(keyword=query, max_num=fetch_count)

        # Gather all downloaded files
        all_paths = sorted(
            p for p in output_dir.iterdir()
            if p.is_file() and p.suffix.lower().lstrip(".") in {"jpg", "jpeg", "png", "bmp", "webp"}
        )

        # Apply filters
        paths = self._filter_by_file_type(all_paths)
        paths = self._filter_by_resolution(paths)
        paths = self._deduplicate(paths)

        # Truncate to requested count
        paths = paths[:num_images]

        logger.info(
            "Acquired %d images (from %d downloaded) for query: %r",
            len(paths), len(all_paths), query,
        )
        return paths

    def _filter_by_resolution(self, paths: list[Path]) -> list[Path]:
        """Keep only images meeting minimum resolution requirements."""
        kept: list[Path] = []
        for p in paths:
            try:
                w, h = get_image_dimensions(p)
                if w >= self.config.min_width and h >= self.config.min_height:
                    kept.append(p)
            except Exception:
                logger.debug("Cannot read dimensions for %s, skipping", p)
        return kept

    def _filter_by_file_type(self, paths: list[Path]) -> list[Path]:
        """Keep only images with allowed file extensions."""
        allowed = {ft.lower().lstrip(".") for ft in self.config.file_types}
        return [p for p in paths if p.suffix.lower().lstrip(".") in allowed]

    def _deduplicate(self, paths: list[Path]) -> list[Path]:
        """Remove perceptual duplicates using average hash."""
        seen_hashes: set[str] = set()
        unique: list[Path] = []
        for p in paths:
            try:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                h = compute_perceptual_hash(img, hash_size=self.config.dedup_hash_size)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    unique.append(p)
            except Exception:
                logger.debug("Cannot hash %s, skipping", p)
        return unique
