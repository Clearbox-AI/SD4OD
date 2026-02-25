"""Tests for synthdet.acquire.web_scraper — all mocked, no real downloads."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from synthdet.acquire.web_scraper import WebScraper
from synthdet.config import WebScraperConfig


def _create_test_image(path: Path, w: int = 500, h: int = 400) -> None:
    """Create a dummy image file on disk."""
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


class TestWebScraperProtocol:
    def test_satisfies_image_acquirer_protocol(self):
        """WebScraper should satisfy the ImageAcquirer protocol."""
        from synthdet.acquire.base import ImageAcquirer

        scraper = WebScraper()
        assert isinstance(scraper, ImageAcquirer)


class TestWebScraperInit:
    def test_default_config(self):
        scraper = WebScraper()
        assert scraper.config.search_engine == "google"

    def test_unsupported_engine_raises(self):
        config = WebScraperConfig(search_engine="yahoo")
        with pytest.raises(ValueError, match="Unsupported search engine"):
            WebScraper(config)

    def test_bing_engine_accepted(self):
        config = WebScraperConfig(search_engine="bing")
        scraper = WebScraper(config)
        assert scraper.config.search_engine == "bing"


class TestResolutionFilter:
    def test_keeps_large_images(self, tmp_path: Path):
        scraper = WebScraper(WebScraperConfig(min_width=200, min_height=200))
        p = tmp_path / "big.jpg"
        _create_test_image(p, 500, 400)

        result = scraper._filter_by_resolution([p])
        assert len(result) == 1

    def test_rejects_small_images(self, tmp_path: Path):
        scraper = WebScraper(WebScraperConfig(min_width=600, min_height=600))
        p = tmp_path / "small.jpg"
        _create_test_image(p, 100, 100)

        result = scraper._filter_by_resolution([p])
        assert len(result) == 0


class TestFileTypeFilter:
    def test_keeps_allowed_types(self, tmp_path: Path):
        scraper = WebScraper(WebScraperConfig(file_types=["jpg", "png"]))
        paths = [tmp_path / "a.jpg", tmp_path / "b.png", tmp_path / "c.bmp"]
        for p in paths:
            p.touch()

        result = scraper._filter_by_file_type(paths)
        assert len(result) == 2

    def test_rejects_disallowed_types(self, tmp_path: Path):
        scraper = WebScraper(WebScraperConfig(file_types=["png"]))
        paths = [tmp_path / "a.jpg"]
        for p in paths:
            p.touch()

        result = scraper._filter_by_file_type(paths)
        assert len(result) == 0


class TestDedup:
    def test_removes_duplicates(self, tmp_path: Path):
        scraper = WebScraper()
        # Create two identical images
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        p1 = tmp_path / "dup1.jpg"
        p2 = tmp_path / "dup2.jpg"
        cv2.imwrite(str(p1), img)
        cv2.imwrite(str(p2), img)

        result = scraper._deduplicate([p1, p2])
        assert len(result) == 1

    def test_keeps_different_images(self, tmp_path: Path):
        scraper = WebScraper()
        p1 = tmp_path / "a.jpg"
        p2 = tmp_path / "b.jpg"
        _create_test_image(p1, 100, 100)
        # Create a very different image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(p2), img)

        result = scraper._deduplicate([p1, p2])
        assert len(result) == 2


class TestAcquireIntegration:
    @patch("synthdet.acquire.web_scraper.icrawler", create=True)
    def test_acquire_with_mocked_crawler(self, _mock_icrawler, tmp_path: Path):
        """Test the full acquire flow with mocked icrawler."""
        output_dir = tmp_path / "output"

        # Pre-populate output dir to simulate crawler results
        output_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            _create_test_image(output_dir / f"img_{i:06d}.jpg", 500, 400)

        config = WebScraperConfig(max_images=3, min_width=200, min_height=200)
        scraper = WebScraper(config)

        # Mock the crawler to be a no-op (images already "downloaded")
        mock_crawler_cls = MagicMock()
        mock_instance = MagicMock()
        mock_crawler_cls.return_value = mock_instance

        with patch.dict("sys.modules", {
            "icrawler": MagicMock(),
            "icrawler.builtin": MagicMock(
                GoogleImageCrawler=mock_crawler_cls,
                BingImageCrawler=MagicMock(),
            ),
        }):
            result = scraper.acquire("laptop", 3, output_dir)

        assert len(result) <= 3
        assert all(p.exists() for p in result)

    def test_lazy_icrawler_import(self):
        """WebScraper should not import icrawler at construction time."""
        # This should succeed even without icrawler installed
        scraper = WebScraper()
        assert scraper is not None
