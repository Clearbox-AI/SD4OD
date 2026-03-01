"""Tests for synthdet.utils.image."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from synthdet.utils.image import (
    compute_image_stats,
    compute_perceptual_hash,
    find_image_files,
    get_image_dimensions,
    group_augmentation_variants,
    load_image,
    resize_if_needed,
)


@pytest.fixture
def sample_image_path(tmp_path) -> Path:
    """Create a small test image."""
    img = np.random.randint(0, 255, (40, 50, 3), dtype=np.uint8)
    path = tmp_path / "test.jpg"
    cv2.imwrite(str(path), img)
    return path


class TestLoadImage:
    def test_load_bgr(self, sample_image_path):
        img = load_image(sample_image_path, mode="bgr")
        assert img.shape == (40, 50, 3)

    def test_load_rgb(self, sample_image_path):
        img = load_image(sample_image_path, mode="rgb")
        assert img.shape == (40, 50, 3)

    def test_load_gray(self, sample_image_path):
        img = load_image(sample_image_path, mode="gray")
        assert img.ndim == 2
        assert img.shape == (40, 50)

    def test_invalid_mode(self, sample_image_path):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(sample_image_path, mode="cmyk")

    def test_nonexistent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "nope.jpg")


class TestGetImageDimensions:
    def test_dimensions(self, sample_image_path):
        w, h = get_image_dimensions(sample_image_path)
        assert w == 50
        assert h == 40


class TestComputeImageStats:
    def test_color_image(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        stats = compute_image_stats(img)
        assert "brightness" in stats
        assert "contrast" in stats
        assert "blue_mean" in stats
        assert stats["brightness"] == pytest.approx(128.0, abs=1)

    def test_grayscale(self):
        img = np.full((10, 10), 200, dtype=np.uint8)
        stats = compute_image_stats(img)
        assert stats["brightness"] == pytest.approx(200.0)
        assert stats["contrast"] == pytest.approx(0.0)


class TestFindImageFiles:
    def test_finds_images(self, tmp_path):
        (tmp_path / "a.jpg").touch()
        (tmp_path / "b.png").touch()
        (tmp_path / "c.txt").touch()
        files = find_image_files(tmp_path)
        assert len(files) == 2
        names = [f.name for f in files]
        assert "a.jpg" in names
        assert "b.png" in names
        assert "c.txt" not in names

    def test_nonexistent_dir(self, tmp_path):
        files = find_image_files(tmp_path / "nope")
        assert files == []


class TestPerceptualHash:
    def test_same_image_same_hash(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        h1 = compute_perceptual_hash(img)
        h2 = compute_perceptual_hash(img)
        assert h1 == h2

    def test_returns_hex_string(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        h = compute_perceptual_hash(img)
        assert isinstance(h, str)
        int(h, 16)  # Should not raise


class TestResizeIfNeeded:
    def test_none_returns_original(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = resize_if_needed(img, None)
        assert result is img

    def test_already_small_enough(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = resize_if_needed(img, 200)
        assert result is img

    def test_long_edge_is_width(self):
        img = np.zeros((100, 400, 3), dtype=np.uint8)
        result = resize_if_needed(img, 200)
        assert result.shape[1] == 200  # width capped
        assert result.shape[0] == 50   # height scaled proportionally

    def test_long_edge_is_height(self):
        img = np.zeros((400, 100, 3), dtype=np.uint8)
        result = resize_if_needed(img, 200)
        assert result.shape[0] == 200  # height capped
        assert result.shape[1] == 50   # width scaled proportionally

    def test_aspect_ratio_preserved(self):
        img = np.zeros((300, 600, 3), dtype=np.uint8)
        result = resize_if_needed(img, 300)
        # Original aspect ratio: 600/300 = 2.0
        assert result.shape[1] == 300
        assert result.shape[0] == 150
        assert result.shape[1] / result.shape[0] == pytest.approx(2.0, abs=0.05)

    def test_square_image(self):
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        result = resize_if_needed(img, 250)
        assert result.shape == (250, 250, 3)

    def test_grayscale(self):
        img = np.zeros((400, 200), dtype=np.uint8)
        result = resize_if_needed(img, 200)
        assert result.shape == (200, 100)


class TestGroupAugmentationVariants:
    def test_roboflow_naming(self):
        paths = [
            Path("laptop_a.rf.abc123.jpg"),
            Path("laptop_a.rf.def456.jpg"),
            Path("laptop_b.rf.ghi789.jpg"),
        ]
        groups = group_augmentation_variants(paths)
        assert len(groups) == 2
        assert len(groups["laptop_a"]) == 2
        assert len(groups["laptop_b"]) == 1

    def test_non_roboflow(self):
        paths = [Path("img001.jpg"), Path("img002.jpg")]
        groups = group_augmentation_variants(paths)
        assert len(groups) == 2

    def test_mixed(self):
        paths = [
            Path("laptop_a.rf.abc123.jpg"),
            Path("random.jpg"),
        ]
        groups = group_augmentation_variants(paths)
        assert len(groups) == 2
