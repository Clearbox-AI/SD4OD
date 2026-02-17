"""Tests for classical augmentation with bbox awareness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from synthdet.augment.classical import ClassicalAugmenter
from synthdet.config import AugmentationConfig
from synthdet.types import AnnotationSource, BBox, ImageRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def augmenter() -> ClassicalAugmenter:
    """An augmenter with moderate settings."""
    config = AugmentationConfig(
        horizontal_flip_p=0.5,
        brightness_contrast_p=0.3,
        hue_saturation_p=0.2,
        noise_p=0.2,
        blur_p=0.15,
        shift_scale_rotate_p=0.3,
        rotate_limit=10,
    )
    return ClassicalAugmenter(config)


@pytest.fixture
def sample_record() -> ImageRecord:
    """A record with an image and 2 bboxes."""
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    return ImageRecord(
        image_path=Path("test_img.jpg"),
        bboxes=[
            BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.2,
                 source=AnnotationSource.compositor),
            BBox(class_id=1, x_center=0.3, y_center=0.7, width=0.15, height=0.1,
                 source=AnnotationSource.compositor),
        ],
        image=img,
    )


@pytest.fixture
def negative_record() -> ImageRecord:
    """A negative record (no bboxes)."""
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    return ImageRecord(
        image_path=Path("neg_img.jpg"),
        bboxes=[],
        image=img,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClassicalAugmenter:
    def test_augment_preserves_bbox_count(self, augmenter, sample_record):
        """Augmentation should preserve bbox count (with min_visibility filter, might lose some)."""
        # Use gentle augmentation to avoid bbox loss
        gentle = ClassicalAugmenter(AugmentationConfig(
            horizontal_flip_p=0.5,
            brightness_contrast_p=0.3,
            hue_saturation_p=0.0,
            noise_p=0.0,
            blur_p=0.0,
            shift_scale_rotate_p=0.0,
        ))
        np.random.seed(42)
        result = gentle.augment(sample_record)
        # With only flip + brightness, bbox count should be preserved
        assert len(result.bboxes) == len(sample_record.bboxes)

    def test_augment_bbox_format_valid(self, augmenter, sample_record):
        """All bboxes should remain in [0, 1] range after augmentation."""
        np.random.seed(42)
        for _ in range(5):  # Run a few times since augmentation is random
            result = augmenter.augment(sample_record)
            for bbox in result.bboxes:
                assert 0 <= bbox.x_center <= 1, f"x_center={bbox.x_center}"
                assert 0 <= bbox.y_center <= 1, f"y_center={bbox.y_center}"
                assert 0 < bbox.width <= 1, f"width={bbox.width}"
                assert 0 < bbox.height <= 1, f"height={bbox.height}"

    def test_augment_returns_new_record(self, augmenter, sample_record):
        result = augmenter.augment(sample_record)
        assert result is not sample_record
        assert result.image is not sample_record.image
        assert "augmented" in result.metadata

    def test_augment_batch_multiplies_count(self, augmenter, sample_record):
        records = [sample_record]
        variants = 3
        result = augmenter.augment_batch(records, variants_per_image=variants)
        assert len(result) == variants  # Only augmented, not originals

    def test_augment_negative_image(self, augmenter, negative_record):
        """Augmenting a negative image should keep it negative."""
        result = augmenter.augment(negative_record)
        assert result.is_negative
        assert len(result.bboxes) == 0
        assert result.image is not None

    def test_augment_image_shape_preserved(self, augmenter, sample_record):
        result = augmenter.augment(sample_record)
        assert result.image.shape == sample_record.image.shape

    def test_augment_batch_unique_paths(self, augmenter, sample_record):
        """Each augmented variant should have a unique path."""
        result = augmenter.augment_batch([sample_record], variants_per_image=3)
        paths = [str(r.image_path) for r in result]
        assert len(set(paths)) == len(paths)
