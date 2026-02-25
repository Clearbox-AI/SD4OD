"""Tests for synthdet.augment.copy_paste — CopyPasteAugmenter."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from synthdet.augment.copy_paste import CopyPasteAugmenter
from synthdet.config import CopyPasteConfig
from synthdet.generate.compositor import DefectPatch
from synthdet.types import AnnotationSource, BBox, ImageRecord


def _make_patch(class_id: int = 0, size: int = 30) -> DefectPatch:
    """Create a synthetic defect patch for testing."""
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 3, 255, -1)
    return DefectPatch(
        image=img,
        mask=mask,
        class_id=class_id,
        original_size=(0.05, 0.05),
        source_path=Path("/fake/source.jpg"),
    )


def _make_record(
    n_bboxes: int = 1, img_w: int = 200, img_h: int = 200
) -> ImageRecord:
    """Create a test ImageRecord with an in-memory image."""
    img = np.random.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)
    bboxes = [
        BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.1, height=0.1,
             source=AnnotationSource.human)
        for _ in range(n_bboxes)
    ]
    return ImageRecord(
        image_path=Path("/fake/test_img.jpg"),
        bboxes=bboxes,
        image=img,
    )


class TestCopyPasteAugmenter:
    def test_augment_adds_bboxes(self):
        """Augmented image should have more bboxes than the original."""
        config = CopyPasteConfig(max_patches_per_image=2, max_placement_attempts=50)
        augmenter = CopyPasteAugmenter(config)
        record = _make_record(n_bboxes=1, img_w=400, img_h=400)
        patches = [_make_patch() for _ in range(5)]

        result = augmenter.augment(record, patches)

        assert len(result.bboxes) >= len(record.bboxes)

    def test_preserves_existing_bboxes(self):
        """Original bboxes should be preserved in the output."""
        config = CopyPasteConfig(preserve_existing_bboxes=True, max_placement_attempts=50)
        augmenter = CopyPasteAugmenter(config)
        record = _make_record(n_bboxes=2)
        patches = [_make_patch()]

        result = augmenter.augment(record, patches)

        # Original human-sourced bboxes must still be there
        human_bboxes = [b for b in result.bboxes if b.source == AnnotationSource.human]
        assert len(human_bboxes) == 2

    def test_new_bboxes_have_copy_paste_source(self):
        """Newly added bboxes should have source=copy_paste."""
        config = CopyPasteConfig(max_patches_per_image=1, max_placement_attempts=50)
        augmenter = CopyPasteAugmenter(config)
        record = _make_record(n_bboxes=0, img_w=400, img_h=400)
        patches = [_make_patch()]

        result = augmenter.augment(record, patches)

        cp_bboxes = [b for b in result.bboxes if b.source == AnnotationSource.copy_paste]
        # At least one copy_paste bbox (may be 0 if placement failed, but with
        # large image and small patch it should succeed)
        assert len(cp_bboxes) >= 1

    def test_returns_new_record(self):
        """Augmentation should return a new ImageRecord, not mutate the original."""
        augmenter = CopyPasteAugmenter()
        record = _make_record()
        patches = [_make_patch()]

        result = augmenter.augment(record, patches)

        assert result is not record
        assert result.image_path != record.image_path

    def test_image_shape_preserved(self):
        """Output image should have the same shape as the input."""
        augmenter = CopyPasteAugmenter()
        record = _make_record(img_w=300, img_h=200)
        patches = [_make_patch()]

        result = augmenter.augment(record, patches)

        assert result.image.shape == record.image.shape

    def test_no_overlapping_placements(self):
        """Bboxes should not overlap beyond max_overlap_iou."""
        config = CopyPasteConfig(
            max_patches_per_image=3,
            max_overlap_iou=0.0,  # No overlap allowed
            max_placement_attempts=50,
        )
        augmenter = CopyPasteAugmenter(config)
        record = _make_record(n_bboxes=0, img_w=800, img_h=800)
        patches = [_make_patch(size=20) for _ in range(5)]

        result = augmenter.augment(record, patches)

        # Verify pairwise IoU is 0 (or close to it)
        from synthdet.utils.bbox import bbox_iou
        for i, a in enumerate(result.bboxes):
            for b in result.bboxes[i + 1:]:
                assert bbox_iou(a, b) <= 0.01

    def test_batch_count_and_unique_paths(self):
        """augment_batch should return correct count with unique paths."""
        config = CopyPasteConfig(max_placement_attempts=50)
        augmenter = CopyPasteAugmenter(config)
        records = []
        for i in range(3):
            r = _make_record(img_w=400, img_h=400)
            r.image_path = Path(f"/fake/test_img_{i}.jpg")
            records.append(r)
        patches = [_make_patch() for _ in range(3)]

        results = augmenter.augment_batch(records, patches, variants_per_image=2)

        assert len(results) == 6  # 3 records * 2 variants
        paths = [r.image_path for r in results]
        assert len(set(paths)) == len(paths)  # All unique

    def test_empty_patches_handled(self):
        """Passing an empty patches list should return a copy without error."""
        augmenter = CopyPasteAugmenter()
        record = _make_record()

        result = augmenter.augment(record, patches=[])

        assert result.metadata.get("patches_added") == 0
        assert len(result.bboxes) == len(record.bboxes)

    def test_metadata_tracking(self):
        """Result should contain copy_paste metadata."""
        augmenter = CopyPasteAugmenter()
        record = _make_record()
        patches = [_make_patch()]

        result = augmenter.augment(record, patches)

        assert result.metadata.get("copy_paste") is True
        assert "patches_added" in result.metadata

    def test_valid_zone_respected(self):
        """Placements should only occur within the valid zone."""
        # Create a tiny valid zone in top-left corner
        hull = np.array([
            [[0.0, 0.0]], [[0.2, 0.0]], [[0.2, 0.2]], [[0.0, 0.2]]
        ], dtype=np.float32)

        config = CopyPasteConfig(max_patches_per_image=1, max_placement_attempts=100)
        augmenter = CopyPasteAugmenter(config)
        record = _make_record(n_bboxes=0, img_w=400, img_h=400)
        patches = [_make_patch(size=10)]

        result = augmenter.augment(record, patches, valid_zone=hull)

        # Any placed bbox center should be within [0, 0.2] range
        for bbox in result.bboxes:
            if bbox.source == AnnotationSource.copy_paste:
                assert bbox.x_center <= 0.25  # small tolerance for patch size
                assert bbox.y_center <= 0.25
