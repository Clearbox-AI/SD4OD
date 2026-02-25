"""Copy-paste augmentation for object detection.

Pastes defect patches onto *existing annotated images* (preserving original
bboxes), complementing the compositor which uses clean backgrounds. Reuses
Poisson blending, rotation, and placement logic from the compositor pipeline.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import cv2
import numpy as np

from synthdet.config import CopyPasteConfig
from synthdet.generate.compositor import (
    DefectPatch,
    _alpha_blend,
    _poisson_blend,
    _rotate_patch,
)
from synthdet.generate.placement import check_placement_valid, point_in_valid_zone
from synthdet.types import AnnotationSource, BBox, ImageRecord
from synthdet.utils.bbox import clip_bbox

logger = logging.getLogger(__name__)


class CopyPasteAugmenter:
    """Paste defect patches onto existing annotated images.

    Unlike the compositor (which composites onto *clean* backgrounds), this
    augmenter pastes onto images that already have annotations, preserving
    the original bboxes and adding new ones with ``source=copy_paste``.
    """

    def __init__(self, config: CopyPasteConfig | None = None) -> None:
        self.config = config or CopyPasteConfig()

    def augment(
        self,
        record: ImageRecord,
        patches: list[DefectPatch],
        valid_zone: np.ndarray | None = None,
    ) -> ImageRecord:
        """Augment a single image by pasting defect patches onto it.

        Args:
            record: Input image with existing annotations.
            patches: Pool of defect patches to paste.
            valid_zone: Optional convex hull restricting placement.

        Returns:
            New ImageRecord with pasted patches and merged bboxes.
        """
        if not patches:
            return ImageRecord(
                image_path=Path(f"cp_{record.image_path.stem}.jpg"),
                bboxes=list(record.bboxes),
                image=record.image.copy() if record.image is not None else record.load_image().copy(),
                metadata={**record.metadata, "copy_paste": True, "patches_added": 0},
            )

        img = record.image if record.image is not None else record.load_image()
        composite = img.copy()
        img_h, img_w = composite.shape[:2]

        # Start with existing bboxes
        bboxes: list[BBox] = list(record.bboxes) if self.config.preserve_existing_bboxes else []

        num_to_paste = random.randint(1, self.config.max_patches_per_image)
        patches_added = 0

        for _ in range(num_to_paste):
            patch = random.choice(patches)

            # Apply scale + rotation jitter
            jittered_img, jittered_mask = self._apply_jitter(patch)
            p_h, p_w = jittered_img.shape[:2]

            # Clamp patch to fit within image
            if p_w >= img_w - 4 or p_h >= img_h - 4:
                continue

            # Find a valid placement
            placement = self._find_placement(
                p_w, p_h, img_w, img_h, bboxes, valid_zone
            )
            if placement is None:
                continue

            cx, cy = placement

            # Poisson blend
            composite = _poisson_blend(
                composite, jittered_img, jittered_mask, cx, cy, self.config.blend_mode
            )

            # Record new bbox
            norm_w = p_w / img_w
            norm_h = p_h / img_h
            norm_cx = cx / img_w
            norm_cy = cy / img_h

            new_bbox = clip_bbox(BBox(
                class_id=patch.class_id,
                x_center=norm_cx,
                y_center=norm_cy,
                width=norm_w,
                height=norm_h,
                source=AnnotationSource.copy_paste,
            ))
            bboxes.append(new_bbox)
            patches_added += 1

        return ImageRecord(
            image_path=Path(f"cp_{record.image_path.stem}.jpg"),
            bboxes=bboxes,
            image=composite,
            metadata={**record.metadata, "copy_paste": True, "patches_added": patches_added},
        )

    def augment_batch(
        self,
        records: list[ImageRecord],
        patches: list[DefectPatch],
        variants_per_image: int = 1,
        valid_zone: np.ndarray | None = None,
    ) -> list[ImageRecord]:
        """Generate copy-paste variants for a batch of images.

        Args:
            records: Input images.
            patches: Pool of defect patches.
            variants_per_image: Number of variants per input image.
            valid_zone: Optional convex hull restricting placement.

        Returns:
            List of augmented ImageRecords (does NOT include originals).
        """
        results: list[ImageRecord] = []
        for record in records:
            for v in range(variants_per_image):
                aug = self.augment(record, patches, valid_zone)
                aug.image_path = Path(f"cp_{record.image_path.stem}_v{v}.jpg")
                results.append(aug)

        logger.info(
            "Copy-paste augmented %d images x %d variants = %d new images",
            len(records), variants_per_image, len(results),
        )
        return results

    def _apply_jitter(
        self, patch: DefectPatch
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply scale and rotation jitter to a patch.

        Returns:
            (jittered_image, jittered_mask)
        """
        scale = random.uniform(*self.config.scale_jitter)
        new_w = max(4, int(patch.image.shape[1] * scale))
        new_h = max(4, int(patch.image.shape[0] * scale))

        scaled_img = cv2.resize(patch.image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        scaled_mask = cv2.resize(patch.mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if self.config.rotation_jitter > 0:
            angle = random.uniform(-self.config.rotation_jitter, self.config.rotation_jitter)
            scaled_img, scaled_mask = _rotate_patch(scaled_img, scaled_mask, angle)

        return scaled_img, scaled_mask

    def _find_placement(
        self,
        patch_w: int,
        patch_h: int,
        img_w: int,
        img_h: int,
        existing_bboxes: list[BBox],
        valid_zone: np.ndarray | None,
    ) -> tuple[int, int] | None:
        """Find a valid placement center avoiding overlaps and respecting valid zone."""
        half_w = patch_w // 2
        half_h = patch_h // 2

        x_lo = half_w + 1
        x_hi = img_w - half_w - 1
        y_lo = half_h + 1
        y_hi = img_h - half_h - 1

        if x_lo >= x_hi or y_lo >= y_hi:
            return None

        for _ in range(self.config.max_placement_attempts):
            cx = random.randint(x_lo, x_hi)
            cy = random.randint(y_lo, y_hi)

            # Check valid zone
            if not point_in_valid_zone(cx / img_w, cy / img_h, valid_zone):
                continue

            # Check overlap
            candidate = BBox(
                class_id=0,
                x_center=cx / img_w,
                y_center=cy / img_h,
                width=patch_w / img_w,
                height=patch_h / img_h,
            )
            if check_placement_valid(candidate, existing_bboxes, self.config.max_overlap_iou):
                return (cx, cy)

        return None
