"""Classical data augmentation with bbox-aware transforms via albumentations."""

from __future__ import annotations

import logging
from pathlib import Path

import albumentations as A
import numpy as np

from synthdet.config import AugmentationConfig
from synthdet.types import AnnotationSource, BBox, ImageRecord

logger = logging.getLogger(__name__)


class ClassicalAugmenter:
    """Bbox-aware image augmentation using albumentations."""

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        self.config = config or AugmentationConfig()
        self._transform = _build_transform(self.config)

    def augment(self, record: ImageRecord) -> ImageRecord:
        """Augment a single image, preserving bbox annotations.

        Args:
            record: Input ImageRecord (must have .image loaded).

        Returns:
            New ImageRecord with augmented image and updated bboxes.
        """
        img = record.image
        if img is None:
            img = record.load_image()

        # Convert BGR to RGB for albumentations
        img_rgb = img[:, :, ::-1].copy()

        # Prepare bboxes for albumentations (yolo format: x_center, y_center, w, h)
        bboxes_albu = []
        class_ids = []
        sources = []
        for bbox in record.bboxes:
            bboxes_albu.append([bbox.x_center, bbox.y_center, bbox.width, bbox.height])
            class_ids.append(bbox.class_id)
            sources.append(bbox.source)

        transformed = self._transform(
            image=img_rgb,
            bboxes=bboxes_albu,
            class_ids=class_ids,
        )

        # Convert back to BGR
        aug_img = transformed["image"][:, :, ::-1].copy()

        # Reconstruct BBox objects
        aug_bboxes = []
        for i, (bcoords, cls_id) in enumerate(
            zip(transformed["bboxes"], transformed["class_ids"])
        ):
            x_c, y_c, w, h = bcoords
            aug_bboxes.append(BBox(
                class_id=cls_id,
                x_center=float(x_c),
                y_center=float(y_c),
                width=float(w),
                height=float(h),
                source=sources[i] if i < len(sources) else AnnotationSource.compositor,
            ))

        return ImageRecord(
            image_path=Path(f"aug_{record.image_path.stem}.jpg"),
            bboxes=aug_bboxes,
            image=aug_img,
            metadata={
                **record.metadata,
                "augmented": True,
                "original_path": str(record.image_path),
            },
        )

    def augment_batch(
        self, records: list[ImageRecord], variants_per_image: int = 2
    ) -> list[ImageRecord]:
        """Generate multiple augmented variants per input image.

        Args:
            records: Input ImageRecords.
            variants_per_image: Number of variants to generate per image.

        Returns:
            List of augmented ImageRecords (does NOT include originals).
        """
        augmented: list[ImageRecord] = []
        for record in records:
            for v in range(variants_per_image):
                aug = self.augment(record)
                # Give unique path
                aug.image_path = Path(
                    f"aug_{record.image_path.stem}_v{v}.jpg"
                )
                augmented.append(aug)

        logger.info(
            "Augmented %d images x %d variants = %d new images",
            len(records), variants_per_image, len(augmented),
        )
        return augmented


def _build_transform(config: AugmentationConfig) -> A.Compose:
    """Build an albumentations pipeline with bbox support."""
    transforms = [
        A.HorizontalFlip(p=config.horizontal_flip_p),
        A.RandomBrightnessContrast(p=config.brightness_contrast_p),
        A.HueSaturationValue(p=config.hue_saturation_p),
        A.GaussNoise(p=config.noise_p),
        A.GaussianBlur(blur_limit=3, p=config.blur_p),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=config.rotate_limit,
            p=config.shift_scale_rotate_p,
            border_mode=0,  # cv2.BORDER_CONSTANT
        ),
    ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            "yolo",
            label_fields=["class_ids"],
            min_visibility=0.3,
        ),
    )
