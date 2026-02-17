"""Shared placement logic for compositor and inpainting generators.

Extracted from compositor.py so that both the compositor and the inpainting
pipeline can place defects (or masks) consistently.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

from synthdet.types import BBox, BBoxSizeBucket, SIZE_BUCKET_THRESHOLDS
from synthdet.utils.bbox import bbox_iou


def point_in_valid_zone(
    x: float, y: float, hull: np.ndarray | None
) -> bool:
    """Check if a normalized (x, y) point is inside the valid zone hull."""
    if hull is None:
        return True  # No hull = no restriction
    result = cv2.pointPolygonTest(hull, (x, y), measureDist=False)
    return result >= 0  # >= 0 means inside or on edge


def check_placement_valid(
    new_bbox: BBox, existing: list[BBox], max_iou: float
) -> bool:
    """Check that a new placement doesn't overlap too much with existing bboxes."""
    for eb in existing:
        if bbox_iou(new_bbox, eb) > max_iou:
            return False
    return True


def determine_center(
    region: "SpatialRegion | None",
    patch_w: int,
    patch_h: int,
    img_w: int,
    img_h: int,
    existing_bboxes: list[BBox],
    max_attempts: int,
    max_iou: float,
    valid_zone: np.ndarray | None = None,
) -> tuple[int, int] | None:
    """Find a valid placement center within a target region, avoiding overlaps.

    If *valid_zone* is provided, only placements whose normalised center falls
    inside the convex hull are accepted.
    """
    from synthdet.types import SpatialRegion

    half_w = patch_w // 2
    half_h = patch_h // 2

    if region is not None:
        regions = list(SpatialRegion)
        idx = regions.index(region)
        col = idx % 3
        row = idx // 3

        rx1 = int(col / 3 * img_w) + half_w
        rx2 = int((col + 1) / 3 * img_w) - half_w
        ry1 = int(row / 3 * img_h) + half_h
        ry2 = int((row + 1) / 3 * img_h) - half_h
    else:
        rx1 = half_w
        rx2 = img_w - half_w
        ry1 = half_h
        ry2 = img_h - half_h

    rx1 = max(half_w, min(rx1, img_w - half_w))
    rx2 = max(rx1 + 1, min(rx2, img_w - half_w))
    ry1 = max(half_h, min(ry1, img_h - half_h))
    ry2 = max(ry1 + 1, min(ry2, img_h - half_h))

    for _ in range(max_attempts):
        cx = random.randint(rx1, rx2)
        cy = random.randint(ry1, ry2)

        if not point_in_valid_zone(cx / img_w, cy / img_h, valid_zone):
            continue

        norm_w = patch_w / img_w
        norm_h = patch_h / img_h
        candidate = BBox(
            class_id=0,
            x_center=cx / img_w,
            y_center=cy / img_h,
            width=norm_w,
            height=norm_h,
        )
        if check_placement_valid(candidate, existing_bboxes, max_iou):
            return (cx, cy)

    return None


def sample_bbox_dimensions(
    target_bucket: BBoxSizeBucket,
    img_w: int,
    img_h: int,
    aspect_ratio_range: tuple[float, float] = (0.3, 3.0),
) -> tuple[int, int]:
    """Sample pixel (width, height) for a target size bucket.

    Picks a random normalised area within the bucket thresholds, a random
    aspect ratio, then converts to pixel dimensions clamped to the image.
    """
    lo, hi = SIZE_BUCKET_THRESHOLDS[target_bucket]
    # Clamp upper bound for large bucket
    if target_bucket == BBoxSizeBucket.large:
        hi = min(hi, 0.2)
    if target_bucket == BBoxSizeBucket.tiny:
        lo = max(lo, 0.0005)

    target_area = random.uniform(lo, hi)
    ar = random.uniform(*aspect_ratio_range)

    # area = norm_w * norm_h, ar = norm_w / norm_h
    norm_h = math.sqrt(target_area / ar)
    norm_w = target_area / norm_h

    px_w = max(4, min(int(norm_w * img_w), img_w - 4))
    px_h = max(4, min(int(norm_h * img_h), img_h - 4))
    return px_w, px_h
