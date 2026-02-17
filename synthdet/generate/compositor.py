"""Defect compositor — deterministic synthetic data with pixel-perfect annotations.

Workflow:
    1. Extract defect patches from annotated images (DefectPatchExtractor)
    2. Generate clean backgrounds via inpainting (BackgroundGenerator)
    3. Composite patches onto backgrounds with Poisson blending (DefectCompositor)
    4. Annotations are known because we placed the defects

This is the most reliable annotation strategy: no detection model needed.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from synthdet.config import AugmentationConfig, CompositorConfig
from synthdet.types import (
    AnnotationSource,
    BBox,
    BBoxSizeBucket,
    Dataset,
    GenerationTask,
    ImageRecord,
    SIZE_BUCKET_THRESHOLDS,
    SpatialRegion,
    SynthesisStrategy,
)
from synthdet.utils.bbox import bbox_iou, clip_bbox
from synthdet.utils.image import group_augmentation_variants

# Import shared placement functions. The canonical definitions live in
# placement.py; compositor.py keeps backward-compat aliases so existing
# callers (tests importing _determine_center etc.) keep working.
from synthdet.generate.placement import (  # noqa: F401
    check_placement_valid as _public_check_placement_valid,
    determine_center as _public_determine_center,
    point_in_valid_zone,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Valid zone (convex hull of existing annotation centers)
# ---------------------------------------------------------------------------


def compute_valid_zone(
    dataset: Dataset, margin: float = 0.05
) -> np.ndarray | None:
    """Compute the convex hull of bbox centers from the training set.

    The hull defines where defects naturally appear (e.g., the laptop surface).
    Placements outside this zone are rejected.

    Args:
        dataset: Source dataset.
        margin: Fractional margin to expand the hull outward, so placements
                near the edge are still allowed.

    Returns:
        Convex hull as an Nx1x2 array of float32 points (normalized 0-1),
        or None if fewer than 3 unique annotation centers exist.
    """
    centers = []
    for record in dataset.train:
        for bbox in record.bboxes:
            centers.append([bbox.x_center, bbox.y_center])

    if len(centers) < 3:
        return None

    pts = np.array(centers, dtype=np.float32)
    hull = cv2.convexHull(pts)

    if margin > 0:
        hull = _expand_hull(hull, margin)

    logger.info(
        "Computed valid zone from %d annotation centers (%d hull vertices, margin=%.2f)",
        len(centers), len(hull), margin,
    )
    return hull


def _expand_hull(hull: np.ndarray, margin: float) -> np.ndarray:
    """Expand a convex hull outward by a fractional margin.

    Moves each vertex away from the centroid by `margin` (in normalized coords).
    Clamps to [0, 1].
    """
    # hull shape: (N, 1, 2)
    points = hull.reshape(-1, 2)
    centroid = points.mean(axis=0)
    expanded = []
    for pt in points:
        direction = pt - centroid
        length = np.linalg.norm(direction)
        if length > 0:
            expanded_pt = pt + direction / length * margin
        else:
            expanded_pt = pt
        expanded.append(np.clip(expanded_pt, 0.0, 1.0))
    return np.array(expanded, dtype=np.float32).reshape(-1, 1, 2)


# point_in_valid_zone is imported from placement.py at module level.


# ---------------------------------------------------------------------------
# DefectPatch data type
# ---------------------------------------------------------------------------


@dataclass
class DefectPatch:
    """A cropped defect patch ready for compositing."""

    image: np.ndarray  # BGR crop with margin
    mask: np.ndarray  # uint8 mask (0-255), feathered edges for blending
    class_id: int
    original_size: tuple[float, float]  # (norm_width, norm_height)
    source_path: Path


# ---------------------------------------------------------------------------
# DefectPatchExtractor
# ---------------------------------------------------------------------------


class DefectPatchExtractor:
    """Extract defect patches from annotated dataset images."""

    def __init__(self, margin: float = 0.15, min_patch_pixels: int = 16) -> None:
        self.margin = margin
        self.min_patch_pixels = min_patch_pixels

    def extract_patches(self, dataset: Dataset) -> list[DefectPatch]:
        """Extract all defect patches from the train split.

        Uses augmentation variant grouping to avoid duplicate patches
        from Roboflow-augmented images. Only the first variant per
        source group is used.
        """
        groups = group_augmentation_variants(
            [r.image_path for r in dataset.train]
        )
        # Build a set of paths to use (first variant per group)
        use_paths = {paths[0] for paths in groups.values()}

        # Build path -> record lookup
        path_to_record = {r.image_path: r for r in dataset.train}

        patches: list[DefectPatch] = []
        for path in sorted(use_paths):
            record = path_to_record[path]
            img = record.load_image()
            img_h, img_w = img.shape[:2]
            for bbox in record.bboxes:
                patch = self._extract_single(img, bbox, img_w, img_h, path)
                if patch is not None:
                    patches.append(patch)

        logger.info("Extracted %d defect patches from %d source images", len(patches), len(use_paths))
        return patches

    def _extract_single(
        self,
        image: np.ndarray,
        bbox: BBox,
        img_w: int,
        img_h: int,
        source_path: Path,
    ) -> DefectPatch | None:
        """Extract a single patch with margin and create a feathered mask."""
        # Convert normalized bbox to pixel coordinates with margin
        margin_w = bbox.width * self.margin
        margin_h = bbox.height * self.margin

        x1 = max(0, int((bbox.x_center - bbox.width / 2 - margin_w) * img_w))
        y1 = max(0, int((bbox.y_center - bbox.height / 2 - margin_h) * img_h))
        x2 = min(img_w, int((bbox.x_center + bbox.width / 2 + margin_w) * img_w))
        y2 = min(img_h, int((bbox.y_center + bbox.height / 2 + margin_h) * img_h))

        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w < self.min_patch_pixels or crop_h < self.min_patch_pixels:
            logger.debug("Skipping too-small patch: %dx%d", crop_w, crop_h)
            return None

        crop = image[y1:y2, x1:x2].copy()
        mask = _create_feathered_mask(crop_h, crop_w)

        return DefectPatch(
            image=crop,
            mask=mask,
            class_id=bbox.class_id,
            original_size=(bbox.width, bbox.height),
            source_path=source_path,
        )


# ---------------------------------------------------------------------------
# BackgroundGenerator
# ---------------------------------------------------------------------------


class BackgroundGenerator:
    """Generate clean backgrounds by inpainting defects from annotated images."""

    def __init__(self, inpaint_radius: int = 5, method: str = "telea") -> None:
        self.inpaint_radius = inpaint_radius
        self.method = method

    def generate_from_dataset(self, dataset: Dataset) -> list[np.ndarray]:
        """Generate clean backgrounds from unique source images in the train split.

        For each unique source image, inpaints all annotated defect regions
        to produce a clean background.
        """
        groups = group_augmentation_variants(
            [r.image_path for r in dataset.train]
        )
        use_paths = {paths[0] for paths in groups.values()}
        path_to_record = {r.image_path: r for r in dataset.train}

        inpaint_flag = (
            cv2.INPAINT_TELEA if self.method == "telea" else cv2.INPAINT_NS
        )

        backgrounds: list[np.ndarray] = []
        for path in sorted(use_paths):
            record = path_to_record[path]
            img = record.load_image().copy()
            img_h, img_w = img.shape[:2]

            if not record.bboxes:
                # Already clean
                backgrounds.append(img)
                continue

            # Create mask covering all bbox regions (dilated slightly)
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for bbox in record.bboxes:
                x1 = max(0, int((bbox.x_center - bbox.width / 2) * img_w) - 2)
                y1 = max(0, int((bbox.y_center - bbox.height / 2) * img_h) - 2)
                x2 = min(img_w, int((bbox.x_center + bbox.width / 2) * img_w) + 2)
                y2 = min(img_h, int((bbox.y_center + bbox.height / 2) * img_h) + 2)
                mask[y1:y2, x1:x2] = 255

            # Dilate mask slightly for better inpainting
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations=1)

            bg = cv2.inpaint(img, mask, self.inpaint_radius, inpaint_flag)
            backgrounds.append(bg)

        logger.info("Generated %d clean backgrounds", len(backgrounds))
        return backgrounds

    def load_from_directory(self, directory: Path) -> list[np.ndarray]:
        """Load user-provided background images from a directory."""
        from synthdet.utils.image import find_image_files, load_image

        paths = find_image_files(directory)
        backgrounds = [load_image(p) for p in paths]
        logger.info("Loaded %d backgrounds from %s", len(backgrounds), directory)
        return backgrounds


# ---------------------------------------------------------------------------
# DefectCompositor
# ---------------------------------------------------------------------------


class DefectCompositor:
    """Composite defect patches onto backgrounds with Poisson blending.

    Implements the ImageGenerator protocol.
    """

    def __init__(self, config: CompositorConfig | None = None) -> None:
        self.config = config or CompositorConfig()

    def generate(
        self,
        task: GenerationTask,
        *,
        patches: list[DefectPatch],
        backgrounds: list[np.ndarray],
        img_size: tuple[int, int] = (860, 640),
        class_names: list[str] | None = None,
        valid_zone: np.ndarray | None = None,
        **kwargs: object,
    ) -> list[ImageRecord]:
        """Generate synthetic images for a task.

        Args:
            task: Generation task with targeting parameters.
            patches: Available defect patches.
            backgrounds: Clean background images.
            img_size: (width, height) for output images.
            class_names: Class name list for metadata.
            valid_zone: Convex hull of annotation centers (normalized 0-1).
                        Placements outside this zone are rejected.

        Returns:
            List of ImageRecord with composited images and annotations.
        """
        if not backgrounds:
            logger.warning("No backgrounds available, skipping task %s", task.task_id)
            return []

        img_w, img_h = img_size
        records: list[ImageRecord] = []

        # Negative example task (no defects)
        is_negative = len(task.target_classes) == 0

        for i in range(task.num_images):
            bg = random.choice(backgrounds).copy()
            # Resize background to target size if needed
            if bg.shape[:2] != (img_h, img_w):
                bg = cv2.resize(bg, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

            if is_negative:
                record = ImageRecord(
                    image_path=Path(f"synth_{task.task_id}_{i:04d}.jpg"),
                    bboxes=[],
                    image=bg,
                    metadata={
                        "source": "compositor",
                        "task_id": task.task_id,
                        "is_negative": True,
                    },
                )
                records.append(record)
                continue

            # Determine number of defects for this image
            num_defects = random.randint(1, self.config.max_defects_per_image)

            # Filter patches by target classes
            eligible = [p for p in patches if p.class_id in task.target_classes]
            if not eligible:
                eligible = patches  # Fallback to all patches

            bboxes: list[BBox] = []
            composite = bg.copy()

            for _ in range(num_defects):
                patch = random.choice(eligible)

                # Scale patch to target size bucket
                target_bucket = (
                    random.choice(task.target_size_buckets)
                    if task.target_size_buckets
                    else None
                )
                scaled_img, scaled_mask, scale_w, scale_h = _scale_patch(
                    patch, target_bucket, img_w, img_h, self.config.scale_jitter
                )

                # Determine target region
                target_region = (
                    random.choice(task.target_regions)
                    if task.target_regions
                    else None
                )

                # Find placement
                placement = _determine_center(
                    target_region,
                    scaled_img.shape[1],
                    scaled_img.shape[0],
                    img_w,
                    img_h,
                    bboxes,
                    self.config.max_placement_attempts,
                    self.config.max_overlap_iou,
                    valid_zone=valid_zone,
                )
                if placement is None:
                    continue

                cx, cy = placement

                # Apply rotation jitter
                if self.config.rotation_jitter > 0:
                    angle = random.uniform(
                        -self.config.rotation_jitter, self.config.rotation_jitter
                    )
                    scaled_img, scaled_mask = _rotate_patch(
                        scaled_img, scaled_mask, angle
                    )

                # Poisson blend
                composite = _poisson_blend(
                    composite, scaled_img, scaled_mask, cx, cy, self.config.blend_mode
                )

                # Record bbox (normalized YOLO format)
                norm_w = scaled_img.shape[1] / img_w
                norm_h = scaled_img.shape[0] / img_h
                norm_cx = cx / img_w
                norm_cy = cy / img_h

                new_bbox = clip_bbox(BBox(
                    class_id=patch.class_id,
                    x_center=norm_cx,
                    y_center=norm_cy,
                    width=norm_w,
                    height=norm_h,
                    source=AnnotationSource.compositor,
                ))
                bboxes.append(new_bbox)

            record = ImageRecord(
                image_path=Path(f"synth_{task.task_id}_{i:04d}.jpg"),
                bboxes=bboxes,
                image=composite,
                metadata={
                    "source": "compositor",
                    "task_id": task.task_id,
                    "num_defects": len(bboxes),
                },
            )
            records.append(record)

        logger.info(
            "Generated %d images for task %s (%d with defects, %d negative)",
            len(records),
            task.task_id,
            sum(1 for r in records if not r.is_negative),
            sum(1 for r in records if r.is_negative),
        )
        return records


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _create_feathered_mask(h: int, w: int) -> np.ndarray:
    """Create an elliptical mask with Gaussian feathering for seamless blending."""
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (max(1, w // 2 - 2), max(1, h // 2 - 2))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    # Feather edges with Gaussian blur
    ksize = max(3, min(h, w) // 4) | 1  # Ensure odd
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    return mask


def _scale_patch(
    patch: DefectPatch,
    target_bucket: BBoxSizeBucket | None,
    img_w: int,
    img_h: int,
    scale_jitter: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Scale a patch to match a target size bucket.

    Returns (scaled_image, scaled_mask, patch_w_pixels, patch_h_pixels).
    """
    if target_bucket is not None:
        lo, hi = SIZE_BUCKET_THRESHOLDS[target_bucket]
        # Target area in the middle of the bucket range
        target_area = (lo + hi) / 2
        if target_bucket == BBoxSizeBucket.large:
            target_area = (lo + min(hi, 0.2)) / 2
        if target_bucket == BBoxSizeBucket.tiny:
            target_area = max(target_area, 0.001)

        # Current normalized area from patch original size
        current_area = patch.original_size[0] * patch.original_size[1]
        if current_area > 0:
            scale_factor = (target_area / current_area) ** 0.5
        else:
            scale_factor = 1.0

        # Apply jitter
        jitter = random.uniform(*scale_jitter)
        scale_factor *= jitter

        new_w = max(4, int(patch.image.shape[1] * scale_factor))
        new_h = max(4, int(patch.image.shape[0] * scale_factor))
    else:
        # Just apply jitter to original size
        jitter = random.uniform(*scale_jitter)
        new_w = max(4, int(patch.image.shape[1] * jitter))
        new_h = max(4, int(patch.image.shape[0] * jitter))

    # Clamp to image bounds
    new_w = min(new_w, img_w - 4)
    new_h = min(new_h, img_h - 4)

    scaled_img = cv2.resize(patch.image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scaled_mask = cv2.resize(patch.mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return scaled_img, scaled_mask, new_w, new_h


def _determine_center(
    region: SpatialRegion | None,
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

    If valid_zone is provided, only placements whose normalized center falls
    inside the convex hull are accepted.
    """
    half_w = patch_w // 2
    half_h = patch_h // 2

    if region is not None:
        # Map region to pixel bounds
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

    # Ensure valid range
    rx1 = max(half_w, min(rx1, img_w - half_w))
    rx2 = max(rx1 + 1, min(rx2, img_w - half_w))
    ry1 = max(half_h, min(ry1, img_h - half_h))
    ry2 = max(ry1 + 1, min(ry2, img_h - half_h))

    for _ in range(max_attempts):
        cx = random.randint(rx1, rx2)
        cy = random.randint(ry1, ry2)

        # Check valid zone (normalized coordinates)
        if not point_in_valid_zone(cx / img_w, cy / img_h, valid_zone):
            continue

        # Check overlap with existing bboxes
        norm_w = patch_w / img_w
        norm_h = patch_h / img_h
        candidate = BBox(
            class_id=0,
            x_center=cx / img_w,
            y_center=cy / img_h,
            width=norm_w,
            height=norm_h,
        )
        if _check_placement_valid(candidate, existing_bboxes, max_iou):
            return (cx, cy)

    return None


def _check_placement_valid(
    new_bbox: BBox, existing: list[BBox], max_iou: float
) -> bool:
    """Check that a new placement doesn't overlap too much with existing bboxes."""
    for eb in existing:
        if bbox_iou(new_bbox, eb) > max_iou:
            return False
    return True


def _rotate_patch(
    image: np.ndarray, mask: np.ndarray, angle: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a patch and its mask by the given angle (degrees)."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    rotated_mask = cv2.warpAffine(mask, M, (w, h), borderValue=0)
    return rotated_img, rotated_mask


def _poisson_blend(
    background: np.ndarray,
    patch: np.ndarray,
    mask: np.ndarray,
    cx: int,
    cy: int,
    blend_mode: str,
) -> np.ndarray:
    """Composite a patch onto a background using Poisson blending.

    Falls back to alpha blending if seamlessClone fails (e.g. patch at image edge).
    """
    flag = cv2.MIXED_CLONE if blend_mode == "mixed" else cv2.NORMAL_CLONE
    bg_h, bg_w = background.shape[:2]
    p_h, p_w = patch.shape[:2]

    # Ensure center is valid for seamlessClone (patch must fit within background)
    # Clamp center so patch region stays within bounds
    cx = max(p_w // 2 + 1, min(cx, bg_w - p_w // 2 - 1))
    cy = max(p_h // 2 + 1, min(cy, bg_h - p_h // 2 - 1))

    # Ensure mask has non-zero area
    if mask.max() == 0:
        return background

    try:
        result = cv2.seamlessClone(patch, background, mask, (cx, cy), flag)
        return result
    except cv2.error:
        # Fallback: alpha blend
        logger.debug("seamlessClone failed, falling back to alpha blend")
        return _alpha_blend(background, patch, mask, cx, cy)


def _alpha_blend(
    background: np.ndarray,
    patch: np.ndarray,
    mask: np.ndarray,
    cx: int,
    cy: int,
) -> np.ndarray:
    """Simple alpha blending as fallback for Poisson blending."""
    bg_h, bg_w = background.shape[:2]
    p_h, p_w = patch.shape[:2]

    x1 = cx - p_w // 2
    y1 = cy - p_h // 2
    x2 = x1 + p_w
    y2 = y1 + p_h

    # Clip to image bounds
    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(bg_w, x2)
    dst_y2 = min(bg_h, y2)
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return background

    alpha = mask[src_y1:src_y2, src_x1:src_x2].astype(np.float32) / 255.0
    if alpha.ndim == 2:
        alpha = alpha[:, :, np.newaxis]

    result = background.copy()
    roi = result[dst_y1:dst_y2, dst_x1:dst_x2]
    src = patch[src_y1:src_y2, src_x1:src_x2]
    result[dst_y1:dst_y2, dst_x1:dst_x2] = (
        src * alpha + roi * (1 - alpha)
    ).astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# Top-level pipeline function
# ---------------------------------------------------------------------------


def run_compositor_pipeline(
    dataset: Dataset,
    strategy: SynthesisStrategy,
    config: CompositorConfig,
    output_dir: Path,
    augment_config: AugmentationConfig | None = None,
    seed: int | None = None,
) -> Dataset:
    """Full pipeline: extract patches -> generate backgrounds -> composite -> augment -> write.

    Args:
        dataset: Source YOLO dataset.
        strategy: Synthesis strategy with prioritized tasks.
        config: Compositor configuration.
        output_dir: Directory for output dataset.
        augment_config: Optional augmentation config (None to skip).
        seed: Random seed for reproducibility.

    Returns:
        Dataset object pointing to the generated output.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Get image size from dataset
    if dataset.train:
        sample = dataset.train[0].load_image()
        img_h, img_w = sample.shape[:2]
    else:
        img_w, img_h = 860, 640

    # Step 1: Extract patches
    logger.info("Step 1: Extracting defect patches...")
    extractor = DefectPatchExtractor(
        margin=config.patch_margin,
        min_patch_pixels=config.min_patch_pixels,
    )
    patches = extractor.extract_patches(dataset)

    if not patches:
        logger.warning("No patches extracted, generating negative examples only")

    # Step 2: Generate backgrounds
    logger.info("Step 2: Generating clean backgrounds...")
    bg_gen = BackgroundGenerator(
        inpaint_radius=config.inpaint_radius,
        method=config.inpaint_method,
    )
    backgrounds = bg_gen.generate_from_dataset(dataset)

    if not backgrounds:
        logger.error("No backgrounds generated, cannot proceed")
        return Dataset(
            root=output_dir,
            class_names=dataset.class_names,
            train=[],
            valid=[],
            test=[],
        )

    # Step 3: Compute valid zone
    logger.info("Step 3: Computing valid placement zone...")
    valid_zone = compute_valid_zone(dataset, margin=config.valid_zone_margin)

    # Step 4: Composite
    logger.info("Step 4: Running compositor...")
    compositor = DefectCompositor(config)

    all_records: list[ImageRecord] = []
    compositor_tasks = [
        t for t in strategy.generation_tasks if t.method == "compositor"
    ]

    for task in compositor_tasks:
        records = compositor.generate(
            task,
            patches=patches,
            backgrounds=backgrounds,
            img_size=(img_w, img_h),
            class_names=dataset.class_names,
            valid_zone=valid_zone,
        )
        all_records.extend(records)

    logger.info("Generated %d synthetic images", len(all_records))

    # Step 5: Optional augmentation
    if augment_config is not None and augment_config.enabled and all_records:
        logger.info("Step 5: Applying augmentation...")
        from synthdet.augment.classical import ClassicalAugmenter

        augmenter = ClassicalAugmenter(augment_config)
        augmented = augmenter.augment_batch(
            all_records, variants_per_image=augment_config.variants_per_image
        )
        all_records.extend(augmented)
        logger.info("Added %d augmented variants (total: %d)", len(augmented), len(all_records))

    # Step 6: Write output
    logger.info("Step 6: Writing YOLO dataset...")
    from synthdet.annotate.yolo_writer import write_yolo_dataset

    # Split: 85% train, 15% valid (maintain same proportions)
    random.shuffle(all_records)
    split_idx = int(len(all_records) * 0.85)
    records_by_split = {
        "train": all_records[:split_idx],
        "valid": all_records[split_idx:],
    }

    write_yolo_dataset(records_by_split, output_dir, dataset.class_names)

    # Return new dataset
    output_dataset = Dataset(
        root=output_dir,
        class_names=dataset.class_names,
        train=records_by_split["train"],
        valid=records_by_split["valid"],
        test=[],
    )

    logger.info(
        "Pipeline complete: %d train, %d valid images in %s",
        len(records_by_split["train"]),
        len(records_by_split["valid"]),
        output_dir,
    )
    return output_dataset
