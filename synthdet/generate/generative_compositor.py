"""Generative defect compositor — API-generated patches + Poisson blending.

Hybrid approach that combines the strengths of both pipelines:
    1. **API generation**: Imagen ``generate_image`` creates isolated defect
       patches on neutral backgrounds (no inpainting conflict).
    2. **Local compositing**: Poisson blending (from compositor.py) merges
       the patches onto real backgrounds with natural lighting adaptation.

This avoids the fundamental inpainting problem where the model reconstructs
clean surfaces instead of adding visible defects.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from synthdet.generate.compositor import (
    BackgroundGenerator,
    DefectPatch,
    _alpha_blend,
    _create_feathered_mask,
    _poisson_blend,
    _rotate_patch,
    compute_valid_zone,
)
from synthdet.generate.errors import InpaintingAPIError
from synthdet.generate.placement import (
    check_placement_valid,
    determine_center,
    sample_bbox_dimensions,
)
from synthdet.types import (
    AnnotationSource,
    BBox,
    BBoxSizeBucket,
    Dataset,
    GenerationTask,
    ImageRecord,
    SpatialRegion,
    SynthesisStrategy,
)
from synthdet.utils.bbox import clip_bbox
from synthdet.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defect prompt library
# ---------------------------------------------------------------------------

DEFAULT_DEFECT_PROMPTS: dict[str, list[str]] = {
    "broken": [
        "A piece of broken plastic casing, cracked into fragments, top-down close-up",
        "A shattered section of a hard surface with jagged crack lines",
        "Broken material with visible fracture lines and separated pieces",
    ],
    "hard scratch": [
        "A deep scratch gouged into a metal surface, with visible groove depth",
        "Multiple deep parallel scratches on brushed aluminum, harsh lighting",
        "A prominent scratch mark on a dark matte surface revealing lighter material underneath",
    ],
    "minor crack": [
        "A thin hairline crack on a plastic surface, barely visible but distinct",
        "A fine crack line on a smooth painted surface, branching slightly at one end",
        "A subtle crack on a laptop lid surface, running diagonally",
    ],
    "minor dent": [
        "A small circular dent on a flat metal surface, with slight shadow",
        "A shallow dent impression on brushed aluminum, showing distorted reflection",
        "A small impact dent on a smooth surface, approximately 5mm diameter",
    ],
    "minor scratch": [
        "A light surface scratch on brushed metal, barely catching the light",
        "A thin cosmetic scratch on a dark matte surface",
        "A fine linear scratch mark on an anodized aluminum surface",
    ],
    "sticker marks": [
        "Adhesive residue marks on a smooth surface, slightly cloudy and irregular",
        "Sticker removal residue with partial paper remnants on a flat surface",
        "Glue marks and sticky residue patches on a clean surface",
    ],
}


# ---------------------------------------------------------------------------
# GenerativeDefectCompositor
# ---------------------------------------------------------------------------


class GenerativeDefectCompositor:
    """Generate synthetic defect images using API-generated patches.

    Combines Imagen ``generate_image`` for patch creation with Poisson
    blending for compositing onto real backgrounds.
    """

    def __init__(
        self,
        provider: Any,  # ImagenGenerateProvider
        *,
        class_prompts: dict[str, list[str]] | None = None,
        blend_mode: str = "mixed",
        max_defects_per_image: int = 3,
        max_placement_attempts: int = 20,
        max_overlap_iou: float = 0.3,
        rotation_jitter: float = 15.0,
        requests_per_minute: float = 15.0,
        max_cost_usd: float = 10.0,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        output_format: str = "jpg",
        output_quality: int = 95,
    ) -> None:
        self.provider = provider
        self.class_prompts = class_prompts or DEFAULT_DEFECT_PROMPTS
        self.blend_mode = blend_mode
        self.max_defects_per_image = max_defects_per_image
        self.max_placement_attempts = max_placement_attempts
        self.max_overlap_iou = max_overlap_iou
        self.rotation_jitter = rotation_jitter
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.output_format = output_format
        self.output_quality = output_quality
        self.max_cost_usd = max_cost_usd

        self._rate_limiter = RateLimiter(requests_per_minute)
        self._cumulative_cost = 0.0

    @property
    def cumulative_cost(self) -> float:
        return self._cumulative_cost

    def generate(
        self,
        task: GenerationTask,
        *,
        backgrounds: list[np.ndarray],
        img_size: tuple[int, int] = (860, 640),
        class_names: list[str] | None = None,
        valid_zone: np.ndarray | None = None,
        output_dir: Path | None = None,
        seed: int | None = None,
        **kwargs: object,
    ) -> list[ImageRecord]:
        """Generate synthetic images for a task.

        For each image:
            1. Pick a random background
            2. For each defect:
                a. Generate an isolated defect patch via API
                b. Extract mask via background subtraction
                c. Scale + rotate to target size
                d. Poisson-blend onto background
            3. Record the bbox annotations

        Args:
            task: Generation task with targeting parameters.
            backgrounds: Clean background images.
            img_size: (width, height) for output images.
            class_names: Class name list for prompt lookup.
            valid_zone: Convex hull for placement validation.
            output_dir: If set, save each image incrementally.
            seed: Random seed for generation.

        Returns:
            List of ImageRecord with composited images and annotations.
        """
        if not backgrounds:
            logger.warning("No backgrounds available, skipping task %s", task.task_id)
            return []

        img_w, img_h = img_size
        records: list[ImageRecord] = []
        is_negative = len(task.target_classes) == 0

        for i in range(task.num_images):
            # Cost guard
            if self.max_cost_usd > 0 and self._cumulative_cost >= self.max_cost_usd:
                logger.warning(
                    "Cost limit reached ($%.2f >= $%.2f), stopping",
                    self._cumulative_cost, self.max_cost_usd,
                )
                break

            bg = random.choice(backgrounds).copy()
            if bg.shape[:2] != (img_h, img_w):
                bg = cv2.resize(bg, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

            # Negative example
            if is_negative:
                record = ImageRecord(
                    image_path=Path(f"synth_{task.task_id}_{i:04d}.jpg"),
                    bboxes=[],
                    image=bg,
                    metadata={
                        "source": "generative_compositor",
                        "task_id": task.task_id,
                        "is_negative": True,
                    },
                )
                _save_incremental(record, output_dir, self.output_format, self.output_quality)
                records.append(record)
                continue

            # Generate defects
            num_defects = random.randint(1, self.max_defects_per_image)
            composite = bg.copy()
            bboxes: list[BBox] = []

            for d in range(num_defects):
                if self.max_cost_usd > 0 and self._cumulative_cost >= self.max_cost_usd:
                    break

                # Pick targets
                target_bucket = (
                    random.choice(task.target_size_buckets)
                    if task.target_size_buckets else None
                )
                target_region = (
                    random.choice(task.target_regions)
                    if task.target_regions else None
                )
                class_id = random.choice(task.target_classes)
                class_name = (
                    class_names[class_id]
                    if class_names and class_id < len(class_names)
                    else "defect"
                )

                # Sample target pixel dimensions
                if target_bucket is not None:
                    px_w, px_h = sample_bbox_dimensions(target_bucket, img_w, img_h)
                else:
                    px_w = random.randint(max(4, img_w // 10), img_w // 3)
                    px_h = random.randint(max(4, img_h // 10), img_h // 3)

                # Find placement first (before API call to save cost)
                center = determine_center(
                    target_region, px_w, px_h, img_w, img_h,
                    bboxes, self.max_placement_attempts,
                    self.max_overlap_iou, valid_zone=valid_zone,
                )
                if center is None:
                    logger.debug("Could not find valid placement, skipping defect")
                    continue

                cx, cy = center

                # Generate defect patch via API
                prompt = self._pick_prompt(class_name)
                patch_bgr, _unused_mask = self._call_provider(
                    prompt, size=(px_w, px_h), seed=seed,
                )
                if patch_bgr is None:
                    continue

                # Ensure patch is the right size
                if patch_bgr.shape[:2] != (px_h, px_w):
                    patch_bgr = cv2.resize(patch_bgr, (px_w, px_h), interpolation=cv2.INTER_LINEAR)

                # Always use a feathered elliptical mask for blending.
                # Poisson MIXED_CLONE transfers gradients (texture), not
                # absolute colours, so the gray background in the generated
                # image is naturally adapted to the target surface.
                # Background-subtracted masks cause visible halos/artifacts.
                patch_mask = _create_feathered_mask(px_h, px_w)

                # Apply rotation jitter
                if self.rotation_jitter > 0:
                    angle = random.uniform(-self.rotation_jitter, self.rotation_jitter)
                    patch_bgr, patch_mask = _rotate_patch(patch_bgr, patch_mask, angle)

                # Poisson blend onto composite
                composite = _poisson_blend(
                    composite, patch_bgr, patch_mask, cx, cy, self.blend_mode,
                )

                # Record bbox
                norm_w = px_w / img_w
                norm_h = px_h / img_h
                norm_cx = cx / img_w
                norm_cy = cy / img_h
                new_bbox = clip_bbox(BBox(
                    class_id=class_id,
                    x_center=norm_cx,
                    y_center=norm_cy,
                    width=norm_w,
                    height=norm_h,
                    source=AnnotationSource.compositor,
                ))
                bboxes.append(new_bbox)

            if bboxes:
                record = ImageRecord(
                    image_path=Path(f"synth_{task.task_id}_{i:04d}.jpg"),
                    bboxes=bboxes,
                    image=composite,
                    metadata={
                        "source": "generative_compositor",
                        "task_id": task.task_id,
                        "num_defects": len(bboxes),
                    },
                )
                _save_incremental(record, output_dir, self.output_format, self.output_quality)
                records.append(record)

        logger.info(
            "Generated %d images for task %s (cost: $%.2f)",
            len(records), task.task_id, self._cumulative_cost,
        )
        return records

    def _pick_prompt(self, class_name: str) -> str:
        """Pick a random prompt for the given class name."""
        key_lower = class_name.lower()
        for k, prompts in self.class_prompts.items():
            if k.lower() == key_lower:
                return random.choice(prompts)
        # Fallback: use class name directly
        return f"A realistic {class_name} defect on a surface, close-up photograph"

    def _call_provider(
        self,
        prompt: str,
        size: tuple[int, int],
        seed: int | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Call the provider with rate limiting and retries."""
        for attempt in range(self.max_retries):
            if not self._rate_limiter.acquire(timeout=60.0):
                logger.warning("Rate limiter timeout, skipping")
                return None, None
            try:
                patch_bgr, mask = self.provider.generate_defect_patch(
                    prompt, size=size, seed=seed,
                )
                self._cumulative_cost += self.provider.cost_per_image
                return patch_bgr, mask
            except InpaintingAPIError as e:
                if e.retryable and attempt < self.max_retries - 1:
                    delay = self.retry_delay_seconds * (2 ** attempt)
                    logger.warning(
                        "Retryable error (attempt %d): %s — retrying in %.1fs",
                        attempt + 1, e, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error("Generation API error: %s", e)
                    return None, None
        return None, None


# ---------------------------------------------------------------------------
# Incremental save
# ---------------------------------------------------------------------------


def _save_incremental(
    record: ImageRecord,
    output_dir: Path | None,
    output_format: str,
    output_quality: int,
) -> None:
    """Save a single record to disk immediately."""
    if output_dir is None:
        return

    images_dir = output_dir / "incremental" / "images"
    labels_dir = output_dir / "incremental" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stem = record.image_path.stem
    ext = f".{output_format}"
    img_path = images_dir / f"{stem}{ext}"
    label_path = labels_dir / f"{stem}.txt"

    if record.image is not None:
        if output_format == "jpg":
            cv2.imwrite(str(img_path), record.image, [cv2.IMWRITE_JPEG_QUALITY, output_quality])
        else:
            cv2.imwrite(str(img_path), record.image)

    lines = [bbox.to_yolo_line() for bbox in record.bboxes]
    label_path.write_text("\n".join(lines) + "\n" if lines else "")
    logger.debug("Saved %s (%d bboxes)", img_path.name, len(record.bboxes))


# ---------------------------------------------------------------------------
# Top-level pipeline function
# ---------------------------------------------------------------------------


def run_generative_compositor_pipeline(
    dataset: Dataset,
    strategy: SynthesisStrategy,
    *,
    class_prompts: dict[str, list[str]] | None = None,
    output_dir: Path,
    blend_mode: str = "mixed",
    max_defects_per_image: int = 3,
    requests_per_minute: float = 15.0,
    max_cost_usd: float = 10.0,
    seed: int | None = None,
    # Provider kwargs
    model: str = "imagen-3.0-generate-002",
    project: str | None = None,
    location: str | None = None,
    bg_color: tuple[int, int, int] = (220, 220, 220),
    threshold_delta: int = 30,
) -> Dataset:
    """Full generative compositor pipeline.

    Args:
        dataset: Source YOLO dataset.
        strategy: Synthesis strategy with prioritized tasks.
        class_prompts: Per-class prompt lists. Defaults to built-in library.
        output_dir: Output directory.
        blend_mode: Poisson blend mode ('mixed' or 'normal').
        max_defects_per_image: Max defects per synthetic image.
        requests_per_minute: API rate limit.
        max_cost_usd: Cost safety limit.
        seed: Random seed.
        model: Imagen model ID.
        project: GCP project ID (or GOOGLE_CLOUD_PROJECT env var).
        location: GCP region (or GOOGLE_CLOUD_LOCATION env var).
        bg_color: RGB background color for generation prompts.
        threshold_delta: Mask extraction threshold.

    Returns:
        Dataset object pointing to the generated output.
    """
    from synthdet.generate.providers.imagen_generate import ImagenGenerateProvider

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Get image size
    if dataset.train:
        sample = dataset.train[0].load_image()
        img_h, img_w = sample.shape[:2]
    else:
        img_w, img_h = 860, 640

    # Step 1: Generate backgrounds
    logger.info("Step 1: Generating clean backgrounds...")
    bg_gen = BackgroundGenerator(inpaint_radius=5, method="telea")
    backgrounds = bg_gen.generate_from_dataset(dataset)

    if not backgrounds:
        logger.error("No backgrounds generated, cannot proceed")
        return Dataset(root=output_dir, class_names=dataset.class_names, train=[], valid=[], test=[])

    # Step 2: Compute valid zone
    logger.info("Step 2: Computing valid placement zone...")
    valid_zone = compute_valid_zone(dataset, margin=0.05)

    # Step 3: Create provider + compositor
    logger.info("Step 3: Creating generative defect provider...")
    provider = ImagenGenerateProvider(
        model=model,
        project=project,
        location=location,
        bg_color=bg_color,
        threshold_delta=threshold_delta,
    )

    compositor = GenerativeDefectCompositor(
        provider,
        class_prompts=class_prompts,
        blend_mode=blend_mode,
        max_defects_per_image=max_defects_per_image,
        requests_per_minute=requests_per_minute,
        max_cost_usd=max_cost_usd,
    )

    # Step 4: Generate
    logger.info("Step 4: Running generative compositor...")
    all_records: list[ImageRecord] = []

    # Use all generation tasks (both compositor and inpainting tasks)
    for task in strategy.generation_tasks:
        if len(task.target_classes) == 0:
            continue  # Skip negative-only tasks for now
        records = compositor.generate(
            task,
            backgrounds=backgrounds,
            img_size=(img_w, img_h),
            class_names=dataset.class_names,
            valid_zone=valid_zone,
            output_dir=output_dir,
            seed=seed,
        )
        all_records.extend(records)

    logger.info(
        "Generated %d synthetic images (total cost: $%.2f)",
        len(all_records), compositor.cumulative_cost,
    )

    # Step 5: Write output
    logger.info("Step 5: Writing YOLO dataset...")
    from synthdet.annotate.yolo_writer import write_yolo_dataset

    random.shuffle(all_records)
    split_idx = int(len(all_records) * 0.85)
    records_by_split = {
        "train": all_records[:split_idx],
        "valid": all_records[split_idx:],
    }

    write_yolo_dataset(records_by_split, output_dir, dataset.class_names)

    return Dataset(
        root=output_dir,
        class_names=dataset.class_names,
        train=records_by_split["train"],
        valid=records_by_split["valid"],
        test=[],
    )
