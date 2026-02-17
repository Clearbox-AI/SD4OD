"""API-based inpainting generation with mask-defined annotations.

Workflow:
    1. Pick a clean background (reused from compositor)
    2. Place mask(s) where defects should go (MaskPlacer)
    3. Send background + mask + prompt to an InpaintingProvider
    4. Annotations are the mask regions — near-perfect by construction

The mask region IS the bbox, same principle as the compositor.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from synthdet.config import AugmentationConfig, InpaintingConfig
from synthdet.generate.compositor import BackgroundGenerator
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
# InpaintingProvider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class InpaintingProvider(Protocol):
    """Protocol for inpainting API providers."""

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        *,
        seed: int | None = None,
        num_images: int = 1,
    ) -> list[np.ndarray]: ...

    @property
    def cost_per_image(self) -> float: ...


# ---------------------------------------------------------------------------
# MaskPlacement
# ---------------------------------------------------------------------------


@dataclass
class MaskPlacement:
    """A binary mask and its corresponding YOLO bbox."""

    mask: np.ndarray  # uint8, 255=edit region, 0=preserve
    bbox: BBox  # YOLO-format bbox derived from mask coordinates


# ---------------------------------------------------------------------------
# MaskPlacer
# ---------------------------------------------------------------------------


class MaskPlacer:
    """Place masks onto images targeting specific size buckets and spatial regions."""

    def __init__(self, config: InpaintingConfig) -> None:
        self.config = config

    def create_mask(
        self,
        img_w: int,
        img_h: int,
        target_size_bucket: BBoxSizeBucket | None,
        target_region: SpatialRegion | None,
        class_id: int,
        existing_bboxes: list[BBox],
        valid_zone: np.ndarray | None = None,
    ) -> MaskPlacement | None:
        """Create a mask placement on an image.

        Returns None if no valid placement can be found.
        """
        # Sample dimensions for the mask
        if target_size_bucket is not None:
            px_w, px_h = sample_bbox_dimensions(target_size_bucket, img_w, img_h)
        else:
            px_w = random.randint(max(4, img_w // 10), img_w // 3)
            px_h = random.randint(max(4, img_h // 10), img_h // 3)

        # Apply padding so the mask extends beyond the strict bbox
        pad_w = int(px_w * self.config.mask_padding)
        pad_h = int(px_h * self.config.mask_padding)
        mask_w = min(px_w + 2 * pad_w, img_w - 4)
        mask_h = min(px_h + 2 * pad_h, img_h - 4)

        # Find a valid center
        center = determine_center(
            target_region,
            mask_w,
            mask_h,
            img_w,
            img_h,
            existing_bboxes,
            self.config.max_placement_attempts,
            self.config.max_overlap_iou,
            valid_zone=valid_zone,
        )
        if center is None:
            return None

        cx, cy = center

        # Build the full-image binary mask
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        x1 = max(0, cx - mask_w // 2)
        y1 = max(0, cy - mask_h // 2)
        x2 = min(img_w, x1 + mask_w)
        y2 = min(img_h, y1 + mask_h)

        if self.config.mask_shape == "ellipse":
            ecx = (x1 + x2) // 2
            ecy = (y1 + y2) // 2
            axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
            cv2.ellipse(mask, (ecx, ecy), axes, 0, 0, 360, 255, -1)
        else:
            mask[y1:y2, x1:x2] = 255

        # The bbox is derived from the actual filled region (inner, no padding)
        bbox_x1 = max(0, cx - px_w // 2)
        bbox_y1 = max(0, cy - px_h // 2)
        bbox_x2 = min(img_w, bbox_x1 + px_w)
        bbox_y2 = min(img_h, bbox_y1 + px_h)

        norm_cx = (bbox_x1 + bbox_x2) / 2 / img_w
        norm_cy = (bbox_y1 + bbox_y2) / 2 / img_h
        norm_w = (bbox_x2 - bbox_x1) / img_w
        norm_h = (bbox_y2 - bbox_y1) / img_h

        bbox = clip_bbox(BBox(
            class_id=class_id,
            x_center=norm_cx,
            y_center=norm_cy,
            width=norm_w,
            height=norm_h,
            source=AnnotationSource.inpainting,
        ))

        return MaskPlacement(mask=mask, bbox=bbox)


# ---------------------------------------------------------------------------
# InpaintingGenerator
# ---------------------------------------------------------------------------


class InpaintingGenerator:
    """Generate synthetic images via inpainting.  Satisfies ImageGenerator protocol."""

    def __init__(
        self,
        provider: InpaintingProvider,
        config: InpaintingConfig,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.rate_limiter = rate_limiter or RateLimiter(config.requests_per_minute)
        self._cumulative_cost = 0.0
        self._mask_placer = MaskPlacer(config)

    def generate(
        self,
        task: GenerationTask,
        *,
        backgrounds: list[np.ndarray],
        img_size: tuple[int, int] = (860, 640),
        class_names: list[str] | None = None,
        valid_zone: np.ndarray | None = None,
        dry_run: bool = False,
        seed: int | None = None,
        output_dir: Path | None = None,
        **kwargs: object,
    ) -> list[ImageRecord]:
        """Generate synthetic images for a task via inpainting.

        If *output_dir* is provided, each image is saved to disk immediately
        after generation so progress is not lost on crash.
        """
        if not backgrounds:
            logger.warning("No backgrounds available, skipping task %s", task.task_id)
            return []

        img_w, img_h = img_size
        records: list[ImageRecord] = []
        is_negative = len(task.target_classes) == 0

        for i in range(task.num_images):
            # Cost guard
            if self.config.max_cost_usd > 0 and self._cumulative_cost >= self.config.max_cost_usd:
                logger.warning(
                    "Cost limit reached ($%.2f >= $%.2f), stopping generation",
                    self._cumulative_cost, self.config.max_cost_usd,
                )
                break

            bg = random.choice(backgrounds).copy()
            if bg.shape[:2] != (img_h, img_w):
                bg = cv2.resize(bg, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

            # Negative example: just return the background
            if is_negative:
                record = ImageRecord(
                    image_path=Path(f"synth_{task.task_id}_{i:04d}.jpg"),
                    bboxes=[],
                    image=bg,
                    metadata={"source": "inpainting", "task_id": task.task_id, "is_negative": True},
                )
                _save_record_incremental(record, output_dir, self.config)
                records.append(record)
                continue

            if dry_run:
                est_cost = self.provider.cost_per_image * min(
                    self.config.max_defects_per_image, len(task.target_classes) or 1
                )
                logger.info(
                    "[dry-run] Task %s image %d: ~%d defects, est. cost $%.3f",
                    task.task_id, i, self.config.max_defects_per_image, est_cost,
                )
                self._cumulative_cost += est_cost
                continue

            # Generate defects sequentially
            num_defects = random.randint(1, self.config.max_defects_per_image)
            current_img = bg.copy()
            bboxes: list[BBox] = []

            for d in range(num_defects):
                # Cost guard (per-defect)
                if self.config.max_cost_usd > 0 and self._cumulative_cost >= self.config.max_cost_usd:
                    break

                target_bucket = (
                    random.choice(task.target_size_buckets) if task.target_size_buckets else None
                )
                target_region = (
                    random.choice(task.target_regions) if task.target_regions else None
                )
                class_id = random.choice(task.target_classes)

                placement = self._mask_placer.create_mask(
                    img_w, img_h, target_bucket, target_region,
                    class_id, bboxes, valid_zone,
                )
                if placement is None:
                    continue

                class_name = class_names[class_id] if class_names and class_id < len(class_names) else "defect"
                prompt = _build_inpainting_prompt(self.config, class_name, task.suggested_prompts)

                # Rate limit + API call with retries
                result_img = self._call_provider(
                    current_img, placement.mask, prompt, seed=seed,
                )
                if result_img is None:
                    continue

                current_img = result_img
                bboxes.append(placement.bbox)

            if bboxes:
                record = ImageRecord(
                    image_path=Path(f"synth_{task.task_id}_{i:04d}.jpg"),
                    bboxes=bboxes,
                    image=current_img,
                    metadata={
                        "source": "inpainting",
                        "task_id": task.task_id,
                        "num_defects": len(bboxes),
                    },
                )
                _save_record_incremental(record, output_dir, self.config)
                records.append(record)

        logger.info(
            "Generated %d images for task %s (cumulative cost: $%.2f)",
            len(records), task.task_id, self._cumulative_cost,
        )
        return records

    def _call_provider(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        seed: int | None = None,
    ) -> np.ndarray | None:
        """Call the provider with rate limiting and retry logic."""
        for attempt in range(self.config.max_retries):
            if not self.rate_limiter.acquire(timeout=60.0):
                logger.warning("Rate limiter timeout, skipping")
                return None
            try:
                results = self.provider.inpaint(image, mask, prompt, seed=seed, num_images=1)
                self._cumulative_cost += self.provider.cost_per_image
                return results[0] if results else None
            except InpaintingAPIError as e:
                if e.retryable and attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay_seconds * (2 ** attempt)
                    logger.warning("Retryable error (attempt %d): %s — retrying in %.1fs", attempt + 1, e, delay)
                    time.sleep(delay)
                else:
                    logger.error("Inpainting API error: %s", e)
                    return None
        return None


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _build_inpainting_prompt(
    config: InpaintingConfig,
    class_name: str,
    task_prompts: list[str],
) -> str:
    """Build the final prompt sent to the inpainting API.

    Resolution order for the base prompt:
      1. ``config.class_prompts[class_name]`` (case-insensitive lookup)
      2. ``config.default_prompts``
      3. ``task_prompts`` (from the strategy's suggested_prompts)

    The chosen prompt is then formatted through ``config.prompt_template``
    with ``{prompt}`` and ``{class_name}`` placeholders.
    """
    # Resolve base prompt list
    prompts: list[str] = []
    if config.class_prompts:
        # Case-insensitive lookup
        key_lower = class_name.lower()
        for k, v in config.class_prompts.items():
            if k.lower() == key_lower:
                prompts = v
                break
    if not prompts:
        prompts = config.default_prompts
    if not prompts:
        prompts = task_prompts or ["surface defect"]

    base = random.choice(prompts)

    try:
        return config.prompt_template.format(prompt=base, class_name=class_name)
    except KeyError:
        # Template uses an unknown placeholder — fall back to plain prompt
        return base


# ---------------------------------------------------------------------------
# Incremental save
# ---------------------------------------------------------------------------


def _save_record_incremental(
    record: ImageRecord, output_dir: Path | None, config: InpaintingConfig,
) -> None:
    """Save a single record to disk immediately (crash-safe incremental writes).

    Writes to ``output_dir/train/images/`` and ``output_dir/train/labels/``.
    All incrementally saved images go to the train split; the final pipeline
    step redistributes into train/valid.
    """
    if output_dir is None:
        return

    images_dir = output_dir / "incremental" / "images"
    labels_dir = output_dir / "incremental" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stem = record.image_path.stem
    ext = f".{config.output_format}"
    img_path = images_dir / f"{stem}{ext}"
    label_path = labels_dir / f"{stem}.txt"

    if record.image is not None:
        if config.output_format == "jpg":
            cv2.imwrite(str(img_path), record.image, [cv2.IMWRITE_JPEG_QUALITY, config.output_quality])
        else:
            cv2.imwrite(str(img_path), record.image)

    lines = [bbox.to_yolo_line() for bbox in record.bboxes]
    label_path.write_text("\n".join(lines) + "\n" if lines else "")

    logger.debug("Saved %s (%d bboxes)", img_path.name, len(record.bboxes))


# ---------------------------------------------------------------------------
# Placement zone
# ---------------------------------------------------------------------------


def _compute_placement_zone(
    dataset: Dataset, config: InpaintingConfig,
) -> np.ndarray | None:
    """Compute the rectangular region where defects may be placed.

    Uses ``config.placement_region`` if set explicitly, otherwise derives
    the region from the percentile range of existing annotation centers.
    This is tighter than a convex hull and avoids placing defects on
    the background (desk, wall, etc.).

    Returns:
        4-point rectangular hull as an Nx1x2 float32 array (same format
        as ``compute_valid_zone``), or None if insufficient data.
    """
    if config.placement_region is not None:
        x_min, y_min, x_max, y_max = config.placement_region
    else:
        centers = [
            (b.x_center, b.y_center)
            for r in dataset.train
            for b in r.bboxes
        ]
        if len(centers) < 3:
            return None
        pts = np.array(centers, dtype=np.float32)
        pct = config.placement_percentile
        x_min = float(np.percentile(pts[:, 0], pct))
        x_max = float(np.percentile(pts[:, 0], 100 - pct))
        y_min = float(np.percentile(pts[:, 1], pct))
        y_max = float(np.percentile(pts[:, 1], 100 - pct))

    logger.info(
        "Placement zone: x=[%.3f, %.3f] y=[%.3f, %.3f]",
        x_min, x_max, y_min, y_max,
    )

    # Build a 4-point rectangular hull compatible with cv2.pointPolygonTest
    rect = np.array([
        [[x_min, y_min]],
        [[x_max, y_min]],
        [[x_max, y_max]],
        [[x_min, y_max]],
    ], dtype=np.float32)
    return rect


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def run_inpainting_pipeline(
    dataset: Dataset,
    strategy: SynthesisStrategy,
    config: InpaintingConfig,
    output_dir: Path,
    augment_config: AugmentationConfig | None = None,
    seed: int | None = None,
    dry_run: bool = False,
) -> Dataset:
    """Full inpainting pipeline: backgrounds -> mask -> inpaint -> augment -> write.

    Args:
        dataset: Source YOLO dataset.
        strategy: Synthesis strategy with prioritized tasks.
        config: Inpainting configuration.
        output_dir: Directory for output dataset.
        augment_config: Optional augmentation config (None to skip).
        seed: Random seed for reproducibility.
        dry_run: If True, log plan and estimated cost without calling the API.

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

    # Step 1: Generate backgrounds (reuse compositor's BackgroundGenerator)
    logger.info("Step 1: Generating clean backgrounds...")
    bg_gen = BackgroundGenerator(inpaint_radius=5, method="telea")
    backgrounds = bg_gen.generate_from_dataset(dataset)

    if not backgrounds:
        logger.error("No backgrounds generated, cannot proceed")
        return Dataset(root=output_dir, class_names=dataset.class_names, train=[], valid=[], test=[])

    # Step 2: Compute placement zone
    logger.info("Step 2: Computing placement zone...")
    valid_zone = _compute_placement_zone(dataset, config)

    # Step 3: Create provider (skip for dry-run — no API key needed)
    if dry_run:
        provider = _DryRunProvider(config.cost_per_image)
    else:
        provider = _create_provider(config)

    # Step 4: Create rate limiter and generator
    rate_limiter = RateLimiter(config.requests_per_minute)
    generator = InpaintingGenerator(provider, config, rate_limiter)

    # Step 5: Generate for each inpainting task
    logger.info("Step 3: Running inpainting generator...")
    all_records: list[ImageRecord] = []
    inpainting_tasks = [t for t in strategy.generation_tasks if t.method == "inpainting"]

    for task in inpainting_tasks:
        records = generator.generate(
            task,
            backgrounds=backgrounds,
            img_size=(img_w, img_h),
            class_names=dataset.class_names,
            valid_zone=valid_zone,
            dry_run=dry_run,
            seed=seed,
            output_dir=output_dir,
        )
        all_records.extend(records)

    if dry_run:
        logger.info(
            "Dry run complete: %d tasks, estimated cost $%.2f",
            len(inpainting_tasks), generator._cumulative_cost,
        )
        return Dataset(root=output_dir, class_names=dataset.class_names, train=[], valid=[], test=[])

    logger.info("Generated %d synthetic images", len(all_records))

    # Step 6: Optional augmentation
    if augment_config is not None and augment_config.enabled and all_records:
        logger.info("Step 4: Applying augmentation...")
        from synthdet.augment.classical import ClassicalAugmenter

        augmenter = ClassicalAugmenter(augment_config)
        augmented = augmenter.augment_batch(all_records, variants_per_image=augment_config.variants_per_image)
        all_records.extend(augmented)
        logger.info("Added %d augmented variants (total: %d)", len(augmented), len(all_records))

    # Step 7: Write output
    logger.info("Step 5: Writing YOLO dataset...")
    from synthdet.annotate.yolo_writer import write_yolo_dataset

    random.shuffle(all_records)
    split_idx = int(len(all_records) * 0.85)
    records_by_split = {
        "train": all_records[:split_idx],
        "valid": all_records[split_idx:],
    }

    write_yolo_dataset(records_by_split, output_dir, dataset.class_names)

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


class _DryRunProvider:
    """Stub provider used during --dry-run so no API key is required."""

    def __init__(self, cost: float = 0.02) -> None:
        self._cost = cost

    @property
    def cost_per_image(self) -> float:
        return self._cost

    def inpaint(
        self, image: np.ndarray, mask: np.ndarray, prompt: str,
        *, seed: int | None = None, num_images: int = 1,
    ) -> list[np.ndarray]:
        return [image]


def _create_provider(config: InpaintingConfig) -> InpaintingProvider:
    """Instantiate the appropriate inpainting provider."""
    if config.provider == "imagen":
        from synthdet.generate.providers.imagen import ImagenInpaintingProvider

        return ImagenInpaintingProvider(
            model=config.model,
            guidance_scale=config.guidance_scale,
            api_key_env_var=config.api_key_env_var,
            project=config.project,
            location=config.location,
        )
    elif config.provider == "diffusers":
        from synthdet.generate.providers.diffusers_local import DiffusersInpaintingProvider

        return DiffusersInpaintingProvider(
            model=config.model,
            device=config.device,
            use_fp16=config.use_fp16,
            num_inference_steps=config.num_inference_steps,
            strength=config.strength,
            guidance_scale=config.guidance_scale,
        )
    raise ValueError(f"Unknown inpainting provider: {config.provider!r}")
