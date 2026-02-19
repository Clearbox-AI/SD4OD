"""Modify-and-Annotate pipeline — whole-image transformation + auto-annotation.

Workflow per image:
    1. Pick a clean source image from the dataset (train split)
    2. Build a damage prompt (from class_prompts config)
    3. Call ``ImagenModifierProvider.modify()`` to get a damaged version
    4. Run an auto-annotator (Grounding DINO or OWL-ViT) on the result
    5. Optionally refine bboxes with SAM, verify with CLIP
    6. Return ``ImageRecord`` with auto-detected bboxes
    7. Save incrementally (crash-safe)

This avoids both the inpainting problem (model fills clean) and compositing
artifacts (blending halos). The API handles realistic defect integration
holistically, and the annotator finds where defects appeared.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import cv2
import numpy as np

from synthdet.config import ModifyAnnotateConfig
from synthdet.generate.errors import InpaintingAPIError
from synthdet.types import (
    AnnotationSource,
    BBox,
    Dataset,
    GenerationTask,
    ImageRecord,
    SynthesisStrategy,
)
from synthdet.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default damage prompts
# ---------------------------------------------------------------------------

DEFAULT_DAMAGE_PROMPTS: dict[str, list[str]] = {
    "broken": [
        "Add a clearly visible broken section with cracked and separated plastic pieces to this laptop top cover",
        "Add an obvious shattered area with fragmented material and dark gaps to this laptop surface",
    ],
    "hard scratch": [
        "Add deep, clearly visible white scratch lines gouged into this laptop top cover surface",
        "Add prominent long scratches with visible depth, lighter than the surrounding surface",
        "Add harsh white abrasion lines deeply etched into this laptop surface",
    ],
    "minor crack": [
        "Add a clearly visible thin dark crack line running across this laptop top cover surface",
        "Add a noticeable fine crack with a dark hairline fracture on this laptop surface",
    ],
    "minor dent": [
        "Add a clearly visible shallow dent creating a noticeable shadow on this laptop top cover",
        "Add an obvious small depression catching light differently on this laptop surface",
    ],
    "minor scratch": [
        "Add clearly visible light scratch lines on this laptop top cover surface",
        "Add noticeable thin white scratch marks standing out against this laptop surface finish",
        "Add distinct fine scratches creating visible light-colored lines across this laptop surface",
    ],
    "sticker marks": [
        "Add a clearly visible rectangular patch of sticky adhesive residue on this laptop top cover",
        "Add an obvious discolored area with visible tacky glue residue on this laptop surface",
    ],
}


# ---------------------------------------------------------------------------
# ModifyAndAnnotateGenerator
# ---------------------------------------------------------------------------


class ModifyAndAnnotateGenerator:
    """Generate synthetic defect images via whole-image modification + auto-annotation.

    Sends clean images to an image modification API (Imagen controlled editing)
    with damage prompts, then runs an object detector to find where defects
    appeared in the modified image.
    """

    def __init__(
        self,
        provider: object,  # ImagenModifierProvider
        config: ModifyAnnotateConfig,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.rate_limiter = rate_limiter or RateLimiter(config.requests_per_minute)
        self._cumulative_cost = 0.0
        self._annotator: object | None = None
        self._sam_refiner: object | None = None
        self._clip_verifier: object | None = None

    @property
    def cumulative_cost(self) -> float:
        return self._cumulative_cost

    def _get_annotator(self) -> object:
        """Lazy-load the auto-annotator."""
        if self._annotator is not None:
            return self._annotator

        if self.config.annotator == "grounding_dino":
            from synthdet.annotate.grounding_dino import GroundingDINOAnnotator

            self._annotator = GroundingDINOAnnotator(
                confidence_threshold=self.config.confidence_threshold,
                box_threshold=self.config.confidence_threshold,
            )
        elif self.config.annotator == "owlvit":
            from synthdet.annotate.owlvit import OWLViTAnnotator

            self._annotator = OWLViTAnnotator(
                confidence_threshold=self.config.confidence_threshold,
            )
        else:
            raise ValueError(f"Unknown annotator: {self.config.annotator!r}")

        return self._annotator

    def _get_sam_refiner(self) -> object | None:
        """Lazy-load SAM refiner if enabled."""
        if not self.config.sam_refine:
            return None
        if self._sam_refiner is not None:
            return self._sam_refiner

        from synthdet.annotate.sam_refiner import SAMRefiner

        self._sam_refiner = SAMRefiner()
        return self._sam_refiner

    def _get_clip_verifier(self, class_names: list[str]) -> object | None:
        """Lazy-load CLIP verifier if enabled."""
        if not self.config.clip_verify:
            return None
        if self._clip_verifier is not None:
            return self._clip_verifier

        from synthdet.annotate.verifier import CLIPVerifier

        self._clip_verifier = CLIPVerifier(
            min_confidence=self.config.min_clip_score,
            class_names=class_names,
        )
        return self._clip_verifier

    def generate(
        self,
        task: GenerationTask,
        *,
        source_images: list[ImageRecord],
        class_names: list[str],
        output_dir: Path | None = None,
        seed: int | None = None,
    ) -> list[ImageRecord]:
        """Generate modified images for a task.

        For each image:
            1. Pick a random source image from the dataset
            2. Build a damage prompt for a target class
            3. Call the modifier API to transform the image
            4. Run the annotator to detect defects
            5. Optionally refine with SAM / filter with CLIP
            6. Save incrementally

        Args:
            task: Generation task with targeting parameters.
            source_images: Clean source images to modify (from dataset train split).
            class_names: Class name list for annotator prompts.
            output_dir: If set, save each image incrementally.
            seed: Random seed.

        Returns:
            List of ImageRecord with modified images and auto-detected bboxes.
        """
        if not source_images:
            logger.warning("No source images available, skipping task %s", task.task_id)
            return []

        records: list[ImageRecord] = []

        for i in range(task.num_images):
            # Cost guard
            if self.config.max_cost_usd > 0 and self._cumulative_cost >= self.config.max_cost_usd:
                logger.warning(
                    "Cost limit reached ($%.2f >= $%.2f), stopping",
                    self._cumulative_cost, self.config.max_cost_usd,
                )
                break

            # Pick a source image
            source_rec = random.choice(source_images)
            source_img = source_rec.load_image()

            # Pick a target class and build prompt
            if task.target_classes:
                class_id = random.choice(task.target_classes)
                class_name = (
                    class_names[class_id]
                    if class_id < len(class_names)
                    else "defect"
                )
            else:
                class_name = "defect"

            prompt = self._build_prompt(class_name, task.suggested_prompts)

            # Call modifier API
            modified_img = self._call_provider(source_img, prompt, seed=seed)
            if modified_img is None:
                continue

            # Run annotator on modified image
            bboxes = self._annotate(modified_img, class_names)

            # Optional SAM refinement
            sam = self._get_sam_refiner()
            if sam is not None and bboxes:
                bboxes = sam.refine(modified_img, bboxes)

            # Optional CLIP verification
            clip = self._get_clip_verifier(class_names)
            if clip is not None and bboxes:
                scored = clip.verify(modified_img, bboxes)
                bboxes = [
                    bbox for bbox, score in scored
                    if score >= self.config.min_clip_score
                ]

            # Tag all bboxes with modify_annotate source
            bboxes = [
                BBox(
                    class_id=b.class_id,
                    x_center=b.x_center,
                    y_center=b.y_center,
                    width=b.width,
                    height=b.height,
                    confidence=b.confidence,
                    source=AnnotationSource.modify_annotate,
                )
                for b in bboxes
            ]

            record = ImageRecord(
                image_path=Path(f"modanno_{task.task_id}_{i:04d}.jpg"),
                bboxes=bboxes,
                image=modified_img,
                metadata={
                    "source": "modify_annotate",
                    "task_id": task.task_id,
                    "source_image": str(source_rec.image_path.name),
                    "prompt": prompt,
                    "num_detections": len(bboxes),
                },
            )
            _save_record_incremental(record, output_dir, self.config)
            records.append(record)

        logger.info(
            "Generated %d images for task %s (cost: $%.2f)",
            len(records), task.task_id, self._cumulative_cost,
        )
        return records

    def modify_single(
        self,
        image: np.ndarray,
        prompt: str,
        *,
        seed: int | None = None,
    ) -> np.ndarray | None:
        """Modify a single image (convenience method for testing).

        Returns the modified BGR image, or None on failure.
        """
        return self._call_provider(image, prompt, seed=seed)

    def annotate_single(
        self,
        image: np.ndarray,
        class_names: list[str],
    ) -> list[BBox]:
        """Run annotation on a single image (convenience method for testing).

        Returns list of detected BBox objects.
        """
        return self._annotate(image, class_names)

    def _build_prompt(self, class_name: str, task_prompts: list[str]) -> str:
        """Build a damage prompt for the given class."""
        # Check config class_prompts first (case-insensitive)
        prompts: list[str] = []
        if self.config.class_prompts:
            key_lower = class_name.lower()
            for k, v in self.config.class_prompts.items():
                if k.lower() == key_lower:
                    prompts = v
                    break

        # Fall back to default prompts from config
        if not prompts:
            prompts = self.config.default_prompts

        # Fall back to built-in defaults
        if not prompts:
            key_lower = class_name.lower()
            for k, v in DEFAULT_DAMAGE_PROMPTS.items():
                if k.lower() == key_lower or key_lower in k.lower():
                    prompts = v
                    break

        # Fall back to task prompts or generic
        if not prompts:
            prompts = task_prompts or [
                f"Add realistic {class_name} damage to this surface"
            ]

        return random.choice(prompts)

    def _call_provider(
        self,
        image: np.ndarray,
        prompt: str,
        seed: int | None = None,
    ) -> np.ndarray | None:
        """Call the modifier provider with rate limiting and retries."""
        for attempt in range(self.config.max_retries):
            if not self.rate_limiter.acquire(timeout=60.0):
                logger.warning("Rate limiter timeout, skipping")
                return None
            try:
                results = self.provider.modify(image, prompt, seed=seed, num_images=1)
                self._cumulative_cost += self.provider.cost_per_image
                return results[0] if results else None
            except InpaintingAPIError as e:
                if e.retryable and attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay_seconds * (2 ** attempt)
                    logger.warning(
                        "Retryable error (attempt %d): %s — retrying in %.1fs",
                        attempt + 1, e, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error("Modifier API error: %s", e)
                    return None
        return None

    def _annotate(self, image: np.ndarray, class_names: list[str]) -> list[BBox]:
        """Run the auto-annotator on an image."""
        annotator = self._get_annotator()
        try:
            return annotator.annotate(image, class_names)
        except Exception as exc:
            logger.error("Annotation failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Incremental save
# ---------------------------------------------------------------------------


def _save_record_incremental(
    record: ImageRecord,
    output_dir: Path | None,
    config: ModifyAnnotateConfig,
) -> None:
    """Save a single record to disk immediately (crash-safe)."""
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
# Top-level pipeline
# ---------------------------------------------------------------------------


def run_modify_annotate_pipeline(
    dataset: Dataset,
    strategy: SynthesisStrategy,
    config: ModifyAnnotateConfig,
    output_dir: Path,
    *,
    class_prompts: dict[str, list[str]] | None = None,
    seed: int | None = None,
) -> Dataset:
    """Full modify-and-annotate pipeline.

    Args:
        dataset: Source YOLO dataset (clean images come from train split).
        strategy: Synthesis strategy with generation tasks.
        config: ModifyAnnotateConfig with provider/annotator settings.
        output_dir: Output directory for generated dataset.
        class_prompts: Override per-class damage prompts.
        seed: Random seed for reproducibility.

    Returns:
        Dataset object pointing to the generated output.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Override class_prompts if provided
    if class_prompts is not None:
        config = config.model_copy(update={"class_prompts": class_prompts})

    # Step 1: Create modifier provider
    logger.info("Step 1: Creating image modifier provider...")
    provider = _create_provider(config)

    # Step 2: Create rate limiter and generator
    logger.info("Step 2: Setting up generator...")
    rate_limiter = RateLimiter(config.requests_per_minute)
    generator = ModifyAndAnnotateGenerator(provider, config, rate_limiter)

    # Step 3: Use all non-negative source images from train split
    source_images = [r for r in dataset.train if not r.is_negative]
    if not source_images:
        # Fall back to all train images
        source_images = list(dataset.train)
    logger.info("Using %d source images from train split", len(source_images))

    # Step 4: Generate for each task
    logger.info("Step 3: Running modify-and-annotate generator...")
    all_records: list[ImageRecord] = []

    for task in strategy.generation_tasks:
        if len(task.target_classes) == 0:
            continue  # Skip negative-only tasks
        records = generator.generate(
            task,
            source_images=source_images,
            class_names=dataset.class_names,
            output_dir=output_dir,
            seed=seed,
        )
        all_records.extend(records)

    logger.info(
        "Generated %d synthetic images (total cost: $%.2f)",
        len(all_records), generator.cumulative_cost,
    )

    # Step 5: Write output
    logger.info("Step 4: Writing YOLO dataset...")
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


def _create_provider(config: ModifyAnnotateConfig) -> object:
    """Instantiate the modifier provider from config."""
    if config.provider == "imagen":
        from synthdet.generate.providers.imagen_modifier import ImagenModifierProvider

        return ImagenModifierProvider(
            model=config.model,
            control_type=config.control_type,
            api_key_env_var=config.api_key_env_var,
            project=config.project,
            location=config.location,
        )
    raise ValueError(f"Unknown modifier provider: {config.provider!r}")
