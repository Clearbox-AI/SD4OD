"""Pipeline orchestrator — unified analysis → generation → augmentation → validation.

Calls generator classes directly (not ``run_*_pipeline()`` functions) so that
results from all methods are merged *before* the train/valid split.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from synthdet.analysis.loader import load_yolo_dataset
from synthdet.analysis.statistics import compute_dataset_statistics
from synthdet.analysis.strategy import generate_synthesis_strategy
from synthdet.config import SynthDetConfig
from synthdet.generate.compositor import (
    BackgroundGenerator,
    DefectCompositor,
    DefectPatchExtractor,
    compute_valid_zone,
)
from synthdet.pipeline.config_schema import PipelineConfig
from synthdet.pipeline.validator import ValidationReport, validate_dataset
from synthdet.types import (
    Dataset,
    GenerationTask,
    ImageRecord,
    SynthesisStrategy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Outcome of a full pipeline run."""

    output_dataset: Dataset
    output_dir: Path
    methods_used: list[str]
    records_per_method: dict[str, int]
    total_records: int
    train_count: int
    valid_count: int
    total_cost_usd: float
    cost_per_method: dict[str, float]
    validation_report: ValidationReport | None
    dry_run: bool


# ---------------------------------------------------------------------------
# Strategy scaling
# ---------------------------------------------------------------------------


def _scale_strategy(strategy: SynthesisStrategy, num_methods: int) -> SynthesisStrategy:
    """Divide each task's ``num_images`` by *num_methods* to avoid overproduction."""
    if num_methods <= 1:
        return strategy
    scaled = copy.deepcopy(strategy)
    for task in scaled.generation_tasks:
        task.num_images = max(1, task.num_images // num_methods)
    return scaled


# ---------------------------------------------------------------------------
# Per-method runners
# ---------------------------------------------------------------------------


def _run_compositor(
    dataset: Dataset,
    strategy: SynthesisStrategy,
    config: SynthDetConfig,
    patches,
    backgrounds: list[np.ndarray],
    valid_zone: np.ndarray | None,
    img_size: tuple[int, int],
    seed: int | None,
) -> tuple[list[ImageRecord], float]:
    """Run the compositor generator and return (records, cost)."""
    compositor = DefectCompositor(config.compositor)
    tasks = [t for t in strategy.generation_tasks if t.method == "compositor"]
    all_records: list[ImageRecord] = []

    for task in tasks:
        records = compositor.generate(
            task,
            patches=patches,
            backgrounds=backgrounds,
            img_size=img_size,
            class_names=dataset.class_names,
            valid_zone=valid_zone,
        )
        all_records.extend(records)

    return all_records, 0.0  # compositor is local, zero cost


def _run_inpainting(
    dataset: Dataset,
    strategy: SynthesisStrategy,
    config: SynthDetConfig,
    backgrounds: list[np.ndarray],
    valid_zone: np.ndarray | None,
    output_dir: Path,
    img_size: tuple[int, int],
    seed: int | None,
    dry_run: bool,
) -> tuple[list[ImageRecord], float]:
    """Run the inpainting generator and return (records, cost)."""
    from synthdet.generate.inpainting import (
        InpaintingGenerator,
        MaskPlacer,
        _DryRunProvider,
        _compute_placement_zone,
        _create_provider,
    )
    from synthdet.utils.rate_limiter import RateLimiter

    if dry_run:
        provider = _DryRunProvider(config.inpainting.cost_per_image)
    else:
        provider = _create_provider(config.inpainting)

    rate_limiter = RateLimiter(config.inpainting.requests_per_minute)
    generator = InpaintingGenerator(provider, config.inpainting, rate_limiter)

    inp_zone = _compute_placement_zone(dataset, config.inpainting)
    zone = inp_zone if inp_zone is not None else valid_zone

    tasks = [t for t in strategy.generation_tasks if t.method == "inpainting"]
    all_records: list[ImageRecord] = []

    for task in tasks:
        records = generator.generate(
            task,
            backgrounds=backgrounds,
            img_size=img_size,
            class_names=dataset.class_names,
            valid_zone=zone,
            dry_run=dry_run,
            seed=seed,
            output_dir=output_dir,
        )
        all_records.extend(records)

    return all_records, generator._cumulative_cost


def _run_generative_compositor(
    dataset: Dataset,
    strategy: SynthesisStrategy,
    config: SynthDetConfig,
    backgrounds: list[np.ndarray],
    valid_zone: np.ndarray | None,
    output_dir: Path,
    img_size: tuple[int, int],
    seed: int | None,
) -> tuple[list[ImageRecord], float]:
    """Run the generative compositor and return (records, cost)."""
    from synthdet.generate.generative_compositor import GenerativeDefectCompositor
    from synthdet.generate.providers.imagen_generate import ImagenGenerateProvider

    provider = ImagenGenerateProvider(
        model="imagen-3.0-generate-002",
        project=config.inpainting.project,
        location=config.inpainting.location,
    )
    compositor = GenerativeDefectCompositor(
        provider,
        max_cost_usd=config.inpainting.max_cost_usd,
        requests_per_minute=config.inpainting.requests_per_minute,
    )

    all_records: list[ImageRecord] = []
    for task in strategy.generation_tasks:
        if not task.target_classes:
            continue
        records = compositor.generate(
            task,
            backgrounds=backgrounds,
            img_size=img_size,
            class_names=dataset.class_names,
            valid_zone=valid_zone,
            output_dir=output_dir,
            seed=seed,
        )
        all_records.extend(records)

    return all_records, compositor.cumulative_cost


def _run_modify_annotate(
    dataset: Dataset,
    strategy: SynthesisStrategy,
    config: SynthDetConfig,
    output_dir: Path,
    seed: int | None,
) -> tuple[list[ImageRecord], float]:
    """Run the modify-annotate generator and return (records, cost)."""
    from synthdet.generate.modify_annotate import (
        ModifyAndAnnotateGenerator,
        _create_provider,
    )
    from synthdet.utils.rate_limiter import RateLimiter

    provider = _create_provider(config.modify_annotate)
    rate_limiter = RateLimiter(config.modify_annotate.requests_per_minute)
    generator = ModifyAndAnnotateGenerator(provider, config.modify_annotate, rate_limiter)

    all_records: list[ImageRecord] = []
    for task in strategy.generation_tasks:
        records = generator.generate(
            task,
            dataset=dataset,
            output_dir=output_dir,
            seed=seed,
        )
        all_records.extend(records)

    return all_records, generator.cumulative_cost


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


_METHOD_TO_PREFERRED = {
    "compositor": "compositor",
    "inpainting": "inpainting",
    "generative_compositor": "compositor",
    "modify_annotate": "inpainting",
}


def run_pipeline(
    data_yaml: Path,
    output_dir: Path,
    config: PipelineConfig | None = None,
    seed: int | None = None,
) -> PipelineResult:
    """Run the full SynthDet pipeline.

    Args:
        data_yaml: Path to the YOLO dataset ``data.yaml``.
        output_dir: Where to write the generated dataset.
        config: Pipeline configuration. Uses defaults when *None*.
        seed: Random seed for reproducibility.

    Returns:
        PipelineResult with details about the run.
    """
    if config is None:
        config = PipelineConfig()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    methods = config.methods
    num_methods = len(methods)
    synthdet = config.synthdet

    # ------------------------------------------------------------------
    # Step 1: Load dataset → compute statistics
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading dataset and computing statistics...")
    dataset = load_yolo_dataset(data_yaml)
    stats = compute_dataset_statistics(dataset, synthdet.analysis)

    # Determine image size from dataset
    if dataset.train:
        sample = dataset.train[0].load_image()
        img_h, img_w = sample.shape[:2]
    else:
        img_w, img_h = 860, 640
    img_size = (img_w, img_h)

    # ------------------------------------------------------------------
    # Step 2: Compute shared resources (backgrounds, valid zone, patches)
    # ------------------------------------------------------------------
    needs_backgrounds = any(
        m in ("compositor", "inpainting", "generative_compositor") for m in methods
    )
    backgrounds: list[np.ndarray] = []
    valid_zone: np.ndarray | None = None
    patches = []

    if needs_backgrounds:
        logger.info("Step 2: Computing backgrounds and valid zone...")
        bg_gen = BackgroundGenerator(
            inpaint_radius=synthdet.compositor.inpaint_radius,
            method=synthdet.compositor.inpaint_method,
        )
        backgrounds = bg_gen.generate_from_dataset(dataset)
        valid_zone = compute_valid_zone(dataset, margin=synthdet.compositor.valid_zone_margin)

    if "compositor" in methods:
        extractor = DefectPatchExtractor(
            margin=synthdet.compositor.patch_margin,
            min_patch_pixels=synthdet.compositor.min_patch_pixels,
        )
        patches = extractor.extract_patches(dataset)

    # ------------------------------------------------------------------
    # Step 3: For each method, generate strategy → scale → run generator
    # ------------------------------------------------------------------
    all_records: list[ImageRecord] = []
    records_per_method: dict[str, int] = {}
    cost_per_method: dict[str, float] = {}

    for method in methods:
        logger.info("Step 3 [%s]: Generating strategy and running...", method)
        # Generate strategy with the appropriate preferred_method
        analysis_cfg = synthdet.analysis.model_copy(
            update={"preferred_method": _METHOD_TO_PREFERRED.get(method, "compositor")}
        )
        strategy = generate_synthesis_strategy(dataset, stats, analysis_cfg)
        strategy = _scale_strategy(strategy, num_methods)

        records: list[ImageRecord] = []
        cost = 0.0

        if method == "compositor":
            records, cost = _run_compositor(
                dataset, strategy, synthdet, patches, backgrounds,
                valid_zone, img_size, seed,
            )
        elif method == "inpainting":
            records, cost = _run_inpainting(
                dataset, strategy, synthdet, backgrounds, valid_zone,
                output_dir, img_size, seed, config.dry_run,
            )
        elif method == "generative_compositor":
            records, cost = _run_generative_compositor(
                dataset, strategy, synthdet, backgrounds, valid_zone,
                output_dir, img_size, seed,
            )
        elif method == "modify_annotate":
            records, cost = _run_modify_annotate(
                dataset, strategy, synthdet, output_dir, seed,
            )
        else:
            raise ValueError(f"Unknown method: {method!r}")

        all_records.extend(records)
        records_per_method[method] = len(records)
        cost_per_method[method] = cost
        logger.info(
            "  %s: %d records, cost $%.2f", method, len(records), cost,
        )

    total_cost = sum(cost_per_method.values())

    # Dry run: return early with cost estimate
    if config.dry_run:
        logger.info("Dry run complete — total estimated cost: $%.2f", total_cost)
        return PipelineResult(
            output_dataset=Dataset(
                root=output_dir, class_names=dataset.class_names,
                train=[], valid=[], test=[],
            ),
            output_dir=output_dir,
            methods_used=methods,
            records_per_method=records_per_method,
            total_records=0,
            train_count=0,
            valid_count=0,
            total_cost_usd=total_cost,
            cost_per_method=cost_per_method,
            validation_report=None,
            dry_run=True,
        )

    # ------------------------------------------------------------------
    # Step 4: Optional augmentation
    # ------------------------------------------------------------------
    if config.augment and all_records:
        logger.info("Step 4: Applying classical augmentation...")
        from synthdet.augment.classical import ClassicalAugmenter

        augmenter = ClassicalAugmenter(synthdet.augmentation)
        augmented = augmenter.augment_batch(
            all_records, variants_per_image=synthdet.augmentation.variants_per_image,
        )
        all_records.extend(augmented)
        logger.info("Added %d augmented variants (total: %d)", len(augmented), len(all_records))

    # ------------------------------------------------------------------
    # Step 5: Shuffle + split train/valid
    # ------------------------------------------------------------------
    logger.info("Step 5: Splitting and writing dataset...")
    random.shuffle(all_records)
    split_idx = int(len(all_records) * config.train_split_ratio)
    train_records = all_records[:split_idx]
    valid_records = all_records[split_idx:]

    # ------------------------------------------------------------------
    # Step 6: Write output
    # ------------------------------------------------------------------
    from synthdet.annotate.yolo_writer import write_yolo_dataset

    records_by_split = {"train": train_records, "valid": valid_records}
    write_yolo_dataset(records_by_split, output_dir, dataset.class_names)

    output_dataset = Dataset(
        root=output_dir,
        class_names=dataset.class_names,
        train=train_records,
        valid=valid_records,
        test=[],
    )

    # ------------------------------------------------------------------
    # Step 7: Optional validation
    # ------------------------------------------------------------------
    validation_report: ValidationReport | None = None
    if config.validate_output:
        logger.info("Step 6: Validating output dataset...")
        validation_report = validate_dataset(output_dir)
        if validation_report.is_valid:
            logger.info("Validation passed (%d images)", validation_report.total_images)
        else:
            logger.warning(
                "Validation found %d error(s):\n%s",
                len(validation_report.errors),
                validation_report.summary(),
            )

    logger.info(
        "Pipeline complete: %d train, %d valid images in %s",
        len(train_records), len(valid_records), output_dir,
    )

    return PipelineResult(
        output_dataset=output_dataset,
        output_dir=output_dir,
        methods_used=methods,
        records_per_method=records_per_method,
        total_records=len(all_records),
        train_count=len(train_records),
        valid_count=len(valid_records),
        total_cost_usd=total_cost,
        cost_per_method=cost_per_method,
        validation_report=validation_report,
        dry_run=False,
    )
