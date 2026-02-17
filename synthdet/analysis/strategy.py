"""Synthesis strategy generation — converts dataset gaps into prioritized tasks.

In Phase 1, the strategy is purely gap-driven (exploration). When a
ModelPerformanceProfile is provided (Phase 4+), tasks are also generated
from model weaknesses (exploitation), with a priority boost.
"""

from __future__ import annotations

from synthdet.config import AnalysisConfig
from synthdet.types import (
    ActiveLearningSignal,
    BBoxSizeBucket,
    Dataset,
    DatasetStatistics,
    GenerationTask,
    ModelPerformanceProfile,
    QualityMetrics,
    SpatialRegion,
    SynthesisStrategy,
)
from synthdet.analysis.statistics import (
    identify_aspect_ratio_gaps,
    identify_size_bucket_gaps,
    identify_spatial_gaps,
)


def _class_name_prompts(class_names: list[str]) -> list[str]:
    """Simple prompts derived from class names.

    These serve as fallback hints; the inpainting generator builds the
    actual API prompt from ``InpaintingConfig.class_prompts`` /
    ``default_prompts`` / ``prompt_template``.
    """
    if not class_names:
        return ["Surface defect"]
    return [f"{name} on the object surface" for name in class_names]


def _create_generation_tasks(
    stats: DatasetStatistics,
    config: AnalysisConfig,
    model_profile: ModelPerformanceProfile | None = None,
) -> list[GenerationTask]:
    """Convert gap analysis into prioritized GenerationTask objects."""
    tasks: list[GenerationTask] = []
    task_counter = 0
    method = config.preferred_method

    class_names = [d.class_name for d in stats.class_distributions]
    all_class_ids = [d.class_id for d in stats.class_distributions]

    # --- Priority 1.0: Zero-representation size buckets ---
    size_gaps = identify_size_bucket_gaps(stats, config.min_per_bucket)
    for bucket, deficit in size_gaps.items():
        if stats.overall_bucket_counts.get(bucket, 0) == 0:
            task_counter += 1
            tasks.append(GenerationTask(
                task_id=f"gap-size-zero-{bucket.value}-{task_counter}",
                priority=1.0,
                num_images=deficit,
                target_classes=all_class_ids,
                target_size_buckets=[bucket],
                target_regions=list(SpatialRegion),
                suggested_prompts=_class_name_prompts(class_names),
                rationale=f"Size bucket '{bucket.value}' has zero representations",
                method=method,
            ))

    # --- Priority 0.9: Spatial gaps (zero representation) ---
    spatial_gaps = identify_spatial_gaps(stats, config.min_per_region)
    zero_regions = [r for r, d in spatial_gaps.items()
                    if stats.overall_region_counts.get(r, 0) == 0]
    if zero_regions:
        task_counter += 1
        max_deficit = max(spatial_gaps[r] for r in zero_regions)
        tasks.append(GenerationTask(
            task_id=f"gap-spatial-zero-{task_counter}",
            priority=0.9,
            num_images=max_deficit,
            target_classes=all_class_ids,
            target_size_buckets=list(BBoxSizeBucket),
            target_regions=zero_regions,
            suggested_prompts=_class_name_prompts(class_names),
            rationale=f"Spatial regions with zero annotations: {[r.value for r in zero_regions]}",
            method="compositor",
        ))

    # --- Priority 0.8: Negative examples (images with no defects) ---
    if stats.negative_ratio < config.negative_ratio:
        target_negatives = int(
            config.negative_ratio * stats.total_images * config.multiplier
        )
        current_negatives = stats.negative_images
        deficit = max(0, target_negatives - current_negatives)
        if deficit > 0:
            task_counter += 1
            tasks.append(GenerationTask(
                task_id=f"gap-negative-{task_counter}",
                priority=0.8,
                num_images=deficit,
                target_classes=[],
                target_size_buckets=[],
                target_regions=[],
                suggested_prompts=["Clean surface without defects"],
                rationale=(
                    f"Negative ratio is {stats.negative_ratio:.1%} "
                    f"(target: {config.negative_ratio:.1%})"
                ),
                method=method,
            ))

    # --- Priority 0.7: Underrepresented size buckets (non-zero) ---
    for bucket, deficit in size_gaps.items():
        if stats.overall_bucket_counts.get(bucket, 0) > 0:
            task_counter += 1
            tasks.append(GenerationTask(
                task_id=f"gap-size-under-{bucket.value}-{task_counter}",
                priority=0.7,
                num_images=deficit,
                target_classes=all_class_ids,
                target_size_buckets=[bucket],
                target_regions=list(SpatialRegion),
                suggested_prompts=_class_name_prompts(class_names),
                rationale=(
                    f"Size bucket '{bucket.value}' has {stats.overall_bucket_counts[bucket]} "
                    f"annotations (min: {config.min_per_bucket})"
                ),
                method=method,
            ))

    # --- Priority 0.6: Underrepresented spatial regions (non-zero) ---
    under_regions = [r for r, d in spatial_gaps.items()
                     if stats.overall_region_counts.get(r, 0) > 0]
    if under_regions:
        task_counter += 1
        max_deficit = max(spatial_gaps[r] for r in under_regions)
        tasks.append(GenerationTask(
            task_id=f"gap-spatial-under-{task_counter}",
            priority=0.6,
            num_images=max_deficit,
            target_classes=all_class_ids,
            target_size_buckets=list(BBoxSizeBucket),
            target_regions=under_regions,
            suggested_prompts=_class_name_prompts(class_names),
            rationale=f"Underrepresented spatial regions: {[r.value for r in under_regions]}",
            method="compositor",
        ))

    # --- Priority 0.5: Aspect ratio gaps ---
    ar_gaps = identify_aspect_ratio_gaps(stats, config.min_per_aspect_bin)
    for lo, hi, deficit in ar_gaps:
        task_counter += 1
        tasks.append(GenerationTask(
            task_id=f"gap-aspect-{task_counter}",
            priority=0.5,
            num_images=deficit,
            target_classes=all_class_ids,
            target_size_buckets=list(BBoxSizeBucket),
            target_regions=list(SpatialRegion),
            suggested_prompts=_class_name_prompts(class_names),
            rationale=f"Aspect ratio bin [{lo:.2f}, {hi:.2f}] has deficit of {deficit}",
            method="compositor",
        ))

    # --- Priority 0.3: General count increase ---
    target_total = int(stats.total_images * config.multiplier)
    planned_images = sum(t.num_images for t in tasks)
    general_deficit = target_total - stats.total_images - planned_images
    if general_deficit > 0:
        task_counter += 1
        tasks.append(GenerationTask(
            task_id=f"general-increase-{task_counter}",
            priority=0.3,
            num_images=general_deficit,
            target_classes=all_class_ids,
            target_size_buckets=list(BBoxSizeBucket),
            target_regions=list(SpatialRegion),
            suggested_prompts=_class_name_prompts(class_names),
            rationale=(
                f"General dataset expansion: {stats.total_images} → {target_total} images"
            ),
            method="compositor",
        ))

    # --- Model feedback boost (+0.1, capped at 1.0) ---
    if model_profile is not None:
        _apply_model_feedback_boost(tasks, model_profile)

    # Sort by priority (highest first)
    tasks.sort(key=lambda t: t.priority, reverse=True)
    return tasks


def _apply_model_feedback_boost(
    tasks: list[GenerationTask],
    profile: ModelPerformanceProfile,
) -> None:
    """Boost priority of tasks that align with model weaknesses."""
    # Find weak buckets and regions (mAP below 0.5)
    weak_buckets = {b for b, m in profile.per_bucket_map.items() if m < 0.5}
    weak_regions = {r for r, m in profile.per_region_map.items() if m < 0.5}

    for task in tasks:
        has_overlap = (
            any(b in weak_buckets for b in task.target_size_buckets)
            or any(r in weak_regions for r in task.target_regions)
        )
        if has_overlap:
            task_dict = task.__dict__
            task_dict["priority"] = min(1.0, task.priority + 0.1)
            task_dict["rationale"] += " [boosted: model feedback]"


def _generate_active_learning_signals(
    model_profile: ModelPerformanceProfile | None = None,
    quality_metrics: QualityMetrics | None = None,
) -> list[ActiveLearningSignal]:
    """Generate active learning signals from model feedback and quality metrics.

    In Phase 1, both inputs are None → returns empty list.
    In later phases, this drives exploitation of known model weaknesses.
    """
    signals: list[ActiveLearningSignal] = []

    if model_profile is not None:
        # Signal from weak size buckets
        weak_buckets = [b for b, m in model_profile.per_bucket_map.items() if m < 0.5]
        if weak_buckets:
            signals.append(ActiveLearningSignal(
                target_classes=list(model_profile.per_class_map.keys()),
                target_size_buckets=weak_buckets,
                target_regions=list(SpatialRegion),
                priority=0.9,
                rationale=(
                    f"Model struggles with size buckets: "
                    f"{[b.value for b in weak_buckets]} (mAP < 0.5)"
                ),
                source="model_feedback",
            ))

        # Signal from weak spatial regions
        weak_regions = [r for r, m in model_profile.per_region_map.items() if m < 0.5]
        if weak_regions:
            signals.append(ActiveLearningSignal(
                target_classes=list(model_profile.per_class_map.keys()),
                target_size_buckets=list(BBoxSizeBucket),
                target_regions=weak_regions,
                priority=0.85,
                rationale=(
                    f"Model struggles with spatial regions: "
                    f"{[r.value for r in weak_regions]} (mAP < 0.5)"
                ),
                source="model_feedback",
            ))

    if quality_metrics is not None:
        # Signal from SPC alerts
        if quality_metrics.alerts:
            signals.append(ActiveLearningSignal(
                target_classes=[],
                target_size_buckets=list(BBoxSizeBucket),
                target_regions=list(SpatialRegion),
                priority=0.95,
                rationale=f"SPC alert: {'; '.join(quality_metrics.alerts[:3])}",
                source="spc_alert",
            ))

    return signals


def generate_synthesis_strategy(
    dataset: Dataset,
    stats: DatasetStatistics,
    config: AnalysisConfig | None = None,
    model_profile: ModelPerformanceProfile | None = None,
    quality_metrics: QualityMetrics | None = None,
) -> SynthesisStrategy:
    """Generate a synthesis strategy from dataset statistics and optional model feedback.

    Args:
        dataset: The loaded YOLO dataset.
        stats: Pre-computed dataset statistics.
        config: Analysis configuration (uses defaults if None).
        model_profile: Optional model evaluation results for active learning.
        quality_metrics: Optional SPC quality metrics.

    Returns:
        SynthesisStrategy with prioritized generation tasks.
    """
    if config is None:
        config = AnalysisConfig()

    # Compute gaps
    size_gaps = identify_size_bucket_gaps(stats, config.min_per_bucket)
    spatial_gaps = identify_spatial_gaps(stats, config.min_per_region)
    ar_gaps = identify_aspect_ratio_gaps(stats, config.min_per_aspect_bin)

    # Generate tasks
    tasks = _create_generation_tasks(stats, config, model_profile)

    # Active learning signals
    al_signals = _generate_active_learning_signals(model_profile, quality_metrics)

    # Target class counts (proportional to current distribution, scaled by multiplier)
    target_class_counts: dict[int, int] = {}
    for dist in stats.class_distributions:
        target_class_counts[dist.class_id] = int(dist.count * config.multiplier)

    return SynthesisStrategy(
        target_total_images=int(stats.total_images * config.multiplier),
        target_class_counts=target_class_counts,
        negative_ratio=config.negative_ratio,
        size_bucket_gaps=size_gaps,
        spatial_gaps=spatial_gaps,
        aspect_ratio_gaps=ar_gaps,
        generation_tasks=tasks,
        active_learning_signals=al_signals,
        quality_baselines=quality_metrics,
    )
