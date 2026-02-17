"""CLI entrypoint for dataset analysis.

Usage:
    python -m synthdet.analysis <data.yaml> [--config config.yaml] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from synthdet.analysis.loader import load_yolo_dataset
from synthdet.analysis.statistics import compute_dataset_statistics
from synthdet.analysis.strategy import generate_synthesis_strategy
from synthdet.config import AnalysisConfig, SynthDetConfig
from synthdet.types import BBoxSizeBucket, SpatialRegion

console = Console()


def _build_overview_panel(stats, dataset) -> Panel:
    """Build the dataset overview panel."""
    lines = [
        f"[bold]Root:[/bold] {dataset.root}",
        f"[bold]Classes:[/bold] {', '.join(dataset.class_names)} ({len(dataset.class_names)})",
        f"[bold]Total images:[/bold] {stats.total_images}",
    ]
    for split, count in stats.split_image_counts.items():
        ann = stats.split_annotation_counts.get(split, 0)
        lines.append(f"  {split}: {count} images, {ann} annotations")
    lines.append(f"[bold]Total annotations:[/bold] {stats.total_annotations}")
    lines.append(f"[bold]Unique source images:[/bold] {stats.unique_source_images}")

    if stats.image_width and stats.image_height:
        lines.append(f"[bold]Image dimensions:[/bold] {stats.image_width}x{stats.image_height}")

    # Negative examples warning
    if stats.negative_ratio == 0:
        lines.append(
            "[bold red]Negative examples: 0 (0.0%)[/bold red] "
            "[yellow]-- CRITICAL GAP: no negative examples![/yellow]"
        )
    else:
        lines.append(
            f"[bold]Negative examples:[/bold] {stats.negative_images} "
            f"({stats.negative_ratio:.1%})"
        )

    return Panel("\n".join(lines), title="Dataset Overview", border_style="blue")


def _build_class_table(stats) -> Table:
    """Build the class distribution table."""
    table = Table(title="Class Distribution", show_lines=True)
    table.add_column("ID", style="cyan", width=4)
    table.add_column("Class", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Fraction", justify="right")
    table.add_column("Distribution", width=30)

    max_count = max((d.count for d in stats.class_distributions), default=1)
    for dist in stats.class_distributions:
        bar_len = int(25 * dist.count / max_count) if max_count > 0 else 0
        bar = "[green]" + "\u2588" * bar_len + "[/green]"
        table.add_row(
            str(dist.class_id),
            dist.class_name,
            str(dist.count),
            f"{dist.fraction:.1%}",
            bar,
        )
    return table


def _build_size_bucket_table(stats) -> Table:
    """Build the size bucket distribution table."""
    table = Table(
        title=f"Size Bucket Distribution (uniformity: {stats.bucket_uniformity:.2f})",
        show_lines=True,
    )
    table.add_column("Bucket", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Distribution", width=30)

    max_count = max(stats.overall_bucket_counts.values(), default=1)
    for bucket in BBoxSizeBucket:
        count = stats.overall_bucket_counts.get(bucket, 0)
        bar_len = int(25 * count / max_count) if max_count > 0 else 0
        color = "red" if count == 0 else "yellow" if count < 20 else "green"
        bar = f"[{color}]" + "\u2588" * bar_len + f"[/{color}]"
        table.add_row(bucket.value, str(count), bar)
    return table


def _build_spatial_heatmap(stats) -> Panel:
    """Build a 3x3 spatial region heatmap."""
    grid_vals = []
    for region in SpatialRegion:
        grid_vals.append(stats.overall_region_counts.get(region, 0))

    max_val = max(grid_vals) if grid_vals else 1

    rows_text = []
    for row in range(3):
        cells = []
        for col in range(3):
            idx = row * 3 + col
            val = grid_vals[idx]
            # Color code: red=0, yellow=low, green=high
            if val == 0:
                color = "red"
            elif val < max_val * 0.3:
                color = "yellow"
            else:
                color = "green"
            cells.append(f"[{color}]{val:>4}[/{color}]")
        rows_text.append("  ".join(cells))

    content = "\n".join(rows_text)
    return Panel(
        content,
        title=f"Spatial Heatmap (uniformity: {stats.region_uniformity:.2f})",
        border_style="magenta",
    )


def _build_annotation_density_panel(stats) -> Panel:
    """Build annotation density statistics panel."""
    lines = [
        f"[bold]Mean annotations/image:[/bold] {stats.annotations_per_image_mean:.2f}",
        f"[bold]Std:[/bold] {stats.annotations_per_image_std:.2f}",
        f"[bold]Max:[/bold] {stats.annotations_per_image_max}",
        "",
        "[bold]Distribution (annotations per image → count):[/bold]",
    ]
    for n_ann in sorted(stats.annotations_per_image_histogram.keys()):
        count = stats.annotations_per_image_histogram[n_ann]
        bar = "\u2588" * min(count, 50)
        lines.append(f"  {n_ann:>2}: {count:>3} {bar}")

    return Panel("\n".join(lines), title="Annotation Density", border_style="cyan")


def _build_strategy_table(strategy) -> Table:
    """Build the synthesis strategy task table."""
    table = Table(title="Synthesis Strategy — Generation Tasks", show_lines=True)
    table.add_column("Priority", justify="center", width=8)
    table.add_column("Task ID", style="cyan")
    table.add_column("Images", justify="right", width=8)
    table.add_column("Method", width=12)
    table.add_column("Rationale")

    for task in strategy.generation_tasks:
        prio_color = (
            "red" if task.priority >= 0.9
            else "yellow" if task.priority >= 0.7
            else "green" if task.priority >= 0.5
            else "dim"
        )
        table.add_row(
            f"[{prio_color}]{task.priority:.1f}[/{prio_color}]",
            task.task_id,
            str(task.num_images),
            task.method,
            task.rationale,
        )
    return table


def _build_strategy_summary(strategy, stats) -> Panel:
    """Build strategy summary panel."""
    lines = [
        f"[bold]Current images:[/bold] {stats.total_images}",
        f"[bold]Target total:[/bold] {strategy.target_total_images}",
        f"[bold]Planned synthetic:[/bold] {strategy.total_synthetic_images}",
        f"[bold]Target negative ratio:[/bold] {strategy.negative_ratio:.1%}",
        f"[bold]Generation tasks:[/bold] {len(strategy.generation_tasks)}",
    ]
    return Panel("\n".join(lines), title="Strategy Summary", border_style="green")


def _build_active_learning_panel(strategy) -> Panel:
    """Build active learning status panel."""
    if strategy.has_model_feedback:
        lines = [f"[green]Model feedback available[/green]"]
        for signal in strategy.active_learning_signals:
            lines.append(f"  [{signal.source}] {signal.rationale} (priority: {signal.priority})")
    else:
        lines = [
            "[yellow]No model feedback available[/yellow]",
            "",
            "Strategy is based on dataset gap analysis only (exploration).",
            "To enable active learning (exploitation), provide model evaluation:",
            "  python -m synthdet.analysis data.yaml --model-eval eval.json",
            "",
            "Feedback loop: train YOLO -> evaluate -> provide eval.json -> regenerate strategy",
        ]
    return Panel("\n".join(lines), title="Active Learning Status", border_style="yellow")


def _build_quality_panel(config: SynthDetConfig) -> Panel:
    """Build quality monitoring status panel."""
    qc = config.quality_monitoring
    lines = [
        "[yellow]Quality baselines not yet established[/yellow]",
        "",
        "SPC monitoring will activate after the first generation cycle.",
        "",
        "[bold]Configuration:[/bold]",
        f"  Control limit: {qc.control_limit_sigma:.1f} sigma (Shewhart)",
        f"  Monitored layers: {', '.join(qc.activation_layers)}",
        f"  Snapshot percentiles: {qc.snapshot_percentiles}",
        f"  Trend window: {qc.trend_window} (Western Electric rule)",
    ]
    return Panel("\n".join(lines), title="Quality Monitoring (SPC)", border_style="red")


def _stats_to_dict(stats, strategy) -> dict:
    """Convert statistics and strategy to a JSON-serializable dict."""
    return {
        "dataset": {
            "total_images": stats.total_images,
            "total_annotations": stats.total_annotations,
            "negative_images": stats.negative_images,
            "negative_ratio": stats.negative_ratio,
            "unique_source_images": stats.unique_source_images,
            "split_image_counts": stats.split_image_counts,
            "split_annotation_counts": stats.split_annotation_counts,
            "image_dimensions": (
                {"width": stats.image_width, "height": stats.image_height}
                if stats.image_width else None
            ),
        },
        "class_distributions": [
            {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "count": d.count,
                "fraction": d.fraction,
                "bucket_counts": {k.value: v for k, v in d.bucket_counts.items()},
                "region_counts": {k.value: v for k, v in d.region_counts.items()},
            }
            for d in stats.class_distributions
        ],
        "size_buckets": {k.value: v for k, v in stats.overall_bucket_counts.items()},
        "spatial_regions": {k.value: v for k, v in stats.overall_region_counts.items()},
        "uniformity": {
            "bucket": stats.bucket_uniformity,
            "region": stats.region_uniformity,
        },
        "annotations_per_image": {
            "mean": stats.annotations_per_image_mean,
            "std": stats.annotations_per_image_std,
            "max": stats.annotations_per_image_max,
            "histogram": stats.annotations_per_image_histogram,
        },
        "strategy": {
            "target_total_images": strategy.target_total_images,
            "target_class_counts": strategy.target_class_counts,
            "negative_ratio": strategy.negative_ratio,
            "total_synthetic_images": strategy.total_synthetic_images,
            "num_tasks": len(strategy.generation_tasks),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "priority": t.priority,
                    "num_images": t.num_images,
                    "method": t.method,
                    "rationale": t.rationale,
                }
                for t in strategy.generation_tasks
            ],
            "has_model_feedback": strategy.has_model_feedback,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a YOLO dataset and generate a synthesis strategy.",
        prog="python -m synthdet.analysis",
    )
    parser.add_argument("data_yaml", type=Path, help="Path to data.yaml")
    parser.add_argument("--config", type=Path, default=None, help="SynthDet config YAML")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of rich report")
    parser.add_argument(
        "--model-eval", type=Path, default=None,
        help="Path to model evaluation JSON (for active learning)",
    )
    args = parser.parse_args(argv)

    if not args.data_yaml.is_file():
        console.print(f"[red]Error: {args.data_yaml} not found[/red]")
        return 1

    # Load config
    config = SynthDetConfig.from_yaml(args.config) if args.config else SynthDetConfig.default()

    # Load dataset
    if not args.json:
        console.print("[bold]Loading dataset...[/bold]")
    dataset = load_yolo_dataset(args.data_yaml)

    # Compute statistics
    if not args.json:
        console.print("[bold]Computing statistics...[/bold]")
    stats = compute_dataset_statistics(dataset, config.analysis)

    # Generate strategy
    if not args.json:
        console.print("[bold]Generating synthesis strategy...[/bold]")
    strategy = generate_synthesis_strategy(
        dataset, stats, config.analysis,
        model_profile=None,  # Phase 1: no model feedback
        quality_metrics=None,
    )

    if args.json:
        output = _stats_to_dict(stats, strategy)
        print(json.dumps(output, indent=2, default=str))
        return 0

    # Rich report
    console.print()
    console.rule("[bold blue]SynthDet Dataset Analysis Report[/bold blue]")
    console.print()

    console.print(_build_overview_panel(stats, dataset))
    console.print()

    console.print(_build_class_table(stats))
    console.print()

    console.print(_build_size_bucket_table(stats))
    console.print()

    console.print(_build_spatial_heatmap(stats))
    console.print()

    console.print(_build_annotation_density_panel(stats))
    console.print()

    console.rule("[bold green]Synthesis Strategy[/bold green]")
    console.print()

    console.print(_build_strategy_summary(strategy, stats))
    console.print()

    console.print(_build_strategy_table(strategy))
    console.print()

    console.print(_build_active_learning_panel(strategy))
    console.print()

    console.print(_build_quality_panel(config))
    console.print()

    console.rule("[bold]Analysis complete[/bold]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
