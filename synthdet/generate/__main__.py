"""CLI entrypoint for synthetic data generation.

Usage:
    python -m synthdet.generate <data.yaml> --output <dir> [--config config.yaml]
                                [--method compositor|inpainting|both]
                                [--augment] [--seed 42] [--dry-run] [--json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from synthdet.analysis.loader import load_yolo_dataset
from synthdet.analysis.statistics import compute_dataset_statistics
from synthdet.analysis.strategy import generate_synthesis_strategy
from synthdet.config import SynthDetConfig
from synthdet.generate.compositor import run_compositor_pipeline

console = Console()


def _build_summary_panel(output_dataset, output_dir: Path, method: str) -> Panel:
    """Build a summary panel for the generation results."""
    total = len(output_dataset.train) + len(output_dataset.valid)
    negatives = sum(
        1 for r in output_dataset.all_records if r.is_negative
    )
    with_defects = total - negatives

    lines = [
        f"[bold]Output directory:[/bold] {output_dir}",
        f"[bold]Method:[/bold] {method}",
        f"[bold]Total images:[/bold] {total}",
        f"  Train: {len(output_dataset.train)}",
        f"  Valid: {len(output_dataset.valid)}",
        f"[bold]With defects:[/bold] {with_defects}",
        f"[bold]Negative (clean):[/bold] {negatives}",
        f"[bold]Classes:[/bold] {', '.join(output_dataset.class_names)}",
    ]
    return Panel("\n".join(lines), title="Generation Complete", border_style="green")


def _build_task_table(strategy, method: str) -> Table:
    """Build a table showing per-task breakdown."""
    table = Table(title="Per-Task Breakdown", show_lines=True)
    table.add_column("Task ID", style="cyan")
    table.add_column("Method", width=12)
    table.add_column("Priority", justify="center", width=8)
    table.add_column("Images", justify="right", width=8)
    table.add_column("Rationale")

    methods = {"compositor", "inpainting"} if method == "both" else {method}
    for task in strategy.generation_tasks:
        if task.method not in methods:
            continue
        table.add_row(
            task.task_id,
            task.method,
            f"{task.priority:.1f}",
            str(task.num_images),
            task.rationale,
        )
    return table


def _results_to_dict(output_dataset, strategy, output_dir: Path, method: str) -> dict:
    """Convert results to JSON-serializable dict."""
    total = len(output_dataset.train) + len(output_dataset.valid)
    negatives = sum(1 for r in output_dataset.all_records if r.is_negative)

    methods = {"compositor", "inpainting"} if method == "both" else {method}
    return {
        "output_dir": str(output_dir),
        "method": method,
        "total_images": total,
        "train_images": len(output_dataset.train),
        "valid_images": len(output_dataset.valid),
        "with_defects": total - negatives,
        "negative_images": negatives,
        "class_names": output_dataset.class_names,
        "tasks": [
            {
                "task_id": t.task_id,
                "priority": t.priority,
                "num_images": t.num_images,
                "method": t.method,
                "rationale": t.rationale,
            }
            for t in strategy.generation_tasks
            if t.method in methods
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic object detection data.",
        prog="python -m synthdet.generate",
    )
    parser.add_argument("data_yaml", type=Path, help="Path to data.yaml")
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output directory for generated dataset",
    )
    parser.add_argument("--config", type=Path, default=None, help="SynthDet config YAML")
    parser.add_argument(
        "--method", choices=["compositor", "inpainting", "both"],
        default="compositor", help="Generation method (default: compositor)",
    )
    parser.add_argument("--augment", action="store_true", help="Apply classical augmentation")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost without API calls")
    parser.add_argument("--json", action="store_true", help="Output JSON summary")
    args = parser.parse_args(argv)

    # Configure logging so pipeline progress is visible
    if not args.json:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[RichHandler(console=console, show_time=True, show_path=False)],
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    if not args.data_yaml.is_file():
        console.print(f"[red]Error: {args.data_yaml} not found[/red]")
        return 1

    config = SynthDetConfig.from_yaml(args.config) if args.config else SynthDetConfig.default()

    # Override preferred_method so the strategy generates tasks for the chosen method
    config.analysis.preferred_method = args.method if args.method != "both" else "compositor"

    # Step 1: Load dataset
    if not args.json:
        console.print("[bold]Loading dataset...[/bold]")
    dataset = load_yolo_dataset(args.data_yaml)

    # Step 2: Compute statistics
    if not args.json:
        console.print("[bold]Computing statistics...[/bold]")
    stats = compute_dataset_statistics(dataset, config.analysis)

    # Step 3: Generate strategy
    if not args.json:
        console.print("[bold]Generating synthesis strategy...[/bold]")
    strategy = generate_synthesis_strategy(dataset, stats, config.analysis)

    augment_config = config.augmentation if args.augment else None
    method = args.method

    # Step 4: Run pipeline(s)
    from synthdet.types import Dataset

    output_dataset = Dataset(
        root=args.output, class_names=dataset.class_names,
        train=[], valid=[], test=[],
    )

    if method in ("compositor", "both"):
        if not args.json:
            console.print("[bold]Running compositor pipeline...[/bold]")
        output_dataset = run_compositor_pipeline(
            dataset=dataset,
            strategy=strategy,
            config=config.compositor,
            output_dir=args.output,
            augment_config=augment_config if method == "compositor" else None,
            seed=args.seed,
        )

    if method in ("inpainting", "both"):
        if not args.json:
            console.print("[bold]Running inpainting pipeline...[/bold]")
        from synthdet.generate.inpainting import run_inpainting_pipeline

        # For "both", generate a separate strategy with method="inpainting"
        if method == "both":
            config.analysis.preferred_method = "inpainting"
            inpaint_strategy = generate_synthesis_strategy(dataset, stats, config.analysis)
        else:
            inpaint_strategy = strategy

        inpaint_output_dir = args.output if method == "inpainting" else args.output / "inpainting"
        inpaint_dataset = run_inpainting_pipeline(
            dataset=dataset,
            strategy=inpaint_strategy,
            config=config.inpainting,
            output_dir=inpaint_output_dir,
            augment_config=augment_config if method == "inpainting" else None,
            seed=args.seed,
            dry_run=args.dry_run,
        )

        if method == "inpainting":
            output_dataset = inpaint_dataset
        else:
            # Merge inpainting results into compositor results
            output_dataset.train.extend(inpaint_dataset.train)
            output_dataset.valid.extend(inpaint_dataset.valid)
            # Merge tasks for display
            strategy.generation_tasks.extend(inpaint_strategy.generation_tasks)

    # Output
    if args.json:
        result = _results_to_dict(output_dataset, strategy, args.output, method)
        print(json.dumps(result, indent=2))
    else:
        console.print()
        console.rule("[bold green]SynthDet Generation Report[/bold green]")
        console.print()
        console.print(_build_summary_panel(output_dataset, args.output, method))
        console.print()
        console.print(_build_task_table(strategy, method))
        console.print()
        console.rule("[bold]Generation complete[/bold]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
