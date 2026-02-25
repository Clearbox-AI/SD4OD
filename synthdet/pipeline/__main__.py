"""CLI entrypoint for the SynthDet pipeline orchestrator.

Usage:
    python -m synthdet.pipeline <data.yaml> --output <dir>
        [--config config.yaml]
        [--method {all,compositor,inpainting,generative,modify_annotate}]
        [--augment] [--seed 42] [--dry-run] [--validate] [--json]
        [--train] [--train-epochs N] [--active-learning N]
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

from synthdet.pipeline.config_schema import VALID_METHODS, PipelineConfig
from synthdet.pipeline.orchestrator import PipelineResult, run_pipeline

console = Console()

# Map CLI shorthand to actual method names
_METHOD_ALIASES: dict[str, list[str]] = {
    "all": sorted(VALID_METHODS),
    "compositor": ["compositor"],
    "inpainting": ["inpainting"],
    "generative": ["generative_compositor"],
    "modify_annotate": ["modify_annotate"],
}


def _build_summary_panel(result: PipelineResult) -> Panel:
    """Build a rich summary panel."""
    lines = [
        f"[bold]Output directory:[/bold] {result.output_dir}",
        f"[bold]Methods:[/bold] {', '.join(result.methods_used)}",
        f"[bold]Total images:[/bold] {result.total_records}",
        f"  Train: {result.train_count}",
        f"  Valid: {result.valid_count}",
    ]

    if result.total_cost_usd > 0:
        lines.append(f"[bold]Estimated cost:[/bold] ${result.total_cost_usd:.2f}")

    if result.validation_report is not None:
        status = "[green]VALID[/green]" if result.validation_report.is_valid else "[red]INVALID[/red]"
        lines.append(f"[bold]Validation:[/bold] {status}")

    title = "Dry Run Estimate" if result.dry_run else "Pipeline Complete"
    border = "yellow" if result.dry_run else "green"
    return Panel("\n".join(lines), title=title, border_style=border)


def _build_cost_table(result: PipelineResult) -> Table:
    """Build a per-method cost/count breakdown table."""
    table = Table(title="Per-Method Breakdown", show_lines=True)
    table.add_column("Method", style="cyan")
    table.add_column("Records", justify="right", width=10)
    table.add_column("Cost (USD)", justify="right", width=12)

    for method in result.methods_used:
        count = result.records_per_method.get(method, 0)
        cost = result.cost_per_method.get(method, 0.0)
        table.add_row(method, str(count), f"${cost:.2f}")

    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{result.total_records}[/bold]",
        f"[bold]${result.total_cost_usd:.2f}[/bold]",
    )
    return table


def _result_to_dict(result: PipelineResult) -> dict:
    """Convert PipelineResult to a JSON-serializable dict."""
    d = {
        "output_dir": str(result.output_dir),
        "methods": result.methods_used,
        "dry_run": result.dry_run,
        "total_records": result.total_records,
        "train_count": result.train_count,
        "valid_count": result.valid_count,
        "total_cost_usd": round(result.total_cost_usd, 4),
        "records_per_method": result.records_per_method,
        "cost_per_method": {k: round(v, 4) for k, v in result.cost_per_method.items()},
    }
    if result.validation_report is not None:
        d["validation"] = {
            "is_valid": result.validation_report.is_valid,
            "errors": len(result.validation_report.errors),
            "warnings": len(result.validation_report.warnings),
            "total_images": result.validation_report.total_images,
            "total_labels": result.validation_report.total_labels,
        }
    return d


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SynthDet pipeline — unified synthetic data generation.",
        prog="python -m synthdet.pipeline",
    )
    parser.add_argument("data_yaml", type=Path, help="Path to data.yaml")
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output directory for generated dataset",
    )
    parser.add_argument("--config", type=Path, default=None, help="Pipeline config YAML")
    parser.add_argument(
        "--method",
        choices=list(_METHOD_ALIASES.keys()),
        default="compositor",
        help="Generation method (default: compositor)",
    )
    parser.add_argument("--augment", action="store_true", help="Apply classical augmentation")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost without API calls")
    parser.add_argument("--validate", action="store_true", help="Validate output dataset")
    parser.add_argument("--json", action="store_true", help="Output JSON summary")
    parser.add_argument("--train", action="store_true", help="Train YOLO after generation")
    parser.add_argument("--train-epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument(
        "--active-learning", type=int, default=0, metavar="N",
        help="Run N active learning iterations (implies --train)",
    )
    args = parser.parse_args(argv)

    # Configure logging
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

    # Build config
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()

    # CLI flags override config
    config.methods = _METHOD_ALIASES[args.method]
    if args.augment:
        config.augment = True
    if args.dry_run:
        config.dry_run = True
    if args.validate:
        config.validate_output = True
    if args.train:
        config.train = True
    if args.active_learning > 0:
        config.active_learning_iterations = args.active_learning
        config.train = True
    if args.train_epochs is not None:
        config.training.epochs = args.train_epochs

    # --- Execution branch: Active Learning ---
    if config.active_learning_iterations > 0:
        return _run_active_learning(args, config)

    # --- Execution branch: Generate (+ optional train) ---
    result = run_pipeline(args.data_yaml, args.output, config=config, seed=args.seed)

    # Output generation results
    out = _result_to_dict(result)

    if not args.json:
        console.print()
        console.rule("[bold green]SynthDet Pipeline Report[/bold green]")
        console.print()
        console.print(_build_summary_panel(result))
        console.print()
        console.print(_build_cost_table(result))
        console.print()
        if result.validation_report and not result.validation_report.is_valid:
            console.print("[red]Validation issues:[/red]")
            for issue in result.validation_report.errors[:10]:
                console.print(f"  [{issue.category}] {issue.message}")
            console.print()

    # --- Optional single train after generation ---
    if config.train and not result.dry_run and result.total_records > 0:
        from synthdet.training.trainer import YOLOTrainer

        trainer = YOLOTrainer(config.training)
        data_yaml_out = result.output_dir / "data.yaml"
        train_result = trainer.train(data_yaml_out)

        out["training"] = {
            "epochs_completed": train_result.epochs_completed,
            "best_map50": round(train_result.best_map50, 4),
            "best_map50_95": round(train_result.best_map50_95, 4),
            "training_time_seconds": round(train_result.training_time_seconds, 1),
            "best_weights": str(train_result.best_weights),
        }

        if not args.json:
            console.print()
            console.print(Panel(
                f"[bold]Epochs:[/bold] {train_result.epochs_completed}\n"
                f"[bold]mAP50:[/bold] {train_result.best_map50:.3f}\n"
                f"[bold]mAP50-95:[/bold] {train_result.best_map50_95:.3f}\n"
                f"[bold]Time:[/bold] {train_result.training_time_seconds:.1f}s\n"
                f"[bold]Weights:[/bold] {train_result.best_weights}",
                title="Training Complete",
                border_style="blue",
            ))

    if args.json:
        print(json.dumps(out, indent=2))
    elif not config.train:
        console.rule("[bold]Pipeline complete[/bold]")
    else:
        console.rule("[bold]Pipeline + Training complete[/bold]")

    return 0


def _run_active_learning(args, config: PipelineConfig) -> int:
    """Execute the active learning loop and print results."""
    from synthdet.training.loop import ActiveLearningLoop

    config.active_learning.max_iterations = config.active_learning_iterations

    loop = ActiveLearningLoop(
        data_yaml=args.data_yaml,
        output_dir=args.output,
        pipeline_config=config,
        training_config=config.training,
        al_config=config.active_learning,
        seed=args.seed,
    )
    al_result = loop.run()

    if args.json:
        out = {
            "active_learning": {
                "iterations": len(al_result.iterations),
                "final_map50": round(al_result.final_map50, 4),
                "stopped_reason": al_result.stopped_reason,
                "total_training_time_seconds": round(al_result.total_training_time_seconds, 1),
                "total_cost_usd": round(al_result.total_cost_usd, 4),
                "final_weights": str(al_result.final_weights),
                "per_iteration": [
                    {
                        "iteration": ir.iteration,
                        "map50": round(ir.map50, 4),
                        "map50_improvement": round(ir.map50_improvement, 4),
                        "records_generated": ir.pipeline_result.total_records,
                    }
                    for ir in al_result.iterations
                ],
            },
        }
        print(json.dumps(out, indent=2))
    else:
        console.print()
        console.rule("[bold green]Active Learning Report[/bold green]")
        console.print()

        table = Table(title="Iteration Summary", show_lines=True)
        table.add_column("Iter", justify="right", width=5)
        table.add_column("Records", justify="right", width=10)
        table.add_column("mAP50", justify="right", width=10)
        table.add_column("Improvement", justify="right", width=12)

        for ir in al_result.iterations:
            table.add_row(
                str(ir.iteration),
                str(ir.pipeline_result.total_records),
                f"{ir.map50:.3f}",
                f"{ir.map50_improvement:+.3f}",
            )
        console.print(table)
        console.print()
        console.print(Panel(
            f"[bold]Final mAP50:[/bold] {al_result.final_map50:.3f}\n"
            f"[bold]Iterations:[/bold] {len(al_result.iterations)}\n"
            f"[bold]Stopped:[/bold] {al_result.stopped_reason}\n"
            f"[bold]Total training time:[/bold] {al_result.total_training_time_seconds:.1f}s\n"
            f"[bold]Total cost:[/bold] ${al_result.total_cost_usd:.2f}\n"
            f"[bold]Final weights:[/bold] {al_result.final_weights}",
            title="Active Learning Complete",
            border_style="green",
        ))
        console.rule("[bold]Done[/bold]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
