"""CLI entrypoint for the SynthDet pipeline orchestrator.

Usage:
    python -m synthdet.pipeline <data.yaml> --output <dir>
        [--config config.yaml]
        [--method {all,compositor,inpainting,generative,modify_annotate}]
        [--augment] [--seed 42] [--dry-run] [--validate] [--json]
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

    # Run pipeline
    result = run_pipeline(args.data_yaml, args.output, config=config, seed=args.seed)

    # Output
    if args.json:
        print(json.dumps(_result_to_dict(result), indent=2))
    else:
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
        console.rule("[bold]Pipeline complete[/bold]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
