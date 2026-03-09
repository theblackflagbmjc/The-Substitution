#!/usr/bin/env python3
"""
The Substitution v1.0.0 — Evaluation Report Generator
Brandon Michael Jeanpierre Corporation d/b/a The Black Flag
Organization: theblackflagbmjc

Reads evaluation results from run_benchmarks.py output and generates
a formatted summary report comparing against baseline scores.

Usage:
    python evaluation/generate_report.py

    # Custom input/output paths
    python evaluation/generate_report.py \
        --input ./output/evaluation/evaluation_report.json \
        --output ./output/evaluation/summary_report.md
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "output" / "evaluation"
LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "generate_report.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("report_generator")
console = Console()


def load_baselines() -> dict:
    """Load baseline scores from evaluation config."""
    config_path = PROJECT_ROOT / "evaluation" / "evaluation_config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("baselines", {}).get("qwen2.5_7b_instruct", {})
    return {}


def compute_delta(score: float, baseline: float) -> str:
    """Compute and format the delta between score and baseline."""
    delta = score - baseline
    if delta > 0:
        return f"[green]+{delta:.1f}[/green]"
    elif delta < 0:
        return f"[red]{delta:.1f}[/red]"
    else:
        return "[dim]0.0[/dim]"


def generate_terminal_report(report: dict, baselines: dict):
    """Print a formatted report to the terminal."""
    console.print()
    console.print(Panel(
        "[bold cyan]The Substitution v1.0.0\n"
        "Evaluation Report\n"
        f"[dim]{report.get('timestamp', 'N/A')}[/dim]",
        title="BMJC d/b/a The Black Flag",
        subtitle="theblackflagbmjc",
    ))
    console.print()

    # --- Benchmark Results Table ---
    table = Table(title="Benchmark Results vs. Base Model", show_lines=True)
    table.add_column("Benchmark", style="cyan")
    table.add_column("Metric", style="dim")
    table.add_column("The Substitution", justify="right", style="bold")
    table.add_column("Qwen2.5-7B Base", justify="right", style="dim")
    table.add_column("Delta", justify="right")
    table.add_column("Samples", justify="right", style="dim")
    table.add_column("Runtime", justify="right", style="dim")

    benchmarks = report.get("benchmarks", {})

    for name, result in benchmarks.items():
        baseline = baselines.get(name, None)
        delta = compute_delta(result["score"], baseline) if baseline else "—"
        baseline_str = f"{baseline}%" if baseline else "—"
        runtime = f"{result.get('runtime_seconds', '?')}s"

        table.add_row(
            name,
            result["metric"],
            f"{result['score']}%",
            baseline_str,
            delta,
            str(result["total"]),
            runtime,
        )

    console.print(table)
    console.print()

    # --- Summary ---
    summary = report.get("summary", {})
    avg_score = summary.get("average_score", 0)
    console.print(f"[bold]Average Score: {avg_score}%[/bold]")

    if baselines:
        baseline_avg = sum(baselines.values()) / len(baselines)
        avg_delta = avg_score - baseline_avg
        sign = "+" if avg_delta >= 0 else ""
        color = "green" if avg_delta >= 0 else "red"
        console.print(f"[bold]Baseline Average: {baseline_avg:.1f}%[/bold]")
        console.print(f"[bold {color}]Overall Delta: {sign}{avg_delta:.1f}%[/bold {color}]")

    console.print()

    # --- MMLU Subject Breakdown (if available) ---
    mmlu_result = benchmarks.get("mmlu", {})
    subject_accuracies = mmlu_result.get("subject_accuracies", {})

    if subject_accuracies:
        console.print("[bold]MMLU Subject Breakdown (Top 10 / Bottom 10):[/bold]")

        sorted_subjects = sorted(subject_accuracies.items(), key=lambda x: -x[1])

        sub_table = Table(show_lines=False)
        sub_table.add_column("Subject", style="cyan")
        sub_table.add_column("Accuracy", justify="right")

        console.print("[dim]  Top 10:[/dim]")
        for subject, acc in sorted_subjects[:10]:
            sub_table.add_row(f"  {subject}", f"{acc}%")

        if len(sorted_subjects) > 10:
            sub_table.add_row("  ...", "...")

            console.print()
            console.print("[dim]  Bottom 10:[/dim]")
            for subject, acc in sorted_subjects[-10:]:
                sub_table.add_row(f"  {subject}", f"{acc}%")

        console.print(sub_table)
        console.print()


def generate_markdown_report(report: dict, baselines: dict, output_path: Path):
    """Generate a Markdown-formatted evaluation report."""
    lines = []
    lines.append("# The Substitution v1.0.0 — Evaluation Report")
    lines.append("")
    lines.append(f"**Organization:** theblackflagbmjc")
    lines.append(f"**Date:** {report.get('timestamp', 'N/A')}")
    lines.append(f"**Model:** {report.get('model', 'N/A')}")
    lines.append("")
    lines.append("## Benchmark Results")
    lines.append("")
    lines.append("| Benchmark | Metric | Score | Baseline | Delta |")
    lines.append("|-----------|--------|------:|----------:|------:|")

    benchmarks = report.get("benchmarks", {})
    for name, result in benchmarks.items():
        baseline = baselines.get(name, None)
        baseline_str = f"{baseline}%" if baseline else "—"
        if baseline:
            delta = result["score"] - baseline
            delta_str = f"{'+' if delta >= 0 else ''}{delta:.1f}%"
        else:
            delta_str = "—"

        lines.append(f"| {name} | {result['metric']} | {result['score']}% | {baseline_str} | {delta_str} |")

    lines.append("")

    # Summary
    summary = report.get("summary", {})
    lines.append(f"**Average Score:** {summary.get('average_score', 0)}%")
    lines.append("")

    # MMLU breakdown
    mmlu_result = benchmarks.get("mmlu", {})
    subject_accuracies = mmlu_result.get("subject_accuracies", {})
    if subject_accuracies:
        lines.append("## MMLU Subject Breakdown")
        lines.append("")
        lines.append("| Subject | Accuracy |")
        lines.append("|---------|--------:|")
        for subject, acc in sorted(subject_accuracies.items(), key=lambda x: -x[1]):
            lines.append(f"| {subject} | {acc}% |")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by The Substitution evaluation pipeline.*")
    lines.append(f"*Brandon Michael Jeanpierre Corporation d/b/a The Black Flag*")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    console.print(f"[bold]Markdown report saved: {output_path}[/bold]")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument(
        "--input", type=str,
        default=str(EVAL_DIR / "evaluation_report.json"),
        help="Path to evaluation results JSON",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(EVAL_DIR / "summary_report.md"),
        help="Path for Markdown output report",
    )
    args = parser.parse_args()

    console.print()
    console.print("[bold cyan]The Substitution v1.0.0 — Report Generator[/bold cyan]")
    console.print()

    # Load results
    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Results file not found: {input_path}[/red]")
        console.print("[dim]Run evaluation/run_benchmarks.py first.[/dim]")
        sys.exit(1)

    with open(input_path, "r") as f:
        report = json.load(f)

    # Load baselines
    baselines = load_baselines()

    # Generate reports
    generate_terminal_report(report, baselines)
    generate_markdown_report(report, baselines, Path(args.output))

    console.print()
    console.print("[green bold]Report generation complete.[/green bold]")


if __name__ == "__main__":
    main()
