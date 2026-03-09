#!/usr/bin/env python3
"""
The Substitution v1.0.0 — Dataset Acquisition Pipeline
Brandon Michael Jeanpierre Corporation d/b/a The Black Flag
Organization: theblackflagbmjc

Downloads all required training datasets from Hugging Face and stores them
in the project data directory with verification logging.

Usage:
    python datasets/download_datasets.py

Datasets:
    - Instruction: OpenHermes-2.5, UltraChat, OpenOrca
    - Coding: CodeSearchNet
    - Mathematics: GSM8K
    - Evaluation: HumanEval, MMLU

All datasets are MIT or permissively licensed.
"""

import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from datasets import load_dataset, get_dataset_config_names
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "download_datasets.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("dataset_downloader")

console = Console()

# --- Dataset Registry ---
# Each entry defines a dataset to download with its configuration.
# Categories align with the Master Engineering Prompt spec:
#   - instruction: general instruction-following capability
#   - coding: programming and code generation
#   - math: mathematical reasoning
#   - evaluation: benchmark datasets (not used in training)

DATASET_REGISTRY = {
    # =========================================================================
    # INSTRUCTION DATASETS
    # =========================================================================
    "openhermes": {
        "hf_id": "teknium/OpenHermes-2.5",
        "category": "instruction",
        "description": "1M+ GPT-4 generated instruction pairs across diverse tasks",
        "split": "train",
        "config": None,
        "max_samples": None,  # Use all
        "license": "custom (open)",
        "priority": 1,
    },
    "ultrachat": {
        "hf_id": "openbmb/UltraChat",
        "category": "instruction",
        "description": "Multi-round dialogue data for conversational ability",
        "split": "train",
        "config": None,
        "max_samples": 500_000,  # Subsample — full dataset is very large
        "license": "MIT",
        "priority": 2,
    },
    "openorca": {
        "hf_id": "Open-Orca/OpenOrca",
        "category": "instruction",
        "description": "Augmented FLAN collection for reasoning and instruction following",
        "split": "train",
        "config": None,
        "max_samples": 500_000,  # Subsample
        "license": "MIT",
        "priority": 3,
    },
    # =========================================================================
    # CODING DATASETS
    # =========================================================================
    "codesearchnet": {
        "hf_id": "code-search-net/code_search_net",
        "category": "coding",
        "description": "2M code-comment pairs across 6 languages",
        "split": "train",
        "config": "python",  # Start with Python (primary language per spec)
        "max_samples": None,
        "license": "custom (open)",
        "priority": 4,
    },
    # =========================================================================
    # MATHEMATICS DATASETS
    # =========================================================================
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "category": "math",
        "description": "8.5K grade school math word problems requiring multi-step reasoning",
        "split": "train",
        "config": "main",
        "max_samples": None,
        "license": "MIT",
        "priority": 5,
    },
    # =========================================================================
    # EVALUATION DATASETS (not used in training — held out for benchmarking)
    # =========================================================================
    "humaneval": {
        "hf_id": "openai/openai_humaneval",
        "category": "evaluation",
        "description": "164 hand-written Python programming problems",
        "split": "test",
        "config": None,
        "max_samples": None,
        "license": "MIT",
        "priority": 6,
    },
    "mmlu": {
        "hf_id": "cais/mmlu",
        "category": "evaluation",
        "description": "Massive multitask benchmark across 57 subjects",
        "split": "test",
        "config": "all",
        "max_samples": None,
        "license": "MIT",
        "priority": 7,
    },
    "gsm8k_eval": {
        "hf_id": "openai/gsm8k",
        "category": "evaluation",
        "description": "GSM8K test split for math reasoning evaluation",
        "split": "test",
        "config": "main",
        "max_samples": None,
        "license": "MIT",
        "priority": 8,
    },
}


def compute_checksum(filepath: Path) -> str:
    """Compute SHA256 checksum for a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_dataset_size(dataset) -> int:
    """Get number of examples in a dataset."""
    try:
        return len(dataset)
    except Exception:
        return -1


def download_single_dataset(name: str, config: dict) -> dict:
    """
    Download a single dataset from Hugging Face.

    Args:
        name: Internal name for the dataset
        config: Dataset configuration from DATASET_REGISTRY

    Returns:
        dict with download metadata (status, path, size, checksum, etc.)
    """
    result = {
        "name": name,
        "hf_id": config["hf_id"],
        "category": config["category"],
        "status": "pending",
        "path": None,
        "num_examples": 0,
        "size_bytes": 0,
        "checksum": None,
        "error": None,
        "download_time_seconds": 0,
    }

    output_dir = DATA_DIR / config["category"] / name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dataset.jsonl"

    # Skip if already downloaded
    if output_path.exists() and output_path.stat().st_size > 0:
        result["status"] = "skipped (already exists)"
        result["path"] = str(output_path)
        result["size_bytes"] = output_path.stat().st_size
        result["checksum"] = compute_checksum(output_path)
        logger.info(f"  [{name}] Already exists, skipping. Size: {result['size_bytes']:,} bytes")
        return result

    start_time = time.time()

    try:
        logger.info(f"  [{name}] Downloading from {config['hf_id']}...")

        # Load dataset
        load_kwargs = {"path": config["hf_id"], "split": config["split"]}
        if config.get("config"):
            load_kwargs["name"] = config["config"]

        # Stream for large datasets to manage memory
        if config.get("max_samples"):
            load_kwargs["streaming"] = True
            dataset = load_dataset(**load_kwargs)

            # Write streamed data to JSONL
            count = 0
            with open(output_path, "w", encoding="utf-8") as f:
                for example in dataset:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    count += 1
                    if count >= config["max_samples"]:
                        break
            result["num_examples"] = count
        else:
            dataset = load_dataset(**load_kwargs)
            result["num_examples"] = get_dataset_size(dataset)

            # Save to JSONL
            with open(output_path, "w", encoding="utf-8") as f:
                for example in dataset:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")

        elapsed = time.time() - start_time
        result["status"] = "success"
        result["path"] = str(output_path)
        result["size_bytes"] = output_path.stat().st_size
        result["checksum"] = compute_checksum(output_path)
        result["download_time_seconds"] = round(elapsed, 2)

        logger.info(
            f"  [{name}] Downloaded {result['num_examples']:,} examples "
            f"({result['size_bytes'] / 1e6:.1f} MB) in {elapsed:.1f}s"
        )

    except Exception as e:
        elapsed = time.time() - start_time
        result["status"] = "failed"
        result["error"] = str(e)
        result["download_time_seconds"] = round(elapsed, 2)
        logger.error(f"  [{name}] Download FAILED: {e}")

    return result


def download_all_datasets() -> list[dict]:
    """Download all datasets in the registry."""
    results = []

    # Sort by priority
    sorted_datasets = sorted(DATASET_REGISTRY.items(), key=lambda x: x[1]["priority"])

    console.print()
    console.print("[bold cyan]The Substitution v1.0.0 — Dataset Acquisition Pipeline[/bold cyan]")
    console.print("[dim]Organization: theblackflagbmjc[/dim]")
    console.print()

    total = len(sorted_datasets)
    for i, (name, config) in enumerate(sorted_datasets, 1):
        console.print(f"[bold][{i}/{total}] {name}[/bold] — {config['description']}")
        console.print(f"  Source: {config['hf_id']} | Category: {config['category']}")
        result = download_single_dataset(name, config)
        results.append(result)
        console.print()

    return results


def generate_manifest(results: list[dict]):
    """Generate a download manifest with checksums and metadata."""
    manifest_path = DATA_DIR / "download_manifest.json"

    manifest = {
        "project": "The Substitution v1.0.0",
        "organization": "theblackflagbmjc",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "datasets": results,
        "summary": {
            "total": len(results),
            "success": sum(1 for r in results if r["status"] in ("success", "skipped (already exists)")),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "total_examples": sum(r["num_examples"] for r in results),
            "total_size_bytes": sum(r["size_bytes"] for r in results),
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"Manifest written to {manifest_path}")
    return manifest


def print_summary(results: list[dict]):
    """Print a summary table of all downloads."""
    table = Table(title="Dataset Download Summary", show_lines=True)
    table.add_column("Dataset", style="cyan")
    table.add_column("Category", style="dim")
    table.add_column("Status", style="bold")
    table.add_column("Examples", justify="right")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Checksum (first 12)", style="dim")

    for r in results:
        status_style = "green" if "success" in r["status"] or "skipped" in r["status"] else "red"
        checksum_short = r["checksum"][:12] if r["checksum"] else "—"
        table.add_row(
            r["name"],
            r["category"],
            f"[{status_style}]{r['status']}[/{status_style}]",
            f"{r['num_examples']:,}" if r["num_examples"] > 0 else "—",
            f"{r['size_bytes'] / 1e6:.1f}" if r["size_bytes"] > 0 else "—",
            checksum_short,
        )

    console.print()
    console.print(table)
    console.print()


def main():
    """Main entry point."""
    logger.info("=" * 70)
    logger.info("The Substitution v1.0.0 — Dataset Acquisition Pipeline")
    logger.info("=" * 70)

    results = download_all_datasets()
    manifest = generate_manifest(results)
    print_summary(results)

    # Final status
    s = manifest["summary"]
    console.print(f"[bold]Total datasets: {s['total']}[/bold]")
    console.print(f"[green]Successful: {s['success']}[/green]")
    if s["failed"] > 0:
        console.print(f"[red]Failed: {s['failed']}[/red]")
    console.print(f"Total examples: {s['total_examples']:,}")
    console.print(f"Total size: {s['total_size_bytes'] / 1e9:.2f} GB")
    console.print()

    if s["failed"] > 0:
        console.print("[yellow]Some downloads failed. Re-run this script to retry.[/yellow]")
        sys.exit(1)

    console.print("[green bold]All datasets acquired successfully.[/green bold]")


if __name__ == "__main__":
    main()
