#!/usr/bin/env python3
"""
The Substitution v1.0.0 — Training Data Preparation
Brandon Michael Jeanpierre Corporation d/b/a The Black Flag
Organization: theblackflagbmjc

Merges all cleaned datasets into a single shuffled training file,
creates train/validation splits, and prepares the final dataset
for tokenization.

Pipeline:
  1. Load all cleaned JSONL files from data/processed/
  2. Apply category-based sampling weights
  3. Shuffle and merge into a single dataset
  4. Create 95/5 train/validation split
  5. Convert to Hugging Face Dataset format
  6. Save to data/processed/final/

Usage:
    python datasets/prepare_training_data.py
"""

import json
import logging
import os
import random
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict
from rich.console import Console
from rich.table import Table

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FINAL_DIR = PROCESSED_DIR / "final"
LOG_DIR = PROJECT_ROOT / "logs"

FINAL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "prepare_training_data.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("data_preparer")
console = Console()

# --- Sampling Configuration ---
# These weights control how much each dataset contributes to the final mix.
# Higher weight = more representation in training.
# Rationale:
#   - instruction datasets are primary (general capability)
#   - coding is critical for the systems engineering role
#   - math supports reasoning ability
SAMPLING_CONFIG = {
    "openhermes_cleaned.jsonl": {
        "weight": 1.0,      # Full inclusion — highest quality instruction data
        "max_samples": None,
        "category": "instruction",
    },
    "ultrachat_cleaned.jsonl": {
        "weight": 0.5,      # Subsample — complementary multi-turn data
        "max_samples": 250_000,
        "category": "instruction",
    },
    "openorca_cleaned.jsonl": {
        "weight": 0.5,      # Subsample — complementary reasoning data
        "max_samples": 250_000,
        "category": "instruction",
    },
    "codesearchnet_cleaned.jsonl": {
        "weight": 1.0,      # Full inclusion — critical for code capability
        "max_samples": None,
        "category": "coding",
    },
    "gsm8k_cleaned.jsonl": {
        "weight": 3.0,      # Oversample — small dataset, high value for reasoning
        "max_samples": None,
        "category": "math",
    },
}

# Train/validation split ratio
TRAIN_RATIO = 0.95
VALIDATION_RATIO = 0.05

# Random seed for reproducibility
RANDOM_SEED = 42


def load_cleaned_dataset(filepath: Path, config: dict) -> list[dict]:
    """
    Load a cleaned JSONL file and apply sampling configuration.

    Args:
        filepath: Path to the cleaned JSONL file
        config: Sampling configuration for this dataset

    Returns:
        List of conversation examples
    """
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return []

    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                example = json.loads(line.strip())
                examples.append(example)
            except json.JSONDecodeError:
                continue

    # Apply max_samples cap
    if config.get("max_samples") and len(examples) > config["max_samples"]:
        random.seed(RANDOM_SEED)
        examples = random.sample(examples, config["max_samples"])

    # Apply oversampling weight (for small high-value datasets like gsm8k)
    weight = config.get("weight", 1.0)
    if weight > 1.0:
        repeat_count = int(weight)
        fractional = weight - repeat_count
        oversampled = examples * repeat_count
        if fractional > 0:
            extra_count = int(len(examples) * fractional)
            random.seed(RANDOM_SEED + 1)
            oversampled.extend(random.sample(examples, min(extra_count, len(examples))))
        examples = oversampled

    return examples


def validate_example(example: dict) -> bool:
    """
    Validate that an example has the correct structure for training.

    Required format:
    {"messages": [{"role": "...", "content": "..."}, ...]}

    Must have at least a user and assistant message.
    """
    messages = example.get("messages", [])
    if not messages or len(messages) < 2:
        return False

    roles = {m.get("role") for m in messages}
    if "user" not in roles or "assistant" not in roles:
        return False

    # Validate each message has role and non-empty content
    for msg in messages:
        if not msg.get("role") or not msg.get("content"):
            return False

    return True


def merge_datasets() -> list[dict]:
    """
    Load and merge all cleaned datasets according to sampling config.

    Returns:
        Merged and shuffled list of training examples
    """
    all_examples = []
    stats = {}

    console.print("[bold]Loading cleaned datasets...[/bold]")

    for filename, config in SAMPLING_CONFIG.items():
        filepath = PROCESSED_DIR / filename
        console.print(f"  Loading: {filename}")
        examples = load_cleaned_dataset(filepath, config)

        # Validate
        valid_examples = [ex for ex in examples if validate_example(ex)]
        invalid_count = len(examples) - len(valid_examples)

        if invalid_count > 0:
            logger.warning(f"  [{filename}] {invalid_count:,} invalid examples removed")

        # Tag source for tracking
        for ex in valid_examples:
            ex["_source"] = filename.replace("_cleaned.jsonl", "")

        all_examples.extend(valid_examples)

        stats[filename] = {
            "loaded": len(examples),
            "valid": len(valid_examples),
            "invalid": invalid_count,
            "category": config["category"],
        }

        console.print(
            f"    Loaded {len(valid_examples):,} examples "
            f"({invalid_count:,} invalid removed)"
        )

    console.print()

    # Shuffle
    console.print("[bold]Shuffling merged dataset...[/bold]")
    random.seed(RANDOM_SEED)
    random.shuffle(all_examples)

    return all_examples, stats


def create_splits(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split examples into train and validation sets.

    Returns:
        (train_examples, val_examples)
    """
    split_idx = int(len(examples) * TRAIN_RATIO)
    train = examples[:split_idx]
    val = examples[split_idx:]
    return train, val


def save_as_hf_dataset(train_examples: list[dict], val_examples: list[dict]):
    """
    Save the prepared data as a Hugging Face DatasetDict.

    This format is directly consumable by the training pipeline.
    """
    console.print("[bold]Converting to Hugging Face Dataset format...[/bold]")

    def examples_to_dict(examples: list[dict]) -> dict:
        """Convert list of examples to columnar format for HF Dataset."""
        return {
            "messages": [json.dumps(ex["messages"], ensure_ascii=False) for ex in examples],
            "source": [ex.get("_source", "unknown") for ex in examples],
        }

    train_dict = examples_to_dict(train_examples)
    val_dict = examples_to_dict(val_examples)

    train_ds = Dataset.from_dict(train_dict)
    val_ds = Dataset.from_dict(val_dict)

    dataset_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
    })

    # Save to disk
    dataset_dict.save_to_disk(str(FINAL_DIR))
    console.print(f"  Saved to: {FINAL_DIR}")

    # Also save as JSONL for easy inspection
    for split_name, examples in [("train", train_examples), ("validation", val_examples)]:
        jsonl_path = FINAL_DIR / f"{split_name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for ex in examples:
                # Remove internal metadata before saving
                clean_ex = {"messages": ex["messages"]}
                f.write(json.dumps(clean_ex, ensure_ascii=False) + "\n")
        console.print(f"  JSONL backup: {jsonl_path}")

    return dataset_dict


def print_category_distribution(examples: list[dict], title: str = "Dataset"):
    """Print distribution of examples by source and category."""
    source_counts = {}
    for ex in examples:
        source = ex.get("_source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    table = Table(title=f"{title} — Source Distribution", show_lines=True)
    table.add_column("Source", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    total = len(examples)
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        table.add_row(source, f"{count:,}", f"{pct:.1f}%")

    table.add_row("[bold]Total[/bold]", f"[bold]{total:,}[/bold]", "[bold]100.0%[/bold]")
    console.print(table)


def main():
    console.print()
    console.print("[bold cyan]The Substitution v1.0.0 — Training Data Preparation[/bold cyan]")
    console.print("[dim]Organization: theblackflagbmjc[/dim]")
    console.print()

    # Step 1: Merge
    all_examples, stats = merge_datasets()
    console.print(f"[bold]Total merged examples: {len(all_examples):,}[/bold]")
    console.print()

    # Step 2: Split
    console.print("[bold]Creating train/validation splits...[/bold]")
    train_examples, val_examples = create_splits(all_examples)
    console.print(f"  Train: {len(train_examples):,} ({TRAIN_RATIO * 100:.0f}%)")
    console.print(f"  Validation: {len(val_examples):,} ({VALIDATION_RATIO * 100:.0f}%)")
    console.print()

    # Step 3: Distribution analysis
    print_category_distribution(train_examples, "Training Set")
    console.print()

    # Step 4: Save
    dataset_dict = save_as_hf_dataset(train_examples, val_examples)
    console.print()

    # Step 5: Manifest
    manifest = {
        "project": "The Substitution v1.0.0",
        "stage": "preparation",
        "total_examples": len(all_examples),
        "train_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "split_ratio": f"{TRAIN_RATIO}/{VALIDATION_RATIO}",
        "random_seed": RANDOM_SEED,
        "source_stats": stats,
        "output_dir": str(FINAL_DIR),
    }

    manifest_path = FINAL_DIR / "preparation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    console.print(f"Manifest saved to: {manifest_path}")
    console.print()
    console.print("[green bold]Training data preparation complete.[/green bold]")
    console.print()
    console.print("[dim]Next step: python tokenizer/build_tokenizer.py[/dim]")


if __name__ == "__main__":
    main()
