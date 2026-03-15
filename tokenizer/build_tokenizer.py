#!/usr/bin/env python3
"""
The Substitution v1.0.0 — Tokenizer Preparation
Brandon Michael Jeanpierre Corporation d/b/a The Black Flag
Organization: theblackflagbmjc

Loads the Qwen2.5-7B-Instruct tokenizer, applies ChatML formatting to the
prepared training data, and saves tokenized datasets ready for training.

Qwen2.5 uses ChatML format natively:
    <|im_start|>system
    You are The Substitution...
    <|im_end|>
    <|im_start|>user
    ...
    <|im_end|>
    <|im_start|>assistant
    ...
    <|im_end|>

Usage:
    python tokenizer/build_tokenizer.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import DatasetDict, load_from_disk
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "final"
TOKENIZED_DIR = PROJECT_ROOT / "data" / "tokenized"
LOG_DIR = PROJECT_ROOT / "logs"

TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "build_tokenizer.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("tokenizer_builder")
console = Console()

# --- Load config from environment or defaults ---
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
# --- Load max_seq_length from training config ---
_train_config_path = PROJECT_ROOT / "training" / "training_config.yaml"
if _train_config_path.exists():
    with open(_train_config_path, "r") as _f:
        _train_config = yaml.safe_load(_f)
    MAX_SEQ_LENGTH = _train_config.get("data", {}).get("max_seq_length", 2048)
else:
    MAX_SEQ_LENGTH = 2048
NUM_PROC = os.cpu_count() or 4


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Load the base model tokenizer.

    Qwen2.5 uses a BPE tokenizer with ChatML special tokens:
        <|im_start|>  — marks the beginning of a turn
        <|im_end|>    — marks the end of a turn
        <|endoftext|> — EOS token

    The tokenizer's apply_chat_template() method handles formatting
    automatically when given a messages list.
    """
    console.print(f"[bold]Loading tokenizer: {model_name}[/bold]")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )

    # Ensure pad token is set (required for batch training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Set pad_token to eos_token: '{tokenizer.eos_token}'")

    # Log tokenizer info
    console.print(f"  Vocab size: {tokenizer.vocab_size:,}")
    console.print(f"  Model max length: {tokenizer.model_max_length:,}")
    console.print(f"  EOS token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
    console.print(f"  PAD token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
    console.print(f"  Chat template available: {tokenizer.chat_template is not None}")

    return tokenizer


def tokenize_example(example: dict, tokenizer: AutoTokenizer) -> dict:
    """
    Tokenize a single example using the Qwen2.5 ChatML template.

    Input format (from prepare_training_data.py):
        {"messages": '[{"role": "system", "content": "..."}, ...]', "source": "openhermes"}

    Output format:
        {"input_ids": [...], "attention_mask": [...], "labels": [...]}

    For instruction tuning, we mask the system and user tokens in the labels
    so the model only learns to predict assistant responses.
    """
    # Parse messages from JSON string
    messages = json.loads(example["messages"])

    # Use the tokenizer's built-in chat template to format the conversation
    # This produces the correct ChatML format for Qwen2.5:
    #   <|im_start|>system\n...\n<|im_end|>\n<|im_start|>user\n...\n<|im_end|>\n...
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Tokenize the full conversation
    tokenized = tokenizer(
        formatted_text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,  # Dynamic padding handled by data collator during training
        return_tensors=None,
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # --- Build labels with masking ---
    # We want the model to learn to predict ONLY the assistant responses.
    # Everything else (system prompt, user messages, special tokens) gets
    # masked with -100 (ignored by CrossEntropyLoss).

    labels = build_labels_with_masking(input_ids, messages, tokenizer)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_labels_with_masking(
    input_ids: list[int],
    messages: list[dict],
    tokenizer: AutoTokenizer,
) -> list[int]:
    """
    Build training labels that mask non-assistant tokens.

    Strategy:
    - Tokenize each message segment individually to find boundaries
    - Set labels to -100 for system and user segments
    - Keep labels = input_ids for assistant segments

    This ensures the model only learns to generate assistant responses,
    not to parrot back system prompts or user queries.
    """
    IGNORE_INDEX = -100
    labels = [IGNORE_INDEX] * len(input_ids)

    # We need to find where each assistant response starts and ends in the
    # full tokenized sequence. The ChatML format uses <|im_start|>assistant\n
    # as the delimiter before each assistant response and <|im_end|> after.

    # Get special token IDs
    im_start_token = "<|im_start|>"
    im_end_token = "<|im_end|>"

    im_start_id = tokenizer.convert_tokens_to_ids(im_start_token)
    im_end_id = tokenizer.convert_tokens_to_ids(im_end_token)

    # Find assistant response regions
    # Pattern: <|im_start|> + "assistant\n" tokens + content + <|im_end|>
    assistant_marker = tokenizer.encode("assistant\n", add_special_tokens=False)

    i = 0
    while i < len(input_ids):
        # Look for <|im_start|>
        if input_ids[i] == im_start_id:
            # Check if followed by "assistant\n" tokens
            marker_end = i + 1 + len(assistant_marker)
            if marker_end <= len(input_ids):
                candidate = input_ids[i + 1 : marker_end]
                if candidate == assistant_marker:
                    # Found assistant turn start
                    # The actual content to predict starts after "assistant\n"
                    content_start = marker_end

                    # Find the matching <|im_end|>
                    content_end = content_start
                    while content_end < len(input_ids) and input_ids[content_end] != im_end_id:
                        content_end += 1

                    # Include the <|im_end|> token in what the model should predict
                    if content_end < len(input_ids):
                        content_end += 1  # Include im_end

                    # Unmask the assistant content (set labels = input_ids)
                    for j in range(content_start, min(content_end, len(input_ids))):
                        labels[j] = input_ids[j]

                    i = content_end
                    continue
        i += 1

    return labels


def verify_tokenization(tokenizer: AutoTokenizer, dataset, num_samples: int = 3):
    """
    Verify tokenization by decoding a few examples and displaying them.
    """
    console.print()
    console.print("[bold]Verification — Sample tokenized examples:[/bold]")

    for idx in range(min(num_samples, len(dataset))):
        example = dataset[idx]
        input_ids = example["input_ids"]
        labels = example["labels"]

        # Decode full input
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)

        # Decode only the labeled (non-masked) portions
        labeled_ids = [tid for tid, lid in zip(input_ids, labels) if lid != -100]
        labeled_text = tokenizer.decode(labeled_ids, skip_special_tokens=False) if labeled_ids else "(none)"

        total_tokens = len(input_ids)
        labeled_tokens = sum(1 for l in labels if l != -100)
        masked_tokens = total_tokens - labeled_tokens

        console.print(f"\n  [cyan]Example {idx + 1}:[/cyan]")
        console.print(f"    Total tokens: {total_tokens}")
        console.print(f"    Labeled (assistant): {labeled_tokens} ({labeled_tokens / total_tokens * 100:.1f}%)")
        console.print(f"    Masked (system/user): {masked_tokens} ({masked_tokens / total_tokens * 100:.1f}%)")
        console.print(f"    Full text (first 200 chars): {full_text[:200]}...")
        console.print(f"    Labeled text (first 200 chars): {labeled_text[:200]}...")


def compute_statistics(dataset) -> dict:
    """Compute token length statistics across the dataset."""
    lengths = [len(ex["input_ids"]) for ex in dataset]
    labeled_counts = [sum(1 for l in ex["labels"] if l != -100) for ex in dataset]

    import statistics
    return {
        "total_examples": len(lengths),
        "avg_length": round(statistics.mean(lengths), 1),
        "median_length": round(statistics.median(lengths), 1),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "stdev_length": round(statistics.stdev(lengths), 1) if len(lengths) > 1 else 0,
        "avg_labeled_tokens": round(statistics.mean(labeled_counts), 1),
        "total_tokens": sum(lengths),
        "total_labeled_tokens": sum(labeled_counts),
        "pct_at_max_length": round(sum(1 for l in lengths if l >= MAX_SEQ_LENGTH) / len(lengths) * 100, 1),
    }


def main():
    console.print()
    console.print("[bold cyan]The Substitution v1.0.0 — Tokenizer Preparation[/bold cyan]")
    console.print("[dim]Organization: theblackflagbmjc[/dim]")
    console.print(f"[dim]Base model: {BASE_MODEL}[/dim]")
    console.print(f"[dim]Max sequence length: {MAX_SEQ_LENGTH}[/dim]")
    console.print()

    # Step 1: Load tokenizer
    tokenizer = load_tokenizer(BASE_MODEL)
    console.print()

    # Step 2: Load prepared dataset
    console.print("[bold]Loading prepared dataset...[/bold]")
    if not PROCESSED_DIR.exists():
        console.print("[red]Prepared dataset not found. Run prepare_training_data.py first.[/red]")
        sys.exit(1)

    dataset_dict = load_from_disk(str(PROCESSED_DIR))
    console.print(f"  Train: {len(dataset_dict['train']):,} examples")
    console.print(f"  Validation: {len(dataset_dict['validation']):,} examples")
    console.print()

    # Step 3: Tokenize
    console.print("[bold]Tokenizing dataset...[/bold]")
    console.print(f"  Using {NUM_PROC} workers")
    console.print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
    console.print()

    tokenized_dict = {}
    for split_name in ["train", "validation"]:
        console.print(f"  Processing {split_name}...")
        tokenized = dataset_dict[split_name].map(
            lambda ex: tokenize_example(ex, tokenizer),
            remove_columns=dataset_dict[split_name].column_names,
            num_proc=NUM_PROC,
            desc=f"Tokenizing {split_name}",
        )
        tokenized_dict[split_name] = tokenized
        console.print(f"    {split_name}: {len(tokenized):,} examples tokenized")

    tokenized_dataset = DatasetDict(tokenized_dict)

    # Step 4: Verify
    verify_tokenization(tokenizer, tokenized_dataset["train"])
    console.print()

    # Step 5: Statistics
    console.print("[bold]Computing statistics...[/bold]")

    for split_name in ["train", "validation"]:
        stats = compute_statistics(tokenized_dataset[split_name])

        table = Table(title=f"{split_name.capitalize()} Set Statistics", show_lines=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        for key, value in stats.items():
            table.add_row(key.replace("_", " ").title(), f"{value:,}" if isinstance(value, int) else str(value))

        console.print(table)
        console.print()

    # Step 6: Save
    console.print("[bold]Saving tokenized dataset...[/bold]")
    tokenized_dataset.save_to_disk(str(TOKENIZED_DIR))
    console.print(f"  Saved to: {TOKENIZED_DIR}")

    # Save tokenizer alongside data (needed for training script)
    tokenizer_save_path = PROJECT_ROOT / "output" / "tokenizer"
    tokenizer_save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(tokenizer_save_path))
    console.print(f"  Tokenizer saved to: {tokenizer_save_path}")

    # Save tokenization config
    config = {
        "base_model": BASE_MODEL,
        "max_seq_length": MAX_SEQ_LENGTH,
        "vocab_size": tokenizer.vocab_size,
        "pad_token": tokenizer.pad_token,
        "eos_token": tokenizer.eos_token,
        "train_examples": len(tokenized_dataset["train"]),
        "val_examples": len(tokenized_dataset["validation"]),
    }
    config_path = TOKENIZED_DIR / "tokenization_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    console.print()
    console.print("[green bold]Tokenization complete.[/green bold]")
    console.print()
    console.print("[dim]Next step: python training/train_model.py[/dim]")


if __name__ == "__main__":
    main()
