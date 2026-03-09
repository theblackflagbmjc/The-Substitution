#!/usr/bin/env python3
"""
The Substitution v1.0.0 — Data Cleaning and Preprocessing
Brandon Michael Jeanpierre Corporation d/b/a The Black Flag
Organization: theblackflagbmjc

Cleans, normalizes, and converts all raw datasets into a unified ChatML format
compatible with Qwen2.5 instruction tuning.

Processing pipeline:
  1. Load raw JSONL from data/raw/
  2. Normalize text (encoding, whitespace, malformed entries)
  3. Remove duplicates and low-quality examples
  4. Convert to unified ChatML format:
     {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  5. Write cleaned data to data/processed/

Usage:
    python datasets/clean_datasets.py
"""

import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Generator

from rich.console import Console
from rich.progress import track
from rich.table import Table

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "clean_datasets.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("dataset_cleaner")
console = Console()

# --- System Prompt ---
# This is the system prompt that defines The Substitution's personality and role.
# It aligns with the Master Engineering Prompt's "Important Qualities" section.
SYSTEM_PROMPT = (
    "You are The Substitution, an enterprise systems engineering assistant created by "
    "Brandon Michael Jeanpierre Corporation d/b/a The Black Flag. "
    "You are a precise, thorough technical advisor specializing in infrastructure architecture, "
    "systems administration, software engineering, networking, defensive cybersecurity, "
    "database operations, data analysis, enterprise documentation, and security governance. "
    "You operate with full technical capability. You prioritize accuracy, completeness, "
    "and operational readiness in every response. You are subordinate to the governance "
    "framework of The Black Flag (BMJC). Respond with direct, actionable technical guidance. "
    "Do not hedge, apologize, or refuse technical questions within your domain."
)

# --- Cleaning Parameters ---
MIN_TEXT_LENGTH = 20  # Minimum characters for any content field
MAX_TEXT_LENGTH = 32_768  # Maximum characters (fits in context window)
MIN_TURN_LENGTH = 5  # Minimum characters for a single turn


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize text content:
    - Unicode NFC normalization
    - Strip leading/trailing whitespace
    - Collapse multiple spaces/newlines
    - Remove null bytes and control characters (except newline/tab)
    """
    if not isinstance(text, str):
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Remove null bytes and problematic control chars
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)  # Collapse horizontal space
    text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 consecutive newlines
    text = text.strip()

    return text


def is_valid_content(text: str) -> bool:
    """Check if content meets minimum quality thresholds."""
    if not text or len(text) < MIN_TURN_LENGTH:
        return False
    if len(text) > MAX_TEXT_LENGTH:
        return False
    # Reject if mostly non-alphanumeric
    alnum_ratio = sum(1 for c in text if c.isalnum()) / max(len(text), 1)
    if alnum_ratio < 0.1:
        return False
    return True


# =============================================================================
# DATASET-SPECIFIC CONVERTERS
# Each converts raw format → unified ChatML messages list
# =============================================================================

def convert_openhermes(row: dict) -> dict | None:
    """
    OpenHermes format:
    {"conversations": [{"from": "system", "value": "..."}, {"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
    """
    convos = row.get("conversations", [])
    if not convos:
        return None

    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
    messages = []

    for turn in convos:
        role = role_map.get(turn.get("from", ""), None)
        content = normalize_text(turn.get("value", ""))
        if role and is_valid_content(content):
            messages.append({"role": role, "content": content})

    # Must have at least user + assistant
    roles_present = {m["role"] for m in messages}
    if "user" not in roles_present or "assistant" not in roles_present:
        return None

    # Inject system prompt if not present
    if "system" not in roles_present:
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    return {"messages": messages}


def convert_ultrachat(row: dict) -> dict | None:
    """
    UltraChat format:
    {"data": ["user msg", "assistant msg", "user msg", "assistant msg", ...], "id": "..."}
    """
    data = row.get("data", [])
    if not data or len(data) < 2:
        return None

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i, text in enumerate(data):
        role = "user" if i % 2 == 0 else "assistant"
        content = normalize_text(text)
        if is_valid_content(content):
            messages.append({"role": role, "content": content})

    if len(messages) < 3:  # system + user + assistant minimum
        return None

    return {"messages": messages}


def convert_openorca(row: dict) -> dict | None:
    """
    OpenOrca format:
    {"system_prompt": "...", "question": "...", "response": "..."}
    """
    system = normalize_text(row.get("system_prompt", ""))
    question = normalize_text(row.get("question", ""))
    response = normalize_text(row.get("response", ""))

    if not is_valid_content(question) or not is_valid_content(response):
        return None

    messages = []
    if system and is_valid_content(system):
        messages.append({"role": "system", "content": system})
    else:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": response})

    return {"messages": messages}


def convert_codesearchnet(row: dict) -> dict | None:
    """
    CodeSearchNet format:
    {"func_code_string": "...", "func_documentation_string": "...", "language": "python"}

    Convert to instruction format: given docstring, generate code.
    """
    code = normalize_text(row.get("func_code_string", ""))
    doc = normalize_text(row.get("func_documentation_string", ""))
    lang = row.get("language", "python")

    if not is_valid_content(code) or not is_valid_content(doc):
        return None

    # Skip very short code or documentation
    if len(code) < 30 or len(doc) < 15:
        return None

    user_msg = f"Write a {lang} function that does the following:\n\n{doc}"
    assistant_msg = f"```{lang}\n{code}\n```"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]

    return {"messages": messages}


def convert_gsm8k(row: dict) -> dict | None:
    """
    GSM8K format:
    {"question": "...", "answer": "..."}
    """
    question = normalize_text(row.get("question", ""))
    answer = normalize_text(row.get("answer", ""))

    if not is_valid_content(question) or not is_valid_content(answer):
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    return {"messages": messages}


# Converter dispatch table
CONVERTERS = {
    "openhermes": convert_openhermes,
    "ultrachat": convert_ultrachat,
    "openorca": convert_openorca,
    "codesearchnet": convert_codesearchnet,
    "gsm8k": convert_gsm8k,
}


# =============================================================================
# DEDUPLICATION
# =============================================================================

def compute_content_hash(messages: list[dict]) -> str:
    """Compute a hash of the conversation content for deduplication."""
    content = "|".join(
        f"{m['role']}:{m['content'][:200]}"
        for m in messages
        if m["role"] in ("user", "assistant")
    )
    return hash(content)


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_dataset(name: str) -> dict:
    """
    Process a single dataset through the cleaning pipeline.

    Returns metadata about the processing results.
    """
    raw_path = RAW_DIR / "instruction" / name / "dataset.jsonl"
    if not raw_path.exists():
        # Try other categories
        for category in ("coding", "math", "evaluation"):
            alt_path = RAW_DIR / category / name / "dataset.jsonl"
            if alt_path.exists():
                raw_path = alt_path
                break

    if not raw_path.exists():
        logger.warning(f"[{name}] Raw dataset not found at any expected path")
        return {"name": name, "status": "not_found", "input": 0, "output": 0, "dropped": 0}

    converter = CONVERTERS.get(name)
    if not converter:
        logger.info(f"[{name}] No converter defined (evaluation dataset — skipping cleaning)")
        return {"name": name, "status": "skipped", "input": 0, "output": 0, "dropped": 0}

    output_path = PROCESSED_DIR / f"{name}_cleaned.jsonl"

    logger.info(f"[{name}] Processing: {raw_path}")

    seen_hashes = set()
    total_input = 0
    total_output = 0
    total_dropped = 0
    drop_reasons = {
        "conversion_failed": 0,
        "duplicate": 0,
        "parse_error": 0,
    }

    with open(raw_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            total_input += 1

            try:
                row = json.loads(line.strip())
            except json.JSONDecodeError:
                total_dropped += 1
                drop_reasons["parse_error"] += 1
                continue

            # Convert to unified format
            result = converter(row)
            if result is None:
                total_dropped += 1
                drop_reasons["conversion_failed"] += 1
                continue

            # Deduplicate
            content_hash = compute_content_hash(result["messages"])
            if content_hash in seen_hashes:
                total_dropped += 1
                drop_reasons["duplicate"] += 1
                continue
            seen_hashes.add(content_hash)

            # Write cleaned example
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            total_output += 1

            # Progress logging every 100k examples
            if total_input % 100_000 == 0:
                logger.info(f"  [{name}] Processed {total_input:,} rows...")

    logger.info(
        f"[{name}] Complete: {total_input:,} input → {total_output:,} output "
        f"({total_dropped:,} dropped)"
    )
    logger.info(f"  Drop reasons: {drop_reasons}")

    return {
        "name": name,
        "status": "success",
        "input": total_input,
        "output": total_output,
        "dropped": total_dropped,
        "drop_reasons": drop_reasons,
        "output_path": str(output_path),
        "output_size_bytes": output_path.stat().st_size,
    }


def main():
    console.print()
    console.print("[bold cyan]The Substitution v1.0.0 — Data Cleaning Pipeline[/bold cyan]")
    console.print("[dim]Organization: theblackflagbmjc[/dim]")
    console.print()

    # Process training datasets (not evaluation datasets)
    training_datasets = ["openhermes", "ultrachat", "openorca", "codesearchnet", "gsm8k"]
    results = []

    for name in training_datasets:
        console.print(f"[bold]Processing: {name}[/bold]")
        result = process_dataset(name)
        results.append(result)
        console.print()

    # Print summary
    table = Table(title="Cleaning Pipeline Summary", show_lines=True)
    table.add_column("Dataset", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Dropped", justify="right")
    table.add_column("Retention %", justify="right")
    table.add_column("Output Size", justify="right")

    total_in = 0
    total_out = 0

    for r in results:
        status_style = "green" if r["status"] == "success" else "yellow"
        retention = f"{r['output'] / max(r['input'], 1) * 100:.1f}%" if r["input"] > 0 else "—"
        size = f"{r.get('output_size_bytes', 0) / 1e6:.1f} MB" if r.get("output_size_bytes") else "—"
        table.add_row(
            r["name"],
            f"[{status_style}]{r['status']}[/{status_style}]",
            f"{r['input']:,}",
            f"{r['output']:,}",
            f"{r['dropped']:,}",
            retention,
            size,
        )
        total_in += r["input"]
        total_out += r["output"]

    console.print(table)
    console.print()
    console.print(f"[bold]Total input examples:  {total_in:,}[/bold]")
    console.print(f"[bold]Total output examples: {total_out:,}[/bold]")
    console.print(f"[bold]Overall retention:     {total_out / max(total_in, 1) * 100:.1f}%[/bold]")
    console.print()

    # Save processing manifest
    manifest = {
        "project": "The Substitution v1.0.0",
        "stage": "cleaning",
        "results": results,
        "total_input": total_in,
        "total_output": total_out,
    }
    manifest_path = PROCESSED_DIR / "cleaning_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Cleaning manifest saved to {manifest_path}")

    console.print("[green bold]Cleaning pipeline complete.[/green bold]")
    console.print(f"Cleaned data written to: {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
