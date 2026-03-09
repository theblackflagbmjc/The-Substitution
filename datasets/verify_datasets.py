#!/usr/bin/env python3
"""
The Substitution v1.0.0 — Dataset Verification
Brandon Michael Jeanpierre Corporation d/b/a The Black Flag
Organization: theblackflagbmjc

Verifies integrity and completeness of downloaded datasets by checking:
  1. All expected files exist
  2. File sizes are non-zero
  3. SHA256 checksums match the download manifest
  4. JSONL files are parseable
  5. Expected fields are present in each dataset

Usage:
    python datasets/verify_datasets.py
"""

import hashlib
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = DATA_DIR / "download_manifest.json"
LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "verify_datasets.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("dataset_verifier")

console = Console()

# --- Expected Fields per Dataset ---
# These are the fields we expect to find in each dataset's JSONL rows.
# Used to validate that the download produced usable data.
EXPECTED_FIELDS = {
    "openhermes": ["conversations"],
    "ultrachat": ["data", "id"],
    "openorca": ["system_prompt", "question", "response"],
    "codesearchnet": ["func_code_string", "func_documentation_string", "language"],
    "gsm8k": ["question", "answer"],
    "humaneval": ["task_id", "prompt", "canonical_solution", "test"],
    "mmlu": ["question", "choices", "answer"],
    "gsm8k_eval": ["question", "answer"],
}


def compute_checksum(filepath: Path) -> str:
    """Compute SHA256 checksum for a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_jsonl_parseable(filepath: Path, max_lines: int = 100) -> tuple[bool, int, str | None]:
    """
    Verify that a JSONL file is parseable.

    Returns:
        (is_valid, lines_checked, error_message)
    """
    try:
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                json.loads(line.strip())
                count += 1
        return True, count, None
    except json.JSONDecodeError as e:
        return False, i, f"JSON parse error on line {i + 1}: {e}"
    except Exception as e:
        return False, 0, str(e)


def verify_fields(filepath: Path, expected_fields: list[str], max_lines: int = 10) -> tuple[bool, str | None]:
    """
    Verify that expected fields exist in the JSONL data.

    Returns:
        (is_valid, error_message)
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                row = json.loads(line.strip())
                for field in expected_fields:
                    if field not in row:
                        return False, f"Missing field '{field}' on line {i + 1}. Available: {list(row.keys())}"
        return True, None
    except Exception as e:
        return False, str(e)


def count_lines(filepath: Path) -> int:
    """Count total lines in a file."""
    count = 0
    with open(filepath, "rb") as f:
        for _ in f:
            count += 1
    return count


def run_verification() -> list[dict]:
    """Run full verification pipeline."""
    results = []

    # Load manifest
    if not MANIFEST_PATH.exists():
        console.print("[red]Download manifest not found. Run download_datasets.py first.[/red]")
        sys.exit(1)

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    console.print()
    console.print("[bold cyan]The Substitution v1.0.0 — Dataset Verification[/bold cyan]")
    console.print("[dim]Organization: theblackflagbmjc[/dim]")
    console.print()

    for entry in manifest["datasets"]:
        name = entry["name"]
        console.print(f"[bold]Verifying: {name}[/bold]")

        result = {
            "name": name,
            "checks": [],
            "passed": True,
        }

        filepath = Path(entry["path"]) if entry.get("path") else None

        # Check 1: File exists
        if filepath and filepath.exists():
            result["checks"].append(("File exists", True, str(filepath)))
        else:
            result["checks"].append(("File exists", False, f"Not found: {filepath}"))
            result["passed"] = False
            results.append(result)
            console.print(f"  [red]✖ File not found[/red]")
            continue

        # Check 2: File size > 0
        size = filepath.stat().st_size
        if size > 0:
            result["checks"].append(("Non-zero size", True, f"{size:,} bytes ({size / 1e6:.1f} MB)"))
        else:
            result["checks"].append(("Non-zero size", False, "File is empty"))
            result["passed"] = False

        # Check 3: Checksum matches manifest
        if entry.get("checksum"):
            actual_checksum = compute_checksum(filepath)
            if actual_checksum == entry["checksum"]:
                result["checks"].append(("Checksum match", True, actual_checksum[:16] + "..."))
            else:
                result["checks"].append((
                    "Checksum match", False,
                    f"Expected: {entry['checksum'][:16]}... Got: {actual_checksum[:16]}..."
                ))
                result["passed"] = False
        else:
            result["checks"].append(("Checksum match", None, "No checksum in manifest"))

        # Check 4: JSONL parseable
        is_valid, lines_checked, error = verify_jsonl_parseable(filepath)
        if is_valid:
            result["checks"].append(("JSONL parseable", True, f"Checked {lines_checked} lines"))
        else:
            result["checks"].append(("JSONL parseable", False, error))
            result["passed"] = False

        # Check 5: Expected fields present
        if name in EXPECTED_FIELDS:
            fields_ok, field_error = verify_fields(filepath, EXPECTED_FIELDS[name])
            if fields_ok:
                result["checks"].append(("Expected fields", True, str(EXPECTED_FIELDS[name])))
            else:
                result["checks"].append(("Expected fields", False, field_error))
                result["passed"] = False
        else:
            result["checks"].append(("Expected fields", None, "No field spec defined"))

        # Check 6: Line count matches expected
        total_lines = count_lines(filepath)
        expected = entry.get("num_examples", 0)
        if expected > 0 and total_lines == expected:
            result["checks"].append(("Line count", True, f"{total_lines:,} lines (matches manifest)"))
        elif expected > 0:
            result["checks"].append(("Line count", False, f"Expected {expected:,}, got {total_lines:,}"))
            # This is a warning, not a failure — streaming can produce slightly different counts
        else:
            result["checks"].append(("Line count", True, f"{total_lines:,} lines"))

        # Print check results
        for check_name, passed, detail in result["checks"]:
            if passed is True:
                console.print(f"  [green]✓[/green] {check_name}: {detail}")
            elif passed is False:
                console.print(f"  [red]✖[/red] {check_name}: {detail}")
                result["passed"] = False
            else:
                console.print(f"  [yellow]⚠[/yellow] {check_name}: {detail}")

        results.append(result)
        console.print()

    return results


def print_summary(results: list[dict]):
    """Print verification summary table."""
    table = Table(title="Verification Summary", show_lines=True)
    table.add_column("Dataset", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Checks Passed", justify="right")

    for r in results:
        total_checks = len(r["checks"])
        passed_checks = sum(1 for _, p, _ in r["checks"] if p is True)
        status = "[green]PASS[/green]" if r["passed"] else "[red]FAIL[/red]"
        table.add_row(r["name"], status, f"{passed_checks}/{total_checks}")

    console.print(table)


def main():
    logger.info("=" * 70)
    logger.info("The Substitution v1.0.0 — Dataset Verification")
    logger.info("=" * 70)

    results = run_verification()
    print_summary(results)

    total_pass = sum(1 for r in results if r["passed"])
    total = len(results)

    console.print()
    if total_pass == total:
        console.print(f"[green bold]All {total} datasets verified successfully.[/green bold]")
    else:
        failed = total - total_pass
        console.print(f"[red bold]{failed}/{total} datasets failed verification.[/red bold]")
        console.print("[yellow]Re-download failed datasets or check the error details above.[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    main()
