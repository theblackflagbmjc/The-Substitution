#!/usr/bin/env python3
"""
The Substitution v1.0.0 — Hugging Face Deployment
Brandon Michael Jeanpierre Corporation d/b/a The Black Flag
Organization: theblackflagbmjc

Pushes the trained model (merged or adapter) to the Hugging Face Hub,
creates the repository if necessary, and uploads the model card.

Usage:
    # Push merged model
    python deployment/huggingface_push.py

    # Push LoRA adapter only (smaller upload)
    python deployment/huggingface_push.py --adapter_only

    # Push to a specific repo
    python deployment/huggingface_push.py --repo_id theblackflagbmjc/the-substitution

    # Dry run (validate without pushing)
    python deployment/huggingface_push.py --dry_run
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, create_repo, upload_folder, login
from rich.console import Console

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "huggingface_push.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("hf_push")
console = Console()

DEFAULT_REPO_ID = "theblackflagbmjc/the-substitution"


def authenticate():
    """Authenticate with Hugging Face Hub."""
    console.print("[bold]Authenticating with Hugging Face...[/bold]")

    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
        console.print("  Authenticated via HF_TOKEN environment variable")
    else:
        # Check if already logged in
        api = HfApi()
        try:
            info = api.whoami()
            console.print(f"  Already authenticated as: {info.get('name', 'unknown')}")
        except Exception:
            console.print("  [yellow]Not authenticated. Running huggingface-cli login...[/yellow]")
            login()

    return HfApi()


def create_or_verify_repo(api: HfApi, repo_id: str, private: bool = True) -> str:
    """Create the repository if it doesn't exist, or verify access."""
    console.print(f"[bold]Repository: {repo_id}[/bold]")

    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="model")
        console.print(f"  Repository exists: {repo_info.id}")
        return repo_id
    except Exception:
        console.print(f"  Creating repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
        console.print(f"  Repository created (private={private})")
        return repo_id


def push_model(
    api: HfApi,
    repo_id: str,
    model_path: str,
    commit_message: str = "Upload The Substitution v1.0.0",
):
    """Upload model files to Hugging Face."""
    model_dir = Path(model_path)
    if not model_dir.exists():
        console.print(f"[red]Model directory not found: {model_dir}[/red]")
        sys.exit(1)

    console.print(f"[bold]Uploading model from: {model_dir}[/bold]")

    # List files to upload
    files = list(model_dir.rglob("*"))
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    file_count = sum(1 for f in files if f.is_file())

    console.print(f"  Files: {file_count}")
    console.print(f"  Total size: {total_size / 1e9:.2f} GB")

    # Upload
    upload_folder(
        repo_id=repo_id,
        folder_path=str(model_dir),
        repo_type="model",
        commit_message=commit_message,
        ignore_patterns=["*.pyc", "__pycache__", "*.log", ".DS_Store"],
    )

    console.print(f"  [green]Upload complete![/green]")
    console.print(f"  URL: https://huggingface.co/{repo_id}")


def push_model_card(api: HfApi, repo_id: str):
    """Upload the model card (README.md) to the repository."""
    model_card_path = PROJECT_ROOT / "model_card" / "README.md"

    if not model_card_path.exists():
        console.print("[yellow]Model card not found. Skipping.[/yellow]")
        return

    console.print("[bold]Uploading model card...[/bold]")

    api.upload_file(
        path_or_fileobj=str(model_card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update model card",
    )

    console.print("  Model card uploaded")


def parse_args():
    parser = argparse.ArgumentParser(description="Push model to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID, help="HF repo ID")
    parser.add_argument("--adapter_only", action="store_true", help="Push LoRA adapter only")
    parser.add_argument("--model_path", type=str, default=None, help="Custom model path")
    parser.add_argument("--private", action="store_true", default=True, help="Make repo private")
    parser.add_argument("--public", action="store_true", help="Make repo public")
    parser.add_argument("--dry_run", action="store_true", help="Validate without pushing")
    return parser.parse_args()


def main():
    args = parse_args()

    console.print()
    console.print("[bold cyan]The Substitution v1.0.0 — Hugging Face Deployment[/bold cyan]")
    console.print("[dim]Organization: theblackflagbmjc[/dim]")
    console.print()

    # Load training config for paths
    config_path = PROJECT_ROOT / "training" / "training_config.yaml"
    with open(config_path, "r") as f:
        train_config = yaml.safe_load(f)

    output_dir = train_config["training"]["output_dir"]

    # Determine model path
    if args.model_path:
        model_path = args.model_path
    elif args.adapter_only:
        model_path = str(PROJECT_ROOT / output_dir / "final_adapter")
    else:
        merged_path = PROJECT_ROOT / output_dir / "merged_model"
        adapter_path = PROJECT_ROOT / output_dir / "final_adapter"
        if merged_path.exists():
            model_path = str(merged_path)
        elif adapter_path.exists():
            model_path = str(adapter_path)
            console.print("[yellow]Merged model not found. Pushing adapter instead.[/yellow]")
        else:
            console.print("[red]No model found. Run training first.[/red]")
            sys.exit(1)

    console.print(f"  Model path: {model_path}")
    console.print(f"  Target repo: {args.repo_id}")
    console.print(f"  Private: {not args.public}")
    console.print()

    if args.dry_run:
        console.print("[yellow]Dry run mode — no files will be uploaded.[/yellow]")

        model_dir = Path(model_path)
        if model_dir.exists():
            files = list(model_dir.rglob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            file_count = sum(1 for f in files if f.is_file())
            console.print(f"  Would upload {file_count} files ({total_size / 1e9:.2f} GB)")
        else:
            console.print(f"  [red]Path does not exist: {model_path}[/red]")

        console.print("[dim]Dry run complete. No changes made.[/dim]")
        return

    # Authenticate
    api = authenticate()
    console.print()

    # Create/verify repo
    is_private = not args.public
    create_or_verify_repo(api, args.repo_id, private=is_private)
    console.print()

    # Push model
    push_model(api, args.repo_id, model_path)
    console.print()

    # Push model card
    push_model_card(api, args.repo_id)
    console.print()

    console.print("[green bold]Deployment complete.[/green bold]")
    console.print(f"  Repository: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
