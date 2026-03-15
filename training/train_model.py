#!/usr/bin/env python3
"""
The Substitution v1.0.0 — Model Training Script
Brandon Michael Jeanpierre Corporation d/b/a The Black Flag
Organization: theblackflagbmjc

Fine-tunes Qwen2.5-7B-Instruct using QLoRA (4-bit quantization + LoRA adapters)
for enterprise systems engineering tasks.

This script:
  1. Loads training and LoRA configurations from YAML
  2. Loads the base model in 4-bit quantization
  3. Applies LoRA adapters to all linear layers
  4. Loads the tokenized dataset
  5. Runs training with gradient checkpointing and accumulation
  6. Saves checkpoints and the final LoRA adapter
  7. Merges the adapter with the base model (optional)

Usage:
    python training/train_model.py

    # With Accelerate (recommended for multi-GPU):
    accelerate launch training/train_model.py

    # Override config values:
    python training/train_model.py --learning_rate 1e-4 --num_train_epochs 5

Hardware targets:
    - Primary: MacBook Pro M1Pro 32GB (MPS backend, single device)
    - Secondary: Linux with NVIDIA GPU (CUDA, single or multi-GPU)
    - Fallback: CPU (functional but very slow)
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import yaml
from datasets import load_from_disk
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "train_model.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("trainer")
console = Console()


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_yaml_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def detect_compute_backend() -> str:
    """Detect the best available compute backend."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info(f"CUDA available: {gpu_name} ({vram:.1f} GB VRAM)")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("MPS (Apple Silicon Metal) available")
        return "mps"
    else:
        logger.warning("No GPU detected. Training will use CPU.")
        return "cpu"


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_base_model(
    model_name: str,
    quant_config: dict,
    compute_backend: str,
) -> AutoModelForCausalLM:
    """
    Load the base model with optional 4-bit quantization.

    For CUDA systems: uses bitsandbytes 4-bit quantization (QLoRA)
    For MPS/CPU: loads in bfloat16 without quantization (bitsandbytes
    doesn't support MPS). LoRA still provides parameter efficiency.
    """
    console.print(f"[bold]Loading base model: {model_name}[/bold]")

    load_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "trust_remote_code": True,
        "use_cache": False,  # Incompatible with gradient checkpointing
    }

    if compute_backend == "cuda" and quant_config.get("load_in_4bit", False):
        # CUDA: Full QLoRA with bitsandbytes 4-bit quantization
        console.print("  Quantization: 4-bit NF4 (QLoRA)")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, quant_config.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    elif compute_backend == "mps":
        # MPS: Load in bfloat16 (no bitsandbytes support on macOS)
        # Training still works with LoRA — just uses more memory than QLoRA
        console.print("  Precision: bfloat16 (MPS — no quantization)")
        load_kwargs["dtype"] = torch.bfloat16
        load_kwargs["device_map"] = {"": "mps"}
    else:
        # CPU fallback
        console.print("  Precision: float32 (CPU fallback)")
        load_kwargs["dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

    # Prepare for k-bit training if quantized
    if compute_backend == "cuda" and quant_config.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"  Total parameters: {total_params:,}")
    console.print(f"  Model dtype: {next(model.parameters()).dtype}")

    return model


def apply_lora(model: AutoModelForCausalLM, lora_config_dict: dict) -> AutoModelForCausalLM:
    """
    Apply LoRA adapters to the model.

    This wraps the base model with trainable low-rank adapter matrices
    while freezing the original weights.
    """
    console.print("[bold]Applying LoRA adapters...[/bold]")

    lora_params = lora_config_dict.get("lora", lora_config_dict)

    config = LoraConfig(
        r=lora_params.get("r", 64),
        lora_alpha=lora_params.get("lora_alpha", 128),
        lora_dropout=lora_params.get("lora_dropout", 0.05),
        target_modules=lora_params.get("target_modules", "all-linear"),
        task_type=TaskType.CAUSAL_LM,
        bias=lora_params.get("bias", "none"),
        modules_to_save=lora_params.get("modules_to_save", None),
    )

    model = get_peft_model(model, config)

    # Print trainable parameters
    model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct = trainable_params / total_params * 100
    console.print(f"  Trainable: {trainable_params:,} / {total_params:,} ({pct:.2f}%)")

    return model


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data(data_path: str) -> tuple:
    """Load the tokenized training and validation datasets."""
    console.print(f"[bold]Loading tokenized dataset: {data_path}[/bold]")

    dataset_dict = load_from_disk(data_path)

    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["validation"]

    console.print(f"  Train: {len(train_dataset):,} examples")
    console.print(f"  Validation: {len(eval_dataset):,} examples")

    return train_dataset, eval_dataset


# =============================================================================
# TRAINING
# =============================================================================

def build_training_args(train_config: dict, output_dir: str) -> TrainingArguments:
    """Build TrainingArguments from the YAML configuration."""
    tc = train_config.get("training", train_config)

    args = TrainingArguments(
        output_dir=tc.get("output_dir", output_dir),
        num_train_epochs=tc.get("num_train_epochs", 3),
        per_device_train_batch_size=tc.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=tc.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=tc.get("gradient_accumulation_steps", 32),
        learning_rate=tc.get("learning_rate", 2e-4),
        weight_decay=tc.get("weight_decay", 0.01),
        warmup_steps=tc.get("warmup_steps", 100),
        lr_scheduler_type=tc.get("lr_scheduler_type", "cosine"),
        max_grad_norm=tc.get("max_grad_norm", 1.0),
        bf16=tc.get("bf16", True),
        fp16=tc.get("fp16", False),
        tf32=tc.get("tf32", False),
        gradient_checkpointing=tc.get("gradient_checkpointing", True),
        optim=tc.get("optim", "adamw_torch"),
        logging_steps=tc.get("logging_steps", 10),
        logging_first_step=tc.get("logging_first_step", True),
        log_level=tc.get("log_level", "info"),
        eval_strategy=tc.get("eval_strategy", "steps"),
        eval_steps=tc.get("eval_steps", 500),
        save_strategy=tc.get("save_strategy", "steps"),
        save_steps=tc.get("save_steps", 500),
        save_total_limit=tc.get("save_total_limit", 3),
        load_best_model_at_end=tc.get("load_best_model_at_end", True),
        metric_for_best_model=tc.get("metric_for_best_model", "eval_loss"),
        greater_is_better=tc.get("greater_is_better", False),
        seed=tc.get("seed", 42),
        data_seed=tc.get("data_seed", 42),
        report_to=tc.get("report_to", "wandb"),
        dataloader_num_workers=tc.get("dataloader_num_workers", 4),
        dataloader_pin_memory=tc.get("dataloader_pin_memory", True),
        remove_unused_columns=tc.get("remove_unused_columns", True),
    )

    return args


def main():
    console.print()
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  The Substitution v1.0.0 — Model Training[/bold cyan]")
    console.print("[bold cyan]  Brandon Michael Jeanpierre Corporation[/bold cyan]")
    console.print("[bold cyan]  d/b/a The Black Flag[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print()

    # --- Load Configurations ---
    config_dir = PROJECT_ROOT / "training"
    train_config = load_yaml_config(str(config_dir / "training_config.yaml"))
    lora_config = load_yaml_config(str(config_dir / "lora_config.yaml"))

    model_name = train_config["model"]["base_model"]
    data_path = train_config["data"]["tokenized_dataset_path"]
    output_dir = train_config["training"]["output_dir"]

    console.print(f"  Base model:  {model_name}")
    console.print(f"  Data path:   {data_path}")
    console.print(f"  Output dir:  {output_dir}")
    console.print()

    # --- Detect Compute ---
    compute_backend = detect_compute_backend()
    console.print(f"  Compute backend: [bold]{compute_backend}[/bold]")
    console.print()

    # --- Load Tokenizer ---
    console.print("[bold]Loading tokenizer...[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    console.print(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size:,}")
    console.print()

    # --- Load Base Model ---
    model = load_base_model(
        model_name,
        train_config.get("quantization", {}),
        compute_backend,
    )
    console.print()

    # --- Apply LoRA ---
    model = apply_lora(model, lora_config)
    console.print()

    # --- Load Data ---
    train_dataset, eval_dataset = load_training_data(data_path)
    console.print()

    # --- Data Collator ---
    # Handles dynamic padding and label alignment for causal LM training
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=train_config["data"]["max_seq_length"],
        pad_to_multiple_of=8,  # Optimal for tensor core alignment
        label_pad_token_id=-100,  # Ignore padded positions in loss
    )

    # --- Build Training Arguments ---
    training_args = build_training_args(train_config, output_dir)

    # Compute warmup_steps from warmup_ratio if not explicitly set in config
    tc = train_config.get("training", train_config)
    if "warmup_ratio" in tc and "warmup_steps" not in tc:
        total_steps = (
            len(train_dataset)
            // training_args.per_device_train_batch_size
            // training_args.gradient_accumulation_steps
            * int(training_args.num_train_epochs)
        )
        training_args.warmup_steps = max(1, int(total_steps * tc["warmup_ratio"]))
        console.print(f"  Warmup steps (from {tc['warmup_ratio']} ratio): {training_args.warmup_steps}")

    # Adjust for compute backend
    if compute_backend == "mps":
        # MPS doesn't support some operations
        training_args.bf16 = False
        training_args.fp16 = False
        training_args.dataloader_pin_memory = False
        logger.info("Adjusted training args for MPS backend")
    elif compute_backend == "cpu":
        training_args.bf16 = False
        training_args.fp16 = False
        training_args.dataloader_pin_memory = False
        training_args.per_device_train_batch_size = 1
        training_args.gradient_accumulation_steps = 64
        logger.info("Adjusted training args for CPU backend")

    # --- Initialize Trainer ---
    console.print("[bold]Initializing Trainer...[/bold]")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Print training plan
    total_steps = (
        len(train_dataset)
        // training_args.per_device_train_batch_size
        // training_args.gradient_accumulation_steps
        * training_args.num_train_epochs
    )
    console.print(f"  Estimated total training steps: {total_steps:,}")
    console.print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    console.print(f"  Epochs: {training_args.num_train_epochs}")
    console.print()

    # --- Train ---
    console.print("[bold green]Starting training...[/bold green]")
    console.print()

    train_result = trainer.train()

    # --- Save Results ---
    console.print()
    console.print("[bold]Saving final model...[/bold]")

    # Save the LoRA adapter
    adapter_path = Path(output_dir) / "final_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    console.print(f"  LoRA adapter saved: {adapter_path}")

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Run final evaluation
    console.print("[bold]Running final evaluation...[/bold]")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # --- Summary ---
    console.print()
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  Training Complete[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print()
    console.print(f"  Train loss:      {metrics.get('train_loss', 'N/A')}")
    console.print(f"  Train runtime:   {metrics.get('train_runtime', 'N/A'):.1f}s")
    console.print(f"  Eval loss:       {eval_metrics.get('eval_loss', 'N/A')}")
    console.print(f"  Adapter saved:   {adapter_path}")
    console.print()
    console.print("[dim]Next step: python evaluation/run_benchmarks.py[/dim]")
    console.print("[dim]Or merge:  python training/train_model.py --merge[/dim]")


if __name__ == "__main__":
    # Check if merge mode requested
    if "--merge" in sys.argv:
        console.print("[bold]Merging LoRA adapter with base model...[/bold]")

        config_dir = PROJECT_ROOT / "training"
        train_config = load_yaml_config(str(config_dir / "training_config.yaml"))
        model_name = train_config["model"]["base_model"]
        output_dir = train_config["training"]["output_dir"]
        adapter_path = Path(output_dir) / "final_adapter"
        merged_path = Path(output_dir) / "merged_model"

        from peft import PeftModel

        console.print(f"  Loading base model: {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto",
        )

        console.print(f"  Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, str(adapter_path))

        console.print("  Merging weights...")
        merged_model = model.merge_and_unload()

        console.print(f"  Saving merged model: {merged_path}")
        merged_path.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(merged_path))

        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(merged_path))

        console.print("[green bold]Merge complete.[/green bold]")
        console.print(f"  Merged model at: {merged_path}")
    else:
        main()
