#!/usr/bin/env python3
"""
The Substitution v1.0.0 — Benchmark Runner
Brandon Michael Jeanpierre Corporation d/b/a The Black Flag
Organization: theblackflagbmjc

Runs evaluation benchmarks against the trained model to measure performance
across coding, mathematical reasoning, and general knowledge domains.

Benchmarks:
  - HumanEval: Python code generation (pass@1, pass@10)
  - GSM8K: Multi-step mathematical reasoning (accuracy)
  - MMLU: Massive multitask language understanding (accuracy per subject)

Usage:
    # Evaluate the LoRA adapter (loads base + adapter)
    python evaluation/run_benchmarks.py

    # Evaluate a merged model
    python evaluation/run_benchmarks.py --model_path ./output/the-substitution-v1/merged_model

    # Run specific benchmarks only
    python evaluation/run_benchmarks.py --benchmarks humaneval gsm8k

    # Quick smoke test (small sample)
    python evaluation/run_benchmarks.py --smoke_test
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import yaml
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
EVAL_DIR = PROJECT_ROOT / "output" / "evaluation"
LOG_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "run_benchmarks.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("benchmark_runner")
console = Console()


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_for_eval(
    model_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
) -> tuple:
    """
    Load model and tokenizer for evaluation.

    Supports three modes:
      1. Merged model path (single directory with full weights)
      2. Adapter path (loads base model + LoRA adapter)
      3. Default (loads from training output directory)
    """
    config_path = PROJECT_ROOT / "training" / "training_config.yaml"
    with open(config_path, "r") as f:
        train_config = yaml.safe_load(f)

    base_model_name = train_config["model"]["base_model"]

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32  # MPS bfloat16 support is limited
    else:
        device = "cpu"
        dtype = torch.float32

    console.print(f"[bold]Loading model for evaluation...[/bold]")
    console.print(f"  Device: {device}")

    if model_path:
        # Mode 1: Load merged model directly
        console.print(f"  Loading merged model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else {"": device},
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        # Mode 2: Load base model + LoRA adapter
        if adapter_path is None:
            adapter_path = str(
                PROJECT_ROOT / train_config["training"]["output_dir"] / "final_adapter"
            )

        console.print(f"  Loading base model: {base_model_name}")
        console.print(f"  Loading adapter: {adapter_path}")

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else {"": device},
        )

        if Path(adapter_path).exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
            console.print("  Adapter merged for evaluation")
        else:
            console.print(f"  [yellow]Adapter not found at {adapter_path}. Evaluating base model.[/yellow]")

        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    console.print(f"  Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer, device


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    device: str = "cuda",
) -> str:
    """Generate a single response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3584)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


# =============================================================================
# BENCHMARK: HUMANEVAL (Coding)
# =============================================================================

def run_humaneval(model, tokenizer, device: str, max_samples: Optional[int] = None) -> dict:
    """
    Run HumanEval coding benchmark.

    HumanEval contains 164 Python programming problems. For each problem,
    the model generates a function body given the function signature and
    docstring, then the generated code is tested against unit tests.

    Metrics: pass@1 (greedy), pass@10 (with sampling)
    """
    console.print("[bold]Running HumanEval benchmark...[/bold]")

    from datasets import load_dataset
    dataset = load_dataset("openai/openai_humaneval", split="test")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    total = len(dataset)
    passed = 0
    results = []

    for i, example in enumerate(dataset):
        task_id = example["task_id"]
        prompt = example["prompt"]
        test_code = example["test"]
        entry_point = example["entry_point"]

        # Format prompt for the model
        chat_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a Python programming assistant. Write only the function body to complete the given function. Do not include the function signature or any explanation."},
                {"role": "user", "content": f"Complete the following Python function:\n\n{prompt}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate completion
        completion = generate_response(model, tokenizer, chat_prompt, max_new_tokens=512, device=device)

        # Extract code from response (handle markdown code blocks)
        code = extract_code(completion)

        # Build full program for testing
        full_code = prompt + code

        # Test the generated code
        passed_test = execute_test(full_code, test_code, entry_point)

        if passed_test:
            passed += 1

        results.append({
            "task_id": task_id,
            "prompt": prompt[:100],
            "completion": code[:200],
            "passed": passed_test,
        })

        if (i + 1) % 20 == 0:
            console.print(f"  Progress: {i + 1}/{total} | Pass rate: {passed / (i + 1) * 100:.1f}%")

    pass_at_1 = passed / total * 100

    console.print(f"  [bold]HumanEval pass@1: {pass_at_1:.1f}%[/bold] ({passed}/{total})")

    return {
        "benchmark": "humaneval",
        "metric": "pass@1",
        "score": round(pass_at_1, 2),
        "passed": passed,
        "total": total,
        "details": results,
    }


def extract_code(response: str) -> str:
    """Extract Python code from a model response, handling markdown blocks."""
    # Try to extract from code blocks
    code_block = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()

    # If no code block, try to extract lines that look like code
    lines = response.split("\n")
    code_lines = []
    for line in lines:
        stripped = line.rstrip()
        # Skip obvious non-code lines
        if stripped.startswith("#") or stripped == "":
            code_lines.append(stripped)
        elif any(stripped.startswith(kw) for kw in [
            "def ", "return ", "if ", "for ", "while ", "class ", "import ",
            "from ", "try:", "except", "with ", "    ", "\t", "else:", "elif ",
            "raise ", "yield ", "pass", "break", "continue", "assert ",
        ]):
            code_lines.append(stripped)
        elif code_lines:  # Continue collecting after first code line
            code_lines.append(stripped)

    return "\n".join(code_lines) if code_lines else response


def execute_test(code: str, test_code: str, entry_point: str, timeout: int = 10) -> bool:
    """Execute generated code against test cases with a timeout."""
    import multiprocessing
    import io
    import contextlib

    full_program = code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"

    def run_test(result_queue):
        try:
            exec_globals = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(full_program, exec_globals)
            result_queue.put(True)
        except Exception:
            result_queue.put(False)

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=run_test, args=(result_queue,))
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    try:
        return result_queue.get_nowait()
    except Exception:
        return False


# =============================================================================
# BENCHMARK: GSM8K (Math Reasoning)
# =============================================================================

def run_gsm8k(model, tokenizer, device: str, max_samples: Optional[int] = None) -> dict:
    """
    Run GSM8K mathematical reasoning benchmark.

    GSM8K contains grade school math problems requiring multi-step reasoning.
    The model must generate a step-by-step solution and arrive at the correct
    numerical answer.

    Metric: Exact match accuracy on the final numerical answer.
    """
    console.print("[bold]Running GSM8K benchmark...[/bold]")

    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    total = len(dataset)
    correct = 0
    results = []

    for i, example in enumerate(dataset):
        question = example["question"]
        answer_text = example["answer"]

        # Extract the ground truth numerical answer (after ####)
        gt_answer = extract_gsm8k_answer(answer_text)

        # Format prompt
        chat_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a math tutor. Solve the following problem step by step. End your response with the final numerical answer on its own line, preceded by ####."},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate solution
        response = generate_response(model, tokenizer, chat_prompt, max_new_tokens=512, device=device)

        # Extract predicted answer
        predicted = extract_gsm8k_answer(response)

        is_correct = predicted is not None and gt_answer is not None and abs(predicted - gt_answer) < 1e-6

        if is_correct:
            correct += 1

        results.append({
            "question": question[:100],
            "ground_truth": gt_answer,
            "predicted": predicted,
            "correct": is_correct,
        })

        if (i + 1) % 100 == 0:
            console.print(f"  Progress: {i + 1}/{total} | Accuracy: {correct / (i + 1) * 100:.1f}%")

    accuracy = correct / total * 100
    console.print(f"  [bold]GSM8K accuracy: {accuracy:.1f}%[/bold] ({correct}/{total})")

    return {
        "benchmark": "gsm8k",
        "metric": "accuracy",
        "score": round(accuracy, 2),
        "correct": correct,
        "total": total,
        "details": results,
    }


def extract_gsm8k_answer(text: str) -> Optional[float]:
    """Extract the numerical answer from a GSM8K response (after ####)."""
    if not text:
        return None

    # Look for #### delimiter
    match = re.search(r"####\s*([-\d,\.]+)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Fallback: look for the last number in the text
    numbers = re.findall(r"[-\d,]+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass

    return None


# =============================================================================
# BENCHMARK: MMLU (General Reasoning)
# =============================================================================

def run_mmlu(model, tokenizer, device: str, max_samples: Optional[int] = None) -> dict:
    """
    Run MMLU (Massive Multitask Language Understanding) benchmark.

    MMLU tests knowledge across 57 subjects including STEM, humanities,
    social sciences, and more. Each question is multiple choice (A/B/C/D).

    Metric: Accuracy (correct choice selection).
    """
    console.print("[bold]Running MMLU benchmark...[/bold]")

    from datasets import load_dataset
    dataset = load_dataset("cais/mmlu", "all", split="test")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    total = len(dataset)
    correct = 0
    subject_scores = {}
    results = []

    choice_labels = ["A", "B", "C", "D"]

    for i, example in enumerate(dataset):
        question = example["question"]
        choices = example["choices"]
        answer_idx = example["answer"]
        subject = example.get("subject", "unknown")

        # Format choices
        choices_text = "\n".join(
            f"{label}. {choice}" for label, choice in zip(choice_labels, choices)
        )

        # Format prompt
        chat_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are taking a multiple choice exam. Answer with only the letter (A, B, C, or D) of the correct answer. Do not explain."},
                {"role": "user", "content": f"{question}\n\n{choices_text}\n\nAnswer:"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate answer
        response = generate_response(
            model, tokenizer, chat_prompt,
            max_new_tokens=5,
            device=device,
        )

        # Extract predicted choice
        predicted_idx = extract_mmlu_answer(response)
        is_correct = predicted_idx == answer_idx

        if is_correct:
            correct += 1

        # Track per-subject scores
        if subject not in subject_scores:
            subject_scores[subject] = {"correct": 0, "total": 0}
        subject_scores[subject]["total"] += 1
        if is_correct:
            subject_scores[subject]["correct"] += 1

        results.append({
            "question": question[:80],
            "subject": subject,
            "ground_truth": choice_labels[answer_idx],
            "predicted": choice_labels[predicted_idx] if predicted_idx is not None else "?",
            "correct": is_correct,
        })

        if (i + 1) % 500 == 0:
            console.print(f"  Progress: {i + 1}/{total} | Accuracy: {correct / (i + 1) * 100:.1f}%")

    accuracy = correct / total * 100

    # Compute per-subject accuracies
    subject_accuracies = {}
    for subject, scores in subject_scores.items():
        if scores["total"] > 0:
            subject_accuracies[subject] = round(scores["correct"] / scores["total"] * 100, 1)

    console.print(f"  [bold]MMLU accuracy: {accuracy:.1f}%[/bold] ({correct}/{total})")

    return {
        "benchmark": "mmlu",
        "metric": "accuracy",
        "score": round(accuracy, 2),
        "correct": correct,
        "total": total,
        "subject_accuracies": subject_accuracies,
        "details": results[:100],  # Limit stored details for MMLU (large dataset)
    }


def extract_mmlu_answer(response: str) -> Optional[int]:
    """Extract the choice index from an MMLU response."""
    if not response:
        return None

    response = response.strip().upper()

    # Direct letter match
    for i, letter in enumerate(["A", "B", "C", "D"]):
        if response.startswith(letter):
            return i

    # Search for letter in response
    match = re.search(r"\b([ABCD])\b", response)
    if match:
        return ["A", "B", "C", "D"].index(match.group(1))

    return None


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(all_results: list[dict], model_info: str) -> dict:
    """Generate a comprehensive evaluation report."""
    report = {
        "project": "The Substitution v1.0.0",
        "organization": "theblackflagbmjc",
        "model": model_info,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "benchmarks": {},
        "summary": {},
    }

    table = Table(title="Evaluation Results", show_lines=True)
    table.add_column("Benchmark", style="cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Details", justify="right")

    for result in all_results:
        name = result["benchmark"]
        report["benchmarks"][name] = result

        detail_str = f"{result.get('correct', result.get('passed', '?'))}/{result['total']}"
        table.add_row(name, result["metric"], f"{result['score']}%", detail_str)

    console.print()
    console.print(table)

    # Summary statistics
    scores = [r["score"] for r in all_results]
    report["summary"] = {
        "num_benchmarks": len(all_results),
        "average_score": round(sum(scores) / len(scores), 2) if scores else 0,
        "scores": {r["benchmark"]: r["score"] for r in all_results},
    }

    return report


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="The Substitution v1.0.0 — Benchmark Runner")
    parser.add_argument("--model_path", type=str, default=None, help="Path to merged model directory")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter directory")
    parser.add_argument(
        "--benchmarks", nargs="+", default=["humaneval", "gsm8k", "mmlu"],
        choices=["humaneval", "gsm8k", "mmlu"],
        help="Which benchmarks to run",
    )
    parser.add_argument("--smoke_test", action="store_true", help="Run with small sample for quick validation")
    parser.add_argument("--output", type=str, default=None, help="Output report path")
    return parser.parse_args()


def main():
    args = parse_args()

    console.print()
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  The Substitution v1.0.0 — Benchmark Evaluation[/bold cyan]")
    console.print("[bold cyan]  Brandon Michael Jeanpierre Corporation[/bold cyan]")
    console.print("[bold cyan]  d/b/a The Black Flag[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print()

    # Load model
    model, tokenizer, device = load_model_for_eval(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
    )
    console.print()

    # Determine sample limits
    max_samples = None
    if args.smoke_test:
        max_samples = 20
        console.print("[yellow]Smoke test mode: limited to 20 samples per benchmark[/yellow]")
        console.print()

    # Run benchmarks
    all_results = []

    benchmark_runners = {
        "humaneval": lambda: run_humaneval(model, tokenizer, device, max_samples),
        "gsm8k": lambda: run_gsm8k(model, tokenizer, device, max_samples),
        "mmlu": lambda: run_mmlu(model, tokenizer, device, max_samples),
    }

    for benchmark_name in args.benchmarks:
        if benchmark_name in benchmark_runners:
            console.print()
            start_time = time.time()
            result = benchmark_runners[benchmark_name]()
            result["runtime_seconds"] = round(time.time() - start_time, 1)
            all_results.append(result)
        else:
            console.print(f"[yellow]Unknown benchmark: {benchmark_name}[/yellow]")

    # Generate and save report
    model_info = args.model_path or args.adapter_path or "default (training output)"
    report = generate_report(all_results, model_info)

    output_path = args.output or str(EVAL_DIR / "evaluation_report.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    console.print()
    console.print(f"[bold]Report saved: {output_path}[/bold]")
    console.print()
    console.print(f"[bold]Average score: {report['summary']['average_score']}%[/bold]")
    console.print()
    console.print("[dim]Next step: python inference/local_inference.py[/dim]")


if __name__ == "__main__":
    main()
