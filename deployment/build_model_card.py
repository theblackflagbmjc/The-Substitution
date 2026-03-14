#!/usr/bin/env python3
"""
The Substitution v1.0.0 — Model Card Builder
Brandon Michael Jeanpierre Corporation d/b/a The Black Flag
Organization: theblackflagbmjc

Generates a Hugging Face model card (README.md) from evaluation results
and training configuration.

Usage:
    python deployment/build_model_card.py
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("model_card_builder")
console = Console()


def load_eval_results() -> dict:
    """Load evaluation results if available."""
    eval_path = PROJECT_ROOT / "output" / "evaluation" / "evaluation_report.json"
    if eval_path.exists():
        with open(eval_path, "r") as f:
            return json.load(f)
    return {}


def load_training_config() -> dict:
    """Load training configuration."""
    config_path = PROJECT_ROOT / "training" / "training_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_lora_config() -> dict:
    """Load LoRA configuration."""
    config_path = PROJECT_ROOT / "training" / "lora_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model_card() -> str:
    """Build the model card content."""
    train_config = load_training_config()
    lora_config = load_lora_config()
    eval_results = load_eval_results()

    base_model = train_config["model"]["base_model"]
    lora_params = lora_config.get("lora", lora_config)

    # Build evaluation table
    eval_table = ""
    benchmarks = eval_results.get("benchmarks", {})
    if benchmarks:
        eval_table = "| Benchmark | Metric | Score |\n|-----------|--------|------:|\n"
        for name, result in benchmarks.items():
            eval_table += f"| {name} | {result['metric']} | {result['score']}% |\n"

    card = f"""---
license: apache-2.0
base_model: {base_model}
tags:
  - enterprise
  - systems-engineering
  - cybersecurity
  - infrastructure
  - code-generation
  - peft
  - lora
  - qlora
language:
  - en
pipeline_tag: text-generation
library_name: transformers
---

# The Substitution v1.0.0

**Enterprise Systems Engineering LLM**

Created by **Brandon Michael Jeanpierre Corporation d/b/a The Black Flag**
Organization: [theblackflagbmjc](https://huggingface.co/theblackflagbmjc)

## Model Description

The Substitution is a fine-tuned language model specialized for enterprise systems engineering tasks. Built on {base_model} using QLoRA (4-bit quantization with Low-Rank Adaptation), it serves as a technical advisor for infrastructure architecture, systems administration, defensive cybersecurity, and enterprise operations.

### Capabilities

- Infrastructure architecture design and review
- Systems administration across macOS, Linux, and Windows
- Python-primary software engineering with multi-language support
- Defensive cybersecurity analysis and security governance (NIST CSF, SP 800-53, SP 800-171)
- Network engineering: subnetting, routing, topology design
- Database operations: MySQL, MSSQL, GraphQL
- Enterprise documentation and technical writing
- Cryptographic architecture and key management

### Intended Use

This model is intended for use within The Black Flag's organizational infrastructure as a technical reasoning assistant. It operates under the governance framework of BMJC and the Doctrine of Inherent Entanglement.

## Training Details

### Base Model
- **Architecture:** {base_model}
- **Parameters:** 7.6B

### Fine-tuning Method
- **Method:** QLoRA (4-bit NF4 quantization + LoRA)
- **LoRA Rank (r):** {lora_params.get('r', 64)}
- **LoRA Alpha:** {lora_params.get('lora_alpha', 128)}
- **Target Modules:** {lora_params.get('target_modules', 'all-linear')}
- **Trainable Parameters:** ~160M (~2.1% of total)

### Training Configuration
- **Epochs:** {train_config['training']['num_train_epochs']}
- **Learning Rate:** {train_config['training']['learning_rate']}
- **Scheduler:** {train_config['training']['lr_scheduler_type']}
- **Effective Batch Size:** {train_config['training']['per_device_train_batch_size'] * train_config['training']['gradient_accumulation_steps']}
- **Max Sequence Length:** {train_config['data']['max_seq_length']}
- **Precision:** bfloat16

### Training Data
- OpenHermes-2.5 (instruction following)
- UltraChat (multi-turn dialogue)
- OpenOrca (reasoning)
- CodeSearchNet (code generation)
- GSM8K (mathematical reasoning)

## Evaluation Results

{eval_table if eval_table else "*Evaluation pending — run `python evaluation/run_benchmarks.py` after training.*"}

## Usage

### Local Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "theblackflagbmjc/the-substitution",
    trust_remote_code=True,
    dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "theblackflagbmjc/the-substitution",
    trust_remote_code=True,
)

messages = [
    {{"role": "system", "content": "You are The Substitution, an enterprise systems engineering assistant."}},
    {{"role": "user", "content": "Design a VLAN segmentation strategy for a small office network with 50 users."}},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1024)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

### API Server

```bash
python inference/api_server.py --host 0.0.0.0 --port 8000
```

Then use the OpenAI-compatible endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "the-substitution",
    "messages": [{{"role": "user", "content": "Explain NIST SP 800-53 access control families."}}]
  }}'
```

## Hardware Requirements

| Environment | Minimum | Recommended |
|-------------|---------|-------------|
| Apple Silicon | M1 Pro 16GB | M1 Pro 32GB+ |
| NVIDIA GPU | 8GB VRAM (4-bit) | 16GB+ VRAM |
| CPU | 16GB RAM | 32GB+ RAM |

## Limitations

- Specialized for enterprise systems engineering; general conversational ability may be reduced compared to the base model
- Code generation is Python-primary; other languages are supported but with less depth
- Security guidance is defensive and governance-focused; this model does not generate offensive security tools or exploits
- Knowledge is bounded by training data cutoff and base model capabilities

## License

Apache 2.0 (inherited from base model)

## Citation

```
@misc{{the-substitution-2026,
  title={{The Substitution: Enterprise Systems Engineering LLM}},
  author={{Brandon Michael Jeanpierre}},
  year={{2026}},
  publisher={{Brandon Michael Jeanpierre Corporation d/b/a The Black Flag}},
  organization={{theblackflagbmjc}},
  url={{https://huggingface.co/theblackflagbmjc/the-substitution}},
}}
```

---

*Brandon Michael Jeanpierre Corporation d/b/a The Black Flag*
*Delaware Entity #7336243 | IRS 501(c)(3) EIN: 92-2858861*
*8 The Green, Ste A, Dover, DE 19901*
"""

    return card


def main():
    console.print()
    console.print("[bold cyan]The Substitution v1.0.0 — Model Card Builder[/bold cyan]")
    console.print()

    card_content = build_model_card()

    output_path = PROJECT_ROOT / "model_card" / "README.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(card_content)

    console.print(f"[bold]Model card generated: {output_path}[/bold]")
    console.print(f"  Length: {len(card_content):,} characters")
    console.print()
    console.print("[green bold]Model card ready.[/green bold]")


if __name__ == "__main__":
    main()
