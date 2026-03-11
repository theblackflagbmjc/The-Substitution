Ya want something done right, ya gotta do it yerself.

---

# The Substitution v1.0.0

**Enterprise Systems Engineering LLM**

Brandon Michael Jeanpierre Corporation d/b/a The Black Flag
Delaware Entity #7336243 | IRS 501(c)(3) EIN: 92-2858861
Organization: [theblackflagbmjc](https://huggingface.co/theblackflagbmjc)

---

## Overview

The Substitution is a fine-tuned language model specialized for enterprise systems engineering. It is built on [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) using QLoRA (4-bit quantization with Low-Rank Adaptation) and trained on a curated mix of instruction, coding, mathematics, and technical documentation datasets.

This repository contains the complete, end-to-end pipeline: environment setup, dataset acquisition, data cleaning, tokenization, model training, evaluation, local inference, API serving, and deployment to Hugging Face Hub.

**22 source files. 5,821 lines. Zero external dependencies on proprietary tooling.**

| Spec | Value |
|------|-------|
| Base Model | Qwen/Qwen2.5-7B-Instruct (7.6B params) |
| License | Apache 2.0 |
| Method | QLoRA — 4-bit NF4 + LoRA (r=64, α=128, all-linear) |
| Trainable Params | ~160M (~2.1% of total) |
| Target Hardware | MacBook Pro M1Pro 32GB / NVIDIA CUDA / CPU |
| Training Format | ChatML (Qwen2.5 native) |
| Context Length | 4,096 tokens |

## Capabilities

- Infrastructure architecture design and review
- Systems administration (macOS, Linux, Windows, Kali)
- Python-primary software engineering with 8 secondary languages (Swift, Bash, Java, JavaScript, HTML, C++, C#, PowerShell)
- Defensive cybersecurity and authorized security testing
- Network engineering (subnetting, routing, topology, hybrid cloud)
- Database operations (MySQL, MSSQL, GraphQL)
- Security governance (NIST CSF, SP 800-53, SP 800-171, RMF)
- Enterprise documentation and technical writing
- Cryptographic architecture and key management
- Data analysis and anomaly detection

## Project Structure

```
the-substitution/
├── bootstrap/
│   ├── install_macos.sh            # Apple Silicon environment installer
│   ├── install_linux.sh            # Ubuntu/Debian installer (CUDA/ROCm/CPU auto-detect)
│   └── verify_environment.sh       # 6-section environment validation
├── datasets/
│   ├── download_datasets.py        # Acquire 8 datasets from Hugging Face
│   ├── verify_datasets.py          # SHA256 checksums + field validation
│   ├── clean_datasets.py           # Normalize, deduplicate, convert to ChatML
│   └── prepare_training_data.py    # Merge, weighted sampling, 95/5 train/val split
├── tokenizer/
│   └── build_tokenizer.py          # Qwen2.5 ChatML tokenization, assistant-only label masking
├── training/
│   ├── train_model.py              # QLoRA training with --merge support
│   ├── training_config.yaml        # Hyperparameters (tuned for M1Pro 32GB)
│   └── lora_config.yaml            # LoRA r=64, alpha=128, all-linear
├── evaluation/
│   ├── run_benchmarks.py           # HumanEval, GSM8K, MMLU benchmarks
│   ├── generate_report.py          # Baseline comparison reporting
│   └── evaluation_config.yaml      # Benchmark configuration + baselines
├── inference/
│   ├── local_inference.py          # Interactive CLI with streaming output
│   └── api_server.py               # FastAPI server (OpenAI-compatible endpoints)
├── deployment/
│   ├── huggingface_push.py         # Push to Hugging Face Hub
│   └── build_model_card.py         # Auto-generate model card from configs
├── model_card/
│   └── README.md                   # Hugging Face model card
├── requirements.txt                # 40+ pinned dependencies
├── pyproject.toml                  # Project metadata and build config
└── .gitignore
```

## Quick Start

### 1. Install the Environment

**macOS (Apple Silicon):**
```bash
git clone https://github.com/theblackflagbmjc/The-Substitution.git
cd The-Substitution
bash bootstrap/install_macos.sh
```

**Linux (Ubuntu/Debian):**
```bash
git clone https://github.com/theblackflagbmjc/The-Substitution.git
cd The-Substitution
bash bootstrap/install_linux.sh
```

The Linux installer auto-detects NVIDIA GPU (CUDA), AMD GPU (ROCm), or CPU-only and installs the appropriate PyTorch build.

### 2. Activate and Verify

```bash
source activate.sh
bash bootstrap/verify_environment.sh
```

### 3. Authenticate with Hugging Face

```bash
hf auth logout #clear session
hf auth login #fresh login with access token
```

Token available at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 4. Acquire and Prepare Training Data

```bash
python datasets/download_datasets.py        # Download all datasets
python datasets/verify_datasets.py           # Verify checksums + structure
python datasets/clean_datasets.py            # Clean, normalize, convert to ChatML
python datasets/prepare_training_data.py     # Merge, sample, shuffle, split
```

### 5. Tokenize

```bash
python tokenizer/build_tokenizer.py
```

### 6. Train

```bash
python training/train_model.py
```

Training uses QLoRA with gradient checkpointing and accumulation. On the M1Pro 32GB target hardware, the memory footprint is approximately:

| Component | Memory |
|-----------|-------:|
| Base model (4-bit) | ~4.0 GB |
| LoRA adapters | ~0.2 GB |
| Optimizer states | ~2.0 GB |
| Activations (gradient checkpointing) | ~4.0 GB |
| Overhead | ~2.0 GB |
| **Total** | **~12.2 GB** |

### 7. Evaluate

```bash
python evaluation/run_benchmarks.py          # Run HumanEval, GSM8K, MMLU
python evaluation/generate_report.py         # Generate comparison report
```

Quick smoke test (20 samples per benchmark):
```bash
python evaluation/run_benchmarks.py --smoke_test
```

### 8. Run Locally

**Interactive CLI:**
```bash
python inference/local_inference.py
```

Commands: `/quit`, `/clear`, `/system <msg>`, `/temp <value>`, `/tokens <n>`, `/save <path>`

**Single prompt:**
```bash
python inference/local_inference.py --prompt "Design a VLAN segmentation strategy for 50 users"
```

**API server:**
```bash
python inference/api_server.py --host 0.0.0.0 --port 8000
```

The API is OpenAI-compatible:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "the-substitution",
    "messages": [{"role": "user", "content": "Explain NIST SP 800-53 access control families."}]
  }'
```

### 9. Deploy to Hugging Face

```bash
python training/train_model.py --merge       # Merge LoRA into base weights
python deployment/build_model_card.py        # Generate model card
python deployment/huggingface_push.py        # Push to HF Hub
```

## Training Data

| Dataset | Source | Category | Usage |
|---------|--------|----------|-------|
| OpenHermes-2.5 | teknium | Instruction | 1M+ GPT-4 instruction pairs (full) |
| UltraChat | openbmb | Instruction | Multi-turn dialogue (500K cap) |
| OpenOrca | Open-Orca | Instruction | Augmented FLAN reasoning (500K cap) |
| CodeSearchNet | code-search-net | Coding | Python code-comment pairs (full) |
| GSM8K | openai | Math | 8.5K multi-step reasoning (3× oversampled) |
| HumanEval | openai | Eval (held out) | 164 Python coding problems |
| MMLU | cais | Eval (held out) | 57-subject knowledge benchmark |
| GSM8K test | openai | Eval (held out) | Math reasoning benchmark |

All training data is converted to a unified ChatML format with a system prompt defining The Substitution's role and capabilities. Labels are masked so the model only learns to predict assistant responses.

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank (r) | 64 | High capacity for domain specialization |
| LoRA alpha (α) | 128 | 2× rank scaling for balanced learning |
| Target modules | all-linear | QLoRA paper recommendation |
| Learning rate | 2e-4 | Standard for QLoRA with cosine decay |
| Epochs | 3 | Standard for instruction tuning |
| Effective batch size | 32 | 1 × 32 gradient accumulation |
| Warmup | 3% of steps | Cosine schedule |
| Max sequence length | 4,096 | Balances context vs. memory |
| Precision | bfloat16 | Native on both CUDA and Apple Silicon |
| Gradient checkpointing | Enabled | Memory optimization |

## Compute Backend Auto-Detection

| Backend | Behavior |
|---------|----------|
| **CUDA** (NVIDIA) | Full QLoRA: 4-bit NF4 via bitsandbytes + LoRA |
| **MPS** (Apple Silicon) | bfloat16 + LoRA (no quantization — bitsandbytes unsupported on macOS) |
| **CPU** | float32 fallback (functional, slow) |

The training script, inference scripts, and evaluation pipeline all auto-detect the available backend and adjust accordingly. No manual configuration needed.

## Evaluation Benchmarks

| Benchmark | Metric | Qwen2.5-7B Baseline |
|-----------|--------|---------------------:|
| HumanEval | pass@1 | 79.9% |
| GSM8K | accuracy | 82.6% |
| MMLU | accuracy | 74.2% |

Post-training scores are generated by `evaluation/run_benchmarks.py` and compared against these baselines via `evaluation/generate_report.py`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI-compatible chat completion |
| POST | `/v1/completions` | Raw text completion |
| GET | `/v1/models` | List available models |
| GET | `/health` | Health check (status, device, uptime) |

## Hardware Requirements

| Environment | Minimum | Recommended |
|-------------|---------|-------------|
| Apple Silicon | M1 Pro 16GB | M1 Pro 32GB+ |
| NVIDIA GPU | 8GB VRAM (4-bit) | 16GB+ VRAM |
| CPU | 16GB RAM | 32GB+ RAM |

## History

This repository previously contained mergekit-based experiments (`TheSubstitutionXYXXYYX-mistral-Slurpee-Xb`) that merged ArmurAI/Pentest_AI with ramgpt-13b-awq-gemm using slerp, dare-ties, and passthrough strategies. Those merges produced incoherent outputs due to incompatible architectures (Mistral 7B vs. AWQ-quantized 13B) causing tokenizer conflicts and embedding space misalignment.

v1.0.0 starts clean from a single compatible base model (Qwen2.5-7B-Instruct) with proper supervised fine-tuning via LoRA, eliminating the architectural incompatibility entirely.

## License

Apache 2.0 (inherited from Qwen2.5-7B-Instruct base model).

---

*Brandon Michael Jeanpierre Corporation d/b/a The Black Flag*
*8 The Green, Ste A, Dover, DE 19901*
*[theblackflag.org](https://theblackflag.org)*

*Do no harm. Take no shit.*
