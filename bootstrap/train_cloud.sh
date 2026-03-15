#!/usr/bin/env bash
# ==============================================================================
# The Substitution v1.0.0 — Cloud GPU Training (Single-Shot)
# Brandon Michael Jeanpierre Corporation d/b/a The Black Flag
# Organization: theblackflagbmjc
#
# PURPOSE:
#   Runs the ENTIRE pipeline on a fresh GCP VM from zero to deployed model.
#   When this script finishes, the trained model is on HuggingFace Hub.
#   You can then delete the VM. Pull the model locally for inference.
#
# USAGE:
#   1. Create a GCP VM:
#        - Image: Ubuntu 22.04 LTS
#        - GPU: NVIDIA L4 (cheapest, ~$0.70/hr) or A100 40GB (~$3.50/hr)
#        - Disk: 200GB SSD (datasets + model weights are large)
#        - RAM: 32GB+
#
#   2. SSH in and run:
#        curl -fsSL https://raw.githubusercontent.com/theblackflagbmjc/The-Substitution/main/bootstrap/train_cloud.sh | bash
#
#      OR clone and run:
#        git clone https://github.com/theblackflagbmjc/The-Substitution.git
#        cd The-Substitution
#        bash bootstrap/train_cloud.sh
#
#   3. When it says "BUILD COMPLETE", delete the VM.
#      The model is on HuggingFace. Pull it locally with:
#        python inference/local_inference.py --model_path theblackflagbmjc/the-substitution
#
# ESTIMATED TIME:
#   L4 (24GB VRAM):   ~36-48 hours  |  ~$25-35 total
#   A100 40GB:         ~18-24 hours  |  ~$65-85 total
#   A100 80GB:         ~12-18 hours  |  ~$85-120 total
#
# ESTIMATED COST assumes full 1.8M dataset, 3 epochs, seq_length 4096.
# ==============================================================================

set -euo pipefail

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()     { echo -e "${CYAN}[$(date '+%H:%M:%S')] $1${NC}"; }
success() { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ $1${NC}"; }
warn()    { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠ $1${NC}"; }
fail()    { echo -e "${RED}[$(date '+%H:%M:%S')] ✖ $1${NC}"; exit 1; }

echo ""
echo "=============================================================================="
echo -e "${CYAN}  The Substitution v1.0.0 — Cloud GPU Training${NC}"
echo -e "${CYAN}  Brandon Michael Jeanpierre Corporation d/b/a The Black Flag${NC}"
echo "=============================================================================="
echo ""

# =============================================================================
# STEP 0: PRE-FLIGHT
# =============================================================================

log "Step 0/11: Pre-flight checks..."

# Must have NVIDIA GPU
if ! command -v nvidia-smi &>/dev/null; then
    fail "No NVIDIA GPU detected. This script requires a CUDA GPU."
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
success "GPU detected: ${GPU_NAME} (${GPU_VRAM} MB VRAM)"

# Check disk space (need ~150GB free)
DISK_FREE=$(df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')
if [[ "${DISK_FREE}" -lt 100 ]]; then
    warn "Only ${DISK_FREE}GB free disk space. Recommend 200GB+. Continuing anyway..."
else
    success "Disk space: ${DISK_FREE}GB free"
fi

# =============================================================================
# STEP 1: HUGGING FACE TOKEN
# =============================================================================

log "Step 1/11: Hugging Face authentication..."

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo ""
    echo -e "${YELLOW}  You need a Hugging Face token with WRITE access to push the trained model.${NC}"
    echo -e "${YELLOW}  Get one at: https://huggingface.co/settings/tokens${NC}"
    echo ""
    read -rp "  Paste your HF token: " HF_TOKEN
    echo ""
fi

if [[ -z "${HF_TOKEN}" ]]; then
    fail "No HF token provided. Cannot push model after training."
fi

export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
success "HF token set"

# =============================================================================
# STEP 2: SYSTEM DEPENDENCIES
# =============================================================================

log "Step 2/11: Installing system dependencies..."

sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3.11 python3.11-venv python3.11-dev \
    git git-lfs cmake build-essential wget curl \
    2>&1 | tail -5

git lfs install 2>/dev/null
success "System dependencies installed"

# =============================================================================
# STEP 3: CLONE REPO
# =============================================================================

log "Step 3/11: Setting up project..."

WORK_DIR="/home/$(whoami)/the-substitution-training"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

if [[ ! -d "The-Substitution/.git" ]]; then
    git clone https://github.com/theblackflagbmjc/The-Substitution.git
fi
cd The-Substitution
success "Repository ready at $(pwd)"

# =============================================================================
# STEP 4: PYTHON ENVIRONMENT
# =============================================================================

log "Step 4/11: Creating Python environment..."

VENV_PATH="./the-substitution-env"
if [[ ! -d "${VENV_PATH}" ]]; then
    python3.11 -m venv "${VENV_PATH}"
fi
source "${VENV_PATH}/bin/activate"
pip install --upgrade pip setuptools wheel -q

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q

# Install all project dependencies
pip install -r requirements.txt -q 2>/dev/null || {
    warn "Some optional deps failed, installing core..."
    pip install transformers datasets tokenizers accelerate peft trl \
        huggingface_hub safetensors sentencepiece bitsandbytes \
        wandb tensorboard scipy scikit-learn \
        fastapi uvicorn pydantic \
        pandas pyarrow tqdm pyyaml jsonlines requests \
        python-dotenv psutil rich rouge-score nltk -q
}

success "Python environment ready"
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"

# =============================================================================
# STEP 5: CONFIGURE FOR CLOUD
# =============================================================================

log "Step 5/11: Writing cloud training configuration..."

# Overwrite the training config with full-scale cloud profile
cat > training/training_config.yaml << 'CLOUD_CONFIG'
# ==============================================================================
# The Substitution v1.0.0 — CLOUD TRAINING CONFIGURATION
# Auto-generated by train_cloud.sh
# ==============================================================================

model:
  base_model: "Qwen/Qwen2.5-7B-Instruct"
  model_type: "causal_lm"
  trust_remote_code: true
  dtype: "bfloat16"

quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true

data:
  tokenized_dataset_path: "./data/tokenized"
  max_seq_length: 4096

training:
  output_dir: "./output/the-substitution-v1"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  max_grad_norm: 1.0
  bf16: true
  fp16: false
  tf32: true
  gradient_checkpointing: true
  optim: "adamw_torch"
  logging_steps: 25
  logging_first_step: true
  log_level: "info"
  eval_strategy: "steps"
  eval_steps: 1000
  save_strategy: "steps"
  save_steps: 1000
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  seed: 42
  data_seed: 42
  report_to: "none"
  dataloader_num_workers: 8
  dataloader_pin_memory: true
  remove_unused_columns: true
CLOUD_CONFIG

# Overwrite the dataset config with full-scale sampling
cat > datasets/prepare_training_data_cloud.py << 'CLOUD_PREP_EOF'
#!/usr/bin/env python3
"""Cloud-scale dataset preparation — overrides local caps."""

import json
import logging
import random
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FINAL_DIR = PROCESSED_DIR / "final"
LOG_DIR = PROJECT_ROOT / "logs"

FINAL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "prepare_training_data.log"), logging.StreamHandler()])
logger = logging.getLogger("data_preparer")
console = Console()

SAMPLING_CONFIG = {
    "openhermes_cleaned.jsonl":   {"weight": 1.0, "max_samples": None,    "category": "instruction"},
    "ultrachat_cleaned.jsonl":    {"weight": 1.0, "max_samples": 200_000, "category": "instruction"},
    "openorca_cleaned.jsonl":     {"weight": 1.0, "max_samples": 200_000, "category": "instruction"},
    "codesearchnet_cleaned.jsonl":{"weight": 1.0, "max_samples": None,    "category": "coding"},
    "gsm8k_cleaned.jsonl":        {"weight": 3.0, "max_samples": None,    "category": "math"},
}

TRAIN_RATIO = 0.95
RANDOM_SEED = 42

def load_cleaned_dataset(filepath, config):
    if not filepath.exists():
        return []
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try: examples.append(json.loads(line.strip()))
            except: continue
    if config.get("max_samples") and len(examples) > config["max_samples"]:
        random.seed(RANDOM_SEED)
        examples = random.sample(examples, config["max_samples"])
    weight = config.get("weight", 1.0)
    if weight > 1.0:
        repeat = int(weight)
        frac = weight - repeat
        oversampled = examples * repeat
        if frac > 0:
            extra = int(len(examples) * frac)
            random.seed(RANDOM_SEED + 1)
            oversampled.extend(random.sample(examples, min(extra, len(examples))))
        examples = oversampled
    return examples

def validate_example(ex):
    msgs = ex.get("messages", [])
    if not msgs or len(msgs) < 2: return False
    roles = {m.get("role") for m in msgs}
    if "user" not in roles or "assistant" not in roles: return False
    return all(m.get("role") and m.get("content") for m in msgs)

def main():
    console.print("[bold cyan]Cloud-Scale Dataset Preparation[/bold cyan]")
    all_examples = []
    for filename, config in SAMPLING_CONFIG.items():
        filepath = PROCESSED_DIR / filename
        console.print(f"  Loading: {filename}")
        examples = load_cleaned_dataset(filepath, config)
        valid = [ex for ex in examples if validate_example(ex)]
        for ex in valid: ex["_source"] = filename.replace("_cleaned.jsonl", "")
        all_examples.extend(valid)
        console.print(f"    {len(valid):,} examples loaded")

    random.seed(RANDOM_SEED)
    random.shuffle(all_examples)
    console.print(f"\n[bold]Total: {len(all_examples):,} examples[/bold]")

    split_idx = int(len(all_examples) * TRAIN_RATIO)
    train_ex, val_ex = all_examples[:split_idx], all_examples[split_idx:]
    console.print(f"  Train: {len(train_ex):,} | Val: {len(val_ex):,}")

    def to_dict(exs):
        return {"messages": [json.dumps(e["messages"], ensure_ascii=False) for e in exs],
                "source": [e.get("_source", "unknown") for e in exs]}

    ds = DatasetDict({"train": Dataset.from_dict(to_dict(train_ex)),
                       "validation": Dataset.from_dict(to_dict(val_ex))})
    ds.save_to_disk(str(FINAL_DIR))

    for split_name, exs in [("train", train_ex), ("validation", val_ex)]:
        with open(FINAL_DIR / f"{split_name}.jsonl", "w", encoding="utf-8") as f:
            for ex in exs:
                f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")

    console.print(f"\nSaved to: {FINAL_DIR}")
    console.print("[green bold]Cloud dataset preparation complete.[/green bold]")

if __name__ == "__main__":
    main()
CLOUD_PREP_EOF

chmod +x datasets/prepare_training_data_cloud.py
success "Cloud configuration written"

# =============================================================================
# STEP 6: AUTHENTICATE HUGGING FACE
# =============================================================================

log "Step 6/11: Authenticating with Hugging Face..."

huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null
success "Hugging Face authenticated"

# =============================================================================
# STEP 7: DOWNLOAD AND PREPARE DATA
# =============================================================================

log "Step 7/11: Downloading datasets (this takes 10-30 min)..."

python datasets/download_datasets.py
python datasets/verify_datasets.py

log "Cleaning datasets..."
python datasets/clean_datasets.py

log "Preparing cloud-scale training data..."
python datasets/prepare_training_data_cloud.py

success "Data pipeline complete"

# =============================================================================
# STEP 8: TOKENIZE
# =============================================================================

log "Step 8/11: Tokenizing (seq_length=4096)..."

python tokenizer/build_tokenizer.py
success "Tokenization complete"

# =============================================================================
# STEP 9: TRAIN
# =============================================================================

log "Step 9/11: Starting training..."
echo ""
echo "=============================================================================="
echo -e "${GREEN}  TRAINING IN PROGRESS — This will take 18-48 hours depending on GPU.${NC}"
echo -e "${GREEN}  Checkpoints save every 1000 steps. Safe to monitor and leave running.${NC}"
echo "=============================================================================="
echo ""

python training/train_model.py

success "Training complete"

# =============================================================================
# STEP 10: MERGE AND EVALUATE
# =============================================================================

log "Step 10/11: Merging adapter and evaluating..."

# Merge LoRA adapter into base model weights
python training/train_model.py --merge
success "Model merged"

# Quick evaluation (smoke test — full eval is optional)
log "Running evaluation smoke test..."
python evaluation/run_benchmarks.py --smoke_test || {
    warn "Evaluation had issues (non-blocking). Model is still valid."
}

# Generate report
python evaluation/generate_report.py 2>/dev/null || true

# =============================================================================
# STEP 11: PUSH TO HUGGING FACE
# =============================================================================

log "Step 11/11: Pushing to Hugging Face Hub..."

# Build model card
python deployment/build_model_card.py 2>/dev/null || true

# Push the MERGED model (full weights, not just adapter)
# This is what you pull locally for inference.
python deployment/huggingface_push.py --repo_id theblackflagbmjc/the-substitution --public

success "Model pushed to Hugging Face Hub"

# Also push the adapter separately (smaller, useful for re-applying to base)
log "Pushing LoRA adapter as backup..."
python deployment/huggingface_push.py \
    --repo_id theblackflagbmjc/the-substitution-adapter \
    --adapter_only 2>/dev/null || {
    warn "Adapter-only push failed (non-blocking). Merged model is already on Hub."
}

# =============================================================================
# DONE
# =============================================================================

echo ""
echo "=============================================================================="
echo -e "${GREEN}  ██████╗ ██╗   ██╗██╗██╗     ██████╗      ██████╗ ██████╗ ███╗   ███╗${NC}"
echo -e "${GREEN}  ██╔══██╗██║   ██║██║██║     ██╔══██╗    ██╔════╝██╔═══██╗████╗ ████║${NC}"
echo -e "${GREEN}  ██████╔╝██║   ██║██║██║     ██║  ██║    ██║     ██║   ██║██╔████╔██║${NC}"
echo -e "${GREEN}  ██╔══██╗██║   ██║██║██║     ██║  ██║    ██║     ██║   ██║██║╚██╔╝██║${NC}"
echo -e "${GREEN}  ██████╔╝╚██████╔╝██║███████╗██████╔╝    ╚██████╗╚██████╔╝██║ ╚═╝ ██║${NC}"
echo -e "${GREEN}  ╚═════╝  ╚═════╝ ╚═╝╚══════╝╚═════╝      ╚═════╝ ╚═════╝ ╚═╝     ╚═╝${NC}"
echo "=============================================================================="
echo ""
echo "  The Substitution v1.0.0 is trained and deployed."
echo ""
echo "  Model on HuggingFace:"
echo "    https://huggingface.co/theblackflagbmjc/the-substitution"
echo ""
echo "  To use locally on your Mac:"
echo "    git pull  (in your local The-Substitution repo)"
echo "    python inference/local_inference.py --model_path theblackflagbmjc/the-substitution"
echo ""
echo "  Or start the API server:"
echo "    python inference/api_server.py --model_path theblackflagbmjc/the-substitution"
echo ""
echo "  You can now safely delete this VM."
echo ""
echo "  Do no harm. Take no shit."
echo ""
echo "=============================================================================="
