#!/usr/bin/env bash
# ==============================================================================
# The Substitution v1.0.0 — Train on HuggingFace (CLI Only)
# Brandon Michael Jeanpierre Corporation d/b/a The Black Flag
# Organization: theblackflagbmjc
#
# PURPOSE:
#   Trains The Substitution entirely on HuggingFace infrastructure.
#   No VM, no cloud console, no GUI. One command from your Mac.
#
#   Under the hood it uses HuggingFace AutoTrain, which spins up a
#   GPU-backed Space, trains the model, pushes it to your Hub repo,
#   and shuts down. You pay per-minute for the GPU.
#
# PREREQUISITES:
#   1. pip install autotrain-advanced
#   2. A HuggingFace token with WRITE access
#   3. A payment method on your HF account (Settings → Billing)
#   4. Training data already cleaned and prepared locally
#      (run the data pipeline first if you haven't)
#
# USAGE:
#   bash bootstrap/train_huggingface.sh
#
# WHAT HAPPENS:
#   1. Pushes your cleaned training data to HF as a private dataset
#   2. Writes an AutoTrain config targeting that dataset
#   3. Kicks off training on HF Spaces with an A100 GPU
#   4. Model is pushed to theblackflagbmjc/the-substitution on completion
#   5. The Space auto-pauses (billing stops)
#
# COST ESTIMATE:
#   A100 Large (~$6.30/hr):  ~18-24 hours  →  ~$115-150
#   A10G Large (~$3.15/hr):  ~36-48 hours  →  ~$115-150
#   L4x1 (~$1.05/hr):       ~48-72 hours  →  ~$50-75
#
#   L4x1 is the cheapest option. A100 is fastest.
#
# AFTER TRAINING:
#   python inference/local_inference.py --model_path theblackflagbmjc/the-substitution
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()     { echo -e "${CYAN}[$(date '+%H:%M:%S')] $1${NC}"; }
success() { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ $1${NC}"; }
warn()    { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠ $1${NC}"; }
fail()    { echo -e "${RED}[$(date '+%H:%M:%S')] ✖ $1${NC}"; exit 1; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo ""
echo "=============================================================================="
echo -e "${CYAN}  The Substitution v1.0.0 — Train on HuggingFace${NC}"
echo -e "${CYAN}  Brandon Michael Jeanpierre Corporation d/b/a The Black Flag${NC}"
echo "=============================================================================="
echo ""

# =============================================================================
# STEP 1: CHECK PREREQUISITES
# =============================================================================

log "Step 1/5: Checking prerequisites..."

# Check autotrain is installed
if ! command -v autotrain &>/dev/null; then
    log "Installing autotrain-advanced..."
    pip install autotrain-advanced -q
fi
AUTOTRAIN_VER=$(autotrain --version 2>&1 | head -1)
success "autotrain installed: ${AUTOTRAIN_VER}"

# Check HF token
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo ""
    echo -e "${YELLOW}  You need a HuggingFace token with WRITE access.${NC}"
    echo -e "${YELLOW}  Get one at: https://huggingface.co/settings/tokens${NC}"
    echo ""
    read -rp "  Paste your HF token: " HF_TOKEN
    echo ""
fi

if [[ -z "${HF_TOKEN}" ]]; then
    fail "No HF token provided."
fi

export HF_TOKEN
export HF_USERNAME="theblackflagbmjc"
success "HF token set"

# Check training data exists
TRAIN_DATA="${PROJECT_ROOT}/data/processed/final/train.jsonl"
if [[ ! -f "${TRAIN_DATA}" ]]; then
    fail "Training data not found at ${TRAIN_DATA}. Run the data pipeline first:
    python datasets/download_datasets.py
    python datasets/clean_datasets.py
    python datasets/prepare_training_data.py"
fi

TRAIN_LINES=$(wc -l < "${TRAIN_DATA}")
success "Training data found: ${TRAIN_LINES} examples"

# =============================================================================
# STEP 2: PUSH TRAINING DATA TO HF HUB
# =============================================================================

log "Step 2/5: Pushing training data to HuggingFace Hub..."

python3 << 'PUSH_DATA_EOF'
import json
import os
from pathlib import Path
from datasets import Dataset
from huggingface_hub import HfApi, create_repo

token = os.environ["HF_TOKEN"]
api = HfApi(token=token)

# Create private dataset repo
dataset_repo = "theblackflagbmjc/the-substitution-training-data"
try:
    create_repo(dataset_repo, repo_type="dataset", private=True, token=token, exist_ok=True)
    print(f"  Dataset repo: {dataset_repo} (private)")
except Exception as e:
    print(f"  Repo exists or created: {e}")

# Load the JSONL training data
train_path = Path("data/processed/final/train.jsonl")
val_path = Path("data/processed/final/validation.jsonl")

def load_jsonl_messages(path):
    """Load JSONL and keep messages as a column."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line.strip())
            rows.append({"messages": row["messages"]})
    return rows

print(f"  Loading training data...")
train_rows = load_jsonl_messages(train_path)
print(f"  Train examples: {len(train_rows)}")

train_ds = Dataset.from_list(train_rows)

# Push to hub
print(f"  Uploading to {dataset_repo}...")
train_ds.push_to_hub(dataset_repo, split="train", token=token, private=True)

if val_path.exists():
    val_rows = load_jsonl_messages(val_path)
    val_ds = Dataset.from_list(val_rows)
    val_ds.push_to_hub(dataset_repo, split="validation", token=token, private=True)
    print(f"  Validation examples: {len(val_rows)}")

print("  Dataset uploaded successfully.")
PUSH_DATA_EOF

success "Training data pushed to HuggingFace Hub"

# =============================================================================
# STEP 3: SELECT GPU BACKEND
# =============================================================================

log "Step 3/5: Select GPU backend..."
echo ""
echo "  Available backends (billed per-minute to your HF account):"
echo ""
echo "    1) spaces-l4x1        — NVIDIA L4 24GB       ~\$1.05/hr  (cheapest, ~48-72h)"
echo "    2) spaces-a10g-large  — NVIDIA A10G 24GB     ~\$3.15/hr  (balanced, ~36-48h)"
echo "    3) spaces-a100-large  — NVIDIA A100 80GB     ~\$6.30/hr  (fastest, ~18-24h)"
echo ""

read -rp "  Choose backend [1/2/3] (default: 3): " BACKEND_CHOICE
BACKEND_CHOICE="${BACKEND_CHOICE:-3}"

case "${BACKEND_CHOICE}" in
    1) BACKEND="spaces-l4x1";       GPU_NAME="L4 24GB" ;;
    2) BACKEND="spaces-a10g-large"; GPU_NAME="A10G 24GB" ;;
    3) BACKEND="spaces-a100-large"; GPU_NAME="A100 80GB" ;;
    *) BACKEND="spaces-a100-large"; GPU_NAME="A100 80GB" ;;
esac

success "Backend selected: ${BACKEND} (${GPU_NAME})"

# =============================================================================
# STEP 4: WRITE AUTOTRAIN CONFIG
# =============================================================================

log "Step 4/5: Writing AutoTrain configuration..."

# Determine batch size based on GPU
case "${BACKEND}" in
    spaces-l4x1)        BATCH_SIZE=1; GRAD_ACCUM=16; BLOCK=2048; MAX_LEN=4096 ;;
    spaces-a10g-large)  BATCH_SIZE=2; GRAD_ACCUM=8;  BLOCK=2048; MAX_LEN=4096 ;;
    spaces-a100-large)  BATCH_SIZE=4; GRAD_ACCUM=8;  BLOCK=4096; MAX_LEN=8192 ;;
esac

cat > "${PROJECT_ROOT}/autotrain_config.yaml" << AUTOTRAIN_EOF
# ==============================================================================
# The Substitution v1.0.0 — AutoTrain Config (HuggingFace Remote Training)
# Auto-generated by train_huggingface.sh
# ==============================================================================

task: llm-sft
base_model: Qwen/Qwen2.5-7B-Instruct
project_name: the-substitution

log: tensorboard
backend: ${BACKEND}

data:
  path: theblackflagbmjc/the-substitution-training-data
  train_split: train
  valid_split: validation
  chat_template: chatml
  column_mapping:
    text_column: messages

params:
  block_size: ${BLOCK}
  model_max_length: ${MAX_LEN}
  epochs: 3
  batch_size: ${BATCH_SIZE}
  lr: 2e-4
  peft: true
  quantization: int4
  target_modules: all-linear
  lora_r: 64
  lora_alpha: 128
  lora_dropout: 0.05
  padding: right
  optimizer: paged_adamw_8bit
  scheduler: cosine
  gradient_accumulation: ${GRAD_ACCUM}
  mixed_precision: bf16
  warmup_ratio: 0.03
  weight_decay: 0.01
  max_grad_norm: 1.0
  merge_adapter: true
  logging_steps: 25
  save_total_limit: 3
  eval_strategy: steps
  eval_steps: 1000

hub:
  username: \${HF_USERNAME}
  token: \${HF_TOKEN}
  push_to_hub: true
AUTOTRAIN_EOF

success "Config written: autotrain_config.yaml"

echo ""
echo "  Configuration summary:"
echo "    Base model:    Qwen/Qwen2.5-7B-Instruct"
echo "    Method:        QLoRA (4-bit) + LoRA (r=64, alpha=128, all-linear)"
echo "    Backend:       ${BACKEND} (${GPU_NAME})"
echo "    Epochs:        3"
echo "    Batch size:    ${BATCH_SIZE} × ${GRAD_ACCUM} grad_accum = $(( BATCH_SIZE * GRAD_ACCUM )) effective"
echo "    Block size:    ${BLOCK}"
echo "    Push to hub:   theblackflagbmjc/the-substitution"
echo ""

# =============================================================================
# STEP 5: LAUNCH TRAINING
# =============================================================================

log "Step 5/5: Launching training on HuggingFace..."
echo ""
echo "=============================================================================="
echo -e "${GREEN}  TRAINING IS ABOUT TO START ON HUGGINGFACE SERVERS.${NC}"
echo -e "${GREEN}  Billing begins when the Space boots. Monitor your spend at:${NC}"
echo -e "${GREEN}    https://huggingface.co/settings/billing${NC}"
echo ""
echo -e "${GREEN}  A HuggingFace Space will be created under your org.${NC}"
echo -e "${GREEN}  You can monitor logs at the Space URL that autotrain prints.${NC}"
echo ""
echo -e "${GREEN}  When training completes, the model is pushed to:${NC}"
echo -e "${GREEN}    https://huggingface.co/theblackflagbmjc/the-substitution${NC}"
echo ""
echo -e "${GREEN}  The Space auto-pauses after training. Billing stops.${NC}"
echo "=============================================================================="
echo ""

read -rp "  Confirm launch? [y/N]: " CONFIRM
if [[ "${CONFIRM}" != "y" && "${CONFIRM}" != "Y" ]]; then
    echo ""
    log "Aborted. Config saved at autotrain_config.yaml — launch manually with:"
    echo "    export HF_TOKEN=your_token"
    echo "    export HF_USERNAME=theblackflagbmjc"
    echo "    autotrain --config autotrain_config.yaml"
    exit 0
fi

echo ""
log "Launching autotrain..."
echo ""

# Run autotrain
autotrain --config "${PROJECT_ROOT}/autotrain_config.yaml"

echo ""
echo "=============================================================================="
echo -e "${GREEN}  Training launched on HuggingFace.${NC}"
echo ""
echo "  Monitor at the Space URL printed above."
echo "  When complete, use locally:"
echo ""
echo "    python inference/local_inference.py --model_path theblackflagbmjc/the-substitution"
echo ""
echo "  Do no harm. Take no shit."
echo "=============================================================================="
