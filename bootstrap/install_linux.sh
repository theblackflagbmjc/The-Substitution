#!/usr/bin/env bash
# ==============================================================================
# The Substitution v1.0.0 — Linux (Ubuntu) Environment Installer
# Brandon Michael Jeanpierre Corporation d/b/a The Black Flag
# Organization: theblackflagbmjc
#
# Target: Ubuntu 22.04+ / Debian 12+
# Supports: NVIDIA GPU (CUDA), AMD GPU (ROCm), CPU-only
#
# Usage:
#   bash bootstrap/install_linux.sh
#
# This script installs everything from scratch. No prior ML experience needed.
# ==============================================================================

set -euo pipefail

# --- Configuration ---
PYTHON_VERSION="3.11"
VENV_NAME="the-substitution-env"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/${VENV_NAME}"
LOG_FILE="${PROJECT_ROOT}/bootstrap/install_linux.log"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "${CYAN}${msg}${NC}"
    echo "${msg}" >> "${LOG_FILE}"
}

success() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $1"
    echo -e "${GREEN}${msg}${NC}"
    echo "${msg}" >> "${LOG_FILE}"
}

warn() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ $1"
    echo -e "${YELLOW}${msg}${NC}"
    echo "${msg}" >> "${LOG_FILE}"
}

fail() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ✖ $1"
    echo -e "${RED}${msg}${NC}"
    echo "${msg}" >> "${LOG_FILE}"
    exit 1
}

# --- Pre-flight ---
log "The Substitution v1.0.0 — Linux Installer"
log "Organization: theblackflagbmjc"
log "Project root: ${PROJECT_ROOT}"
echo ""

OS_NAME=$(uname -s)
if [[ "${OS_NAME}" != "Linux" ]]; then
    fail "This installer requires Linux. Detected: ${OS_NAME}"
fi
success "Linux detected"

# Detect GPU
GPU_TYPE="cpu"
if command -v nvidia-smi &>/dev/null; then
    GPU_TYPE="cuda"
    NVIDIA_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "unknown")
    success "NVIDIA GPU detected: ${NVIDIA_INFO}"
elif [[ -d /opt/rocm ]] || command -v rocminfo &>/dev/null; then
    GPU_TYPE="rocm"
    success "AMD GPU (ROCm) detected"
else
    warn "No GPU detected. Training will use CPU (slow but functional)."
fi
log "Compute backend: ${GPU_TYPE}"

# --- Step 1: Install System Dependencies ---
log "Step 1/7: Installing system dependencies..."
sudo apt-get update 2>&1 | tee -a "${LOG_FILE}"
sudo apt-get install -y \
    software-properties-common \
    build-essential \
    cmake \
    git \
    git-lfs \
    wget \
    curl \
    unzip \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    liblzma-dev \
    zlib1g-dev \
    protobuf-compiler \
    2>&1 | tee -a "${LOG_FILE}"

git lfs install 2>&1 | tee -a "${LOG_FILE}"
success "System dependencies installed"

# --- Step 2: Install Python ---
log "Step 2/7: Installing Python ${PYTHON_VERSION}..."
if ! command -v python${PYTHON_VERSION} &>/dev/null; then
    sudo add-apt-repository -y ppa:deadsnakes/ppa 2>&1 | tee -a "${LOG_FILE}"
    sudo apt-get update 2>&1 | tee -a "${LOG_FILE}"
    sudo apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
        2>&1 | tee -a "${LOG_FILE}"
fi
success "Python ${PYTHON_VERSION} installed: $(python${PYTHON_VERSION} --version)"

# --- Step 3: Create Virtual Environment ---
log "Step 3/7: Creating Python virtual environment..."
if [[ -d "${VENV_PATH}" ]]; then
    warn "Virtual environment already exists. Removing..."
    rm -rf "${VENV_PATH}"
fi

python${PYTHON_VERSION} -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
pip install --upgrade pip setuptools wheel 2>&1 | tee -a "${LOG_FILE}"
success "Virtual environment created at ${VENV_PATH}"

# --- Step 4: Install PyTorch ---
log "Step 4/7: Installing PyTorch (backend: ${GPU_TYPE})..."
case "${GPU_TYPE}" in
    cuda)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
            2>&1 | tee -a "${LOG_FILE}"
        ;;
    rocm)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2 \
            2>&1 | tee -a "${LOG_FILE}"
        ;;
    cpu)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
            2>&1 | tee -a "${LOG_FILE}"
        ;;
esac
success "PyTorch installed (${GPU_TYPE})"

# --- Step 5: Install Hugging Face Ecosystem ---
log "Step 5/7: Installing Hugging Face libraries..."
pip install \
    transformers \
    datasets \
    tokenizers \
    accelerate \
    peft \
    trl \
    huggingface_hub \
    safetensors \
    sentencepiece \
    2>&1 | tee -a "${LOG_FILE}"
success "Hugging Face ecosystem installed"

# --- Step 6: Install All Project Dependencies ---
log "Step 6/7: Installing project dependencies..."

# bitsandbytes works natively on Linux with CUDA
if [[ "${GPU_TYPE}" == "cuda" ]]; then
    pip install bitsandbytes 2>&1 | tee -a "${LOG_FILE}"
    success "bitsandbytes installed (CUDA)"
else
    warn "bitsandbytes skipped (requires CUDA). QLoRA will use alternative quantization."
fi

pip install -r "${PROJECT_ROOT}/requirements.txt" 2>&1 | tee -a "${LOG_FILE}" || {
    warn "Some optional packages failed. Installing critical dependencies..."
    pip install \
        wandb tensorboard scipy scikit-learn \
        fastapi uvicorn pydantic \
        pandas pyarrow tqdm \
        pyyaml jsonlines requests python-dotenv psutil rich \
        rouge-score nltk \
        2>&1 | tee -a "${LOG_FILE}"
}
success "Project dependencies installed"

# --- Step 7: Configure Hugging Face CLI ---
log "Step 7/7: Configuring Hugging Face CLI..."
pip install huggingface_hub[cli] 2>&1 | tee -a "${LOG_FILE}"

echo ""
log "Hugging Face authentication:"
log "Run: huggingface-cli login"
log "Token: https://huggingface.co/settings/tokens"
echo ""

# --- Create Activation Script ---
cat > "${PROJECT_ROOT}/activate.sh" << 'ACTIVATE_EOF'
#!/usr/bin/env bash
# Activate The Substitution development environment
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_ROOT}/the-substitution-env/bin/activate"
export SUBSTITUTION_ROOT="${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
echo "The Substitution v1.0.0 environment activated."
echo "Project root: ${PROJECT_ROOT}"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# Detect compute backend
python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS (Apple Silicon) available')
else:
    print('CPU-only mode')
"
ACTIVATE_EOF
chmod +x "${PROJECT_ROOT}/activate.sh"
success "Activation script created"

# --- Create .env Template ---
cat > "${PROJECT_ROOT}/.env" << 'ENV_EOF'
# ==============================================================================
# The Substitution v1.0.0 — Environment Configuration
# ==============================================================================

# Hugging Face
HF_TOKEN=your_token_here
HF_ORGANIZATION=theblackflagbmjc
HF_MODEL_NAME=the-substitution

# Base Model
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct

# Training
WANDB_PROJECT=the-substitution
WANDB_ENTITY=theblackflagbmjc

# Paths
DATA_DIR=./data
OUTPUT_DIR=./output
CHECKPOINT_DIR=./checkpoints
LOG_DIR=./logs
ENV_EOF
success ".env template created"

# --- Prepare Directories ---
mkdir -p "${PROJECT_ROOT}/data/raw"
mkdir -p "${PROJECT_ROOT}/data/processed"
mkdir -p "${PROJECT_ROOT}/data/tokenized"
mkdir -p "${PROJECT_ROOT}/output"
mkdir -p "${PROJECT_ROOT}/checkpoints"
mkdir -p "${PROJECT_ROOT}/logs"
success "Data directories created"

# --- Summary ---
echo ""
echo "=============================================================================="
echo -e "${GREEN}  The Substitution v1.0.0 — Linux Installation Complete${NC}"
echo "=============================================================================="
echo ""
echo "  Project root:     ${PROJECT_ROOT}"
echo "  Virtual env:      ${VENV_PATH}"
echo "  Python:           $(python --version 2>&1)"
echo "  PyTorch:          $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "  Compute backend:  ${GPU_TYPE}"

if [[ "${GPU_TYPE}" == "cuda" ]]; then
    echo "  CUDA version:     $(python -c 'import torch; print(torch.version.cuda)' 2>&1)"
    echo "  GPU:              $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
fi

echo "  Transformers:     $(python -c 'import transformers; print(transformers.__version__)' 2>&1)"
echo ""
echo "  Next steps:"
echo "    1. source activate.sh"
echo "    2. huggingface-cli login"
echo "    3. Edit .env with your tokens"
echo "    4. bash bootstrap/verify_environment.sh"
echo ""
echo "  Organization:     theblackflagbmjc"
echo "  Base model:       Qwen/Qwen2.5-7B-Instruct"
echo "=============================================================================="
