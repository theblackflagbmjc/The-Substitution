#!/usr/bin/env bash
# ==============================================================================
# The Substitution v1.0.0 — macOS (Apple Silicon) Environment Installer
# Brandon Michael Jeanpierre Corporation d/b/a The Black Flag
# Organization: theblackflagbmjc
#
# Target: macOS 26.x+ on Apple Silicon (M1 Pro / M1 Max / M2 / M3 / M4)
# Baseline: MacBook Pro M1Pro 2021, 32GB RAM, 1TB storage
#
# Usage:
#   bash bootstrap/install_macos.sh
#
# This script installs everything from scratch. No prior ML experience needed.
# ==============================================================================

set -euo pipefail

# --- Configuration ---
PYTHON_VERSION="3.11"
VENV_NAME="the-substitution-env"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/${VENV_NAME}"
LOG_FILE="${PROJECT_ROOT}/bootstrap/install_macos.log"

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

# --- Pre-flight Checks ---
log "The Substitution v1.0.0 — macOS Installer"
log "Organization: theblackflagbmjc"
log "Target: Apple Silicon macOS"
log "Project root: ${PROJECT_ROOT}"
echo ""

# Verify Apple Silicon
ARCH=$(uname -m)
if [[ "${ARCH}" != "arm64" ]]; then
    fail "This installer requires Apple Silicon (arm64). Detected: ${ARCH}"
fi
success "Apple Silicon (arm64) detected"

# Verify macOS
OS_NAME=$(uname -s)
if [[ "${OS_NAME}" != "Darwin" ]]; then
    fail "This installer requires macOS. Detected: ${OS_NAME}"
fi
success "macOS detected"

# --- Step 1: Install Homebrew ---
log "Step 1/7: Checking Homebrew..."
if ! command -v brew &>/dev/null; then
    log "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
    success "Homebrew installed"
else
    success "Homebrew already installed: $(brew --version | head -1)"
fi

# --- Step 2: Install System Dependencies ---
log "Step 2/7: Installing system dependencies..."
brew install \
    python@${PYTHON_VERSION} \
    cmake \
    git \
    git-lfs \
    wget \
    openssl@3 \
    protobuf \
    rust \
    2>&1 | tee -a "${LOG_FILE}"

# Initialize git-lfs (required for large model files)
git lfs install 2>&1 | tee -a "${LOG_FILE}"
success "System dependencies installed"

# --- Step 3: Create Python Virtual Environment ---
log "Step 3/7: Creating Python virtual environment..."
PYTHON_BIN=$(brew --prefix python@${PYTHON_VERSION})/bin/python${PYTHON_VERSION}

if [[ ! -f "${PYTHON_BIN}" ]]; then
    fail "Python ${PYTHON_VERSION} not found at ${PYTHON_BIN}"
fi

if [[ -d "${VENV_PATH}" ]]; then
    warn "Virtual environment already exists at ${VENV_PATH}. Removing..."
    rm -rf "${VENV_PATH}"
fi

"${PYTHON_BIN}" -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
success "Virtual environment created at ${VENV_PATH}"

# Upgrade pip and core tools
pip install --upgrade pip setuptools wheel 2>&1 | tee -a "${LOG_FILE}"
success "pip, setuptools, wheel upgraded"

# --- Step 4: Install PyTorch (Apple Silicon MPS) ---
log "Step 4/7: Installing PyTorch with MPS (Metal Performance Shaders) support..."
pip install \
    torch \
    torchvision \
    torchaudio \
    2>&1 | tee -a "${LOG_FILE}"
success "PyTorch installed with MPS backend"

# --- Step 5: Install Hugging Face Ecosystem ---
log "Step 5/7: Installing Hugging Face libraries and ML dependencies..."
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

# --- Step 6: Install Remaining Dependencies ---
log "Step 6/7: Installing remaining project dependencies..."

# Note: bitsandbytes has limited macOS support; we install but training
# will use MPS-native quantization or CPU fallback as needed.
pip install -r "${PROJECT_ROOT}/requirements.txt" 2>&1 | tee -a "${LOG_FILE}" || {
    warn "Some packages may have failed. Attempting critical-only install..."
    pip install \
        wandb tensorboard scipy scikit-learn \
        fastapi uvicorn pydantic \
        pandas pyarrow tqdm \
        pyyaml jsonlines requests python-dotenv psutil rich \
        rouge-score nltk \
        2>&1 | tee -a "${LOG_FILE}"
}
success "Project dependencies installed"

# --- Step 7: Install Hugging Face CLI and Login ---
log "Step 7/7: Configuring Hugging Face CLI..."
pip install huggingface_hub[cli] 2>&1 | tee -a "${LOG_FILE}"

echo ""
log "Hugging Face authentication:"
log "Run the following command to log in:"
echo ""
echo -e "  ${CYAN}huggingface-cli login${NC}"
echo ""
log "You will need your Hugging Face access token."
log "Get your token at: https://huggingface.co/settings/tokens"
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
echo "MPS available: $(python -c 'import torch; print(torch.backends.mps.is_available())')"
ACTIVATE_EOF
chmod +x "${PROJECT_ROOT}/activate.sh"
success "Activation script created: source activate.sh"

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

# --- Prepare Data Directories ---
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
echo -e "${GREEN}  The Substitution v1.0.0 — macOS Installation Complete${NC}"
echo "=============================================================================="
echo ""
echo "  Project root:     ${PROJECT_ROOT}"
echo "  Virtual env:      ${VENV_PATH}"
echo "  Python:           $(python --version 2>&1)"
echo "  PyTorch:          $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "  MPS available:    $(python -c 'import torch; print(torch.backends.mps.is_available())' 2>&1)"
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
