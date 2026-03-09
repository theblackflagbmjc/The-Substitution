#!/usr/bin/env bash
# ==============================================================================
# The Substitution v1.0.0 — Environment Verification
# Brandon Michael Jeanpierre Corporation d/b/a The Black Flag
#
# Usage:
#   bash bootstrap/verify_environment.sh
#
# Validates that all required components are properly installed and functional.
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

check_pass() {
    echo -e "  ${GREEN}✓${NC} $1"
    ((PASS++))
}

check_fail() {
    echo -e "  ${RED}✖${NC} $1"
    ((FAIL++))
}

check_warn() {
    echo -e "  ${YELLOW}⚠${NC} $1"
    ((WARN++))
}

echo ""
echo "=============================================================================="
echo -e "${CYAN}  The Substitution v1.0.0 — Environment Verification${NC}"
echo "=============================================================================="
echo ""

# --- Section 1: Python Environment ---
echo -e "${CYAN}[1/6] Python Environment${NC}"

if command -v python &>/dev/null; then
    PY_VER=$(python --version 2>&1)
    check_pass "Python installed: ${PY_VER}"
else
    check_fail "Python not found in PATH"
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    check_pass "Virtual environment active: ${VIRTUAL_ENV}"
else
    check_warn "No virtual environment active. Run: source activate.sh"
fi

if command -v pip &>/dev/null; then
    PIP_VER=$(pip --version 2>&1 | awk '{print $2}')
    check_pass "pip installed: ${PIP_VER}"
else
    check_fail "pip not found"
fi
echo ""

# --- Section 2: PyTorch and Compute ---
echo -e "${CYAN}[2/6] PyTorch and Compute Backend${NC}"

python << 'PYTORCH_CHECK'
import sys

try:
    import torch
    print(f"  \033[0;32m✓\033[0m PyTorch installed: {torch.__version__}")
except ImportError:
    print(f"  \033[0;31m✖\033[0m PyTorch not installed")
    sys.exit(1)

# Check compute backends
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"  \033[0;32m✓\033[0m CUDA available: {gpu_name} ({vram:.1f} GB VRAM)")
    print(f"  \033[0;32m✓\033[0m CUDA version: {torch.version.cuda}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f"  \033[0;32m✓\033[0m MPS (Apple Silicon Metal) available")
else:
    print(f"  \033[1;33m⚠\033[0m No GPU detected. CPU-only mode (training will be slow)")

# Quick tensor test
try:
    x = torch.randn(2, 3)
    y = torch.randn(3, 2)
    z = torch.mm(x, y)
    print(f"  \033[0;32m✓\033[0m Tensor operations functional")
except Exception as e:
    print(f"  \033[0;31m✖\033[0m Tensor operations failed: {e}")
PYTORCH_CHECK
echo ""

# --- Section 3: Hugging Face Ecosystem ---
echo -e "${CYAN}[3/6] Hugging Face Ecosystem${NC}"

python << 'HF_CHECK'
packages = {
    "transformers": "transformers",
    "datasets": "datasets",
    "tokenizers": "tokenizers",
    "accelerate": "accelerate",
    "peft": "peft",
    "trl": "trl",
    "huggingface_hub": "huggingface_hub",
    "safetensors": "safetensors",
    "sentencepiece": "sentencepiece",
}

for display_name, import_name in packages.items():
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "installed")
        print(f"  \033[0;32m✓\033[0m {display_name}: {ver}")
    except ImportError:
        print(f"  \033[0;31m✖\033[0m {display_name}: NOT INSTALLED")
HF_CHECK
echo ""

# --- Section 4: Training Dependencies ---
echo -e "${CYAN}[4/6] Training Dependencies${NC}"

python << 'TRAIN_CHECK'
deps = {
    "wandb": "wandb",
    "tensorboard": "tensorboard",
    "scipy": "scipy",
    "sklearn": "sklearn",
    "yaml": "yaml",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "rich": "rich",
    "psutil": "psutil",
}

for display_name, import_name in deps.items():
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "installed")
        print(f"  \033[0;32m✓\033[0m {display_name}: {ver}")
    except ImportError:
        print(f"  \033[0;31m✖\033[0m {display_name}: NOT INSTALLED")

# Check bitsandbytes separately (optional on macOS)
try:
    import bitsandbytes
    print(f"  \033[0;32m✓\033[0m bitsandbytes: {bitsandbytes.__version__}")
except ImportError:
    import platform
    if platform.system() == "Darwin":
        print(f"  \033[1;33m⚠\033[0m bitsandbytes: not available on macOS (expected)")
    else:
        print(f"  \033[0;31m✖\033[0m bitsandbytes: NOT INSTALLED (needed for QLoRA on Linux)")
TRAIN_CHECK
echo ""

# --- Section 5: API Server Dependencies ---
echo -e "${CYAN}[5/6] API Server Dependencies${NC}"

python << 'API_CHECK'
deps = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
}

for display_name, import_name in deps.items():
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "installed")
        print(f"  \033[0;32m✓\033[0m {display_name}: {ver}")
    except ImportError:
        print(f"  \033[0;31m✖\033[0m {display_name}: NOT INSTALLED")
API_CHECK
echo ""

# --- Section 6: Project Structure ---
echo -e "${CYAN}[6/6] Project Structure${NC}"

REQUIRED_DIRS=(
    "bootstrap"
    "datasets"
    "tokenizer"
    "training"
    "evaluation"
    "inference"
    "deployment"
    "model_card"
    "data/raw"
    "data/processed"
    "data/tokenized"
    "output"
    "checkpoints"
    "logs"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [[ -d "${PROJECT_ROOT}/${dir}" ]]; then
        check_pass "Directory exists: ${dir}/"
    else
        check_fail "Directory missing: ${dir}/"
    fi
done

REQUIRED_FILES=(
    "requirements.txt"
    "pyproject.toml"
    ".env"
    "activate.sh"
    "bootstrap/install_macos.sh"
    "bootstrap/install_linux.sh"
    "bootstrap/verify_environment.sh"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ -f "${PROJECT_ROOT}/${file}" ]]; then
        check_pass "File exists: ${file}"
    else
        check_fail "File missing: ${file}"
    fi
done
echo ""

# --- Hugging Face Authentication ---
echo -e "${CYAN}[Bonus] Hugging Face Authentication${NC}"
python << 'AUTH_CHECK'
try:
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.whoami()
    print(f"  \033[0;32m✓\033[0m Authenticated as: {info.get('name', 'unknown')}")
    orgs = [o.get('name', '') for o in info.get('orgs', [])]
    if 'theblackflagbmjc' in orgs:
        print(f"  \033[0;32m✓\033[0m Organization access: theblackflagbmjc")
    else:
        print(f"  \033[1;33m⚠\033[0m Not a member of theblackflagbmjc org (or not logged in)")
except Exception as e:
    print(f"  \033[1;33m⚠\033[0m Not authenticated with Hugging Face. Run: huggingface-cli login")
AUTH_CHECK
echo ""

# --- Summary ---
echo "=============================================================================="
echo -e "  ${GREEN}Passed: ${PASS}${NC}  |  ${RED}Failed: ${FAIL}${NC}  |  ${YELLOW}Warnings: ${WARN}${NC}"

if [[ ${FAIL} -eq 0 ]]; then
    echo -e "  ${GREEN}Environment verification PASSED. Ready to proceed.${NC}"
else
    echo -e "  ${RED}Environment verification has failures. Fix issues above before proceeding.${NC}"
fi
echo "=============================================================================="
echo ""
