#!/bin/zsh

# Navigate to the project directory
cd "/Users/brandonjeanpierre/Library/CloudStorage/OneDrive-SharedLibraries-BrandonMichaelJeanpierreCorporation/Lilith - CORE/Organizations/Shell Company rAwRxRaWrXD1/CORE/Security/Digital/The-Substitution/mergekit/"
# Remove the existing virtual environment if it exists
if [ -d ".venv" ]; then
    rm -rf .venv
fi
python3.12 -m venv .venv # Create a new virtual environment with Python 3.12
source .venv/bin/activate # Activate the virtual environment
pip install --upgrade pip # Upgrade pip and install necessary dependencies via pip
pip install pyyaml torch transformers safetensors
pipx upgrade-all
brew update && brew upgrade
sleep 10
cd "/Users/brandonjeanpierre/Library/CloudStorage/OneDrive-SharedLibraries-BrandonMichaelJeanpierreCorporation/Lilith - CORE/Organizations/Shell Company rAwRxRaWrXD1/CORE/Security/Digital/The-Substitution/mergekit/mergekit/witchesbrew/" # Navigate to the directory containing the Python script
python3 mergekit_template_slerp_20241103-1152.py # Run the Python script
sleep 30
# EOF