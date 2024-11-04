#!/bin/bash

# Navigate to the project directory
cd "/Users/brandonjeanpierre/Library/CloudStorage/OneDrive-SharedLibraries-BrandonMichaelJeanpierreCorporation/Lilith - CORE/Organizations/Shell Company rAwRxRaWrXD1/CORE/Security/Digital/The-Substitution/mergekit/"

# Remove the existing virtual environment if it exists
if [ -d ".venv" ]; then
    rm -rf .venv
fi

# Create a new virtual environment with Python 3.12
python3.13 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip and install necessary dependencies via pip
pip install --upgrade pip
brew update
sudo brew upgrade
pip install transformers
pip install torch
pip install safetensors

# Navigate to the directory containing the Python script
cd "mergekit/witchesbrew/"

# Run the Python script
python3 mergekit_template_slerp_20241103-1152.py