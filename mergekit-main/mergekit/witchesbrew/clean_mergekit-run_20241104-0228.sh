#!/bin/zsh

# Display initial working directory.
pwd
# Navigate to deployment path.
cd "/Users/brandonjeanpierre/Library/CloudStorage/OneDrive-SharedLibraries-BrandonMichaelJeanpierreCorporation/Lilith - CORE/Organizations/Shell Company rAwRxRaWrXD1/CORE/Security/Digital/The-Substitution/mergekit-main/"
# Display new working directory.
pwd

# Remove the existing virtual environment if it exists
if [ -d ".venv" ]; then
    deactivate && rm -rf .venv
fi

echo "cleaning up python virtual environment" 

# Uninstall all dependencies installed on previous run in environment to maintain consistency for subsequent runs.
pipx uninstall -y mergekit
echo "cleaning up installation of mergekit"
pip uninstall -y pyyaml
echo "cleaning up installation of pyyaml" 
deactivate
echo "deactivating python virtual environment" 
# sudo rm -fvr .venv
# echo "cleaning up python virtual environment" 
pip uninstall -y safetensors
echo "cleaning up installation of safetensors"
pip uninstall -y torch
echo "cleaning up installation of pytorch"
pip uninstall -y transformers
echo "cleaning up installation of transformers"
pip cache purge
echo "I need a fuckin' Slurpee up this bitch!"

# Navigate to a standard path outside of poroject environment and clear output.
cd ~/Downloads
clear

# Pause 3 Seconds before termination prior to termination.
sleep 3

# EOF