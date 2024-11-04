#!/usr/bin/python

import os
import subprocess
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration for paths and file names
mergekit_path = "/Users/brandonjeanpierre/Library/CloudStorage/OneDrive-SharedLibraries-BrandonMichaelJeanpierreCorporation/Lilith - CORE/Organizations/Shell Company rAwRxRaWrXD1/CORE/Security/Digital/The-Substitution/mergekit/mergekit/witchesbrew/slerp_20241103-0057.yml"
access_token_file = os.path.join(mergekit_path, "api/hftk.txt")
yaml_config_file = os.path.join(mergekit_path, "witchesbrew/slerp_20241103-0057.yml")
mergekit_outpath = os.path.join(mergekit_path, "outhouse")

def ensure_file_exists():
        ensure_file_exists(access_token_file, "Access token file") # Ensure the access token file exists

def execute_command(command, env, description):
        try:
            subprocess.run(command, shell=True, env=env, check=True)
        except subprocess.C:    
            with open(access_token_file, encoding="utf-8") as f: # Retrieve Access Token from the file
                access_token = f.read().strip()

    # Login to Hugging Face
        execute_command(f'huggingface-cli login --token {access_token} --add-to-git-credential', env, "Hugging Face login")


# Set up environment variables for subprocess
env = os.environ.copy()
env["PATH"] = f"/Users/brandonjeanpierre/Library/CloudStorage/OneDrive-SharedLibraries-BrandonMichaelJeanpierreCorporation/Lilith - CORE/Organizations/Shell Company rAwRxRaWrXD1/CORE/Security/Digital/The-Substitution/mergekit/.venv/bin:" + env["PATH"]

# Ensure transformers is installed
execute_command('pip install transformers', env, "verifying installation of transformers")
# Ensure torch is installed
execute_command('pip install torch', env, "verifying installation of torch")
# Ensure safetensors is installed
execute_command('pip install safetensors', env, "verifying installation of safetensors")
# Ensure yq is installed
execute_command('pipx install yq', env, "verifying installation of yq for yaml parsing prior to installation of mergekit")

# Ensure mergekit is installed using pipx
execute_command(f'cd {mergekit_path}', 'pipx install -e .', env=env, check=True)

# Define function to read YAML configuration file
def ensure_file_exists(file_path, file_description):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_description} not found: {file_path}")

def main():            
    # Read YAML configuration file
    ensure_file_exists(yaml_config_file, "YAML configuration file")
    with open(yaml_config_file, encoding="utf-8") as f:
        yaml_config = yaml.safe_load(f)
        branch = 'merge_method'

    # Print loaded YAML configuration for debugging
    print("Loaded YAML Config:", yaml_config)
    
    # Construct the mergekit command based on the branch
    mergekit_command = ''
    if branch == "main":
        mergekit_command = f'mergekit-yaml "{yaml_config_file}" "{mergekit_outpath}" --allow-crimes --trust-remote-code --copy-tokenizer --out-shard-size 1B --clone-tensors -v'
    elif branch == "slerp":
        mergekit_command = f'mergekit-yaml "{yaml_config_file}" "{mergekit_outpath}" --lazy-unpickle --cuda --allow-crimes --trust-remote-code --copy-tokenizer --out-shard-size 1B --clone-tensors -v'

    # Set up environment variables for subprocess
    env = os.environ.copy()
    env["PATH"] = f"/Users/brandonjeanpierre/Library/CloudStorage/OneDrive-SharedLibraries-BrandonMichaelJeanpierreCorporation/Lilith - CORE/Organizations/Shell Company rAwRxRaWrXD1/CORE/Security/Digital/The-Substitution/mergekit/.venv/bin:" + env["PATH"]
    
    # Execute the mergekit command
    if mergekit_command:
        execute_command(mergekit_command, env, "mergekit command execution")

    # Main execution function
    if __name__ == "__main__":
        execute_command('pipx uninstall mergekit', env, "cleaning up installation of mergekit")
        execute_command('pipx uninstall yq', env, "cleaning up installation of yq")
        execute_command('deactivate', env, "deactivating python virtual environment")
        execute_command('sudo rm -fvr .venv', env, "cleaning up python virtual environment")
        execute_command('pip uninstall -y safetensors', env, "cleaning up installation of safetensors")
        execute_command('pip uninstall -y torch', env, "cleaning up installation of pytorch")
        execute_command('pip uninstall -y transformers', env, "cleaning up installation of transformers")
        print("I need a fuckin' Slurpee up this bitch!")