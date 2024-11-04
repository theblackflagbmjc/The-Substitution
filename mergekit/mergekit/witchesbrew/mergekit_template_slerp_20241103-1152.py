#!/usr/bin/python

import os
import subprocess
import logging
import sys
import yaml

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration for paths and file names
mergekit_path = "/Users/brandonjeanpierre/Library/CloudStorage/OneDrive-SharedLibraries-BrandonMichaelJeanpierreCorporation/Lilith - CORE/Organizations/Shell Company rAwRxRaWrXD1/CORE/Security/Digital/The-Substitution/mergekit/"
access_token_file = os.path.join(mergekit_path, "api/hftk.txt")
yaml_config_file = os.path.join(mergekit_path, "witchesbrew/slerp_20241103-0057.yml")
mergekit_outpath = os.path.join(mergekit_path, "outhouse")

def ensure_file_exists(file_path, file_description):
    """ Ensure that a file exists or raise an error. """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_description} not found: {file_path}")

def execute_command(command, env, description):
    """ Execute a shell command with error handling. """
    try:
        subprocess.run(command, shell=True, env=env, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to execute {description}: {e}")
        sys.exit(1)

def main():
    # Ensure the access token file exists
    ensure_file_exists(access_token_file, "Access token file")

    # Retrieve Access Token from the file
    with open(access_token_file, encoding="utf-8") as f:
        access_token = f.read().strip()

    # Set up environment variables for subprocess
    env = os.environ.copy()
    env["PATH"] = f"/Users/brandonjeanpierre/Library/CloudStorage/OneDrive-SharedLibraries-BrandonMichaelJeanpierreCorporation/Lilith - CORE/Organizations/Shell Company rAwRxRaWrXD1/CORE/Security/Digital/The-Substitution/mergekit/.venv/bin:" + env["PATH"]

    # Login to Hugging Face
    execute_command(f'huggingface-cli login --token {access_token} --add-to-git-credential', env, "Hugging Face login")

    # Ensure necessary packages are installed
    execute_command('pip install transformers', env, "verifying installation of transformers")
    execute_command('pip install torch', env, "verifying installation of torch")
    execute_command('pip install safetensors', env, "verifying installation of safetensors")
    execute_command('pip install pyyaml', env, "verifying installation of pyyaml for yaml parsing")

    # Ensure mergekit is installed using pip
    execute_command(f'pip install -e {mergekit_path}', env, "installing mergekit")

    # Read YAML configuration file
    ensure_file_exists(yaml_config_file, "YAML configuration file")
    with open(yaml_config_file, encoding="utf-8") as f:
        yaml_config = yaml.safe_load(f)

    # Print loaded YAML configuration for debugging
    logging.info("Loaded YAML Config: %s", yaml_config)

    # Determine the branch and construct the mergekit command
    branch = yaml_config.get('merge_method', 'slerp')  # Default to 'slerp' if not present
    mergekit_command = ''
    if branch == "main":
        mergekit_command = f'mergekit-yaml "{yaml_config_file}" "{mergekit_outpath}" --allow-crimes --trust-remote-code --copy-tokenizer --out-shard-size 1B --clone-tensors -v'
    elif branch == "slerp":
        mergekit_command = f'mergekit-yaml "{yaml_config_file}" "{mergekit_outpath}" --lazy-unpickle --cuda --allow-crimes --trust-remote-code --copy-tokenizer --out-shard-size 1B --clone-tensors -v'

    # Execute the mergekit command
    if mergekit_command:
        execute_command(mergekit_command, env, "mergekit command execution")

    # Main execution function
    if __name__ == "__main__":
        execute_command('pip uninstall -y mergekit', env, "cleaning up installation of mergekit")
        execute_command('pip uninstall -y pyyaml', env, "cleaning up installation of pyyaml")
        execute_command('deactivate', env, "deactivating python virtual environment")
        execute_command('sudo rm -fvr .venv', env, "cleaning up python virtual environment")
        execute_command('pip uninstall -y safetensors', env, "cleaning up installation of safetensors")
        execute_command('pip uninstall -y torch', env, "cleaning up installation of pytorch")
        execute_command('pip uninstall -y transformers', env, "cleaning up installation of transformers")
        print("I need a fuckin' Slurpee up this bitch!")
        logging.info("I need a Slurpee up this bitch!")