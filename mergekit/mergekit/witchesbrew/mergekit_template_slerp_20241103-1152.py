#!/usr/bin/python

# @title ## Run merge

# @markdown ### Runtime type
# @markdown Select your runtime (CPU, High RAM, GPU)

import os
import yaml

# runtime = "CPU + High-RAM" # @param ["CPU", "CPU + High-RAM", "GPU"]

# @markdown ### Mergekit arguments
# @markdown Use the `main` branch by default, [`mixtral`](https://github.com/cg123/mergekit/blob/mixtral/moe.md) if you want to create a Mixture of Experts.

branch = "slerp" # @param ["main", "mixtral"]
# trust_remote_code = True # @param {type:"boolean"}
mergekit_path = "/Users/brandonjeanpierre/Library/CloudStorage/OneDrive-SharedLibraries-BrandonMichaelJeanpierreCorporation/Lilith - CORE/Organizations/Shell Company rAwRxRaWrXD1/CORE/Security/Digital/The-Substitution/mergekit/"
access_token_file = "mergekit_path/mergekit/witchesbrew/api/hftk.txt"
yaml_config_file = "mergekit_path/mergekit/witchesbrew/slerp_20241103-0057.yml"
mergekit_outpath = "mergekit_path/mergekit/witchesbrew/outhouse"

# Retreive Access Token from file
def access_token():
    with open(access_token_file, encoding="utf-8") as f:
        f.read()
        f.closed
        True # verify that the file has been automatically closed.

# Install mergekit
if branch == "main":
    os.popen('source .venv/bin/activate')
    os.popen('cd "mergekit_path"')
    os.popen('pipx install -e .')
    os.popen('huggingface-cli login --token access_token --add-to-git-credential')
    # os.popen('git clone https://github.com/theblackflagbmjc/The-Substitution.git')
    os.popen('cd mergekit && pip3 install -qqq -e . --progress-bar on')
elif branch == "slerp":
    os.popen('source .venv/bin/activate')
    os.popen('cd "mergekit_path"')
    os.popen('pipx install -e .')
    os.popen('huggingface-cli login --token access_token --add-to-git-credential')
    # os.popen('git clone -b slerp https://github.com/theblackflagbmjc/The-Substitution.git')
    os.popen('cd mergekit && pip3 install -qqq -e . --progress-bar on')
    os.popen('pip3 install -qqq -U transformers --progress-bar on')

# Read config as yaml file
# with open('yaml_config', 'r', encoding="utf-8") as f:
#     f.read(yaml_config)

# Read config as yaml file
def yaml_config():
    with open(yaml_config_file, encoding="utf-8") as f:
        f.read()
        f.closed
        True # verify that the file has been automatically closed.

# Give birth to Frankenstein with mergekit command in CLI.
if branch == "main":
    os.popen('mergekit-yaml "yaml_config" "mergekit_outpath" --allow-crimes --trust-remote-code --copy-tokenizer --out-shard-size 1B --clone-tensors -v')
elif branch == "slerp":
    os.popen('mergekit-yaml "yaml_config" "mergekit_outpath" --lazy-unpickle --cuda --allow-crimes --trust-remote-code --copy-tokenizer --out-shard-size 1B --clone-tensors --v')

# Additional arguments
# if runtime == "CPU":
    # cli += " --allow-crimes --out-shard-size 1B --trust-remote-code"
# elif runtime == "GPU":
    # cli += " --cuda --low-cpu-memory"

# Merge models
# {os.popen('mergekit-yaml "/Users/brandonjeanpierre/Library/CloudStorage/OneDrive-SharedLibraries-BrandonMichaelJeanpierreCorporation/Lilith - CORE/Organizations/Shell Company rAwRxRaWrXD1/CORE/Security/Digital/The-Substitution/mergekit/mergekit/witchesbrew/slerp_20241103-0057.yml" "/Users/brandonjeanpierre/Library/CloudStorage/OneDrive-SharedLibraries-BrandonMichaelJeanpierreCorporation/Lilith - CORE/Organizations/Shell Company rAwRxRaWrXD1/CORE/Security/Digital/The-Substitution/mergekit/mergekit/witchesbrew" --allow-crimes --trust-remote-code --copy-tokenizer --out-shard-size 1B --clone-tensors -v')}

if __name__ == "__main__":
    print("I need a fuckin' Slurpee up this bitch!")