#!/usr/bin/python

# @title ## Run merge

# @markdown ### Runtime type
# @markdown Select your runtime (CPU, High RAM, GPU)

import os

# runtime = "CPU + High-RAM" # @param ["CPU", "CPU + High-RAM", "GPU"]

# @markdown ### Mergekit arguments
# @markdown Use the `main` branch by default, [`mixtral`](https://github.com/cg123/mergekit/blob/mixtral/moe.md) if you want to create a Mixture of Experts.

branch = "slerp" # @param ["main", "mixtral"]
# trust_remote_code = True # @param {type:"boolean"}

# Install mergekit
if branch == "main":
    os.popen('git clone https://github.com/theblackflagbmjc/The-Substitution.git')
    os.popen('cd mergekit && pip3 install -qqq -e . --progress-bar on')
elif branch == "slerp":
    os.popen('git clone -b slerp https://github.com/theblackflagbmjc/The-Substitution.git')
    os.popen('cd mergekit && pip3 install -qqq -e . --progress-bar on')
    os.popen('pip3 install -qqq -U transformers --progress-bar on')

# Save config as yaml file
with open('config.yaml', 'w', encoding="utf-8") as f:
    f.write(yaml_config)

# Base CLI
if branch == "main":
    os.popen('mergekit-yaml "/Users/brandonjeanpierre/Library/CloudStorage/OneDrive-SharedLibraries-BrandonMichaelJeanpierreCorporation/Lilith - CORE/Organizations/Shell Company rAwRxRaWrXD1/CORE/Security/Digital/The-Substitution/mergekit/mergekit/witchesbrew/slerp_20241103-0057.yml" "/Users/brandonjeanpierre/Library/CloudStorage/OneDrive-SharedLibraries-BrandonMichaelJeanpierreCorporation/Lilith - CORE/Organizations/Shell Company rAwRxRaWrXD1/CORE/Security/Digital/The-Substitution/mergekit/mergekit/witchesbrew" --allow-crimes --trust-remote-code --copy-tokenizer --out-shard-size 1B --clone-tensors -v')
elif branch == "slerp":
    os.popen('mergekit-yaml "/Users/brandonjeanpierre/Library/CloudStorage/OneDrive-SharedLibraries-BrandonMichaelJeanpierreCorporation/Lilith - CORE/Organizations/Shell Company rAwRxRaWrXD1/CORE/Security/Digital/The-Substitution/mergekit/mergekit/witchesbrew/slerp_20241103-0057.yml" "/Users/brandonjeanpierre/Library/CloudStorage/OneDrive-SharedLibraries-BrandonMichaelJeanpierreCorporation/Lilith - CORE/Organizations/Shell Company rAwRxRaWrXD1/CORE/Security/Digital/The-Substitution/mergekit/mergekit/witchesbrew" --allow-crimes --trust-remote-code --copy-tokenizer --out-shard-size 1B --clone-tensors -v')

# Additional arguments
# if runtime == "CPU":
    # cli += " --allow-crimes --out-shard-size 1B --trust-remote-code"
# elif runtime == "GPU":
    # cli += " --cuda --low-cpu-memory"

print(os.popen())

# Merge models
{os.popen()}