# !/usr/bin/python3
# @title ## Run merge

# @markdown ### Runtime type
# @markdown Select your runtime (CPU, High RAM, GPU)
import os

runtime = "CPU + High-RAM" # @param ["CPU", "CPU + High-RAM", "GPU"]

# @markdown ### Mergekit arguments
# @markdown Use the `main` branch by default, [`mixtral`](https://github.com/cg123/mergekit/blob/mixtral/moe.md) if you want to create a Mixture of Experts.

branch = "mixtral" # @param ["main", "mixtral"]
trust_remote_code = True # @param {type:"boolean"}
yaml_config = "/Users/brandonjeanpierre/Library/CloudStorage/OneDrive-SharedLibraries-BrandonMichaelJeanpierreCorporation/Lilith - CORE/Organizations/Shell Company rAwRxRaWrXD1/CORE/Resources/CODE/Utilities/The Substitution/witchesbrew/slerp_20241103-0057.yml"

# Install mergekit
if branch == "main":
    os.popen ("git clone https://github.com/arcee-ai/mergekit.git")
    os.popen ("cd mergekit && pipx install -qqq -e . --progress-bar off")
elif branch == "mixtral":
    os.popen ("git clone -b mixtral https://github.com/arcee-ai/mergekit.git")
    os.popen ("cd mergekit && pipx install -qqq -e . --progress-bar off")
    os.popen ("pipx install -qqq -U transformers --progress-bar off")

# Save config as yaml file
with open('config.yaml', 'w', encoding="utf-8") as f:
    f.write(yaml_config)

# Base CLI
if branch == "main":
    cli = "mergekit-yaml config.yaml merge --copy-tokenizer"
elif branch == "mixtral":
    cli = "mergekit-moe config.yaml merge --copy-tokenizer"

# Additional arguments
if runtime == "CPU":
    cli += " --allow-crimes --out-shard-size 1B --lazy-unpickle"
elif runtime == "GPU":
    cli += " --cuda --low-cpu-memory"
if trust_remote_code:
    cli += " --trust-remote-code"

print(cli)

# Merge models
{cli}