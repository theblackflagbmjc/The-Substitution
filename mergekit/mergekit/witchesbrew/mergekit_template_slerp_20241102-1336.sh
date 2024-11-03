# @title ## Run merge

# @markdown ### Runtime type
# @markdown Select your runtime (CPU, High RAM, GPU)

runtime = "CPU + High-RAM" # @param ["CPU", "CPU + High-RAM", "GPU"]

# @markdown ### Mergekit arguments
# @markdown Use the `main` branch by default, [`mixtral`](https://github.com/cg123/mergekit/blob/mixtral/moe.md) if you want to create a Mixture of Experts.

branch = "mixtral" # @param ["main", "mixtral"]
trust_remote_code = True # @param {type:"boolean"}

# Install mergekit
if branch == "main":
    !git clone https://github.com/arcee-ai/mergekit.git
    !cd mergekit && pip install -qqq -e . --progress-bar off
elif branch == "mixtral":
    !git clone -b mixtral https://github.com/arcee-ai/mergekit.git
    !cd mergekit && pip install -qqq -e . --progress-bar off
    !pip install -qqq -U transformers --progress-bar off

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
!{cli}