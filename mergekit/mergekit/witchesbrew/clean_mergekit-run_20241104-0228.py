# Main Cleanup function
def main():
        execute_command('pipx uninstall mergekit', env, "cleaning up installation of mergekit")
        execute_command('pip uninstall pyyaml', env, "cleaning up installation of pyyaml")
        execute_command('deactivate', env, "deactivating python virtual environment")
        execute_command('sudo rm -fvr .venv', env, "cleaning up python virtual environment")
        execute_command('pip uninstall -y safetensors', env, "cleaning up installation of safetensors")
        execute_command('pip uninstall -y torch', env, "cleaning up installation of pytorch")
        execute_command('pip uninstall -y transformers', env, "cleaning up installation of transformers")
        print("I need a fuckin' Slurpee up this bitch!")