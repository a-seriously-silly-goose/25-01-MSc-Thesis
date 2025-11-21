from pathlib import Path
import os
import sys
path = Path(os.getcwd())

sys.path.append(str(path))
print("\n>>>", str(path), "\n")

from dto.config_loader import load_config # noqa

def main():
    """
    Load and display the configuration using the 'v_debug' parameters from the
    'runs.hyperparameters.yml' file.
    """
    # Assuming 'v_debug' is a key in the YAML file and the file path is correct
    config_path = "runs/hyperparameters.yml"
    params_key = "v_debug"

    # Load the configuration
    env_params, algo_params, risk_params = load_config(path=config_path, version=params_key)

    # Display the loaded configuration
    print("Loaded Configuration:")
    print("Environment Parameters:")
    print(env_params)
    print("Algorithm Parameters:")
    print(algo_params)
    print("Risk Parameters:")
    print(risk_params)

if __name__ == "__main__":
    main()
