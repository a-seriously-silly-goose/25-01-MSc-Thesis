import yaml
from pathlib import Path
from DTO.input_dtos import EnvParams, AlgoParams, RiskParams, RunParams


def load_config(path: str, version: str):
    """
    Load YAML config and return DTOs for a specific version.

    Args:
        path (str): Path to the YAML config.
        version (str): Version key in the YAML (e.g. "vCJ21").

    Returns:
        tuple: (EnvParams, AlgoParams, RiskParams, RunParams)

    Raises:
        ValueError: If the requested version does not exist in the config.
    """
    with open(Path(path), "r") as f:
        raw_config = yaml.safe_load(f)

    if version not in raw_config:
        raise ValueError(
            f"Version '{version}' not found in config. "
            f"Available versions: {list(raw_config.keys())}"
        )

    experiment_config = raw_config[version]

    env_params = EnvParams(**experiment_config["envParams"])
    algo_params = AlgoParams(**experiment_config["algoParams"])
    risk_params = RiskParams(**experiment_config["riskParams"])
    # run_params = RunParams(**experiment_config.get("runParams", {}))

    return env_params, algo_params, risk_params #, run_params
