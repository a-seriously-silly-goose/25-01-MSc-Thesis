## Write code to testing/engines/test_environments.py
from pathlib import Path
current_dir = Path().resolve() # noqa E402

import sys # noqa E402
sys.path.append(str(current_dir)) # noqa E402
print(current_dir) # noqa E402


from dto.input_dtos import EnvParams # noqa E402
from dto.config_loader import load_config # noqa E402


## test loading config
def test_load_config():
    env_params, algo_params, risk_params = load_config("runs/hyperparameters.yml", "vTesting")
    assert isinstance(env_params, EnvParams)
    assert env_params.S0 == 10.0
    assert env_params.K == 10.0
    assert env_params.v0 == 0.04
    assert env_params.sigma == 0.2
    assert env_params.kappa == 9
    assert env_params.theta == 0.0625
    assert env_params.eta == 1
    assert env_params.T == 0.08333333
    assert env_params.rho == -0.5
    assert env_params.mu == 0.1
    assert env_params.r == 0.01
    assert env_params.B0 == 0.0
    assert env_params.epsilon == 0.0
    assert env_params.max_alpha == 3
    assert env_params.Ndt == 31

    # Add for algo_params and risk_params as needed
    assert algo_params.Ntrajectories == 500
    assert algo_params.Mtransitions == 500
    assert algo_params.Nepochs == 400
    assert algo_params.gamma == 1
    assert algo_params.Nepochs_V_init == 1500
    assert algo_params.Nepochs_V == 300
    assert algo_params.lr_V == 0.0005
    assert algo_params.batch_V == 200
    assert algo_params.hidden_V == 16
    assert algo_params.layers_V == 4
    assert algo_params.Nepochs_pi == 10
    assert algo_params.lr_pi == 0.0005
    assert algo_params.batch_pi == 200
    assert algo_params.hidden_pi == 16
    assert algo_params.layers_pi == 3
    assert algo_params.seed == 42
    assert algo_params.clip_epsilon == 0.1
    assert algo_params.gae_lambda == 0.95
    assert algo_params.entropy_coef == 0.01
    assert algo_params.n_critic_updates == 10
    assert algo_params.n_actor_updates == 10

    assert risk_params.method == 'mean'


if __name__ == "__main__":
    test_load_config()
    print("All tests passed.")