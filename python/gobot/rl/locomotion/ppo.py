"""Shared PPO defaults for velocity tracking tasks."""
from typing import Any

def rsl_rl_train_cfg(
    *,
    experiment_name: str = "velocity",
    max_iterations: int = 10_000,
    save_interval: int = 50,
    obs_normalization: bool = False,
    clip_actions: float | None = None,
) -> dict[str, Any]:
    return {
        "num_steps_per_env": 24,
        "save_interval": save_interval,
        "max_iterations": max_iterations,
        "obs_groups": {
            "actor": ["actor"],
            "critic": ["critic"],
        },
        "experiment_name": experiment_name,
        "clip_actions": clip_actions,
        "actor": {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": obs_normalization,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": obs_normalization,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "use_clipped_value_loss": True,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "learning_rate": 1.0e-3,
            "max_grad_norm": 1.0,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


__all__ = ["rsl_rl_train_cfg"]
