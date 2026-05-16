"""RSL-RL integration for Gobot vectorized environments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal, Mapping, Tuple

from . import GobotOnPolicyRunner, RslRlVecEnvWrapper


@dataclass
class RslRlModelCfg:
    """Config for a single RSL-RL model."""

    hidden_dims: Tuple[int, ...] = (128, 128, 128)
    activation: str = "elu"
    obs_normalization: bool = False
    cnn_cfg: dict[str, Any] | None = None
    distribution_cfg: dict[str, Any] | None = None
    rnn_type: str | None = None
    rnn_hidden_dim: int | None = None
    rnn_num_layers: int | None = None
    class_name: str = "MLPModel"


@dataclass
class RslRlPpoAlgorithmCfg:
    """PPO algorithm config passed to RSL-RL."""

    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: Literal["adaptive", "fixed"] = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.005
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    normalize_advantage_per_mini_batch: bool = False
    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    share_cnn_encoders: bool = False
    class_name: str = "PPO"


@dataclass
class RslRlBaseRunnerCfg:
    """Base runner config for RSL-RL training."""

    seed: int = 42
    num_steps_per_env: int = 24
    max_iterations: int = 300
    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"actor": ("actor",), "critic": ("critic",)}
    )
    save_interval: int = 50
    experiment_name: str = "cartpole"
    run_name: str = ""
    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    wandb_project: str = "gobot"
    wandb_tags: Tuple[str, ...] = ()
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    clip_actions: float | None = 1.0
    upload_model: bool = False


@dataclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """On-policy actor-critic runner config."""

    class_name: str = "OnPolicyRunner"
    actor: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            }
        )
    )
    critic: RslRlModelCfg = field(default_factory=RslRlModelCfg)
    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)


def rsl_rl_cfg_to_dict(cfg: Mapping[str, Any] | RslRlBaseRunnerCfg) -> dict[str, Any]:
    """Convert dataclass configs to a plain dict accepted by RSL-RL."""

    if is_dataclass(cfg):
        data = asdict(cfg)
    else:
        data = dict(cfg)
    return _drop_nones(_tuple_to_list(data))


def _tuple_to_list(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_tuple_to_list(item) for item in value]
    if isinstance(value, list):
        return [_tuple_to_list(item) for item in value]
    if isinstance(value, dict):
        return {key: _tuple_to_list(item) for key, item in value.items()}
    return value


def _drop_nones(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_nones(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_nones(item) for item in value]
    return value


__all__ = [
    "GobotOnPolicyRunner",
    "RslRlBaseRunnerCfg",
    "RslRlModelCfg",
    "RslRlOnPolicyRunnerCfg",
    "RslRlPpoAlgorithmCfg",
    "RslRlVecEnvWrapper",
    "rsl_rl_cfg_to_dict",
]
