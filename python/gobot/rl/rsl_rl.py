"""RSL-RL integration for Gobot vectorized environments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import time
from typing import Any, Literal, Mapping, Tuple

import numpy as np

from .batch import BatchEnvState


def _writable_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    return array if array.flags.writeable else array.copy()


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
    rnd_cfg: dict[str, Any] | None = None
    symmetry_cfg: dict[str, Any] | None = None
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
    check_for_nan: bool = True
    torch_compile_mode: str | None = None


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


class RslRlVecEnvWrapper:
    """Torch/TensorDict adapter for Gobot's NumPy batch env contract."""

    is_vector_env = True

    def __init__(self, env: Any, *, device: str = "cpu") -> None:
        self.env = env
        self.device = str(device)
        try:
            import torch
            from tensordict import TensorDict
        except ImportError as error:
            raise RuntimeError("RslRlVecEnvWrapper requires gobot[train] dependencies.") from error
        self.torch = torch
        self.TensorDict = TensorDict
        self._episode_length_buf = self.torch.zeros(self.num_envs, dtype=self.torch.long, device=self.device)
        self._last_step_profile_ms: dict[str, float] = {}
        self._sync_steps_from_core()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def num_envs(self) -> int:
        return int(self.env.num_envs)

    @property
    def num_actions(self) -> int:
        return int(self.env.num_actions)

    @property
    def num_obs(self) -> int:
        return int(self.env.num_obs)

    @property
    def num_privileged_obs(self) -> int:
        return int(getattr(self.env, "num_privileged_obs", self.env.num_obs))

    @property
    def episode_length_buf(self):
        return self._episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value) -> None:
        self._episode_length_buf = self.torch.as_tensor(value, dtype=self.torch.long, device=self.device).clone()
        self._sync_steps_to_core()

    @property
    def cfg(self) -> Mapping[str, Any]:
        return self.env.cfg

    @property
    def cfg_obj(self) -> Any:
        return self.env.cfg_obj

    @property
    def seed(self) -> int:
        return int(self.env.seed)

    @property
    def max_episode_length(self) -> int:
        return int(self.env.max_episode_length)

    def reset(self, seed: int | None = None):
        obs, _ = self.env.reset(seed=seed)
        self._sync_steps_from_core()
        return self._tensor_obs(obs)

    def get_observations(self):
        return self._tensor_obs(self._core_state().obs)

    def step(self, actions):
        total_t0 = time.perf_counter()
        t0 = total_t0
        self._sync_steps_to_core()
        sync_to_core_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        core_actions = actions.detach().cpu().numpy() if hasattr(actions, "detach") else np.asarray(actions)
        action_to_numpy_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        state = self.env.step(core_actions)
        env_step_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        self._sync_steps_from_core()
        sync_from_core_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        reward = self.torch.as_tensor(state.reward, dtype=self.torch.float32, device=self.device)
        done = self.torch.as_tensor(state.done, dtype=self.torch.bool, device=self.device)
        extras = self._tensor_extras(state.info)
        extras_to_tensor_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        obs = self._tensor_obs(state.obs)
        obs_to_tensor_ms = (time.perf_counter() - t0) * 1000.0
        self._last_step_profile_ms = {
            "wrapper_total_ms": (time.perf_counter() - total_t0) * 1000.0,
            "sync_to_core_ms": sync_to_core_ms,
            "action_to_numpy_ms": action_to_numpy_ms,
            "env_step_ms": env_step_ms,
            "sync_from_core_ms": sync_from_core_ms,
            "extras_to_tensor_ms": extras_to_tensor_ms,
            "obs_to_tensor_ms": obs_to_tensor_ms,
        }
        return obs, reward, done, extras

    def last_step_profile_ms(self) -> dict[str, float]:
        return dict(self._last_step_profile_ms)

    def close(self) -> None:
        self.env.close()

    def _core_state(self) -> BatchEnvState:
        state = self.env.state
        if state is None:
            state = self.env.init_state()
        return state

    def _sync_steps_from_core(self) -> None:
        state = getattr(self.env, "state", None)
        if state is None:
            return
        steps = state.info.get("steps")
        if steps is None:
            return
        self._episode_length_buf = self.torch.as_tensor(
            np.asarray(steps, dtype=np.int64),
            dtype=self.torch.long,
            device=self.device,
        ).clone()

    def _sync_steps_to_core(self) -> None:
        state = getattr(self.env, "state", None)
        if state is None:
            return
        steps = state.info.get("steps")
        if isinstance(steps, np.ndarray) and steps.shape == (self.num_envs,):
            np.copyto(steps, self._episode_length_buf.detach().cpu().numpy())

    def _tensor_obs(self, obs: Mapping[str, Any]):
        data = {
            key: self.torch.as_tensor(_writable_array(value), dtype=self.torch.float32, device=self.device).clone()
            for key, value in obs.items()
        }
        if "actor" in data and "policy" not in data:
            data["policy"] = data["actor"].clone()
        elif "obs" in data and "policy" not in data:
            data["policy"] = data["obs"].clone()
        return self.TensorDict(data, batch_size=[self.num_envs], device=self.device)

    def _tensor_extras(self, info: Mapping[str, Any]) -> dict[str, Any]:
        extras: dict[str, Any] = {}
        if "time_outs" in info:
            extras["time_outs"] = self.torch.as_tensor(_writable_array(info["time_outs"]), dtype=self.torch.bool, device=self.device)
        elif "truncated" in info:
            extras["time_outs"] = self.torch.as_tensor(_writable_array(info["truncated"]), dtype=self.torch.bool, device=self.device)
        log = info.get("log", {})
        if isinstance(log, Mapping):
            extras["log"] = {
                key: self.torch.as_tensor(_writable_array(value), dtype=self.torch.float32, device=self.device)
                for key, value in log.items()
            }
        reward_terms = info.get("reward_terms")
        if isinstance(reward_terms, Mapping):
            extras["reward_terms"] = {
                key: self.torch.as_tensor(_writable_array(value), dtype=self.torch.float32, device=self.device)
                for key, value in reward_terms.items()
            }
        if "final_observation" in info:
            extras["final_observation"] = self._tensor_obs(info["final_observation"])
        if "_final_observation" in info:
            extras["_final_observation"] = self.torch.as_tensor(
                _writable_array(info["_final_observation"]),
                dtype=self.torch.bool,
                device=self.device,
            )
        return extras


def rsl_rl_cfg_to_dict(cfg: Mapping[str, Any] | RslRlBaseRunnerCfg | type | object) -> dict[str, Any]:
    """Convert Gobot's Python class-style PPO config to the dict RSL-RL expects."""

    if is_dataclass(cfg):
        data = asdict(cfg)
    elif isinstance(cfg, Mapping):
        data = dict(cfg)
    else:
        data = _class_style_ppo_cfg_to_dict(cfg)
    data = _drop_nones(_tuple_to_list(data))
    if isinstance(data.get("algorithm"), dict):
        data["algorithm"].setdefault("rnd_cfg", None)
        data["algorithm"].setdefault("symmetry_cfg", None)
    return data


def rsl_rl_cfg_to_dataclass(cfg: Mapping[str, Any] | RslRlBaseRunnerCfg | type | object) -> RslRlOnPolicyRunnerCfg:
    """Materialize any supported PPO config form as Gobot's runtime dataclass."""

    if isinstance(cfg, RslRlOnPolicyRunnerCfg):
        return cfg
    data = rsl_rl_cfg_to_dict(cfg)
    data = _normalize_dataclass_data(data)
    actor = RslRlModelCfg(**_filter_dataclass_kwargs(RslRlModelCfg, data.get("actor", {})))
    critic = RslRlModelCfg(**_filter_dataclass_kwargs(RslRlModelCfg, data.get("critic", {})))
    algorithm = RslRlPpoAlgorithmCfg(
        **_filter_dataclass_kwargs(RslRlPpoAlgorithmCfg, data.get("algorithm", {}))
    )
    runner = _filter_dataclass_kwargs(RslRlOnPolicyRunnerCfg, data)
    runner.pop("actor", None)
    runner.pop("critic", None)
    runner.pop("algorithm", None)
    return RslRlOnPolicyRunnerCfg(actor=actor, critic=critic, algorithm=algorithm, **runner)


def _class_style_ppo_cfg_to_dict(cfg: type | object) -> dict[str, Any]:
    policy = _cfg_section(cfg, "policy")
    algorithm = _cfg_section(cfg, "algorithm")
    runner = _cfg_section(cfg, "runner")

    requested_model_class = _cfg_get(
        policy,
        "model_class_name",
        _cfg_get(runner, "policy_class_name", "MLPModel"),
    )
    if requested_model_class == "ActorCritic":
        requested_model_class = "MLPModel"

    actor_hidden_dims = _cfg_get(policy, "actor_hidden_dims", _cfg_get(policy, "hidden_dims", [128, 128, 128]))
    critic_hidden_dims = _cfg_get(policy, "critic_hidden_dims", _cfg_get(policy, "hidden_dims", actor_hidden_dims))
    activation = _cfg_get(policy, "activation", "elu")
    init_noise_std = _cfg_get(policy, "init_noise_std", 1.0)
    obs_normalization = _cfg_get(policy, "obs_normalization", False)
    model_class_name = requested_model_class

    actor_cfg = {
        "class_name": _cfg_get(policy, "actor_class_name", model_class_name),
        "hidden_dims": actor_hidden_dims,
        "activation": activation,
        "obs_normalization": obs_normalization,
        "cnn_cfg": _cfg_get(policy, "cnn_cfg", None),
        "distribution_cfg": {
            "class_name": _cfg_get(policy, "distribution_class_name", "GaussianDistribution"),
            "init_std": init_noise_std,
            "std_type": _cfg_get(policy, "std_type", "scalar"),
        },
        "rnn_type": _cfg_get(policy, "rnn_type", None),
        "rnn_hidden_dim": _cfg_get(policy, "rnn_hidden_dim", None),
        "rnn_num_layers": _cfg_get(policy, "rnn_num_layers", None),
    }
    critic_cfg = {
        "class_name": _cfg_get(policy, "critic_class_name", model_class_name),
        "hidden_dims": critic_hidden_dims,
        "activation": activation,
        "obs_normalization": _cfg_get(policy, "critic_obs_normalization", obs_normalization),
        "cnn_cfg": _cfg_get(policy, "critic_cnn_cfg", None),
        "distribution_cfg": None,
        "rnn_type": _cfg_get(policy, "critic_rnn_type", None),
        "rnn_hidden_dim": _cfg_get(policy, "critic_rnn_hidden_dim", None),
        "rnn_num_layers": _cfg_get(policy, "critic_rnn_num_layers", None),
    }

    return {
        "seed": _cfg_get(runner, "seed", _cfg_get(cfg, "seed", 42)),
        "num_steps_per_env": _cfg_get(runner, "num_steps_per_env", 24),
        "max_iterations": _cfg_get(runner, "max_iterations", 300),
        "obs_groups": _cfg_get(runner, "obs_groups", {"actor": ("actor",), "critic": ("critic",)}),
        "save_interval": _cfg_get(runner, "save_interval", 50),
        "experiment_name": _cfg_get(runner, "experiment_name", "gobot"),
        "run_name": _cfg_get(runner, "run_name", ""),
        "logger": _cfg_get(runner, "logger", "tensorboard"),
        "wandb_project": _cfg_get(runner, "wandb_project", "gobot"),
        "wandb_tags": _cfg_get(runner, "wandb_tags", ()),
        "resume": _cfg_get(runner, "resume", False),
        "load_run": _cfg_get(runner, "load_run", ".*"),
        "load_checkpoint": _cfg_get(runner, "load_checkpoint", "model_.*.pt"),
        "clip_actions": _cfg_get(runner, "clip_actions", 1.0),
        "upload_model": _cfg_get(runner, "upload_model", False),
        "check_for_nan": _cfg_get(runner, "check_for_nan", True),
        "torch_compile_mode": _cfg_get(runner, "torch_compile_mode", None),
        "actor": actor_cfg,
        "critic": critic_cfg,
        "algorithm": {
            "class_name": _cfg_get(
                algorithm,
                "class_name",
                _cfg_get(runner, "algorithm_class_name", _cfg_get(algorithm, "algorithm_class_name", "PPO")),
            ),
            "num_learning_epochs": _cfg_get(algorithm, "num_learning_epochs", 5),
            "num_mini_batches": _cfg_get(algorithm, "num_mini_batches", 4),
            "learning_rate": _cfg_get(algorithm, "learning_rate", 1.0e-3),
            "schedule": _cfg_get(algorithm, "schedule", "adaptive"),
            "gamma": _cfg_get(algorithm, "gamma", 0.99),
            "lam": _cfg_get(algorithm, "lam", 0.95),
            "entropy_coef": _cfg_get(algorithm, "entropy_coef", 0.01),
            "desired_kl": _cfg_get(algorithm, "desired_kl", 0.01),
            "max_grad_norm": _cfg_get(algorithm, "max_grad_norm", 1.0),
            "value_loss_coef": _cfg_get(algorithm, "value_loss_coef", 1.0),
            "use_clipped_value_loss": _cfg_get(algorithm, "use_clipped_value_loss", True),
            "clip_param": _cfg_get(algorithm, "clip_param", 0.2),
            "normalize_advantage_per_mini_batch": _cfg_get(
                algorithm,
                "normalize_advantage_per_mini_batch",
                False,
            ),
            "optimizer": _cfg_get(algorithm, "optimizer", "adam"),
            "share_cnn_encoders": _cfg_get(algorithm, "share_cnn_encoders", False),
            "rnd_cfg": _cfg_get(algorithm, "rnd_cfg", None),
            "symmetry_cfg": _cfg_get(algorithm, "symmetry_cfg", None),
        },
    }


def _cfg_section(cfg: Any, name: str) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, {})
    return getattr(cfg, name, type("_EmptyCfg", (), {}))


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _filter_dataclass_kwargs(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    return {key: value for key, value in data.items() if key in fields}


def _normalize_dataclass_data(data: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    if isinstance(normalized.get("obs_groups"), Mapping):
        normalized["obs_groups"] = {
            str(key): tuple(value) for key, value in normalized["obs_groups"].items()
        }
    if "wandb_tags" in normalized:
        normalized["wandb_tags"] = tuple(normalized["wandb_tags"])
    for key in ("actor", "critic"):
        if isinstance(normalized.get(key), Mapping):
            model = dict(normalized[key])
            if "hidden_dims" in model:
                model["hidden_dims"] = tuple(model["hidden_dims"])
            normalized[key] = model
    return normalized


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
    "RslRlBaseRunnerCfg",
    "RslRlModelCfg",
    "RslRlOnPolicyRunnerCfg",
    "RslRlPpoAlgorithmCfg",
    "RslRlVecEnvWrapper",
    "rsl_rl_cfg_to_dataclass",
    "rsl_rl_cfg_to_dict",
]
