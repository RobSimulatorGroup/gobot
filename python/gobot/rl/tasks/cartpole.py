"""CartPole task configs built on the generic Gobot RL environment."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .. import TaskConfig, make_cartpole_target_task
from ..rsl_rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from .registry import register_gobot_task


@dataclass
class CartPoleEnvCfg:
    project_path: str = "/home/wqq/gobot/examples/cartpole"
    scene: str = "res://cartpole.jscn"
    backend: str = "mujoco"
    robot: str = "cartpole"
    num_envs: int = 64
    batch_size: int = 64
    num_workers: int = 0
    seed: int = 42
    max_episode_steps: int = 500
    physics_dt: float = 1.0 / 240.0
    decimation: int = 1
    target_cart_position: float = 0.0
    randomize_target_position: bool = False
    target_cart_position_min: float = -1.0
    target_cart_position_max: float = 1.0
    force_limit: float = 10.0
    cart_position_limit: float = 2.4
    pole_angle_limit: float = 0.7
    initial_angle: float = 0.05
    randomize_initial_angle: bool = True


@dataclass
class CartPoleTrainCfg:
    env: CartPoleEnvCfg = field(default_factory=CartPoleEnvCfg)
    agent: RslRlOnPolicyRunnerCfg = field(default_factory=lambda: make_balance_agent_cfg())
    device: str = "cpu"
    log_root: str = "logs/rsl_rl"


def make_env_cfg(cfg: CartPoleEnvCfg | None = None) -> TaskConfig:
    return make_balance_env_cfg(cfg)


def make_balance_env_cfg(cfg: CartPoleEnvCfg | None = None) -> TaskConfig:
    cfg = replace(cfg or CartPoleEnvCfg(), target_cart_position=0.0, randomize_target_position=False)
    task = _make_base_env_cfg(cfg)
    _configure_balance_rewards(task)
    task.metadata = {
        **dict(task.metadata),
        "objective": "balance",
        "seed": int(cfg.seed),
        "script": "gobot.rl.tasks.cartpole",
    }
    return task


def make_target_env_cfg(cfg: CartPoleEnvCfg | None = None) -> TaskConfig:
    cfg = cfg or CartPoleEnvCfg(target_cart_position=1.0, randomize_target_position=True, force_limit=20.0)
    task = _make_base_env_cfg(cfg)
    task.metadata = {
        **dict(task.metadata),
        "objective": "target",
        "seed": int(cfg.seed),
        "script": "gobot.rl.tasks.cartpole",
    }
    return task


def make_agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return make_target_agent_cfg()


def make_balance_agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        num_steps_per_env=64,
        max_iterations=500,
        save_interval=100,
        experiment_name="cartpole_balance",
        run_name="",
        logger="tensorboard",
        clip_actions=1.0,
        upload_model=False,
        actor=RslRlModelCfg(
            hidden_dims=(128, 128, 128),
            activation="elu",
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.2,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(hidden_dims=(128, 128, 128), activation="elu"),
        algorithm=RslRlPpoAlgorithmCfg(
            num_learning_epochs=4,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            gamma=0.99,
            lam=0.95,
            entropy_coef=0.0,
            desired_kl=0.01,
            value_loss_coef=1.0,
            max_grad_norm=1.0,
        ),
    )


def make_target_agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        num_steps_per_env=24,
        max_iterations=300,
        save_interval=50,
        experiment_name="cartpole_target",
        run_name="",
        logger="tensorboard",
        clip_actions=1.0,
        upload_model=False,
        actor=RslRlModelCfg(
            hidden_dims=(128, 128, 128),
            activation="elu",
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(hidden_dims=(128, 128, 128), activation="elu"),
        algorithm=RslRlPpoAlgorithmCfg(
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            gamma=0.99,
            lam=0.95,
            entropy_coef=0.005,
            desired_kl=0.01,
            value_loss_coef=1.0,
            max_grad_norm=1.0,
        ),
    )


def make_train_cfg() -> CartPoleTrainCfg:
    return CartPoleTrainCfg(env=CartPoleEnvCfg(), agent=make_balance_agent_cfg())


def _make_base_env_cfg(cfg: CartPoleEnvCfg) -> TaskConfig:
    task = make_cartpole_target_task(
        scene=cfg.scene,
        backend=cfg.backend,
        robot=cfg.robot,
        project_path=cfg.project_path,
        num_envs=cfg.num_envs,
        target_cart_position=cfg.target_cart_position,
        randomize_target_position=cfg.randomize_target_position,
        target_cart_position_range=(cfg.target_cart_position_min, cfg.target_cart_position_max),
        max_episode_steps=cfg.max_episode_steps,
        physics_dt=cfg.physics_dt,
        decimation=cfg.decimation,
        force_limit=cfg.force_limit,
        pole_angle_limit=cfg.pole_angle_limit,
        cart_position_limit=cfg.cart_position_limit,
        initial_angle=cfg.initial_angle,
        randomize_initial_angle=cfg.randomize_initial_angle,
    )
    task.simulation = {
        "batch_size": int(cfg.batch_size),
        "num_workers": int(cfg.num_workers),
    }
    return task


def _configure_balance_rewards(task: TaskConfig) -> None:
    for term in task.rewards.get("terms", []):
        name = term.get("name")
        if name == "alive":
            term["weight"] = 2.0
        elif name == "target_distance":
            term["weight"] = -0.5
        elif name == "cart_velocity":
            term["weight"] = -0.05
        elif name == "pole_angle":
            term["weight"] = -30.0
        elif name == "pole_angular_velocity":
            term["weight"] = -0.5
        elif name == "overshoot":
            term["weight"] = 0.0
        elif name == "action_l2":
            term["weight"] = -0.05
        elif name in {
            "target_progress",
            "settle_bonus",
            "fast_reach_bonus",
            "overspeed_penalty",
            "fast_crossing_penalty",
        }:
            term["weight"] = 0.0
        elif name == "failure_penalty":
            term["weight"] = -10.0


register_gobot_task(
    "Gobot-CartPole-Balance",
    env_cfg=CartPoleEnvCfg(target_cart_position=0.0, randomize_target_position=False, force_limit=10.0),
    rl_cfg=make_balance_agent_cfg(),
    env_builder=make_balance_env_cfg,
)

register_gobot_task(
    "Gobot-CartPole-Target",
    env_cfg=CartPoleEnvCfg(
        target_cart_position=1.0,
        randomize_target_position=True,
        force_limit=20.0,
    ),
    rl_cfg=make_target_agent_cfg(),
    env_builder=make_target_env_cfg,
)

__all__ = [
    "CartPoleEnvCfg",
    "CartPoleTrainCfg",
    "make_agent_cfg",
    "make_balance_agent_cfg",
    "make_balance_env_cfg",
    "make_env_cfg",
    "make_target_agent_cfg",
    "make_target_env_cfg",
    "make_train_cfg",
]
