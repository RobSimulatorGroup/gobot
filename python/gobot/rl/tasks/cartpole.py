"""CartPole target-reaching task for Gobot PPO training."""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import TaskConfig, make_cartpole_target_task
from ..rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlModelCfg


@dataclass
class CartPoleEnvCfg:
    project_path: str = "/home/wqq/gobot/examples/cartpole"
    scene: str = "res://cartpole.jscn"
    backend: str = "mujoco"
    num_envs: int = 64
    batch_size: int = 64
    num_workers: int = 0
    seed: int = 42
    max_episode_steps: int = 500
    physics_dt: float = 1.0 / 240.0
    decimation: int = 1
    target_cart_position: float = 1.0
    randomize_target_position: bool = True
    target_cart_position_min: float = -1.0
    target_cart_position_max: float = 1.0
    force_limit: float = 20.0
    cart_position_limit: float = 2.4
    pole_angle_limit: float = 0.7
    initial_angle: float = 0.05
    randomize_initial_angle: bool = True


@dataclass
class CartPoleTrainCfg:
    env: CartPoleEnvCfg = field(default_factory=CartPoleEnvCfg)
    agent: RslRlOnPolicyRunnerCfg = field(default_factory=lambda: make_agent_cfg())
    device: str = "cpu"
    log_root: str = "logs/rsl_rl"


def make_env_cfg(cfg: CartPoleEnvCfg | None = None) -> TaskConfig:
    cfg = cfg or CartPoleEnvCfg()
    task = make_cartpole_target_task(
        scene=cfg.scene,
        backend=cfg.backend,
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
    task.metadata = {
        **dict(task.metadata),
        "seed": int(cfg.seed),
        "script": "gobot.rl.tasks.cartpole",
    }
    return task


def make_agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        num_steps_per_env=24,
        max_iterations=300,
        save_interval=50,
        experiment_name="cartpole",
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
    return CartPoleTrainCfg(env=CartPoleEnvCfg(), agent=make_agent_cfg())


__all__ = [
    "CartPoleEnvCfg",
    "CartPoleTrainCfg",
    "make_agent_cfg",
    "make_env_cfg",
    "make_train_cfg",
]
