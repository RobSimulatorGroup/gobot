"""RSL-RL PPO workflow for the Gobot-owned tracking environment."""

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from gobot.rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
    rsl_rl_cfg_to_dict,
    RslRlVecEnvWrapper,
)
from gobot.rl.training import TrainingSpec
from gobot.rl.training.workflow import config_dict
from gobot.rl.training.status import write_json
from gobot.rl.locomotion.logger import _install_cpu_logger_bookkeeping
from gobot.rl.locomotion.training import snapshot_sources
from .config import TrackingConfig, REFERENCE_REVISION
from .env import TrackingEnv


def default_agent():
    return rsl_rl_cfg_to_dict(
        RslRlOnPolicyRunnerCfg(
            actor=RslRlModelCfg(
                hidden_dims=(512, 256, 128),
                activation="elu",
                obs_normalization=True,
                distribution_cfg={
                    "class_name": "GaussianDistribution",
                    "init_std": 1.0,
                    "std_type": "scalar",
                },
            ),
            critic=RslRlModelCfg(
                hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True
            ),
            num_steps_per_env=24,
            max_iterations=30000,
            save_interval=500,
            experiment_name="g1-tracking",
        )
    )


@dataclass
class Config(TrackingConfig):
    updates: int = 30000
    output: Path = Path("logs/g1-tracking/run")
    resume: Path | None = None
    torch_threads: int = 4
    agent: dict = field(default_factory=default_agent)

    def validate(self):
        super().validate()
        for name in ("updates", "torch_threads"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("num_steps_per_env", "save_interval"):
            if not isinstance(self.agent.get(name), int) or self.agent[name] < 1:
                raise ValueError(f"invalid {name}")
        if self.agent.get("clip_actions") is not None:
            raise ValueError("tracking uses unclipped actions")
        if self.agent.get("logger") != "tensorboard":
            raise ValueError("tracking uses local TensorBoard logs")
        if (
            self.num_envs * self.agent["num_steps_per_env"]
            < self.agent["algorithm"]["num_mini_batches"]
        ):
            raise ValueError("rollout is smaller than minibatch count")
        if not Path(self.project_path).is_dir():
            raise ValueError("prepare tracking assets before training")


def contract(env):
    return dict(
        schema="gobot_mjlab_tracking_v1",
        reference_revision=REFERENCE_REVISION,
        scene_digest=env.batch.artifact.content_digest,
        motion_digest=env.motion.digest,
        layout_digest=env.motion.layout_digest,
        kinematics_digest=env.kin.digest,
        observation=env.observation_spec.metadata(),
        critic=env.critic_spec.metadata(),
        joints=list(env.cfg_obj.joint_names),
        body_names=list(env.cfg_obj.body_names),
        anchor=env.cfg_obj.anchor_link,
        default_position=env.default.tolist(),
        action_scale=env.action_scale.tolist(),
        physics_dt=env.cfg_obj.physics_dt,
        decimation=env.cfg_obj.decimation,
    )


class Runner(OnPolicyRunner):
    def __init__(self, env, agent, output, device, signature, metadata):
        self.signature = signature
        self.metadata = metadata
        super().__init__(env, agent, str(output), device)
        _install_cpu_logger_bookkeeping(self.logger)

    def save(self, path, infos=None):
        info = dict(infos or {})
        info.update(
            tracking_contract=self.metadata,
            training_signature=self.signature,
            tracking_state=self.env.training_state_dict(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(self.device)
            if str(self.device).startswith("cuda")
            else None,
            resume_semantics="optimizer and sampler restored; fresh physics episodes",
        )
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        super().save(str(temporary), info)
        temporary.replace(path)

    def load(self, path, **kwargs):
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if payload["infos"].get("training_signature") != self.signature:
            raise ValueError(
                "tracking checkpoint configuration or motion does not match"
            )
        info = super().load(str(path), map_location=self.device, **kwargs)
        self.current_learning_iteration += 1
        torch.set_rng_state(info["torch_rng"].cpu())
        if info["cuda_rng"] is not None:
            torch.cuda.set_rng_state(info["cuda_rng"].cpu(), self.device)
        self.env.load_training_state_dict(info["tracking_state"])
        return info


def run(cfg):
    cfg.validate()
    torch.set_num_threads(cfg.torch_threads)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    output = Path(cfg.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config_dict(cfg))
    env = None
    started = time.monotonic()
    try:
        env = TrackingEnv(cfg)
        agent = deepcopy(cfg.agent)
        agent["seed"] = cfg.seed
        metadata = contract(env)
        semantic = config_dict(cfg)
        for name in (
            "updates",
            "output",
            "resume",
            "project_path",
            "scene_path",
            "motion_file",
            "motion_layout",
        ):
            semantic.pop(name)
        signature = hashlib.sha256(
            json.dumps(
                dict(config=semantic, contract=metadata), sort_keys=True
            ).encode()
        ).hexdigest()
        write_json(output / "contract.json", metadata)
        snapshot_sources(output, env.project_path)
        runner = Runner(
            RslRlVecEnvWrapper(env, device=env.device),
            agent,
            output,
            env.device,
            signature,
            metadata,
        )
        if cfg.resume is not None:
            runner.load(cfg.resume)
        initial = next(runner.alg.actor.parameters()).detach().clone()
        write_json(
            output / "status.json", dict(state="training", algorithm="rsl_rl.PPO")
        )
        runner.learn(cfg.updates, init_at_random_ep_len=False)
        runner.save(output / "model.pt")
        result = dict(
            state="trained",
            updates=cfg.updates,
            last_iteration=runner.current_learning_iteration,
            samples=cfg.updates * cfg.num_envs * agent["num_steps_per_env"],
            elapsed_seconds=time.monotonic() - started,
            actor_max_weight_change=float(
                (next(runner.alg.actor.parameters()).detach() - initial).abs().max()
            ),
            checkpoint=str(output / "model.pt"),
            exact_physics_resume=False,
        )
        write_json(output / "status.json", result)
        return result
    except BaseException as e:
        write_json(output / "status.json", dict(state="failed", error=str(e)))
        raise
    finally:
        if env is not None:
            env.close()


def training_spec():
    return TrainingSpec(
        Config, run, description="mjlab/BeyondMimic G1 tracking on Gobot scenes"
    )
