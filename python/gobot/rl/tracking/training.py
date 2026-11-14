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
from gobot.rl.locomotion.training import snapshot_sources, update_policy
from .distributed import DistributedTraining, cpu_state, resolve_config, state_digest
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
    batch_scope: str = "per_rank"
    updates: int = 30000
    output: Path = Path("logs/g1-tracking/run")
    resume: Path | None = None
    torch_threads: int = 4
    agent: dict = field(default_factory=default_agent)

    def validate(self):
        super().validate()
        if self.batch_scope not in ("per_rank", "global"):
            raise ValueError("batch_scope must be per_rank or global")
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
    def __init__(self, env, agent, output, device, signature, metadata, group):
        self.group = group
        self.signature = signature
        self.metadata = metadata
        super().__init__(env, agent, str(output), device)
        _install_cpu_logger_bookkeeping(self.logger)

    def _configure_multi_gpu(self):
        # The launcher initializes NCCL (or Gloo for CPU tests) exactly once.
        self.is_distributed = self.group.world_size > 1
        self.gpu_world_size = self.group.world_size
        self.gpu_global_rank = self.group.rank
        self.gpu_local_rank = self.group.local_rank
        self.cfg["multi_gpu"] = self.group.algorithm_config

    def save(self, path, infos=None):
        local = cpu_state(
            dict(
                tracking_state=self.env.training_state_dict(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(self.device)
                if self.device.startswith("cuda")
                else None,
            )
        )
        ranks = self.group.gather(local)

        def write():
            info = dict(infos or {})
            info.update(
                tracking_contract=self.metadata,
                training_signature=self.signature,
                rank_states=ranks,
                world_size=self.group.world_size,
                num_envs_per_rank=self.env.num_envs,
                resume_semantics="shared optimizer and sampler restored; fresh per-rank physics episodes",
            )
            payload = self.alg.save()
            payload.update(iter=self.current_learning_iteration, infos=info)
            target = Path(path)
            temporary = target.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(target)

        self.group.on_main(write)

    def load(self, path, **kwargs):
        error = None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            info = payload["infos"]
            if info.get("training_signature") != self.signature:
                raise ValueError(
                    "tracking checkpoint configuration or motion does not match"
                )
            if (
                info.get("world_size") != self.group.world_size
                or info.get("num_envs_per_rank") != self.env.num_envs
            ):
                raise ValueError(
                    "resume requires the same world size and per-rank environment count"
                )
            if len(info["rank_states"]) != self.group.world_size:
                raise ValueError("checkpoint has missing rank states")
            self.alg.load(payload, kwargs.get("load_cfg"), kwargs.get("strict", True))
            self.current_learning_iteration = int(payload["iter"]) + 1
            local = info["rank_states"][self.group.rank]
            torch.set_rng_state(local["torch_rng"].cpu())
            if local["cuda_rng"] is not None:
                torch.cuda.set_rng_state(local["cuda_rng"].cpu(), self.device)
            self.env.load_training_state_dict(local["tracking_state"])
        except Exception as caught:
            error = caught
        self.group.check_error(error, phase="checkpoint restore")
        return info

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            raise ValueError("tracking uses explicit reference-state episode resets")
        self.alg.train_mode()
        if self.is_distributed:
            self.alg.broadcast_parameters()
        self.group.on_main(self.logger.init_logging_writer)
        observation = self.env.get_observations()
        first = self.current_learning_iteration
        end = first + num_learning_iterations
        try:
            for iteration in range(first, end):
                timing = {}
                observation, losses = update_policy(
                    self.alg,
                    self.env,
                    observation,
                    self.cfg["num_steps_per_env"],
                    distributed=self.group,
                    timings=timing,
                )
                self.env.sampler.synchronize(self.group)
                self.current_learning_iteration = iteration
                episode = self.group.reduce(self.env.consume_episode_statistics())
                averaged = self.group.reduce(list(losses.values()))
                losses = {
                    k: v / self.group.world_size
                    for k, v in zip(losses, averaged, strict=True)
                }
                collect, learn = self.group.reduce(
                    [timing["collection_s"], timing["learning_s"]], maximum=True
                )
                terms = {}
                if episode[0]:
                    terms = {
                        f"Episode_Reward/{k}": v
                        / (episode[0] * self.env.cfg_obj.episode_length_s)
                        for k, v in zip(
                            self.env.cfg_obj.reward_weights, episode[2:], strict=True
                        )
                    }
                    terms["Episode/length"] = episode[1] / episode[0]
                metrics = dict(
                    iteration=iteration,
                    world_size=self.group.world_size,
                    num_envs_per_rank=self.env.num_envs,
                    total_num_envs=self.env.num_envs * self.group.world_size,
                    completed_episodes=int(episode[0]),
                    episode=terms,
                    losses=losses,
                    samples=(iteration + 1)
                    * self.env.num_envs
                    * self.group.world_size
                    * self.cfg["num_steps_per_env"],
                    collection_s=collect,
                    learning_s=learn,
                )

                def log():
                    self.logger.ep_extras = [dict(terms)] if terms else []
                    if episode[0]:
                        self.logger.rewbuffer.clear()
                        self.logger.rewbuffer.append(sum(episode[2:]) / episode[0])
                        self.logger.lenbuffer.clear()
                        self.logger.lenbuffer.append(episode[1] / episode[0])
                    self.logger.log(
                        it=iteration,
                        start_it=first,
                        total_it=end,
                        collect_time=collect,
                        learn_time=learn,
                        loss_dict=losses,
                        learning_rate=self.alg.learning_rate,
                        action_std=self.alg.get_policy().output_std,
                        rnd_weight=None,
                    )
                    directory = Path(self.logger.log_dir)
                    with (directory / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps(metrics) + "\n")
                    write_json(
                        directory / "status.json", dict(state="training", **metrics)
                    )

                self.group.on_main(log)
                if iteration % self.cfg["save_interval"] == 0:
                    self.save(Path(self.logger.log_dir) / f"model_{iteration}.pt")
            self.save(
                Path(self.logger.log_dir)
                / f"model_{self.current_learning_iteration}.pt"
            )
        except BaseException:
            if self.group.main:
                self.logger.abort_logging_writer()
            raise
        if self.group.main:
            self.logger.stop_logging_writer()

    def audit(self):
        # Includes observation normalizers and optimizer state, not just weights.
        digests = {k: state_digest(v) for k, v in self.alg.save().items()}
        digests["sampler"] = state_digest(self.env.sampler.failed)
        ranks = self.group.gather(
            dict(
                rank=self.group.rank,
                device=self.device,
                seed=self.env.seed,
                digests=digests,
            )
        )

        def check():
            if any(r["digests"] != ranks[0]["digests"] for r in ranks):
                raise RuntimeError("distributed models, optimizer or sampler diverged")
            return dict(passed=True, ranks=ranks)

        return self.group.on_main(check)


def run(cfg):
    cfg.validate()
    torch.set_num_threads(cfg.torch_threads)
    group = DistributedTraining.from_environment(
        cfg.device or ("cuda:0" if cfg.backend == "mujoco-warp" else "cpu")
    )
    local = resolve_config(cfg, group)
    output = Path(cfg.output).resolve()
    env = None
    prepared = False
    started = time.monotonic()
    try:
        group.initialize()
        random.seed(local.seed)
        np.random.seed(local.seed)
        torch.manual_seed(local.seed)

        def prepare_output():
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "config.json", config_dict(cfg))
            write_json(
                output / "distributed.json",
                dict(
                    world_size=group.world_size,
                    batch_scope=cfg.batch_scope,
                    num_envs_per_rank=local.num_envs,
                    total_num_envs=local.num_envs * group.world_size,
                    sampler_sync="global failure-count EMA once per PPO rollout",
                ),
            )

        group.on_main(prepare_output)
        prepared = True
        error = None
        try:
            env = TrackingEnv(local)
        except Exception as caught:
            error = caught
        group.check_error(error, phase="environment initialization")
        if group.world_size > 1:
            env.sampler.defer()
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
            "device",
        ):
            semantic.pop(name)
        signature = hashlib.sha256(
            json.dumps(
                dict(
                    config=semantic,
                    contract=metadata,
                    world_size=group.world_size,
                    num_envs_per_rank=local.num_envs,
                ),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        signatures = group.gather(signature)

        def prepare_training():
            if len(set(signatures)) != 1:
                raise ValueError(
                    "tracking configurations or motion differ across ranks"
                )
            write_json(output / "contract.json", metadata)
            write_json(output / "training_signature.json", dict(signature=signature))
            snapshot_sources(output, env.project_path)

        group.on_main(prepare_training)
        runner = Runner(
            RslRlVecEnvWrapper(env, device=env.device),
            agent,
            output,
            env.device,
            signature,
            metadata,
            group,
        )
        if cfg.resume is not None:
            runner.load(cfg.resume)
        if group.world_size > 1:
            runner.alg.broadcast_parameters()
        initial = next(runner.alg.actor.parameters()).detach().clone()
        runner.learn(cfg.updates, init_at_random_ep_len=False)
        audit = runner.audit()
        runner.save(output / "model.pt")
        result = dict(
            state="trained",
            updates=cfg.updates,
            last_iteration=runner.current_learning_iteration,
            world_size=group.world_size,
            num_envs_per_rank=local.num_envs,
            total_num_envs=local.num_envs * group.world_size,
            samples=cfg.updates
            * local.num_envs
            * group.world_size
            * agent["num_steps_per_env"],
            elapsed_seconds=time.monotonic() - started,
            actor_max_weight_change=float(
                (next(runner.alg.actor.parameters()).detach() - initial).abs().max()
            ),
            checkpoint=str(output / "model.pt"),
            exact_physics_resume=False,
        )

        def finish():
            write_json(output / "synchronization.json", audit)
            write_json(output / "status.json", result)
            return result

        return group.on_main(finish)
    except BaseException as error:
        if prepared:
            path = (
                output / "status.json"
                if group.main
                else output / f"rank_{group.rank}_failure.json"
            )
            write_json(path, dict(state="failed", error=str(error), rank=group.rank))
        raise
    finally:
        if env is not None:
            env.close()
        group.close()


def training_spec():
    return TrainingSpec(
        Config,
        run,
        supports_distributed=True,
        description="One motion, one shared PPO policy on Gobot scenes; synchronized multi-GPU supported",
    )
