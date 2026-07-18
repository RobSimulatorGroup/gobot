"""Train the Go1 velocity policy with the Gobot runtime."""

from __future__ import annotations

import argparse
import copy
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from gobot.rl.policy import (
    POLICY_MANIFEST_KEY,
    PolicyManifest,
    scene_bundle_digest,
    write_policy_manifest_sidecar,
)
from gobot.rl.rsl_rl import RslRlVecEnvWrapper

from .go1_velocity_cfg import apply_training_profile, go1_velocity_cfg, rsl_rl_train_cfg
from .go1_velocity_env import Go1VelocityEnv
from .go1_warp_velocity_env import Go1WarpVelocityEnv
from .go1_velocity_video import Go1TrainingVideoCfg, Go1TrainingVideoRecorder, VideoCheckpointRunnerMixin


class Go1OnPolicyRunner(VideoCheckpointRunnerMixin, OnPolicyRunner):
    def __init__(
        self,
        *args,
        video_recorder: Go1TrainingVideoRecorder | None = None,
        checkpoint_infos: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.video_recorder = video_recorder
        self.checkpoint_infos = dict(checkpoint_infos or {})

    def save(self, path: str, infos: dict | None = None) -> None:
        merged_infos = dict(self.checkpoint_infos)
        if infos:
            merged_infos.update(infos)
        merged_infos["env_state"] = self.env.training_state_dict()
        super().save(path, infos=merged_infos)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-batch", action="store_true", help="Use CPU smoke-training defaults: 64 rough-terrain environments and no video.")
    parser.add_argument(
        "--backend",
        choices=("mujoco-warp", "mujoco-cpu"),
        default="mujoco-warp",
        help="Physics/task backend. Selection is explicit and never falls back silently.",
    )
    parser.add_argument("--num-envs", "--num_envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--max-episode-length", type=int, default=None)
    parser.add_argument("--log-dir", "--log_dir", type=str, default="logs/go1_rough_velocity")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--training-profile",
        choices=("balanced", "run"),
        default="balanced",
        help=(
            "Command sampling profile. 'run' keeps normal commands and reserves "
            "60% of environments for a profile-local 1.5-3.5 m/s curriculum."
        ),
    )
    parser.add_argument("--sim-workers", type=int, default=0, help="CPU workers for batched physics stepping. 0 uses hardware concurrency; 1 keeps stepping serial.")
    parser.add_argument("--profile-step", action="store_true", help="Log rolling Go1 env step phase timings.")
    parser.add_argument("--profile-memory", action="store_true", help="Print RSS/CUDA memory after each training iteration.")
    parser.add_argument("--no-step-extras", action="store_true", default=False, help="Disable per-step reward-term/log extras.")
    parser.add_argument("--policy-out", type=str, default="policies/go1_velocity.pt")
    parser.add_argument("--terrain-curriculum", dest="terrain_curriculum", action="store_true", default=None)
    parser.add_argument("--no-terrain-curriculum", dest="terrain_curriculum", action="store_false")
    parser.add_argument("--no-obs-noise", dest="obs_noise", action="store_false", default=True)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest model_*.pt in log-dir.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--render-video-interval", type=int, default=0, help="Write an MP4 every N training iterations. Set 0 to disable.")
    parser.add_argument("--render-video-env-id", type=int, default=0, help="Eval batch env id to capture into the RGB training video.")
    parser.add_argument("--render-video-num-envs", type=int, default=1, help="Number of environments in the independent video eval rollout.")
    parser.add_argument("--render-video-steps", type=int, default=240)
    parser.add_argument("--render-video-fps", type=int, default=30)
    parser.add_argument("--render-video-width", type=int, default=640)
    parser.add_argument("--render-video-height", type=int, default=480)
    parser.add_argument("--render-video-dir", type=str, default=None)
    parser.add_argument("--render-video-debug-arrows", dest="render_video_debug_arrows", action="store_true", default=True)
    parser.add_argument("--no-render-video-debug-arrows", dest="render_video_debug_arrows", action="store_false")
    return parser


def resolve_project_path() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_log_dir(log_dir_arg: str, project_path: Path) -> Path:
    log_dir = Path(log_dir_arg)
    if not log_dir.is_absolute():
        log_dir = project_path / log_dir
    return log_dir


def build_velocity_cfg(args: argparse.Namespace, project_path: Path):
    cfg = go1_velocity_cfg(project_path=project_path)
    apply_training_profile(cfg, args.training_profile)
    if args.terrain_curriculum is not None:
        cfg.terrain_curriculum = bool(args.terrain_curriculum)
    cfg.observations.actor_noise = bool(args.obs_noise)
    return cfg


def build_core_env(args: argparse.Namespace, cfg) -> Go1VelocityEnv | Go1WarpVelocityEnv:
    if args.backend == "mujoco-warp":
        return Go1WarpVelocityEnv(
            cfg,
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed,
            max_episode_length=args.max_episode_length,
            profile_step=args.profile_step,
            collect_step_extras=not args.no_step_extras,
        )
    if args.backend != "mujoco-cpu":
        raise ValueError(f"unsupported Go1 backend {args.backend!r}")
    return Go1VelocityEnv(
        cfg,
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        max_episode_length=args.max_episode_length,
        sim_workers=args.sim_workers,
        profile_step=args.profile_step,
        collect_step_extras=not args.no_step_extras,
        task_runtime="numpy",
    )


def build_env(args: argparse.Namespace, cfg) -> RslRlVecEnvWrapper:
    env = build_core_env(args, cfg)
    return RslRlVecEnvWrapper(env, device=args.device)


def build_train_cfg(args: argparse.Namespace, cfg) -> dict:
    action_clip = getattr(cfg, "action_clip", None)
    train_cfg = rsl_rl_train_cfg(
        experiment_name=cfg.name,
        max_iterations=args.iterations,
        save_interval=max(1, int(args.save_interval)),
        obs_normalization=False,
        clip_actions=None if action_clip is None else float(action_clip),
    )
    train_cfg["seed"] = int(args.seed)
    return train_cfg


def apply_training_seed(seed: int, *, device: str) -> None:
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_video_recorder(args: argparse.Namespace, env: RslRlVecEnvWrapper, log_dir: Path, project_path: Path) -> Go1TrainingVideoRecorder:
    video_dir = Path(args.render_video_dir) if args.render_video_dir else log_dir / "videos"
    if not video_dir.is_absolute():
        video_dir = project_path / video_dir
    return Go1TrainingVideoRecorder(
        env,
        Go1TrainingVideoCfg(
            interval=int(args.render_video_interval),
            env_id=int(args.render_video_env_id),
            num_envs=int(args.render_video_num_envs),
            seed=int(args.seed) + 1_000_003,
            steps=int(args.render_video_steps),
            fps=int(args.render_video_fps),
            width=int(args.render_video_width),
            height=int(args.render_video_height),
            directory=video_dir,
            debug_arrows=bool(args.render_video_debug_arrows),
        ),
    )


def build_runner(
    args: argparse.Namespace,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    log_dir: Path,
    video_recorder: Go1TrainingVideoRecorder,
    checkpoint_infos: dict,
) -> Go1OnPolicyRunner:
    return Go1OnPolicyRunner(
        env,
        copy.deepcopy(train_cfg),
        log_dir=str(log_dir),
        device=args.device,
        video_recorder=video_recorder,
        checkpoint_infos=checkpoint_infos,
    )


def build_policy_manifest(
    env: RslRlVecEnvWrapper,
    cfg,
    train_cfg: dict,
    project_path: Path,
) -> PolicyManifest:
    task_metadata = env.task_runtime_metadata
    scene_digest = scene_bundle_digest(project_path, cfg.scene_path)
    return PolicyManifest(
        task_name=str(cfg.name),
        task_version=str(task_metadata.version),
        observation_spec=env.observation_spec.metadata(),
        action_spec=env.action_spec.metadata(),
        joint_names=tuple(env.joint_names),
        physics_dt=float(env.physics_dt),
        decimation=int(env.decimation),
        control={
            "mode": "position_offset",
            "default_joint_position": np.asarray(env.default_joint_pos, dtype=np.float32),
            "action_scale": np.asarray(env.action_scale, dtype=np.float32),
            "action_clip": cfg.action_clip,
            "kp": cfg.kp,
            "kd": cfg.kd,
            "reset_base_height": float(cfg.base_clearance),
        },
        model={
            "type": "mlp",
            "activation": str(train_cfg.get("actor", {}).get("activation", "elu")),
            "hidden_dims": tuple(train_cfg.get("actor", {}).get("hidden_dims", ())),
        },
        scene_path=str(cfg.scene_path),
        scene_digest=scene_digest,
        extras={
            "robot_name": str(cfg.robot_name),
            "base_link": str(cfg.base_link),
            "training_backend": str(task_metadata.backend),
            "critic_observation_spec": env.critic_observation_spec.metadata(),
            "solver_settings": dict(cfg.mujoco_solver_settings),
            "height_scan_max_distance": float(cfg.observations.terrain_scan_max_distance),
            "training_profile": str(cfg.training_profile),
            "run_environment_ratio": float(cfg.command.rel_run_envs),
            "run_velocity_x": tuple(float(value) for value in cfg.command.run_velocity_x),
            "run_velocity_curriculum": tuple(
                {
                    "step": int(stage.step),
                    "run_velocity_x": tuple(float(value) for value in stage.run_velocity_x),
                }
                for stage in cfg.run_command_curriculum
                if stage.run_velocity_x is not None
            ),
            "bound_gait_reward": float(cfg.rewards.bound_gait),
            "run_progress_reward": float(cfg.rewards.run_progress),
        },
    )


def run_training(args: argparse.Namespace) -> tuple[Path, Path]:
    apply_run_preset(args)
    apply_training_seed(args.seed, device=args.device)
    project_path = resolve_project_path()
    log_dir = resolve_log_dir(args.log_dir, project_path)
    os.makedirs(log_dir, exist_ok=True)

    cfg = build_velocity_cfg(args, project_path)

    print(f"Task: {cfg.name}")
    print(f"Backend: {args.backend}")
    print(f"Device: {args.device}")
    print(f"Envs: {args.num_envs}")
    print(f"Training profile: {cfg.training_profile}")
    print(f"Log dir: {log_dir}")

    if args.profile_memory:
        _print_memory_profile("before_env", args.device)
    env = build_env(args, cfg)
    print(f"Obs actor/critic/actions: {env.num_obs}/{env.num_privileged_obs}/{env.num_actions}")
    print(f"Sim workers: requested={args.sim_workers} resolved={env.resolved_sim_workers}")
    if args.profile_memory:
        _print_memory_profile("after_env", args.device)

    train_cfg = build_train_cfg(args, cfg)
    video_recorder = build_video_recorder(args, env, log_dir, project_path)
    policy_manifest = build_policy_manifest(env, cfg, train_cfg, project_path)
    checkpoint_infos = {
        POLICY_MANIFEST_KEY: policy_manifest.metadata(),
        "task_config": env.cfg,
    }
    runner = build_runner(args, env, train_cfg, log_dir, video_recorder, checkpoint_infos)
    _install_cpu_logger_bookkeeping(runner.logger)
    if args.profile_memory:
        _install_runner_memory_profile(runner, device=args.device)
        _print_memory_profile("after_runner", args.device)
    checkpoint = _resolve_checkpoint(args.checkpoint, log_dir) if args.checkpoint or args.resume else None
    if checkpoint is not None:
        infos = runner.load(str(checkpoint), map_location=args.device)
        common_step_counter = runner.current_learning_iteration * train_cfg["num_steps_per_env"]
        restore_result = None
        if infos and isinstance(infos.get("env_state"), dict):
            env_state = dict(infos["env_state"])
            env_state.setdefault("common_step_counter", common_step_counter)
            restore_result = env.load_training_state_dict(env_state)
        else:
            env.set_training_progress(common_step_counter)
        print(f"Resumed checkpoint: {checkpoint}")
        print(f"Resume iteration: {runner.current_learning_iteration}")
        if infos:
            print(f"Checkpoint infos: {sorted(infos.keys())}")
        if restore_result is None:
            print("Training state: legacy checkpoint; terrain curriculum starts from authored initial levels")
        else:
            print(f"Training state: {restore_result}")

    try:
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)

        final_path = log_dir / "model_final.pt"
        runner.save(str(final_path))
        write_policy_manifest_sidecar(final_path, policy_manifest)

        policy_path = Path(args.policy_out)
        if not policy_path.is_absolute():
            policy_path = project_path / policy_path
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        runner.save(str(policy_path))
        write_policy_manifest_sidecar(policy_path, policy_manifest)

        print_training_summary(
            args=args,
            cfg=cfg,
            env=env,
            train_cfg=train_cfg,
            log_dir=log_dir,
            final_path=final_path,
            policy_path=policy_path,
            video_dir=video_recorder.cfg.directory,
        )
        return final_path, policy_path
    finally:
        env.close()
        video_recorder.close()
        if args.profile_memory:
            _print_memory_profile("after_close", args.device)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    run_training(parse_args(argv))


def _resolve_checkpoint(checkpoint: str | None, log_dir: Path) -> Path:
    if checkpoint:
        path = Path(checkpoint)
        candidates = [path]
        if not path.is_absolute():
            candidates.append(log_dir / path)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        tried = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"checkpoint does not exist; tried: {tried}")
    checkpoints = sorted(log_dir.glob("model_*.pt"), key=_checkpoint_iteration)
    if not checkpoints:
        raise FileNotFoundError(f"--resume requested but no model_*.pt checkpoints found in {log_dir}")
    return checkpoints[-1]


def _checkpoint_iteration(path: Path) -> int:
    suffix = path.stem.removeprefix("model_")
    return int(suffix) if suffix.isdigit() else -1


def apply_run_preset(args: argparse.Namespace) -> None:
    if not args.cpu_batch:
        return
    args.device = "cpu"
    args.backend = "mujoco-cpu"
    args.num_envs = 64
    args.sim_workers = 0
    args.render_video_interval = 0
    args.log_dir = "logs/go1_rough_velocity_cpu_batch"
    args.policy_out = "policies/go1_velocity_cpu_batch.pt"


def print_training_summary(
    *,
    args: argparse.Namespace,
    cfg,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    log_dir: Path,
    final_path: Path,
    policy_path: Path,
    video_dir: Path,
) -> None:
    core_env = env.env
    task_runtime = getattr(core_env, "task_runtime_info", {})
    rows = [
        ("task", cfg.name),
        ("backend", args.backend),
        ("device", args.device),
        ("envs", args.num_envs),
        ("training profile", cfg.training_profile),
        ("iterations", args.iterations),
        ("rollout steps/env", train_cfg["num_steps_per_env"]),
        ("sim workers", f"requested={args.sim_workers}, resolved={env.resolved_sim_workers}"),
        ("task runtime", f"{task_runtime.get('backend', 'unknown')}:{task_runtime.get('mode', 'unknown')}"),
        ("memory profile", "on" if args.profile_memory else "off"),
        ("log dir", log_dir),
        ("final checkpoint", final_path),
        ("editor policy", policy_path),
    ]
    if int(args.render_video_interval) > 0:
        rows.append(("video dir", video_dir))
    width = max(len(name) for name, _ in rows)
    print("\nTraining summary")
    print("----------------")
    for name, value in rows:
        print(f"{name.ljust(width)} : {value}")


def _process_rss_mb() -> float:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return float("nan")
    return float("nan")


def _cuda_memory_stats(device: str) -> dict[str, float] | None:
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return None
    index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    except Exception:
        free_bytes, total_bytes = 0, 0
    return {
        "allocated_mb": torch.cuda.memory_allocated(index) / 1024.0 / 1024.0,
        "reserved_mb": torch.cuda.memory_reserved(index) / 1024.0 / 1024.0,
        "free_mb": free_bytes / 1024.0 / 1024.0,
        "total_mb": total_bytes / 1024.0 / 1024.0,
    }


def _cuda_host_memory_stats(device: str) -> dict[str, float] | None:
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        stats = torch.cuda.host_memory_stats(torch_device.index)
    except Exception:
        try:
            stats = torch.cuda.host_memory_stats()
        except Exception:
            return None
    return {
        "host_allocated_mb": float(stats.get("allocated_bytes.all.current", 0.0)) / 1024.0 / 1024.0,
        "host_reserved_mb": float(stats.get("reserved_bytes.all.current", 0.0)) / 1024.0 / 1024.0,
        "host_active_mb": float(stats.get("active_bytes.all.current", 0.0)) / 1024.0 / 1024.0,
    }


def _trim_cpu_allocator() -> None:
    return


def _synchronize_device(device: str) -> None:
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize(torch_device)
    except Exception:
        torch.cuda.synchronize()


def _print_memory_profile(tag: str, device: str) -> None:
    _synchronize_device(device)
    parts = [f"tag={tag}", f"rss_mb={_process_rss_mb():.1f}"]
    cuda_stats = _cuda_memory_stats(device)
    if cuda_stats is not None:
        parts.extend(
            [
                f"cuda_alloc_mb={cuda_stats['allocated_mb']:.1f}",
                f"cuda_reserved_mb={cuda_stats['reserved_mb']:.1f}",
                f"cuda_free_mb={cuda_stats['free_mb']:.1f}",
                f"cuda_total_mb={cuda_stats['total_mb']:.1f}",
            ]
        )
    cuda_host_stats = _cuda_host_memory_stats(device)
    if cuda_host_stats is not None:
        parts.extend(
            [
                f"cuda_host_alloc_mb={cuda_host_stats['host_allocated_mb']:.1f}",
                f"cuda_host_reserved_mb={cuda_host_stats['host_reserved_mb']:.1f}",
                f"cuda_host_active_mb={cuda_host_stats['host_active_mb']:.1f}",
            ]
        )
    print("[memory] " + " ".join(parts), flush=True)


def _install_runner_memory_profile(runner: Go1OnPolicyRunner, *, device: str) -> None:
    original_log = runner.logger.log
    original_save = runner.save

    def log_with_memory_profile(*args, **kwargs):
        result = original_log(*args, **kwargs)
        iteration = kwargs.get("it", args[0] if args else runner.current_learning_iteration)
        wrapper_stats = {}
        if hasattr(runner.env, "memory_profile"):
            wrapper_stats = runner.env.memory_profile()
        logger_stats = {
            "logger_ep_extras": len(getattr(runner.logger, "ep_extras", ())),
            "logger_rewbuffer": len(getattr(runner.logger, "rewbuffer", ())),
            "logger_lenbuffer": len(getattr(runner.logger, "lenbuffer", ())),
        }
        if wrapper_stats or logger_stats:
            stat_parts = [f"{key}={value}" for key, value in {**wrapper_stats, **logger_stats}.items()]
            print("[memory-detail] " + " ".join(stat_parts), flush=True)
        _print_memory_profile(f"iteration_{int(iteration):06d}", device)
        return result

    def save_with_memory_profile(*args, **kwargs):
        path = kwargs.get("path", args[0] if args else "")
        path_name = Path(path).name if path else "unknown"
        _print_memory_profile(f"before_save:{path_name}", device)
        result = original_save(*args, **kwargs)
        _print_memory_profile(f"after_save:{path_name}", device)
        return result

    runner.logger.log = log_with_memory_profile
    runner.save = save_with_memory_profile


def _install_cpu_logger_bookkeeping(logger) -> None:
    logger.device = "cpu"
    logger.cur_reward_sum = logger.cur_reward_sum.cpu()
    logger.cur_episode_length = logger.cur_episode_length.cpu()
    if hasattr(logger, "cur_ereward_sum"):
        logger.cur_ereward_sum = logger.cur_ereward_sum.cpu()
        logger.cur_ireward_sum = logger.cur_ireward_sum.cpu()

    def process_env_step_cpu(rewards, dones, extras, intrinsic_rewards=None) -> None:
        if logger.writer is None:
            return
        if "episode" in extras:
            logger.ep_extras.append(_cpu_log_mapping(extras["episode"]))
        elif "log" in extras:
            logger.ep_extras.append(_cpu_log_mapping(extras["log"]))

        rewards_cpu = _cpu_tensor(rewards, dtype=torch.float32)
        dones_cpu = _cpu_tensor(dones, dtype=torch.bool)
        if intrinsic_rewards is not None:
            intrinsic_rewards_cpu = _cpu_tensor(intrinsic_rewards, dtype=torch.float32)
            logger.cur_ereward_sum += rewards_cpu
            logger.cur_ireward_sum += intrinsic_rewards_cpu
            logger.cur_reward_sum += rewards_cpu + intrinsic_rewards_cpu
        else:
            logger.cur_reward_sum += rewards_cpu
        logger.cur_episode_length += 1

        new_ids = (dones_cpu > 0).nonzero(as_tuple=False)
        logger.rewbuffer.extend(logger.cur_reward_sum[new_ids][:, 0].numpy().tolist())
        logger.lenbuffer.extend(logger.cur_episode_length[new_ids][:, 0].numpy().tolist())
        logger.cur_reward_sum[new_ids] = 0
        logger.cur_episode_length[new_ids] = 0
        if intrinsic_rewards is not None:
            logger.erewbuffer.extend(logger.cur_ereward_sum[new_ids][:, 0].numpy().tolist())
            logger.irewbuffer.extend(logger.cur_ireward_sum[new_ids][:, 0].numpy().tolist())
            logger.cur_ereward_sum[new_ids] = 0
            logger.cur_ireward_sum[new_ids] = 0

    logger.process_env_step = process_env_step_cpu


def _cpu_tensor(value, *, dtype):
    if hasattr(value, "detach"):
        return value.detach().to(device="cpu", dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _cpu_log_mapping(values) -> dict:
    if not isinstance(values, dict):
        return values
    return {key: _cpu_log_value(value) for key, value in values.items()}


def _cpu_log_value(value):
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return float(tensor.reshape(-1)[0])
        return tensor
    return value


if __name__ == "__main__":
    main()
