"""Train the Go1 velocity policy with the Gobot runtime."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

try:
    from .go1_velocity_cfg import go1_velocity_cfg, rsl_rl_train_cfg
    from .go1_velocity_env import Go1VelocityEnv
    from .go1_velocity_video import Go1TrainingVideoCfg, Go1TrainingVideoRecorder, VideoCheckpointRunnerMixin
except ImportError:
    from go1_velocity_cfg import go1_velocity_cfg, rsl_rl_train_cfg
    from go1_velocity_env import Go1VelocityEnv
    from go1_velocity_video import Go1TrainingVideoCfg, Go1TrainingVideoRecorder, VideoCheckpointRunnerMixin


class Go1OnPolicyRunner(VideoCheckpointRunnerMixin, OnPolicyRunner):
    def __init__(self, *args, video_recorder: Go1TrainingVideoRecorder | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.video_recorder = video_recorder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="go1_rough", help="Go1 velocity task name: go1_rough or go1_flat.")
    parser.add_argument("--num-envs", "--num_envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--max-episode-length", type=int, default=None)
    parser.add_argument("--log-dir", "--log_dir", type=str, default="logs/go1_velocity")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-workers", type=int, default=0, help="CPU workers for batched physics stepping. 0 uses hardware concurrency; 1 keeps stepping serial.")
    parser.add_argument("--profile-step", action="store_true", help="Log rolling Go1 env step phase timings.")
    parser.add_argument("--policy-out", type=str, default="policies/go1_velocity.pt")
    parser.add_argument("--no-terrain-curriculum", dest="terrain_curriculum", action="store_false", default=True)
    parser.add_argument("--no-obs-noise", dest="obs_noise", action="store_false", default=True)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest model_*.pt in log-dir.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--render-video-interval", type=int, default=100, help="Write an MP4 every N training iterations. Set 0 to disable.")
    parser.add_argument("--render-video-env-id", type=int, default=0, help="Eval batch env id to capture into the RGB training video.")
    parser.add_argument("--render-video-num-envs", type=int, default=1, help="Number of environments in the independent video eval rollout.")
    parser.add_argument("--render-video-steps", type=int, default=240)
    parser.add_argument("--render-video-fps", type=int, default=30)
    parser.add_argument("--render-video-width", type=int, default=640)
    parser.add_argument("--render-video-height", type=int, default=480)
    parser.add_argument("--render-video-dir", type=str, default=None)
    args = parser.parse_args()

    project_path = Path(__file__).resolve().parents[1]
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = project_path / log_dir
    os.makedirs(log_dir, exist_ok=True)

    cfg = go1_velocity_cfg(args.task, project_path=project_path)
    cfg.terrain_curriculum = bool(args.terrain_curriculum)
    cfg.observations.actor_noise = bool(args.obs_noise)
    cfg.terrain_curriculum_steps = max(1, int(args.iterations * args.num_envs * 24 * 0.6))

    print(f"Task: {cfg.name}")
    print(f"Device: {args.device}")
    print(f"Envs: {args.num_envs}")
    print(f"Log dir: {log_dir}")

    env = Go1VelocityEnv(
        cfg,
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        max_episode_length=args.max_episode_length,
        sim_workers=args.sim_workers,
        profile_step=args.profile_step,
    )
    print(f"Obs actor/critic/actions: {env.num_obs}/{env.num_privileged_obs}/{env.num_actions}")
    print(f"Sim workers: requested={args.sim_workers} resolved={env.resolved_sim_workers}")

    train_cfg = rsl_rl_train_cfg(
        experiment_name=cfg.name,
        max_iterations=args.iterations,
        save_interval=max(1, int(args.render_video_interval)) if args.render_video_interval > 0 else 50,
        obs_normalization=False,
    )
    video_dir = Path(args.render_video_dir) if args.render_video_dir else log_dir / "videos"
    if not video_dir.is_absolute():
        video_dir = project_path / video_dir
    video_recorder = Go1TrainingVideoRecorder(
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
        ),
    )
    runner = Go1OnPolicyRunner(
        env,
        copy.deepcopy(train_cfg),
        log_dir=str(log_dir),
        device=args.device,
        video_recorder=video_recorder,
    )
    checkpoint = _resolve_checkpoint(args.checkpoint, log_dir) if args.checkpoint or args.resume else None
    if checkpoint is not None:
        infos = runner.load(str(checkpoint), map_location=args.device)
        env.set_training_progress(
            runner.current_learning_iteration * args.num_envs * train_cfg["num_steps_per_env"]
        )
        print(f"Resumed checkpoint: {checkpoint}")
        print(f"Resume iteration: {runner.current_learning_iteration}")
        if infos:
            print(f"Checkpoint infos: {sorted(infos.keys())}")

    try:
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=checkpoint is None)

        final_path = log_dir / "model_final.pt"
        runner.save(str(final_path), infos={"gobot_go1_velocity": env.cfg})

        policy_path = Path(args.policy_out)
        if not policy_path.is_absolute():
            policy_path = project_path / policy_path
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        runner.save(str(policy_path), infos={"gobot_go1_velocity": env.cfg})

        print(f"Saved final model to {final_path}")
        print(f"Saved editor policy to {policy_path}")
    finally:
        env.close()
        video_recorder.close()


def _resolve_checkpoint(checkpoint: str | None, log_dir: Path) -> Path:
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = log_dir / path
        if not path.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        return path
    checkpoints = sorted(log_dir.glob("model_*.pt"), key=_checkpoint_iteration)
    if not checkpoints:
        raise FileNotFoundError(f"--resume requested but no model_*.pt checkpoints found in {log_dir}")
    return checkpoints[-1]


def _checkpoint_iteration(path: Path) -> int:
    suffix = path.stem.removeprefix("model_")
    return int(suffix) if suffix.isdigit() else -1


if __name__ == "__main__":
    main()
