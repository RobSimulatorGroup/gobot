"""Evaluate a Gobot tracking checkpoint headlessly, optionally capture RGB video."""

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import torch
from rsl_rl.runners import OnPolicyRunner
from gobot.rl.rsl_rl import RslRlVecEnvWrapper
from gobot.rl.training.workflow import update_config
from gobot.rl.training.status import write_json
from .training import Config, contract
from .env import TrackingEnv


@torch.inference_mode()
def evaluate(checkpoint, *, steps=500, backend="mujoco-cpu", device=None, video=None):
    checkpoint = Path(checkpoint).resolve()
    if steps < 1:
        raise ValueError("steps must be positive")
    cfg = update_config(
        Config(), json.loads(checkpoint.with_name("config.json").read_text())
    )
    torch.set_num_threads(cfg.torch_threads)
    cfg = replace(
        cfg,
        num_envs=1,
        backend=backend,
        device=device or ("cuda:0" if backend == "mujoco-warp" else "cpu"),
        observation_noise=False,
        domain_randomization=False,
        reset_noise=False,
        pushes=False,
        sampling_mode="start",
    )
    env = TrackingEnv(cfg, autoreset=False)
    writer = None
    try:
        payload = torch.load(checkpoint, map_location=env.device, weights_only=True)
        if payload["infos"].get("tracking_contract") != contract(env):
            raise ValueError("checkpoint scene, motion or observation contract differs")
        wrapper = RslRlVecEnvWrapper(env, device=env.device)
        runner = OnPolicyRunner(
            wrapper, deepcopy(cfg.agent), log_dir=None, device=env.device
        )
        runner.alg.load(payload, {"actor": True}, True)
        policy = runner.get_inference_policy(env.device)
        if video is not None:
            import imageio.v2 as imageio

            writer = imageio.get_writer(str(video), fps=1 / env.control_dt)
        position_error = []
        total_reward = 0.0
        terminated = False
        wrapped = False
        for _ in range(steps):
            action = policy(wrapper.get_observations())
            state = env.step(action)
            reference = env.motion.bodies(state.info["target_frame"])
            position_error.append(
                float(
                    (reference[0][:, env.anchor] - env.reward_bodies[0][:, env.anchor])
                    .norm(dim=-1)
                    .mean()
                )
            )
            total_reward += float(state.reward.mean())
            if writer is not None:
                import gobot

                env.batch.sync_scene()
                p = env.batch.read_state().base_pose[0, :3].cpu().tolist()
                image = gobot.render.capture_rgb(
                    root=env.batch.context.root,
                    width=960,
                    height=720,
                    eye=(p[0] + 2.4, p[1] - 3.0, p[2] + 0.8),
                    target=p,
                    fov_y=45,
                )
                writer.append_data(image)
            terminated = bool(state.terminated.any())
            wrapped = bool(state.info["motion_wrap"].any())
            if state.done.any() or wrapped:
                break
        return dict(
            steps=len(position_error),
            seconds=len(position_error) * env.control_dt,
            terminated=terminated,
            motion_completed=wrapped,
            truncated=bool(state.truncated.any()),
            termination_reasons=[
                k for k, v in state.info["termination_terms"].items() if bool(v.any())
            ],
            reference_duration_s=(env.motion.frames - 1) * env.control_dt,
            step_limit_reached=len(position_error) == steps
            and not bool(state.done.any())
            and not wrapped,
            mean_anchor_position_error_m=sum(position_error) / len(position_error),
            reward=total_reward,
        )
    finally:
        if writer is not None:
            writer.close()
            import gobot

            gobot.render.shutdown()
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument(
        "--backend", default="mujoco-cpu", choices=("mujoco-cpu", "mujoco-warp")
    )
    p.add_argument("--device")
    p.add_argument("--video", type=Path)
    p.add_argument("--output", type=Path)
    args = vars(p.parse_args())
    output = args.pop("output")
    result = evaluate(**args)
    if output:
        write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
