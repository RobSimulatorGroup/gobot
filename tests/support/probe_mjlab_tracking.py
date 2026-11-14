"""Native deterministic transition probes, recorded before resets/next commands."""

import argparse
from pathlib import Path
import numpy as np
from mjlab_tracking_reference import reference, write_json


def probe(source, motion, output, device="cuda:0", randomized=False):
    identity = reference(source)
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends(allow_tf32=False)
    cfg = unitree_g1_flat_tracking_env_cfg()
    cfg.scene.num_envs = 4
    cfg.seed = 42
    cfg.commands["motion"].motion_file = str(Path(motion).resolve())
    if not randomized:
        cfg.events = {}
        cfg.commands["motion"].pose_range = {}
        cfg.commands["motion"].velocity_range = {}
        cfg.commands["motion"].joint_position_range = (0.0, 0.0)
        cfg.observations["actor"].enable_corruption = False
    cfg.auto_reset = randomized
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 500
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        command = env.command_manager.get_term("motion")
        robot = env.scene["robot"]
        frames = torch.tensor([0, 1200, 6000, 10000], device=device)
        obs, _ = env.reset()
        if not randomized:
            for lane, frame in enumerate(frames.tolist()):
                command.reset_to_frame(torch.tensor([lane], device=device), frame)
            env.sim.forward()
            command.update_relative_body_poses()
            env.sim.sense()
            obs = env.observation_manager.compute(update_history=True)
        else:
            frames = command.time_steps.clone()
        arrays = {
            "frames": frames.cpu().numpy(),
            "initial_actor": obs["actor"].cpu().numpy().copy(),
            "initial_critic": obs["critic"].cpu().numpy().copy(),
        }
        terms = []
        bodies = []
        observations = []
        ends = []
        actions = []
        rngs = []
        reference_frames = []
        arrays["initial_rng"] = torch.cuda.get_rng_state().cpu().numpy()
        original = env.reward_manager.compute

        def capture(dt):
            value = original(dt)
            terms.append(env.reward_manager.term_rewards.clone().cpu().numpy())
            bodies.append(
                torch.cat(
                    (
                        command.robot_body_pos_w,
                        command.robot_body_quat_w,
                        command.robot_body_lin_vel_w,
                        command.robot_body_ang_vel_w,
                    ),
                    -1,
                )
                .cpu()
                .numpy()
                .copy()
            )
            return value

        env.reward_manager.compute = capture
        generator = torch.Generator().manual_seed(913)
        for step in range(12 if randomized else 3):
            action = (torch.rand(4, 29, generator=generator) * 2 - 1) * 0.05
            actions.append(action.numpy().copy())
            obs, reward, terminated, truncated, info = env.step(action.to(device))
            observations.append(
                torch.cat((obs["actor"], obs["critic"]), -1).cpu().numpy()
            )
            ends.append(torch.stack((terminated, truncated), -1).cpu().numpy())
            rngs.append(torch.cuda.get_rng_state().cpu().numpy())
            reference_frames.append(command.time_steps.clone().cpu().numpy())
            if terminated.any() and not randomized:
                break
        arrays.update(
            actions=np.stack(actions),
            rng=np.stack(rngs),
            reference_frames=np.stack(reference_frames),
            rewards=np.stack(terms),
            bodies=np.stack(bodies),
            observations=np.stack(observations),
            terminations=np.stack(ends),
        )
        out = Path(output)
        out.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(out / "probe.npz", **arrays)
        write_json(
            out / "probe.json",
            dict(
                reference_revision=identity["reference_revision"],
                reward_names=list(env.reward_manager.active_terms),
                frames=frames.tolist(),
                steps=len(actions),
                randomized=randomized,
                body_names=list(command.cfg.body_names),
                joint_names=list(robot.joint_names),
                actor_dim=obs["actor"].shape[-1],
                critic_dim=obs["critic"].shape[-1],
            ),
        )
        print(out)
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[2] / "mjlab"
    )
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--randomized", action="store_true")
    probe(**vars(p.parse_args()))
