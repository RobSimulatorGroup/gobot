"""Local-checkpoint, full-clip evaluation of the pinned native mjlab baseline."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from mjlab_tracking_reference import reference, sha256, write_json, configure_deterministic_physics


def evaluate(source, checkpoint, motion, output, device="cuda:0", trajectory=None):
    reference(source)
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
    from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
    from mjlab.utils.torch import configure_torch_backends

    torch.set_num_threads(4)
    configure_torch_backends(allow_tf32=False)
    saved = json.loads(Path(checkpoint).parent.joinpath("comparison.json").read_text())
    if sha256(motion) != saved["motion_sha256"]:
        raise ValueError("evaluation motion differs from training")
    cfg = unitree_g1_flat_tracking_env_cfg()
    cfg.scene.num_envs = 1
    cfg.seed = saved["seed"]
    cfg.scene.sensors[0].reduce = saved.get("self_contact_reduction", "none")
    cfg.auto_reset = False
    cfg.sim.nconmax = saved["nconmax"]
    cfg.sim.njmax = saved["njmax"]
    cfg.episode_length_s = 1e9
    cfg.events = {}
    cfg.observations["actor"].enable_corruption = False
    motion_cfg = cfg.commands["motion"]
    motion_cfg.motion_file = str(Path(motion).resolve())
    motion_cfg.sampling_mode = "start"
    motion_cfg.pose_range = {}
    motion_cfg.velocity_range = {}
    motion_cfg.joint_position_range = (0.0, 0.0)
    configure_deterministic_physics(cfg.sim, saved.get("deterministic_physics", False))
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        wrapped = RslRlVecEnvWrapper(env)
        agent = unitree_g1_tracking_ppo_runner_cfg()
        agent.logger = "tensorboard"
        agent.upload_model = False
        runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device=device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        runner.alg.load(payload, {"actor": True}, True)
        policy = runner.get_inference_policy(device)
        command = env.command_manager.get_term("motion")
        metrics = []
        rewards = []
        trace = None
        if trajectory is not None:
            import mujoco
            import numpy as np

            trajectory = Path(trajectory)
            trajectory.parent.mkdir(parents=True, exist_ok=True)
            if trajectory.exists() or trajectory.with_suffix(".mjb").exists():
                raise FileExistsError(f"trajectory already exists: {trajectory}")
            mujoco.mj_saveModel(env.sim.mj_model, str(trajectory.with_suffix(".mjb")))
            indexing = command.robot.indexing
            root_qadr = indexing.free_joint_q_adr.cpu().numpy()
            joint_qadr = indexing.joint_q_adr.cpu().numpy()
            with np.load(motion) as data:
                ref_qpos = np.tile(env.sim.mj_model.qpos0, (command.motion.time_step_total, 1))
                pelvis = list(data["body_names"]).index("pelvis")
                ref_qpos[:, root_qadr[:3]] = data["body_pos_w"][:, pelvis]
                ref_qpos[:, root_qadr[3:7]] = data["body_quat_w"][:, pelvis]
                ref_qpos[:, joint_qadr] = data["joint_pos"]
            ref_qpos[:, root_qadr[:3]] += env.scene.env_origins[0].cpu().numpy()
            trace = {
                "qpos": [env.sim.data.qpos[0].cpu().numpy().copy()],
                "reference_frame": [0],
            }
        original = env.reward_manager.compute

        def capture(dt):
            reward = original(dt)
            metrics.append(
                float(
                    (command.anchor_pos_w - command.robot_anchor_pos_w)
                    .norm(dim=-1)
                    .mean()
                )
            )
            rewards.append(float(reward.mean()))
            if trace is not None:
                # Capture before command advancement can wrap/reset the motion.
                trace["qpos"].append(env.sim.data.qpos[0].cpu().numpy().copy())
                trace["reference_frame"].append(int(command.time_steps[0]))
            return reward

        env.reward_manager.compute = capture
        obs = wrapped.get_observations()
        completed = False
        terminated = False
        with torch.inference_mode():
            for _ in range(command.motion.time_step_total - 1):
                frame = int(command.time_steps[0])
                obs, _, done, _ = wrapped.step(policy(obs))
                terminated = bool(env.termination_manager.terminated.any())
                completed = (
                    frame >= command.motion.time_step_total - 1 and not terminated
                )
                if len(metrics) % 500 == 0:
                    print(f"Evaluation: {len(metrics) * env.step_dt:.1f}s / "
                          f"{(command.motion.time_step_total - 1) * env.step_dt:.1f}s", flush=True)
                if done.any() or completed:
                    break
        result = dict(
            engine="mjlab",
            motion_sha256=sha256(motion),
            checkpoint_sha256=sha256(checkpoint),
            steps=len(metrics),
            seconds=len(metrics) * env.step_dt,
            terminated=terminated,
            motion_completed=completed,
            reference_duration_s=(command.motion.time_step_total - 1) * env.step_dt,
            mean_anchor_position_error_m=sum(metrics) / len(metrics),
            reward=sum(rewards),
            termination_reasons=[
                k
                for k, v in env.termination_manager._term_dones.items()
                if bool(v.any())
            ],
            protocol="frame0 state, frame1 first target; no DR/noise/push; first termination or full motion",
        )
        if trace is not None:
            frames = np.asarray(trace["reference_frame"])
            np.savez_compressed(
                trajectory,
                qpos=np.stack(trace["qpos"]),
                reference_qpos=ref_qpos[frames],
                reference_frame=frames,
                time_s=np.arange(len(frames)) * env.step_dt,
                anchor_position_error_m=np.asarray([0.0, *metrics]),
                step_dt=env.step_dt,
            )
            result["trajectory"] = str(trajectory.resolve())
        write_json(output, result)
        print(json.dumps(result, indent=2))
        return result
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[2] / "mjlab"
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--trajectory", type=Path,
                   help="Optionally save physical/reference poses and a render-only model")
    evaluate(**vars(p.parse_args()))
