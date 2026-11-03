"""CPU/Warp short-horizon parity for the shared G1 task."""
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.g1.config import g1_velocity_cfg
from gobot.rl.locomotion import make_velocity_env

def cfg():
    value = g1_velocity_cfg()
    value.observations.actor_noise = value.domain_randomization.enabled = value.push_enabled = False
    value.spawn_jitter = 0.
    value.reset_z_range = (.03, .03)
    value.randomize_reset_yaw = False
    value.command.heading_command = False
    value.command.rel_standing_envs = value.command.rel_heading_envs = value.command.rel_forward_envs = 0.
    value.command.ranges.lin_vel_x = (.5, .5)
    value.command.ranges.lin_vel_y = value.command.ranges.ang_vel_z = (0., 0.)
    return value

def collect(backend):
    env = make_velocity_env(cfg(), backend=backend, num_envs=2, seed=19)
    try:
        env.set_terrain_assignments([0, 0], [0, 0])
        obs, _ = env.reset(seed=19)
        initial = obs['actor'].detach().cpu().numpy() if backend.endswith('warp') else obs['actor'].copy()
        history = []
        for _ in range(4):
            state = env.step(np.zeros((2, 29), dtype=np.float32))
            data = env.locomotion_state()
            data["reward_terms"] = env._reward_terms if backend.endswith("warp") else env.backend.state.reward_terms
            history.append({name: value.detach().cpu().numpy().copy() if hasattr(value, 'detach') else value.copy() for name, value in data.items()})
        if backend.endswith('warp'):
            model = env.provider
            pointer = model.arrays['qpos'].data_ptr()
            env.reset([1], seed=19)
            assert model.arrays['qpos'].data_ptr() == pointer
            ids = list(env._held_layout.actuator_ids)
            np.testing.assert_array_equal(model.arrays['ctrl'][:, ids].cpu().numpy(), 0.)
            model.assert_finite()
            model.assert_no_overflow()
        return initial, history
    finally:
        env.close()

def main():
    if not torch.cuda.is_available() or not (Path(__file__).resolve().parents[2] / "examples/g1/assets/meshes/pelvis.STL").exists():
        raise SystemExit(77)
    cpu_obs, cpu = collect('mujoco-cpu')
    gpu_obs, gpu = collect('mujoco-warp')
    np.testing.assert_allclose(cpu_obs, gpu_obs, atol=2e-4, rtol=2e-4)
    for step, (frame_cpu, frame_gpu) in enumerate(zip(cpu, gpu, strict=True)):
        print("step", step, {name: float(np.max(np.abs(frame_cpu[name] - frame_gpu[name]))) for name in ("base_position", "joint_position", "joint_velocity", "angular_momentum")}, flush=True)
        for name in ('base_position', 'base_quaternion', 'joint_position', 'joint_velocity', 'angular_momentum', 'reward_terms'):
            np.testing.assert_allclose(frame_cpu[name], frame_gpu[name], atol=2e-3, rtol=2e-3, err_msg=name)
    print('G1 CPU/Warp reset observations and short-horizon state parity passed')

if __name__ == '__main__':
    main()
