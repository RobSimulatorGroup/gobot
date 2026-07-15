from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

import gobot


OPTIONAL_DEPENDENCY_SKIP_CODE = 77
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _skip(reason: str) -> int:
    print(f"Go1 MuJoCo Warp integration skipped: {reason}")
    return OPTIONAL_DEPENDENCY_SKIP_CODE


def _deterministic_cfg():
    from examples.go1.train.go1_velocity_cfg import go1_velocity_cfg

    cfg = go1_velocity_cfg(project_path=REPO_ROOT / "examples/go1")
    cfg.observations.actor_noise = False
    cfg.domain_randomization.enabled = False
    cfg.push_enabled = False
    cfg.spawn_jitter = 0.0
    cfg.reset_z_range = (0.03, 0.03)
    cfg.randomize_reset_yaw = False
    cfg.max_init_terrain_level = 0
    cfg.command.rel_standing_envs = 0.0
    cfg.command.rel_heading_envs = 0.0
    cfg.command.rel_world_envs = 0.0
    cfg.command.rel_forward_envs = 0.0
    cfg.command.ranges.lin_vel_x = (0.4, 0.4)
    cfg.command.ranges.lin_vel_y = (0.0, 0.0)
    cfg.command.ranges.ang_vel_z = (0.0, 0.0)
    return cfg


def _assert_cpu_warp_short_horizon_parity(torch, warp_env) -> None:
    from examples.go1.train.go1_velocity_env import Go1VelocityEnv

    cpu_env = Go1VelocityEnv(
        _deterministic_cfg(),
        num_envs=1,
        device="cpu",
        seed=7,
        sim_workers=1,
        collect_step_extras=False,
        context=gobot.app.create_context(),
    )
    try:
        cpu_env.reset(seed=7)
        warp_env.reset(seed=7)
        cpu_state = cpu_env.backend.state
        np.testing.assert_allclose(
            cpu_state.base_position,
            warp_env.provider.arrays["qpos"][:, :3].cpu().numpy(),
            rtol=0.0,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            cpu_state.base_quaternion,
            warp_env.provider.arrays["qpos"][:, 3:7].cpu().numpy(),
            rtol=0.0,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            cpu_state.joint_position,
            warp_env.provider.arrays["qpos"][:, warp_env._joint_qpos_ids].cpu().numpy(),
            rtol=0.0,
            atol=1.0e-6,
        )

        cpu_result = cpu_env.step(np.zeros((1, cpu_env.num_actions), dtype=np.float32))
        warp_result = warp_env.step(
            torch.zeros((1, warp_env.num_actions), dtype=torch.float32, device="cuda:0")
        )
        cpu_state = cpu_env.backend.state
        np.testing.assert_allclose(
            cpu_state.base_position,
            warp_env.provider.arrays["qpos"][:, :3].cpu().numpy(),
            rtol=0.0,
            atol=1.2e-3,
        )
        np.testing.assert_allclose(
            cpu_state.base_quaternion,
            warp_env.provider.arrays["qpos"][:, 3:7].cpu().numpy(),
            rtol=0.0,
            atol=1.2e-3,
        )
        np.testing.assert_allclose(
            cpu_state.joint_position,
            warp_env.provider.arrays["qpos"][:, warp_env._joint_qpos_ids].cpu().numpy(),
            rtol=0.0,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            cpu_result.reward,
            warp_result.reward.cpu().numpy(),
            rtol=2.0e-5,
            atol=2.0e-6,
        )
    finally:
        cpu_env.close()


def _assert_warp_device_contract(torch, env) -> None:
    observations, _ = env.reset(seed=11)
    assert observations["actor"].shape == (1, 235)
    assert observations["critic"].shape == (1, 259)
    assert observations["actor"].is_cuda
    assert observations["critic"].is_cuda
    assert bool(torch.isfinite(observations["actor"]).all())
    assert bool(torch.isfinite(observations["critic"]).all())
    assert env.provider.capabilities.graph_capture is True
    assert env.provider.capabilities.device_native is True

    feet = env.provider.contact_sensor("feet_ground_contact")
    terrain = env.provider.raycast_sensor("terrain_scan")
    foot_scan = env.provider.raycast_sensor("foot_height_scan")
    assert feet["found"].shape == (1, 4)
    assert feet["force"].shape == (1, 4, 3)
    assert terrain["distances"].shape == (1, 187)
    assert foot_scan["distances"].shape == (1, 20)

    pointers = {
        name: env.provider.arrays[name].data_ptr()
        for name in ("qpos", "qvel", "ctrl", "sensordata")
    }
    contact_seen = torch.zeros((), dtype=torch.bool, device="cuda:0")
    actions = torch.zeros((1, env.num_actions), dtype=torch.float32, device="cuda:0")
    previous_actor = env.get_observations()["actor"]
    previous_critic = env.get_observations()["critic"]
    previous_actor_values = previous_actor.clone()
    previous_critic_values = previous_critic.clone()
    state = env.step(actions)
    torch.testing.assert_close(previous_actor, previous_actor_values)
    torch.testing.assert_close(previous_critic, previous_critic_values)
    assert state.obs["actor"].data_ptr() != previous_actor.data_ptr()
    assert state.obs["critic"].data_ptr() != previous_critic.data_ptr()

    env._encoder_bias.fill_(0.01)
    env.step(actions)
    torch.testing.assert_close(
        env._joint_targets,
        env._default_joint_pos.unsqueeze(0) - env._encoder_bias,
        rtol=0.0,
        atol=1.0e-7,
    )
    env._encoder_bias.zero_()
    env.reset(seed=11)
    for _ in range(100):
        state = env.step(actions)
        contact_seen |= (feet["found"] > 0).any()
    env.provider.synchronize()
    env.provider.assert_finite()
    env.provider.assert_no_overflow()
    assert bool(contact_seen)
    assert bool(torch.isfinite(state.obs["actor"]).all())
    assert bool(torch.isfinite(state.obs["critic"]).all())
    for name, pointer in pointers.items():
        assert env.provider.arrays[name].data_ptr() == pointer


def main() -> int:
    availability = gobot.rl.MuJoCoWarpProvider.availability()
    if not availability.available:
        return _skip(availability.reason)

    import torch

    if not torch.cuda.is_available():
        return _skip("Torch cannot access CUDA")

    from examples.go1.train.go1_warp_velocity_env import Go1WarpVelocityEnv

    warp_env = Go1WarpVelocityEnv(
        _deterministic_cfg(),
        num_envs=1,
        device="cuda:0",
        seed=7,
        collect_step_extras=False,
        capture_graphs=True,
        context=gobot.app.create_context(),
    )
    try:
        _assert_cpu_warp_short_horizon_parity(torch, warp_env)
        _assert_warp_device_contract(torch, warp_env)
    finally:
        warp_env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
