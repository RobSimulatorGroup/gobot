"""Keep Coriolis coupling stable without clamping physical state or controls."""

import os

import mujoco
import numpy as np
import pytest


# A force-free serial articulation with perpendicular axes. Unlike a standalone
# free body, its velocity-dependent coupling distinguishes implicit from
# implicitfast. It has no robot asset, controller, contact, or learned policy.
_ARTICULATION_XML = """
<mujoco model="coriolis_articulation">
  <option timestep="0.005" integrator="implicit" gravity="0 0 0">
    <flag energy="enable"/>
  </option>
  <worldbody>
    <body name="first">
      <joint name="first_axis" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 .5" size=".04" mass="1"
            contype="0" conaffinity="0"/>
      <body name="second" pos="0 0 .5">
        <joint name="second_axis" axis="1 0 0"/>
        <geom type="box" pos="0 0 .2" size=".03 .05 .2" mass=".6"
              contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
_INITIAL_POSITION = np.array([.4, .8])
_INITIAL_VELOCITY = np.array([20., 40.])
_STEPS = 100


@pytest.mark.parametrize("settings, expected", [
    ({}, "Implicit"),
    ({"iterations": 17}, "Implicit"),
    ({"iterations": 17, "integrator": "ImplicitFast"}, "ImplicitFast"),
])
def test_generic_task_solver_defaults_preserve_explicit_overrides(settings, expected):
    from gobot.rl.locomotion.config import VelocityTaskCfg

    original = dict(settings)
    cfg = VelocityTaskCfg(mujoco_solver_settings=settings)
    assert cfg.mujoco_solver_settings["integrator"] == expected
    assert cfg.mujoco_solver_settings == {"integrator": expected, **original}
    assert settings == original


@pytest.mark.parametrize("settings, expected", [
    ({"iterations": 17}, mujoco.mjtIntegrator.mjINT_IMPLICIT),
    ({"iterations": 17, "integrator": "ImplicitFast"}, mujoco.mjtIntegrator.mjINT_IMPLICITFAST),
])
def test_custom_task_compiles_selected_integrator(tmp_path, settings, expected):
    from dataclasses import replace

    import gobot
    from gobot.rl.locomotion.scene import prepare_velocity_scene
    from test_generic_velocity_training import custom_task

    task = replace(custom_task(tmp_path / "project"), mujoco_solver_settings=settings)
    assert task.robot_name == "walker"
    context, _, _ = prepare_velocity_scene(task, context=gobot.app.create_context())
    artifact = context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
    model = mujoco.MjModel.from_xml_string(artifact["content"])
    assert model.opt.integrator == expected
    assert model.opt.iterations == 17


def _cpu_trajectory(integrator):
    model = mujoco.MjModel.from_xml_string(_ARTICULATION_XML)
    model.opt.integrator = integrator
    data = mujoco.MjData(model)
    data.qpos[:] = _INITIAL_POSITION
    data.qvel[:] = _INITIAL_VELOCITY
    positions, velocities, energies = [], [], []
    for step in range(_STEPS + 1):
        if step:
            mujoco.mj_step(model, data)
        # Energy from the integrated state, rather than mj_step's pre-step data.
        mujoco.mj_forward(model, data)
        positions.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        energies.append(data.energy[1])
    return np.array(positions), np.array(velocities), np.array(energies)


def test_full_implicit_bounds_energy_in_force_free_articulation():
    _, velocity, energy = _cpu_trajectory(mujoco.mjtIntegrator.mjINT_IMPLICIT)
    _, fast_velocity, fast_energy = _cpu_trajectory(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)

    assert np.isfinite(velocity).all()
    assert np.max(np.abs(velocity)) < 50.
    # No force supplies energy. Full implicit can dissipate energy, but must not
    # exhibit the artificial growth produced by omitted Coriolis derivatives.
    assert energy.max() <= energy[0] * 1.01
    assert fast_energy.max() > fast_energy[0] * 3.
    assert np.max(np.abs(fast_velocity)) > 100.


@pytest.mark.skipif(
    os.environ.get("GOBOT_RUN_LOCOMOTION_NUMERICS_GPU_TEST") != "1",
    reason="requires CUDA",
)
def test_captured_warp_full_implicit_matches_stable_cpu_trajectory():
    import torch

    from gobot.sim.providers.base import CompiledSceneArtifact, _artifact_digest
    from gobot.sim.providers.mujoco_warp import MuJoCoWarpProvider

    model = mujoco.MjModel.from_xml_string(_ARTICULATION_XML)
    artifact = CompiledSceneArtifact(
        schema_version=3,
        producer="mujoco",
        format="mjcf",
        content=_ARTICULATION_XML,
        content_digest=_artifact_digest(_ARTICULATION_XML),
        producer_version=mujoco.mj_versionString(),
        dimensions={name: int(getattr(model, name)) for name in ("nq", "nv", "nu")},
        robots=(),
        controls=(),
        terrain_geom_groups=(),
    )
    expected_position, expected_velocity, _ = _cpu_trajectory(mujoco.mjtIntegrator.mjINT_IMPLICIT)
    provider = MuJoCoWarpProvider(artifact, num_envs=2, capture_graphs=True)
    try:
        device = provider.arrays["qpos"].device
        provider.reset(
            torch.ones(2, dtype=torch.bool, device=device),
            qpos=torch.as_tensor(_INITIAL_POSITION, device=device).expand(2, -1),
            qvel=torch.as_tensor(_INITIAL_VELOCITY, device=device).expand(2, -1),
        )
        for step in range(1, _STEPS + 1):
            provider.step()
            position = provider.arrays["qpos"].cpu().numpy()
            velocity = provider.arrays["qvel"].cpu().numpy()
            np.testing.assert_allclose(
                position, np.broadcast_to(expected_position[step], position.shape),
                rtol=2.e-4, atol=2.e-4, err_msg=f"position at step {step}",
            )
            np.testing.assert_allclose(
                velocity, np.broadcast_to(expected_velocity[step], velocity.shape),
                rtol=2.e-4, atol=2.e-4, err_msg=f"velocity at step {step}",
            )
        provider.assert_finite()
    finally:
        provider.close()
