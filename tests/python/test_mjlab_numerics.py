"""Numerical behavior against mjlab dcdf6ce519abfd2ea45e9503ae6fb650b75bb869.

Execute the checked-out reference methods with small tensor fixtures. AST loading
avoids importing mjlab's simulator, viewer and CLI dependencies; the tested
reference calculations are not reimplemented in this file.
"""

import ast
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
import torch

from gobot.rl.locomotion.config import VelocityTaskCfg
from gobot.rl.locomotion.cpu_env import _TASK_PARAM
from gobot.rl.locomotion.warp_env import WarpVelocityEnv
from test_mjlab_training import REFERENCE as EXPORTED_REFERENCE, _cpu_probe


REFERENCE = Path(__file__).resolve().parents[2] / "mjlab/src/mjlab"
requires_reference = pytest.mark.skipif(
    not all((REFERENCE / source).is_file() for source in (
        "managers/reward_manager.py", "managers/observation_manager.py",
        "tasks/velocity/velocity_env_cfg.py", "tasks/velocity/mdp/rewards.py",
        "utils/nan_guard.py")),
    reason="requires the pinned mjlab source checkout for independent numerical comparisons",
)


def _execute(nodes, source, namespace=None):
    assert EXPORTED_REFERENCE["reference_revision"] == "dcdf6ce519abfd2ea45e9503ae6fb650b75bb869"
    expected = EXPORTED_REFERENCE["source_sha256"].get("src/mjlab/" + source)
    if expected is not None:
        assert hashlib.sha256((REFERENCE / source).read_bytes()).hexdigest() == expected
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    scope = {"__name__": __name__, "torch": torch, "np": np, "math": math,
             "dataclass": dataclass, "field": field, "contextmanager": contextmanager,
             **(namespace or {})}
    exec(compile(ast.fix_missing_locations(module), str(REFERENCE / source), "exec"), scope)
    return scope


def _reference(source, name, owner=None, namespace=None):
    tree = ast.parse((REFERENCE / source).read_text())
    if owner:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    node = next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
    return _execute([node], source, namespace)[name]


@pytest.mark.parametrize("settings,expected", [
    ({}, "ImplicitFast"), ({"iterations": 17}, "ImplicitFast"),
    ({"iterations": 17, "integrator": "Implicit"}, "Implicit"),
])
def test_generic_velocity_defaults_preserve_explicit_solver_choice(settings, expected):
    original = dict(settings)
    cfg = VelocityTaskCfg(mujoco_solver_settings=settings)
    assert cfg.mujoco_solver_settings == {"integrator": expected, **original}
    assert settings == original


@pytest.mark.parametrize("settings,expected", [
    ({"iterations": 17}, mujoco.mjtIntegrator.mjINT_IMPLICITFAST),
    ({"iterations": 17, "integrator": "Implicit"}, mujoco.mjtIntegrator.mjINT_IMPLICIT),
])
def test_custom_robot_compiles_selected_integrator(tmp_path, settings, expected):
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


@requires_reference
def test_reference_velocity_uses_regular_terminations_and_unsanitized_observations():
    group = _reference("managers/observation_manager.py", "ObservationGroupCfg")
    source = "tasks/velocity/velocity_env_cfg.py"
    tree = ast.parse((REFERENCE / source).read_text())
    factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_velocity_env_cfg")
    assignments = [node for node in factory.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id in ("observations", "terminations")
                           for target in node.targets)]
    funcs = NS(time_out=object(), bad_orientation=object(), out_of_terrain_bounds=object())
    cfg = _execute(assignments, source, {"actor_terms": {}, "critic_terms": {},
        "ObservationGroupCfg": group,
        "TerminationTermCfg": lambda **kwargs: NS(**{"time_out": False, **kwargs}), "mdp": funcs})
    assert set(cfg["terminations"]) == {"time_out", "fell_over", "out_of_terrain_bounds"}
    assert cfg["terminations"]["fell_over"].params["limit_angle"] == math.radians(70)
    assert cfg["terminations"]["time_out"].time_out
    assert cfg["terminations"]["out_of_terrain_bounds"].time_out
    assert cfg["observations"]["actor"].enable_corruption
    assert not cfg["observations"]["critic"].enable_corruption
    handle = _reference("managers/observation_manager.py", "_check_and_handle_nans", "ObservationManager")
    values = torch.tensor([[4000., float("nan"), float("inf"), -float("inf")]])
    for obs_group in cfg["observations"].values():
        assert obs_group.nan_policy == "disabled" and obs_group.history_length is None
        result = handle(None, values, "probe", obs_group.nan_policy)
        assert result is values
        torch.testing.assert_close(result, values, equal_nan=True)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
@requires_reference
def test_reference_nan_guard_is_optional_and_dump_does_not_abort_or_reset(bad_value):
    source = "utils/nan_guard.py"
    cfg_cls = _reference(source, "NanGuardCfg")
    guard_cls = _reference(source, "NanGuard")
    cfg = cfg_cls()
    assert not cfg.enabled
    guard = guard_cls(cfg, 2, None)
    with guard.watch(None):
        pass  # Disabled guard does not inspect data.
    guard = guard_cls.__new__(guard_cls)
    guard.enabled = True
    guard._dumped = False
    guard._dump_buffer = Mock()
    state = NS(**{name: torch.zeros(2, 3) for name in
                  ("qpos", "qvel", "qacc", "qacc_warmstart", "sensordata")})
    state.qvel[0, 0] = 4000.
    assert not guard.check_and_dump(state)
    state.qvel[1, 0] = bad_value
    before = state.qvel.clone()
    assert guard.check_and_dump(state)
    assert not guard.check_and_dump(state)
    guard._dump_buffer.assert_called_once_with([1])
    torch.testing.assert_close(state.qvel, before, equal_nan=True)


def _warp_reward_probe(cpu, state, dt):
    tensor = lambda value: torch.as_tensor(value).clone()
    env = WarpVelocityEnv.__new__(WarpVelocityEnv)
    env.cfg_obj = cpu.cfg_obj
    env.num_envs, env.num_actions, env.step_dt = cpu.num_envs, cpu.num_actions, dt
    env._root_velocity_body = lambda: (tensor(state.base_linear_velocity_body), tensor(state.base_angular_velocity_body))
    env._root_link_velocity_world = lambda: (tensor(state.base_linear_velocity), tensor(state.base_angular_velocity))
    env._foot_velocity_world = lambda: tensor(state.foot_velocity)
    env._arrays = {"qpos": tensor(state.joint_position), "xquat": tensor(state.base_quaternion)[:, None, :],
                   "cvel": torch.cat((tensor(state.base_angular_velocity), torch.zeros(cpu.num_envs, 3)), 1)[:, None, :],
                   "sensordata": tensor(state.angular_momentum)}
    env._base_body_id = env._torso_body_id = 0
    env._joint_qpos_ids = torch.arange(cpu.num_actions)
    env._default_joint_pos = tensor(state.default_joint_position)
    env._terrain_up_world = torch.tensor([[0., 0., 1.]]).expand(cpu.num_envs, -1)
    for name in ("pose_std_standing", "pose_std_walking", "pose_std_running", "action", "previous_action", "command"):
        setattr(env, "_" + name, tensor(getattr(state, name)))
    env._joint_limits = torch.stack((tensor(state.joint_lower_limit), tensor(state.joint_upper_limit)), dim=-1)
    env._feet_contact = {"found": tensor(state.foot_contact), "force": tensor(state.foot_contact_force)}
    env._foot_height, env._foot_peak_height = tensor(state.foot_height), tensor(state.foot_peak_height)
    env._current_contact_time = tensor(state.first_contact) * .005
    env._current_air_time = tensor(state.foot_air_time)
    env._contact_histories = {"self_collision": torch.zeros(cpu.num_envs, 1, 1, 3)}
    env._sensor_slices = (slice(0, 0), slice(0, 0), slice(0, 3))
    env._reward_term_names = cpu._reward_term_names
    env._reward_weights = tensor(state.reward_weights)
    env._reward_terms = torch.zeros_like(tensor(state.reward_terms))
    env._reward = torch.zeros(cpu.num_envs)
    env._episode_returns = torch.zeros(cpu.num_envs)
    return env


def _reference_rewards(momentum, weight, dt):
    angular = _reference("tasks/velocity/mdp/rewards.py", "angular_momentum_penalty")
    env = NS(scene={"momentum": NS(data=torch.tensor(momentum))}, extras={"log": {}})
    bad = lambda env: angular(env, "momentum")
    if weight == 0:
        bad = Mock(side_effect=AssertionError("zero-weight reference term was evaluated"))
    manager = NS(_env=env, _scale_by_dt=True, _reward_buf=torch.zeros(4), _step_reward=torch.zeros(4, 2),
                 _episode_sums={"good": torch.zeros(4), "bad": torch.zeros(4)},
                 _term_names=["good", "bad"], _term_cfgs=[
                     NS(func=lambda env: torch.ones(4), weight=3., params={}),
                     NS(func=bad, weight=weight, params={})],
                 _check_term_shape=lambda name, value: None)
    compute = _reference("managers/reward_manager.py", "compute", "RewardManager")
    result = compute(manager, dt)
    if weight == 0:
        bad.assert_not_called()
    return result, manager._step_reward


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@requires_reference
@pytest.mark.parametrize("magnitude,weight,dt", [
    (2., .7, .02), (float("nan"), .7, .02), (float("inf"), .7, .02),
    (-float("inf"), .7, .02), (1.e10, 1., .02), (1.e10, 1.e20, .02),
    (1.e10, 1., 1.e20), (float("nan"), 0., .02),
])
def test_weighted_reward_terms_match_reference_with_nonfinite_and_overflow(backend, magnitude, weight, dt):
    cpu, state = _cpu_probe()
    state.reward_weights[:] = 0.
    state.reward_weights[0], state.reward_weights[5] = 3., weight
    state.command[:] = state.base_linear_velocity[:] = state.base_linear_velocity_body[:] = 0.
    state.angular_momentum[:] = 0.
    state.angular_momentum[:, 0] = [magnitude, 2., 3., 4.]
    state.task_params[_TASK_PARAM["step_dt"]] = dt
    # The real environment stores dt and weights as float32 in its batch state.
    dt, weight = float(state.task_params[_TASK_PARAM["step_dt"]]), float(state.reward_weights[5])
    expected, expected_rates = _reference_rewards(state.angular_momentum, weight, dt)
    if backend == "numpy":
        with np.errstate(over="ignore", invalid="ignore"):
            cpu._run_velocity_task_numpy()
        actual, rates = torch.tensor(state.reward), torch.tensor(state.reward_terms[:, [0, 5]])
    else:
        env = _warp_reward_probe(cpu, state, dt)
        env._compute_rewards()
        actual, rates = env._reward, env._reward_terms[:, [0, 5]]
    torch.testing.assert_close(actual, expected, rtol=5e-6, atol=1e-7)
    torch.testing.assert_close(rates, expected_rates, rtol=5e-6, atol=1e-7)
    assert torch.isfinite(actual).all()
