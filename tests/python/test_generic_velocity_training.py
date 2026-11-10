"""The shared trainer runs typed tasks without a built-in robot selector.

The integration fixture reuses Go1 dynamics under a different authored robot name.
It tests the generic configuration/training boundary, not new robot dynamics.
"""
from copy import deepcopy
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import mujoco
import pytest
import torch

from gobot.rl.locomotion.config import VelocityTaskCfg
from gobot.rl.locomotion.experiment import RunConfig
from gobot.rl.policy import policy_manifest_from_checkpoint
from gobot.rl.training import launch
from gobot.rl.training.locomotion import training_spec

ROOT = Path(__file__).resolve().parents[2]


def custom_task(project):
    """Prepare test assets once; the trainer receives only a generic typed task."""
    from examples.go1.train.go1_velocity_cfg import go1_velocity_cfg

    source = ROOT / 'examples/go1'
    project.mkdir()
    robot = json.loads((source / 'go1.jscn').read_text())
    old_nodes = robot['__NODES__']
    keep = [i for i, node in enumerate(old_nodes) if node['type'] != 'MeshInstance3D']
    indices = {old: new for new, old in enumerate(keep)}
    robot['__NODES__'] = [old_nodes[i] for i in keep]
    robot['__EXT_RESOURCES__'] = []  # All external resources in this fixture are visual meshes.
    for node in robot['__NODES__']:
        node['parent'] = indices[node['parent']] if node['parent'] >= 0 else -1
        node['properties']['script'] = None
        if 'source_path' in node['properties']:
            node['properties']['source_path'] = ''
    robot['__NODES__'][0]['name'] = 'walker'
    (project / 'robot.jscn').write_text(json.dumps(robot))

    world = json.loads((source / 'go1_flat_scene.jscn').read_text())
    world['__EXT_RESOURCES__'] = [resource for resource in world['__EXT_RESOURCES__']
                                if resource['__TYPE__'] != 'PythonScript']
    for resource in world['__EXT_RESOURCES__']:
        resource['__PATH__'] = ('res://robot.jscn' if resource['__TYPE__'] == 'PackedScene'
                                else 'res://terrain.jres')
    for node in world['__NODES__']:
        node['properties']['script'] = None
        if 'source_path' in node['properties']:
            node['properties']['source_path'] = ''
        if node['type'] == 'Robot3D':
            node['name'] = 'walker'
    world['__NODES__'][0]['name'] = 'world'
    (project / 'world.jscn').write_text(json.dumps(world))
    terrain = json.loads((source / 'terrain/flat_terrain.jres').read_text())
    values = terrain['__RESOURCE__']
    values['resource_name'] = 'Test terrain'
    values['num_rows'] = values['num_cols'] = 1
    values['sub_terrains'] = values['sub_terrains'][:1]
    values['sub_terrains'][0]['proportion'] = 1.
    (project / 'terrain.jres').write_text(json.dumps(terrain))
    (project / 'project.gobot').write_text(json.dumps({'main_scene': 'res://world.jscn'}))

    source_cfg = go1_velocity_cfg(project_path=project)
    values = {field.name: deepcopy(getattr(source_cfg, field.name)) for field in fields(VelocityTaskCfg)}
    values.update(robot=replace(source_cfg.robot, name='walker'), name='custom_velocity',
                  scene_path='res://world.jscn',
                  robot_resource_path='res://robot.jscn', capture_transitions=True,
                  terrain_curriculum=False, terrain_out_of_bounds=False,
                  episode_length_s=.04, command_curriculum=(), run_command_curriculum=())
    task = VelocityTaskCfg(**values)
    task.observations.include_height_scan = task.observations.actor_noise = False
    task.domain_randomization.enabled = task.push_enabled = False
    task.rewards.upright_reference = 'world'
    task.illegal_contact.enabled = False
    task.illegal_contact.terminate_on_thigh = False
    task.rewards.self_collisions = task.rewards.shank_collision = task.rewards.trunk_head_collision = 0.
    return task


@pytest.mark.parametrize('entry', ['typed', 'factory'])
def test_custom_robot_uses_shared_optimizer_and_resume(tmp_path, monkeypatch, entry):
    task = custom_task(tmp_path / 'project')
    settings = RunConfig(stage='flat', num_envs=2, iterations=2, save_interval=1, hours=.1)
    if entry == 'factory':
        module = ModuleType('custom_velocity_test_task')
        module.make_task = lambda run: deepcopy(task)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        spec = training_spec(task_factory=f'{module.__name__}:make_task', experiment=settings, name='custom')
    else:
        spec = training_spec(env=task, experiment=settings, name='custom')
    cfg = spec.config_factory()
    cfg.backend = 'mujoco-cpu'
    cfg.output = tmp_path / 'train'
    cfg.agent = {'num_steps_per_env': 4,
                 'actor': {'hidden_dims': [16, 8]}, 'critic': {'hidden_dims': [16, 8]},
                 'algorithm': {'num_learning_epochs': 1, 'num_mini_batches': 1}}
    assert not hasattr(cfg.experiment, 'robot')
    assert cfg.resolve_env().robot_name == 'walker'
    summary = launch(spec, cfg)
    assert summary['completed_updates'] == 2
    checkpoint = torch.load(cfg.output / 'model_final.pt', map_location='cpu', weights_only=False)
    manifest = policy_manifest_from_checkpoint(checkpoint)
    assert manifest.extras['robot_name'] == 'walker'
    assert manifest.task_name == 'custom_velocity'
    assert manifest.joint_names == task.robot.joint_names
    state = checkpoint['infos']['rank_states'][0]['env_state']
    assert state['common_step_counter'] == 8
    assert checkpoint['optimizer_state_dict']['state']
    metrics = [json.loads(line) for line in (cfg.output / 'metrics.jsonl').read_text().splitlines()]
    assert any(row['log']['Episode_Termination/time_out'] > 0 for row in metrics)

    resumed = replace(cfg, experiment=replace(settings, iterations=1), output=tmp_path / 'resumed',
                      resume=cfg.output / 'model_final.pt')
    result = launch(spec, resumed)
    assert result['last_iteration'] == 2
    restored = torch.load(resumed.output / 'model_final.pt', map_location='cpu', weights_only=False)
    assert restored['infos']['rank_states'][0]['env_state']['common_step_counter'] == 12
    for role in ('actor', 'critic'):
        assert all(torch.isfinite(value).all() for value in restored[f'{role}_state_dict'].values())
    resolved = json.loads((resumed.output / 'config.json').read_text())
    assert 'robot' not in resolved['experiment']
    assert resolved['task']['robot']['name'] == 'walker'


@pytest.mark.parametrize('factory,exception,message', [
    (None, ValueError, 'provide env'),
    ('missing_velocity_test_module:make', ModuleNotFoundError, 'missing_velocity_test_module'),
    ('custom_velocity_bad_task:missing', AttributeError, 'missing'),
    ('custom_velocity_bad_task:wrong_type', TypeError, 'VelocityTaskCfg'),
    ('custom_velocity_bad_task:failure', RuntimeError, 'task configuration failed'),
])
def test_bad_task_factories_fail_before_simulation(tmp_path, monkeypatch, factory, exception, message):
    from gobot.rl.locomotion import training
    module = ModuleType('custom_velocity_bad_task')
    module.wrong_type = lambda run: {'robot': 'pretend'}
    def failure(run):
        raise RuntimeError('task configuration failed')
    module.failure = failure
    monkeypatch.setitem(sys.modules, module.__name__, module)
    def forbidden(*args, **kwargs):
        pytest.fail('invalid task factory reached simulation')
    monkeypatch.setattr(training, 'make_velocity_env', forbidden)
    spec = training_spec(task_factory=factory)
    cfg = spec.config_factory()
    cfg.backend = 'mujoco-cpu'
    cfg.output = tmp_path / 'not-created'
    with pytest.raises(exception, match=message):
        launch(spec, cfg)
    assert not cfg.output.exists()


def test_core_configuration_does_not_import_examples_or_choose_robot_assets(tmp_path):
    script = '''
import importlib.abc
import sys
class NoExamples(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'examples' or fullname.startswith('examples.'):
            raise AssertionError('core trainer imported a robot example: ' + fullname)
sys.meta_path.insert(0, NoExamples())
from gobot.rl.locomotion.experiment import RunConfig
from gobot.rl.locomotion.training import agent_config
from gobot.rl.training.locomotion import training_spec
run = RunConfig()
assert not hasattr(run, 'robot')
assert run.motion_dataset is None
config = training_spec(experiment=run).config_factory()
assert config.task_factory is None and config.env == {}
assert config.evaluation_command == () and config.no_eval
agent = agent_config(run)
assert agent['algorithm']['class_name'] == 'rsl_rl.algorithms.PPO'
assert not any(name == 'examples' or name.startswith('examples.') for name in sys.modules)
'''
    environ = dict(os.environ)
    environ['PYTHONPATH'] = os.pathsep.join(str(Path(path).resolve()) for path in sys.path if path)
    flags = ['-S'] if sys.flags.no_site else []
    result = subprocess.run([sys.executable, *flags, '-c', script], cwd=tmp_path,
                            env=environ, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


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

    task = replace(custom_task(tmp_path / "project"), mujoco_solver_settings=settings)
    assert task.robot_name == "walker"
    context, _, _ = prepare_velocity_scene(task, context=gobot.app.create_context())
    artifact = context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
    model = mujoco.MjModel.from_xml_string(artifact["content"])
    assert model.opt.integrator == expected
    assert model.opt.iterations == 17
