"""A single policy-loading contract serves evaluation and live Editor playback."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import gobot

from test_tracking import tiny_config
from gobot.rl.tracking.training import Config, default_agent, run
from gobot.rl.tracking.inference import load_policy
from gobot.rl.tracking.play import prepare, walk


@pytest.fixture
def checkpoint(tmp_path):
    agent = default_agent()
    agent['actor']['hidden_dims'] = [16]
    agent['critic']['hidden_dims'] = [16]
    agent['num_steps_per_env'] = 4
    agent['algorithm']['num_learning_epochs'] = 1
    agent['algorithm']['num_mini_batches'] = 2
    env_cfg = tiny_config(tmp_path)
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(tmp_path));root=context.load_scene('res://world.jscn')
        base=next(node for node in walk(root) if node.name=='base')
        base.add_child(gobot.create_box_visual('BodyVisual',(.1,.1,.1)))
        gobot.save_scene(root,str(tmp_path/'world.jscn'))
    finally:context.clear_scene()
    cfg = Config(**asdict(env_cfg), updates=1, agent=agent,
                 output=tmp_path/'run', torch_threads=1)
    run(cfg)
    return cfg.output/'model.pt'


def test_editor_project_has_separate_materials_floor_and_current_policy(checkpoint, tmp_path):
    output = prepare(checkpoint, tmp_path/'editor')
    doc = json.loads((output/'world.jscn').read_text())
    robots = [node for node in doc['__NODES__'] if node['type']=='Robot3D']
    assert {node['name'] for node in robots} == {'Policy','Reference'}
    assert any(node['name']=='Floor' for node in doc['__NODES__'])
    materials = [resource for resource in doc['__SUB_RESOURCES__'] if resource['__TYPE__']=='PBRMaterial3D']
    assert len({json.dumps(resource['albedo'],sort_keys=True) for resource in materials})>=2
    settings = json.loads((output/'play.json').read_text())
    assert Path(settings['runtime']['checkpoint'])==checkpoint.resolve()
    assert settings['link_names']==['base','anchor']
    context=gobot.app.create_context()
    try:
        context.set_project_path(str(output));root=context.load_scene('res://world.jscn')
        assert root.find('Policy') is not None and root.find('Reference') is not None
    finally:context.clear_scene()


def test_current_checkpoint_required_without_migration(checkpoint, tmp_path):
    payload=torch.load(checkpoint,weights_only=True)
    payload['infos']['tracking_contract']['schema']='obsolete'
    old=checkpoint.with_name('obsolete.pt');torch.save(payload,old)
    with pytest.raises(ValueError,match='current tracking contract'):
        load_policy(old,backend='mujoco-cpu')


def test_live_policy_advances_physics_and_holds_final_pose(checkpoint, monkeypatch):
    import gobot.rl.tracking.playback as playback
    monkeypatch.setattr(playback,'load_policy',lambda checkpoint,**kw:load_policy(checkpoint,backend='mujoco-cpu'))
    player=playback.TrackingPolicy(checkpoint,device='cpu')
    try:
        initial=player.arrays['policy'].copy()
        player.step(nsteps=3)
        assert player.steps==3 and not player.finished
        assert player.arrays['policy'].shape==(1,2,7)
        assert np.isfinite(player.arrays['reference']).all()
        assert not np.array_equal(initial,player.arrays['policy'])
        player.step(nsteps=20)
        assert player.finished and player.steps<=11
        final=player.arrays['policy'].copy()
        assert not player.env.state.info['motion_wrap'].any() or player.env.pending_reset.all()
        player.step(nsteps=3)
        np.testing.assert_array_equal(final,player.arrays['policy'])
        player.reset([True]);assert player.steps==0 and not player.finished
    finally:player.close()


def test_deployment_preserves_actor_and_loads_from_relocated_project(checkpoint, tmp_path):
    import shutil
    from gobot.rl.tracking.export import export
    target = export(checkpoint, tmp_path / 'deployment')
    source = torch.load(checkpoint, weights_only=True)
    deployed = torch.load(target, weights_only=True)
    assert set(deployed) == {'actor_state_dict', 'infos'}
    for name, value in source['actor_state_dict'].items():
        torch.testing.assert_close(value, deployed['actor_state_dict'][name], rtol=0, atol=0)
    project = tmp_path / 'relocated'
    project.mkdir()
    for name in ('world.jscn', 'motion.npz'):
        shutil.copy2(tmp_path / name, project / name)
    env, wrapper, policy = load_policy(target, backend='mujoco-cpu', project_path=project)
    try:
        assert torch.isfinite(policy(wrapper.get_observations())).all()
        from gobot.rl.tracking.inference import validate_contract
        info = deployed['infos']
        # Compiler-dependent XML/FK hashes may differ, but all authored data
        # and complete physical snapshots must still match.
        info['tracking_contract']['scene_digest'] = 'another-compiler'
        info['tracking_contract']['kinematics_digest'] = 'another-blas'
        validate_contract(info, env)
        info['tracking_contract']['motion_digest'] = 'another-motion'
        with pytest.raises(ValueError, match='motion, observations or controls'):
            validate_contract(info, env)
    finally:
        env.close()
    scene = project / 'world.jscn'
    doc = json.loads(scene.read_text())
    doc['__NODES__'][0]['name'] = 'Changed'
    scene.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='authoring scene or assets'):
        load_policy(target, backend='mujoco-cpu', project_path=project)


def test_deployment_snapshot_tolerates_only_transform_roundoff():
    from copy import deepcopy
    from gobot.rl.tracking.inference import _validate_snapshot
    original = dict(mass=1.0, global_transform=dict(matrix=[[1.0, 0.0], [0.0, 1.0]]))
    rounded = deepcopy(original)
    rounded['global_transform']['matrix'][0][0] += float(np.finfo(np.float32).eps)
    _validate_snapshot(original, rounded)
    for value in (0.001, float('nan'), float('inf')):
        changed = deepcopy(original)
        changed['global_transform']['matrix'][0][1] = value
        with pytest.raises(ValueError, match='deployment physics differs'):
            _validate_snapshot(original, changed)
    changed = deepcopy(original)
    changed['mass'] += 1.e-8
    with pytest.raises(ValueError, match='mass'):
        _validate_snapshot(original, changed)


def test_g1_track_project_is_portable(checkpoint, tmp_path):
    from gobot.rl.tracking.export import export
    from examples.g1_track.prepare import prepare as prepare_example
    from types import SimpleNamespace
    from gobot.rl.tracking.editor_play import Script
    import gobot.rl.tracking.editor_play as editor_play
    from unittest.mock import patch
    deployment = export(checkpoint, tmp_path / 'deployment')
    # A standalone prepared source project, without the trainer outputs.
    source = tmp_path / 'source'
    source.mkdir()
    import shutil
    for name in ('world.jscn', 'motion.npz'):
        shutil.copy2(tmp_path / name, source / name)
    output = prepare_example(deployment, source, output=tmp_path / 'example')
    scene = json.loads((output / 'world.jscn').read_text())
    assert all(r['__PATH__'].startswith('res://') for r in scene['__EXT_RESOURCES__'])
    assert json.loads((output / 'project.gobot').read_text())['project_load_hook'] == 'res://prepare.py'
    settings = json.loads((output / 'play.json').read_text())
    assert settings['runtime']['checkpoint'] == 'res://assets/policy/policy.pt'
    assert (output / 'prepare.py').is_file()
    moved = tmp_path / 'moved'
    output.rename(moved)
    env, wrapper, policy = load_policy(moved / 'assets/policy/policy.pt', backend='mujoco-cpu',
                                      project_path=moved / 'assets/project')
    try:
        assert torch.isfinite(policy(wrapper.get_observations())).all()
    finally:
        env.close()
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(moved))
        root = context.load_scene('res://world.jscn')
        instance = SimpleNamespace(context=context, get_root=lambda: root, _sync=lambda snapshot: None)
        class CaptureRuntime:
            def __init__(self, recipe, **kwargs):
                assert json.loads(recipe.recipe)['arguments']['value']['checkpoint'] == str(moved / 'assets/policy/policy.pt')
                assert json.loads(recipe.recipe)['arguments']['value']['project_path'] == str(moved / 'assets/project')
        class Session:
            def __init__(self, *args, **kwargs): pass
            def start(self): return self
            def set_status(self, status): pass
        with patch.object(editor_play.gobot.sim, 'AsyncSimulationSession', CaptureRuntime), \
             patch.object(editor_play.gobot.sim, 'ProviderPlaySession', Session):
            Script._ready(instance)
    finally:
        context.clear_scene()
