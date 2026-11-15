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
