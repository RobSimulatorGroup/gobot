"""Shared training data must preserve actual geometry, origins and rank sampling."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from gobot.rl.locomotion import make_velocity_env
from examples.locomotion.presets import Experiment, task_config
from gobot.rl.locomotion.terrain_reference import (
    from_environment, load, add_to_spec, initial_assignments, assignment_digest, verify_environment,
)


@pytest.fixture(scope='module')
def terrain_env():
    cfg = task_config(Experiment(robot='g1', stage='rough'))
    env = make_velocity_env(cfg, backend='mujoco-cpu', num_envs=1, collect_step_extras=False)
    yield env
    env.close()


def test_exported_geometry_matches_compiled_scene(terrain_env, tmp_path):
    import mujoco
    env = terrain_env
    data = from_environment(env)
    path = tmp_path/'terrain.json'
    path.write_text(json.dumps(data))
    loaded = load(path)
    verify_environment(env, loaded)
    source = mujoco.MjModel.from_xml_string(env.compiled_artifact.content)
    spec = mujoco.MjSpec()
    add_to_spec(spec, loaded)
    target = spec.compile()
    np.testing.assert_array_equal(source.hfield_data, target.hfield_data)
    np.testing.assert_array_equal(source.hfield_size, target.hfield_size)
    np.testing.assert_array_equal(source.hfield_nrow, target.hfield_nrow)
    np.testing.assert_array_equal(source.hfield_ncol, target.hfield_ncol)
    for geom in range(target.ngeom):
        original = source.geom(target.geom(geom).name).id
        for name in ('type', 'pos', 'quat', 'size', 'contype', 'conaffinity', 'condim',
                     'friction', 'solref', 'solimp', 'solmix', 'margin', 'gap'):
            np.testing.assert_array_equal(getattr(source,'geom_'+name)[original],
                                          getattr(target,'geom_'+name)[geom])
    np.testing.assert_array_equal(np.asarray(data['origins']).reshape(-1,3),
                                  np.asarray(env.terrain_origins,dtype=np.float32))
    loaded['seed'] += 1
    path.write_text(json.dumps(loaded))
    with pytest.raises(ValueError,match='checksum'):load(path)
    loaded=deepcopy(data);loaded['origins'][0][0][2]+=1
    with pytest.raises(ValueError,match='scene terrain'):
        verify_environment(env,{**loaded,'sha256':'changed'})


@pytest.mark.parametrize('count',[3,7,64,1024])
def test_rank_sampling_is_shared_and_population_is_exact(terrain_env, count):
    data=from_environment(terrain_env)
    levels,types=initial_assignments(data,count,42)
    again=initial_assignments(data,count,42)
    assert assignment_digest(levels,types)==assignment_digest(*again)
    assert len(levels)==len(types)==count
    assert (levels>=0).all() and (levels<=data['max_init_level']).all()
    other=initial_assignments(data,count,43)
    assert assignment_digest(levels,types)!=assignment_digest(*other)
    if count>=7:assert (np.bincount(types,minlength=7)>0).all()
    # Both runtime paths call this sampler; the RNG does not consume Torch/global NumPy state.
    before=np.random.get_state()
    initial_assignments(data,count,71)
    after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1])
