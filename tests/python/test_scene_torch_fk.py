"""Scene-derived FK agrees with compiled physics, including offset/fixed joints."""
import numpy as np
import pytest
import torch
import gobot
import mujoco
from gobot.rl.kinematics import SceneKinematics
from gobot.rl.locomotion.math import find_node_by_name


@pytest.fixture
def fixture(tmp_path):
    source = tmp_path / 'robot.xml'
    source.write_text('''<mujoco model="fk_test"><compiler angle="radian"/>
    <worldbody><body name="base" pos="0.1 0.2 1"><freejoint name="root"/>
      <geom type="sphere" size="0.1"/>
      <body name="spacer" pos="0.2 -0.1 0.3" euler="0.1 0.2 0.3">
        <geom type="sphere" size="0.05"/>
        <body name="arm" pos="0.2 0.3 0.1" euler="0.2 0 0.1">
          <joint name="hinge" axis="0 1 0" pos="0.07 0.03 -0.04" range="-2 2"/>
          <geom type="capsule" size="0.04 0.15"/>
          <body name="tip" pos="0.2 0.1 0.3"><joint name="slide" type="slide" axis="1 0 0" range="-0.2 0.2"/>
            <geom type="sphere" size="0.03"/>
          </body>
        </body>
      </body>
    </body></worldbody></mujoco>''')
    gobot.set_project_path(str(tmp_path))
    gobot.import_mjcf_scene(str(source), 'res://robot.jscn', name='fk_test')
    context = gobot.app.context(); context.set_project_path(str(tmp_path))
    context.load_scene('res://robot.jscn')
    # Initial coordinates must not be silently included in the fixed frames.
    find_node_by_name(context.root, 'hinge').initial_position = .37
    snapshot = context.compile_physics_scene_snapshot()
    artifact = context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
    kin = SceneKinematics(snapshot, robot_name='fk_test', base_link='base',
                          joint_names=('slide', 'hinge'), dtype=torch.float64)
    yield kin, mujoco.MjModel.from_xml_string(artifact['content']), snapshot
    context.clear_scene()


def test_random_pose_velocity_and_gradients(fixture):
    kin, model, _ = fixture
    rng = np.random.default_rng(42)
    body_ids = [model.body('fk_test_'+name).id for name in kin.body_names]
    joints = [model.joint('fk_test_'+name) for name in kin.joint_names]
    for _ in range(8):
        data = mujoco.MjData(model)
        data.qpos[:3] = rng.normal(size=3)
        quat = rng.normal(size=4); quat /= np.linalg.norm(quat); data.qpos[3:7] = quat
        for joint in joints:
            data.qpos[joint.qposadr[0]] = rng.uniform(-.15, .15)
        data.qvel[:] = rng.normal(size=model.nv)
        mujoco.mj_forward(model, data)
        root = model.body('fk_test_base').id
        q = np.array([data.qpos[j.qposadr[0]] for j in joints])
        qd = np.array([data.qvel[j.dofadr[0]] for j in joints])
        rotation = data.xmat[root].reshape(3, 3).copy()
        inputs = [data.xpos[root].copy(), rotation, data.qvel[:3].copy(), rotation @ data.qvel[3:6], q, qd]
        tensors = [torch.tensor(value, dtype=torch.float64) for value in inputs]
        p, r, v, w = kin(*tensors, all_bodies=True)
        np.testing.assert_allclose(p.detach(), data.xpos[body_ids], atol=1.e-6)
        np.testing.assert_allclose(r.detach(), data.xmat[body_ids].reshape(-1, 3, 3), atol=1.e-6)
        old_p, old_r = data.xpos[body_ids].copy(), data.xmat[body_ids].reshape(-1, 3, 3).copy()
        dt = 1.e-7
        mujoco.mj_integratePos(model, data.qpos, data.qvel, dt); mujoco.mj_forward(model, data)
        np.testing.assert_allclose(v.detach(), (data.xpos[body_ids]-old_p)/dt, atol=1.e-4)
        rotation_dot = (data.xmat[body_ids].reshape(-1, 3, 3)-old_r)/dt
        skew = rotation_dot @ old_r.transpose(0, 2, 1)
        angular = np.stack((skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]), -1)
        np.testing.assert_allclose(w.detach(), angular, atol=1.e-4)
    q = tensors[4].clone().requires_grad_()
    qd=tensors[5].clone().requires_grad_()
    assert torch.autograd.gradcheck(lambda angles,rates: kin(*tensors[:4],angles,rates), (q,qd), eps=1.e-6, atol=1.e-5)


def test_invalid_topology_rejected(fixture):
    from copy import deepcopy
    _, _, snapshot = fixture
    broken = deepcopy(snapshot)
    robot = next(r for r in broken['robots'] if r['name'] == 'fk_test')
    robot['joints'] = [j for j in robot['joints'] if int(j['type']) != 0]
    with pytest.raises(ValueError, match='disconnected'):
        SceneKinematics(broken, robot_name='fk_test', base_link='base', joint_names=('slide', 'hinge'))


def test_static_world_sibling_is_not_part_of_articulation_fk(fixture):
    from copy import deepcopy
    original, _, snapshot = fixture
    snapshot = deepcopy(snapshot)
    robot = next(r for r in snapshot['robots'] if r['name'] == 'fk_test')
    terrain = deepcopy(robot['links'][0])
    terrain['name'] = 'terrain'
    robot['links'].append(terrain)
    kin = SceneKinematics(snapshot, robot_name='fk_test', base_link='base',
                          joint_names=('slide', 'hinge'), dtype=torch.float64)
    assert kin.body_names == original.body_names
    assert kin.digest == original.digest
    with pytest.raises(ValueError, match='selected articulation'):
        SceneKinematics(snapshot, robot_name='fk_test', base_link='base',
                        joint_names=('slide', 'hinge'), tracked_links=('terrain',))
