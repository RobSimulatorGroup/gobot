import torch
import pytest
import gobot
from gobot.rl.articulation import ArticulationBatch
from gobot.sim.providers.base import RobotBatchSpec


def make_batch(tmp_path, *, backend='mujoco-cpu'):
    source = tmp_path/'robot.xml'
    source.write_text('''<mujoco model="batch_test"><compiler angle="radian"/>
    <worldbody><geom name="ground" type="plane" size="0 0 .05"/>
      <body name="base" pos="0 0 1"><freejoint/>
        <geom type="sphere" size=".1"/>
        <body name="foot" pos=".1 0 -.2"><joint name="hinge" axis="0 1 0" range="-2 2"/>
          <geom type="sphere" size=".05"/>
        </body>
      </body>
    </worldbody><actuator><position joint="hinge" kp="20" kv="1"/></actuator></mujoco>''')
    gobot.set_project_path(str(tmp_path))
    gobot.import_mjcf_scene(str(source), 'res://robot.jscn', name='batch_test')
    return ArticulationBatch(project_path=tmp_path, scene_path='res://robot.jscn',
        spec=RobotBatchSpec('batch_test', 'base', ('hinge',), ('foot',)), num_envs=2,
        backend=backend, contact_links=('foot',), solver_settings={'integrator': 'ImplicitFast'})


def test_masked_reset_and_stable_state(tmp_path):
    batch = make_batch(tmp_path)
    try:
        initial = batch.read_state()
        pointers = [value.data_ptr() for value in initial.__dict__.values()]
        saved = {key: value.clone() for key, value in initial.__dict__.items()}
        batch.set_joint_targets(torch.full((2, 1), .3))
        batch.step_control()
        second = {key: value[1].clone() for key, value in batch.read_state().__dict__.items()}
        batch.reset([True, False], base_pose=saved['base_pose'], base_velocity=saved['base_velocity'],
            joint_position=saved['joint_position'], joint_velocity=saved['joint_velocity'], targets=saved['joint_position'])
        state = batch.read_state()
        assert pointers == [value.data_ptr() for value in state.__dict__.values()]
        for key, value in state.__dict__.items():
            torch.testing.assert_close(value[1], second[key], rtol=0, atol=0)
        torch.testing.assert_close(state.joint_velocity[0], saved['joint_velocity'][0], rtol=0, atol=0)
        torch.testing.assert_close(state.joint_control[0], saved['joint_position'][0])
        with pytest.raises(ValueError, match='finite shape'):
            batch.set_joint_targets([[float('nan')], [0.]])
    finally:
        batch.close()
    with pytest.raises(RuntimeError, match='closed'):
        batch.read_state()


def test_contact_impulse_is_aggregated_over_control_ticks(tmp_path):
    batch = make_batch(tmp_path)
    try:
        state = batch.read_state()
        pose = state.base_pose.clone(); pose[:, 2] = .245
        q = state.joint_position.clone()
        batch.reset([True, True], base_pose=pose, base_velocity=torch.zeros_like(state.base_velocity),
            joint_position=q, joint_velocity=torch.zeros_like(q), targets=q)
        for _ in range(4):
            batch.step_control()
        contacts = batch.read_contacts()
        assert contacts['found'].any()
        assert torch.isfinite(contacts['force']).all()
        assert contacts['impulse'].norm(dim=-1).max() > 0
    finally:
        batch.close()


def test_scene_presentation_does_not_change_saved_asset_or_physics_state(tmp_path):
    batch=make_batch(tmp_path)
    try:
        original=(tmp_path/'robot.jscn').read_bytes()
        batch.set_joint_targets(torch.full((2,1),.2));batch.step_control()
        before={key:value.clone() for key,value in batch.read_state().__dict__.items()}
        batch.sync_scene(1)
        for key,value in batch.read_state().__dict__.items():
            torch.testing.assert_close(value,before[key],rtol=0,atol=0)
        assert (tmp_path/'robot.jscn').read_bytes()==original
        with pytest.raises(IndexError):batch.sync_scene(2)
    finally:batch.close()


def test_substep_observer_reads_preintegration_state_without_changing_dynamics(tmp_path):
    batch=make_batch(tmp_path)
    try:
        initial={key:value.clone() for key,value in batch.read_state().__dict__.items()}
        seen=[]
        def observer(tick):
            batch.refresh_kinematics()
            seen.append((tick,batch.read_state().base_pose.clone()))
        batch.step_control(before_substep=observer)
        final=batch.read_state().base_pose.clone()
        assert [t for t,_ in seen]==list(range(batch.decimation))
        torch.testing.assert_close(seen[0][1],initial['base_pose'],atol=0,rtol=0)
        assert not torch.equal(seen[-1][1],final)
        batch.reset([True,True],base_pose=initial['base_pose'],base_velocity=initial['base_velocity'],
                    joint_position=initial['joint_position'],joint_velocity=initial['joint_velocity'],targets=initial['joint_control'])
        batch.step_control()
        torch.testing.assert_close(batch.read_state().base_pose,final,atol=0,rtol=0)
    finally:
        batch.close()


def test_friction_only_update_does_not_recompute_com_constants():
    from types import SimpleNamespace
    from unittest.mock import Mock
    batch=ArticulationBatch.__new__(ArticulationBatch)
    batch._closed=False;batch.num_envs=2;batch.device='cpu'
    batch._targets=torch.zeros(2,1)
    batch.spec=SimpleNamespace(robot_name='robot')
    batch.snapshot={'robots':[{'name':'robot','links':[{'name':'base','shapes':[{'name':'foot'}]}]}]}
    batch.artifact=SimpleNamespace(robot_prefix=lambda name:'robot_')
    batch._physics_parameters={'link_center_of_mass_offsets':{'base':torch.ones(2,3)*.01},'shape_friction':{}}
    friction=torch.zeros(2,1,3)
    provider=SimpleNamespace(expand_model_fields=Mock(),model_array=Mock(return_value=friction),
        resolve_object_ids=Mock(return_value=[0]),recompute_constants=Mock(),refresh_kinematics=Mock())
    batch._provider=provider
    value=torch.tensor([[.3,.005,.0001],[1.2,.005,.0001]])
    batch.configure_physics(shape_friction={'foot':value})
    provider.recompute_constants.assert_not_called()
    provider.expand_model_fields.assert_called_once_with(['geom_friction'])
    torch.testing.assert_close(friction[:,0],value)
    torch.testing.assert_close(batch._physics_parameters['link_center_of_mass_offsets']['base'],torch.ones(2,3)*.01)
