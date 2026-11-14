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


def test_link_pose_batch_handles_child_before_parent(tmp_path):
    from gobot.rl.locomotion.math import find_node_by_name
    batch = make_batch(tmp_path)
    try:
        root = find_node_by_name(batch.context.root, 'batch_test')
        links = [find_node_by_name(root, name) for name in ('foot', 'base')]
        original = batch.read_state().base_pose.clone()
        for offset in range(32):
            poses = torch.tensor([[offset + 2., 3., 4., 0., 0., 0., 1.],
                                  [offset + 1., 2., 3., 0., 0., 0., 1.]])
            batch.context.apply_link_poses(links, poses.numpy())
            snapshot = batch.context.compile_physics_scene_snapshot()
            robot = next(r for r in snapshot['robots'] if r['name'] == 'batch_test')
            actual = {link['name']: link['global_transform']['position'] for link in robot['links']}
            for index, name in enumerate(('foot', 'base')):
                torch.testing.assert_close(torch.tensor(actual[name], dtype=poses.dtype), poses[index, :3])
        torch.testing.assert_close(batch.read_state().base_pose, original, rtol=0, atol=0)
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


def test_static_world_contacts_exclude_self_and_point_away_from_ground(tmp_path):
    source=tmp_path/'self_contact.xml'
    source.write_text('''<mujoco model="self_test"><option gravity="0 0 0"/>
      <worldbody><geom name="ground" type="plane" size="0 0 .1"/>
        <body name="base" pos="0 0 1"><freejoint/>
          <geom type="sphere" size=".1"/>
          <body name="arm" pos="0 0 .3"><joint name="arm_joint"/>
            <geom type="sphere" size=".02"/>
            <body name="hand" pos=".15 0 -.3"><joint name="hand_joint"/>
              <geom type="sphere" size=".1"/>
            </body>
          </body>
        </body>
      </worldbody><actuator><position joint="arm_joint" kp="1"/>
      <position joint="hand_joint" kp="1"/></actuator></mujoco>''')
    gobot.set_project_path(str(tmp_path))
    gobot.import_mjcf_scene(str(source),'res://self.jscn',name='self_test')
    def create(scope):
        return ArticulationBatch(project_path=tmp_path,scene_path='res://self.jscn',
            spec=RobotBatchSpec('self_test','base',('arm_joint','hand_joint'),('base','arm','hand')),
            num_envs=1,decimation=1,contact_links=('base','hand'),contact_scope=scope,
            self_contact_threshold=.1)
    all_contacts=create('all')
    try:
        all_contacts.step_control()
        assert all_contacts.read_contacts()['force'].abs().max()>.1
        assert all_contacts.read_self_contact_substeps().item()==1
    finally:all_contacts.close()
    world=create('static_world')
    try:
        world.step_control()
        assert world.read_contacts()['force'].abs().max()==0
        assert world.read_self_contact_substeps().item()==1
        state=world.read_state()
        pose=state.base_pose.clone();pose[:,2]=.08
        world.reset([True],base_pose=pose,base_velocity=torch.zeros_like(state.base_velocity),
                    joint_position=state.joint_position,joint_velocity=torch.zeros_like(state.joint_velocity),
                    targets=state.joint_position)
        world.step_control()
        assert world.read_contacts()['force'][0,0,2]>0
    finally:world.close()


def test_masked_reset_validates_only_rows_it_applies(tmp_path):
    batch=make_batch(tmp_path)
    try:
        saved={k:v.clone() for k,v in batch.read_state().__dict__.items()}
        values={k:v.clone() for k,v in saved.items()}
        for value in values.values():value[1]=float('nan')
        arguments=dict(base_pose=values['base_pose'],base_velocity=values['base_velocity'],
            joint_position=values['joint_position'],joint_velocity=values['joint_velocity'],targets=values['joint_control'])
        batch.reset([True,False],**arguments)
        for key,value in batch.read_state().__dict__.items():
            torch.testing.assert_close(value[1],saved[key][1],atol=0,rtol=0)
        with pytest.raises(ValueError,match='base pose must have finite shape'):
            batch.reset([False,True],**arguments)
        bad=dict(arguments);bad['base_pose']=values['base_pose'][:1]
        with pytest.raises(ValueError,match='finite shape'):
            batch.reset([True,False],**bad)
        bad=dict(arguments);bad['base_pose']=saved['base_pose'].clone();bad['base_pose'][0,3:]=0.
        with pytest.raises(ValueError,match='unit xyzw'):
            batch.reset([True,False],**bad)
    finally:batch.close()


def test_base_velocity_write_preserves_pose_control_and_other_lane(tmp_path):
    batch=make_batch(tmp_path)
    try:
        state=batch.read_state()
        pose=state.base_pose.clone()
        pose[:,3:]=torch.tensor([0.,0.,2**-.5,2**-.5])
        batch.reset([True,True],base_pose=pose,base_velocity=state.base_velocity.clone(),
                    joint_position=state.joint_position.clone(),joint_velocity=state.joint_velocity.clone(),
                    targets=state.joint_control.clone())
        before={k:v.clone() for k,v in batch.read_state().__dict__.items()}
        contacts={k:v.clone() for k,v in batch.read_contacts().items()}
        velocity=torch.tensor([[.1,.2,.3,.4,.5,.6],[float('nan')]*6])
        batch.set_base_velocity([True,False],velocity)
        current=batch.read_state()
        torch.testing.assert_close(current.base_velocity[0],velocity[0],atol=1e-6,rtol=1e-6)
        for key,value in current.__dict__.items():
            torch.testing.assert_close(value[1],before[key][1],atol=0,rtol=0)
            if key!='base_velocity':torch.testing.assert_close(value,before[key],atol=0,rtol=0)
        for key,value in batch.read_contacts().items():torch.testing.assert_close(value,contacts[key],atol=0,rtol=0)
        with pytest.raises(ValueError,match='finite'):
            batch.set_base_velocity([True,True],velocity)
    finally:batch.close()


def test_backend_link_twists_match_scene_kinematics(tmp_path):
    from gobot.rl.kinematics import SceneKinematics
    from gobot.rl.tracking.math import matrix
    batch=make_batch(tmp_path)
    try:
        state=batch.read_state()
        velocity=torch.tensor([[.2,-.1,.3,.1,.2,-.4],[-.2,.4,.1,.3,-.2,.1]])
        q=torch.tensor([[.4],[-.2]]);qd=torch.tensor([[1.2],[-.7]])
        batch.reset([True,True],base_pose=state.base_pose.clone(),base_velocity=velocity,
                    joint_position=q,joint_velocity=qd,targets=q)
        state=batch.read_state()
        kin=SceneKinematics(batch.snapshot,robot_name='batch_test',base_link='base',joint_names=('hinge',),tracked_links=('foot',))
        p,r,v,w=kin(state.base_pose[:,:3],matrix(state.base_pose[:,(6,3,4,5)]),velocity[:,:3],velocity[:,3:],q,qd)
        actual=batch.read_link_velocities()
        torch.testing.assert_close(actual[...,:3],v,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(actual[...,3:],w,atol=1e-6,rtol=1e-6)
        pointer=actual.data_ptr()
        batch.step_control(refresh_kinematics=False)
        assert batch.read_link_velocities().data_ptr()==pointer
        before=batch.read_state().link_pose.clone()
        batch.forward()
        assert not torch.equal(before,batch.read_state().link_pose)
    finally:batch.close()
