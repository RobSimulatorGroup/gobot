"""CPU contracts for rope trial orchestration, controller routing and metrics."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import sys
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    print("Rope trial tests skipped: torch is unavailable")
    raise SystemExit(77)
import numpy as np

PROJECT = Path(__file__).resolve().parents[2] / "examples/dual_arm_rope_twist"
sys.path.insert(0, str(PROJECT))
import controllers
import rope_twist_batch as batch
import rope_twist_control as control_module
from rope_twist_metrics import RopeMetrics


def fixture(num_envs=2):
    """Synthetic physical state with known forces, geometry and independent rows."""
    zeros = lambda *shape: torch.zeros((num_envs, *shape), dtype=torch.float32)
    identity = torch.eye(4).repeat(num_envs, 2, 1, 1)
    identity[:, 0, 0, 3] = -.5
    identity[:, 1, 0, 3] = .5
    points = torch.zeros((42, 3))
    for strand in range(3):
        points[14*strand:14*(strand+1), 1] = .1 * np.cos(strand * 2*np.pi/3)
        points[14*strand:14*(strand+1), 2] = .1 * np.sin(strand * 2*np.pi/3)
        points[14*strand:14*strand+7, 0] = -.5
        points[14*strand+7:14*(strand+1), 0] = .5
    arrays = dict(ctrl=zeros(18), qpos=zeros(18), qvel=zeros(18),
                  qfrc_applied=zeros(18), actuator_force=zeros(18),
                  xpos=zeros(4, 3), xipos=zeros(4, 3),
                  xmat=torch.eye(3).repeat(num_envs, 4, 1, 1),
                  xfrc_applied=zeros(4, 6), ipc_positions=points.repeat(num_envs, 1, 1),
                  ipc_contact_forces=zeros(42, 3), ipc_affine_contact_wrenches=zeros(2, 6),
                  ipc_affine_transforms=identity.clone(), ipc_affine_targets=identity.clone())
    arrays['xpos'][:, :2, 0] = torch.tensor([-.5, .5])
    arrays['xpos'][:, 2:, 0] = torch.tensor([-.5, .5])
    arrays['xipos'].copy_(arrays['xpos'])
    entries = [dict(element_offset=i*14, element_count=14) for i in range(3)]
    sensors = [dict(force=zeros(2, 3), found=torch.ones((num_envs, 2)), dist=zeros(2)) for _ in range(2)]
    model_arrays = dict(actuator_forcerange=torch.zeros(18, 2), geom_friction=torch.ones(1, 6, 3),
                        geom_condim=torch.full((6,), 3))
    body_ids = dict(left_rope_fixture=0, right_rope_fixture=1, left_fr3_fr3_link7=2, right_fr3_fr3_link7=3)
    provider = NS(num_envs=num_envs, arrays=arrays, close=Mock(), synchronize=Mock(), sense=Mock(), step=Mock())
    provider.rigid_solver = NS(
        resolve_robot_layout=lambda name, **kw: NS(actuator_ids=tuple(range(0 if name=='left_fr3' else 9, 9 if name=='left_fr3' else 18))),
        resolve_object_ids=lambda kind, names: tuple(body_ids[n] for n in names) if kind=='body' else tuple(range(len(names))),
        contact_sensor=lambda name: sensors[0 if name.startswith('left') else 1],
        model_array=lambda name: model_arrays[name], recompute_constants=Mock(), assert_no_overflow=Mock())
    def view(robot_name, **kwargs):
        offset = 0 if robot_name == 'left_fr3' else 9
        return NS(read_state=lambda: NS(joint_position=arrays['qpos'][:, offset:offset+9].clone(),
                                        joint_velocity=arrays['qvel'][:, offset:offset+9].clone()),
                  set_controls=lambda values: arrays['ctrl'][:, offset:offset+9].copy_(values))
    provider.create_robot_view = view
    provider.ipc_solver = NS(deformable_bodies=entries, shard_count=num_envs)
    provider.reset = Mock(side_effect=lambda mask, **values: [arrays[name].copy_(value) for name,value in values.items()])
    mappings = [NS(robot_name=name, link_name=name, mujoco_body_name=name, ipc_body_index=i,
                   force_scale=1., torque_scale=1., mode='TwoWay')
                for i,name in enumerate(('left_rope_fixture','right_rope_fixture'))]
    collider = dict(disabled=False, shape_type='box', size=(2., 2., 2.),
                    transform=dict(matrix_row_major=torch.eye(4).flatten().tolist()))
    artifact = NS(digest='fixture', coupled_bodies=mappings, collision_ownership={},
                  mujoco=NS(content=''), ipc=NS(manifest_data=dict(robots=[], static_colliders=[collider], deformable_attachments=[None]*6)))
    keys = ('proxy_count', 'static_collider_count', 'coupling_solver', 'coupling_iterations',
            'actual_coupling_iterations', 'relaxation_mode', 'interface_residual', 'interface_residual_l2',
            'aitken_coefficient', 'coupler_graph_captured', 'coupler_graph_capture_error',
            'exact_contact_wrench', 'feedback_source')
    provider.diagnostics = dict.fromkeys(keys, 0)
    provider.diagnostics.update(coupler_graph_capture_reason="", phase_latency_ms={}, convergence_guard={})
    provider.capabilities = NS(exact_contact_wrench=True)
    schedule = controllers.GravityCompensationSchedule(
        offset=np.ones((500, 2, 9)), cosine=np.zeros((500, 2, 9)), sine=np.zeros((500, 2, 9)),
        joint_dof_addresses=(tuple(range(9)),tuple(range(9,18))))
    return provider, artifact, schedule


class RopeTrialTest(unittest.TestCase):
    def make_control(self, provider, artifact, schedule):
        with patch.object(control_module, 'gravity_compensation_schedule', return_value=schedule):
            return control_module.RopeControl(provider, artifact, controllers.make_trial_layout(provider.num_envs),
                    fixed_dt=.002, drive_torque_limit=.02, feedback_enabled=True)

    def test_control_routes_both_robots_and_reset_clears_history(self):
        provider, artifact, schedule = fixture()
        control = self.make_control(provider, artifact, schedule)
        control.reset()
        initial = provider.arrays['ctrl'].clone()
        control.step()
        expected = controllers.nominal_joint_targets(0)
        np.testing.assert_allclose(provider.arrays['ctrl'][0].reshape(2,9), expected, atol=1e-8)
        self.assertEqual(control.controller.tick, 1)
        torch.testing.assert_close(provider.arrays['qfrc_applied'], torch.ones((2,18)))
        control.controller.stalled[:] = True
        control.reset()
        self.assertEqual(control.controller.tick, 0)
        self.assertFalse(control.controller.stalled.any())
        torch.testing.assert_close(provider.arrays['ctrl'], initial)
        self.assertEqual(control.wrist_actuator_ids, (6, 15))

    def test_metrics_measure_completed_state_and_keep_storage(self):
        provider, artifact, schedule = fixture()
        control = self.make_control(provider, artifact, schedule)
        args = batch._parser().parse_args(['--num-envs','2','--steps','1'])
        metrics = RopeMetrics(provider, artifact, control, args, batch._solver_config(args))
        control.controller.tick = 1
        provider.arrays['ipc_contact_forces'][1, 0, 0] = 5
        provider.arrays['xfrc_applied'][1, 0, 5] = .25
        provider.arrays['ipc_affine_contact_wrenches'][1, 0, 5] = .25
        metrics.update()
        report = metrics.report(Path('scene.jscn'), batch.TrialTiming(executed_steps=1, elapsed=2.))
        self.assertEqual(report['peak_contact_force_range_newtons'], [0.,5.])
        self.assertEqual(report['raw_peak_axial_torque_range_newton_meters'], [0.,.25])
        self.assertEqual(report['maximum_coupling_wrench_imbalance_ratio_range'], [0.,0.])
        self.assertEqual(report['maximum_attachment_error_range_meters'], [0.,0.])
        self.assertEqual(report['environment_steps_per_second'], 1.)
        self.assertTrue(report['admission_completed'])
        self.assertTrue(report['ipc_position_storage_stable'])
        self.assertTrue(report['qpos_storage_stable'])
        self.assertEqual(len(report['trials']), 2)
        provider.arrays['ipc_positions'] = provider.arrays['ipc_positions'].clone()
        provider.arrays['qpos'] = provider.arrays['qpos'].clone()
        changed = metrics.report(Path('scene.jscn'), batch.TrialTiming(executed_steps=1, elapsed=2.))
        self.assertFalse(changed['ipc_position_storage_stable'])
        self.assertFalse(changed['qpos_storage_stable'])

    def test_play_and_batch_share_control_and_reset_behavior(self):
        import rope_twist_runtime as runtime_module
        provider, artifact, schedule = fixture(num_envs=1)
        provider.refresh_state = Mock()
        artifact.ipc.deformable_bodies = provider.ipc_solver.deformable_bodies
        batch_provider, batch_artifact, _ = fixture(num_envs=1)
        control = self.make_control(batch_provider, batch_artifact, schedule)
        control.reset()
        with patch.object(control_module, 'gravity_compensation_schedule', return_value=schedule), \
             patch.object(runtime_module, 'SceneStateOutput'):
            runtime = runtime_module.RopeRuntime(provider, artifact, drive_mode='finite-torque',
                                                 wrist_torque_limit=.02, stall_detection_enabled=True)
        for _ in range(5):
            runtime.step(.002)
            control.step()
            for name in ('ctrl', 'qfrc_applied'):
                torch.testing.assert_close(provider.arrays[name], batch_provider.arrays[name])
        runtime.maximum_grip_slip = 1.
        runtime.reset((0,))
        control.reset()
        torch.testing.assert_close(provider.arrays['ctrl'], batch_provider.arrays['ctrl'])
        self.assertEqual(runtime.maximum_grip_slip, 0.)
        self.assertEqual(runtime.control.controller.tick, 0)
        with self.assertRaises(ValueError):
            runtime.reset((1,))

    def test_latency_guard_accounts_for_completed_step(self):
        args = batch._parser().parse_args(['--steps','10','--maximum-step-latency-seconds','.5'])
        provider = NS(step=Mock(), sense=Mock(), synchronize=Mock())
        control = NS(step=Mock(), controller=NS(tick=1, cycle_complete=False))
        metrics = NS(update=Mock())
        with patch.object(batch.time, 'perf_counter', side_effect=(0., 1., 2., 3.)):
            timing = batch._execute_trial(args, provider, control, metrics)
        self.assertEqual(timing.executed_steps, 1)
        self.assertTrue(timing.admission_aborted)
        self.assertEqual(timing.maximum_step_latency_seconds, 1.)
        self.assertEqual(timing.elapsed, 3.)
        metrics.update.assert_called_once()
        provider.synchronize.assert_called_once()

    def test_provider_construction_failure_clears_scene(self):
        args = batch._parser().parse_args([])
        context = NS(clear_scene=Mock())
        available = NS(available=True)
        with patch.object(batch.MuJoCoWarpProvider, 'availability', return_value=available), \
             patch.object(batch.LibuipcBatchSolver, 'availability', return_value=available), \
             patch.object(batch, '_load_artifact', return_value=(context, object())), \
             patch.object(batch, '_create_provider', side_effect=RuntimeError('provider failed')):
            with self.assertRaisesRegex(RuntimeError, 'provider failed'):
                batch.run(args)
        context.clear_scene.assert_called_once()


if __name__ == '__main__':
    unittest.main()
