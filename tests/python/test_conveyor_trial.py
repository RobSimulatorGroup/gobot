"""CPU coverage for shared conveyor control and completed-tick measurements."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[2] / "examples/conveyor_packages"
sys.path.insert(0, str(PROJECT))
import conveyor_config as config
import conveyor_profile as profile
import conveyor_packages_batch as batch
import conveyor_packages_play as play
from conveyor_control import ConveyorControl
from conveyor_metrics import ConveyorMetrics


class Node:
    def __init__(self, name, type_name="Node3D", children=()):
        self.name, self.type_name, self.children = name, type_name, list(children)
        self.position = (0., 0., 0.)
        self.set_position_target = Mock()

    def find(self, path):
        head, _, tail = path.partition("/")
        child = next((node for node in self.children if node.name == head), None)
        return child.find(tail) if child is not None and tail else child


def fixture():
    """Measured rigid/deformable state with finite geometry and nonzero forces."""
    transform = dict(matrix=np.eye(4), quaternion=(1., 0., 0., 0.), position=(0., 0., .56))
    def link(name):
        return dict(name=name, global_transform=deepcopy(transform),
                    linear_velocity=(.1, 0., 0.), angular_velocity=(0., 0., 0.))
    roots, robots = [], []
    for robot_name, joint_names in zip(config.LEAP_ROBOT_NAMES, config.HAND_JOINT_NAMES_BY_SIDE):
        roots.append(Node(robot_name, "Robot3D", [Node(n, "Joint3D") for n in joint_names]))
        robots.append(dict(name=robot_name, links=[link(n) for n in config.LEAP_CONTACT_LINK_NAMES],
                           joints=[dict(name=n, position=0., velocity=0., effort=0., tracking_error=.01)
                                   for n in joint_names]))
    for name in config.RIGID_BOX_NAMES:
        robots.append(dict(name=name, links=[link(name)], joints=[]))
    roots.append(Node("conveyor", "Robot3D", [Node("belt_surface", "Link3D", [Node("belt_marker_00")])]))
    vertices = np.array([[-.1, .4, .55], [.1, .6, .55], [-.1, .4, .65], [.1, .6, .65]])
    bodies = [dict(name=name, stable_id=i, world_vertices=vertices.copy(), local_vertices=vertices.copy(),
                   local_velocities=np.full((4, 3), .1), contact_forces_world=np.tile((0., 0., 2.), (4, 1)),
                   global_transform=deepcopy(transform)) for i, name in enumerate(config.SOFT_PACKAGE_NAMES)]
    state = dict(simulation_time=0., robots=robots, deformables=bodies,
                 contacts=[dict(robot_name=name, link_name=name, other_robot_name="conveyor",
                                other_link_name="belt_surface", distance=-.0001, normal_force=2.)
                           for name in config.RIGID_BOX_NAMES])
    return Node("conveyor_packages", children=roots), state


class Context:
    def __init__(self, state):
        self.initial = deepcopy(state)
        self.state = deepcopy(state)
        self.has_world = True
        self.set_link_external_force = Mock()
        self.set_deformable_external_forces = Mock()
        self.clear_external_forces = Mock()
        self.clear_world = Mock()
        self.clear_scene = Mock()
        self.set_project_path = Mock()
        self.build_world = Mock()
        self.set_superdex_solver_settings = Mock()
        self.get_physics_debug_settings = lambda: {"draw_contact_forces": False}

    def get_superdex_solver_settings(self):
        return {}

    def get_physics_state_view(self):
        return deepcopy(self.state)

    def step_once(self):
        self.state["simulation_time"] += profile.FIXED_DT

    def reset_simulation(self):
        self.state = deepcopy(self.initial)

    def get_solver_diagnostics(self):
        return dict(execution_device="cpu", device_native=False, graph_capture=False)


class ConveyorTrialTest(unittest.TestCase):
    def test_newton_placement_preserves_the_supported_pouch_span(self):
        from conveyor_newton_runtime import ConveyorNewtonController, PLACEMENT_RADIUS
        provider = Mock(tick=0)
        controller = ConveyorNewtonController(provider)
        elapsed = 0.
        for phase, duration, target in controller.segments:
            if phase == "place":
                break
            previous = target
            elapsed += duration
        pivot = previous[:, 1:3].copy()
        pivot[:, 1] -= PLACEMENT_RADIUS
        for fraction in (.0, .25, .5, .75, 1. - 1.e-8):
            provider.tick += 1
            controller.time = elapsed + duration * fraction
            controller.step(.002)
            hands = np.array([call.args[2] for call in provider.set_joint_position_targets.call_args_list[-2:]])
            np.testing.assert_allclose(np.linalg.norm(hands[:, 1:3] - pivot, axis=1),
                                       PLACEMENT_RADIUS, atol=1.e-8)
            np.testing.assert_array_equal(hands[:, 6:], previous[:, 6:])
        np.testing.assert_allclose(hands, target, atol=1.e-8)

    def test_newton_belt_waits_for_hand_clearance_then_accelerates(self):
        from conveyor_newton_runtime import ConveyorNewtonController
        provider = Mock(tick=0)
        controller = ConveyorNewtonController(provider)
        starts, elapsed = {}, 0.
        for name, duration, _ in controller.segments:
            starts[name] = elapsed
            elapsed += duration

        def speed_at(time):
            provider.tick += 1
            controller.time = time
            controller.step(.002)
            return provider.set_surface_velocity.call_args.args[2][0]

        for phase in ("transfer", "place_contact", "place", "place_hold", "release", "clear", "settle_after_release"):
            self.assertEqual(speed_at(starts[phase] + .1), 0.)
        self.assertAlmostEqual(speed_at(starts["outfeed"]), 0.)
        self.assertAlmostEqual(speed_at(starts["outfeed"] + .25), .175)
        self.assertAlmostEqual(speed_at(starts["outfeed"] + .5), .35)
        # Editor playback can continue past the recorded trajectory.
        self.assertAlmostEqual(speed_at(controller.duration + .1), .35)

    def test_newton_transfer_rejects_a_bounce_even_when_final_frame_is_flat(self):
        from conveyor_newton_batch import TrialMetrics
        snapshot = {"deformables": [{"name": "soft_mailer_blue", "model": 0, "tetrahedra": []}]}

        def check(*, lift=0., fold=0., vertical_speed=0.):
            metrics = TrialMetrics(snapshot)
            for i, phase in enumerate(("place_hold", "release", "clear", "outfeed", "outfeed")):
                q = np.array([[i*.15, .3, .561], [i*.15+.2, .5, .641]])
                v = np.tile((.35, 0., 0.), (2, 1))
                if phase == "release":
                    q[:, 2] += lift
                    q[1, 2] += fold
                    v[:, 2] = vertical_speed
                metrics.update({"deformables": {"soft_mailer_blue": {"positions": q, "velocities": v}},
                                "time": 21. + i*.5, "soft_contact_count": 1, "peak_soft_penetration": 0.,
                                "max_soft_penetration": 0., "minimum_tet_volume_ratio": 1.}, phase)
            return metrics.report()["acceptance"]["conveyor_stability"]

        self.assertTrue(check())
        self.assertFalse(check(lift=.02))
        self.assertFalse(check(fold=.05))
        self.assertFalse(check(vertical_speed=.4))

    def test_newton_transport_rejects_both_velocity_impulses_and_sustained_flutter(self):
        from conveyor_newton_batch import TrialMetrics
        snapshot = {"deformables": [{"name": "soft_mailer_blue", "model": 0, "tetrahedra": []}]}

        def check(sustained):
            metrics = TrialMetrics(snapshot)
            # A quiet final frame or averaged velocity cannot excuse an
            # earlier contact impulse, even if bounds remain unchanged.
            for i in range(16):
                q = np.array([[i*.01, .3, .561], [i*.01+.2, .5, .641]])
                v = np.tile((.25, 0., .3 if i == 5 or (sustained and i >= 5) else 0.), (2, 1))
                metrics.update({"deformables": {"soft_mailer_blue": {"positions": q, "velocities": v}},
                                "time": 21. + i*.04, "soft_contact_count": 1, "peak_soft_penetration": 0.,
                                "max_soft_penetration": 0., "minimum_tet_volume_ratio": 1.},
                               "place_hold" if i == 0 else "outfeed")
            return metrics.report()

        impulse = check(False)
        self.assertFalse(impulse["acceptance"]["conveyor_stability"])
        self.assertAlmostEqual(impulse["conveyor_stability"]["maximum_vertical_rms_speed_m_s"], .3)
        self.assertFalse(check(True)["acceptance"]["conveyor_stability"])

    def test_newton_transport_rejects_local_fold_jump_with_unchanged_bounds(self):
        from conveyor_newton_batch import TrialMetrics
        snapshot = {"deformables": [{"name": "soft_mailer_blue", "model": 0, "tetrahedra": []}]}

        for moving_phase in ("place_hold", "release"):
            metrics = TrialMetrics(snapshot)
            q = np.array([[0., .3, .561], [.2, .5, .641], [.1, .4, .59], [.15, .4, .61]])
            for i, phase in enumerate(("place_hold", moving_phase, "outfeed")):
                positions = q.copy()
                positions[:, 0] += i * .01
                if i == 1:
                    # Equal opposing displacements preserve center and bounds.
                    positions[2, 2] += .019
                    positions[3, 2] -= .019
                metrics.update({"deformables": {"soft_mailer_blue": {
                    "positions": positions, "velocities": np.zeros_like(q)}},
                    "time": 18. + i * .04, "soft_contact_count": 1, "peak_soft_penetration": 0.,
                    "max_soft_penetration": 0., "minimum_tet_volume_ratio": 1.}, phase)
            result = metrics.report()
            self.assertAlmostEqual(result["conveyor_stability"]["maximum_top_rise_m"], 0.)
            self.assertAlmostEqual(result["conveyor_stability"]["maximum_center_rise_m"], 0.)
            self.assertFalse(result["acceptance"]["conveyor_stability"])

    def test_newton_transfer_rejects_missed_belt_placement(self):
        from conveyor_newton_batch import TrialMetrics
        snapshot = {"deformables": [{"name": "soft_mailer_blue", "model": 0, "tetrahedra": []}]}

        def transfer(*, y=.4, travel=.3):
            metrics = TrialMetrics(snapshot)
            for time, x in ((22.3, 0.), (23.8, travel)):
                q = np.array([[x - .1, y - .1, .56], [x + .1, y + .1, .64]])
                metrics.update({"deformables": {"soft_mailer_blue": {
                                    "positions": q, "velocities": np.zeros_like(q)}},
                                "time": time, "soft_contact_count": 0, "peak_soft_penetration": 0.,
                                "max_soft_penetration": 0., "minimum_tet_volume_ratio": 1.}, "outfeed")
            return metrics.report()["acceptance"]["conveyor_transfer"]

        self.assertTrue(transfer())
        self.assertFalse(transfer(travel=0.))
        self.assertFalse(transfer(y=-.4))

    def test_newton_unattended_parcel_rejects_late_motion(self):
        from conveyor_newton_batch import TrialMetrics
        names = ("soft_pouch_yellow", "soft_pouch_yellow_fill")
        snapshot = {"deformables": [{"name": name, "model": 0, "tetrahedra": []}
                                    for name in names]}

        def check(*, displacement=0., bulge=0., speed=0., shape_change=0.):
            metrics = TrialMetrics(snapshot)
            for time in (1., 2., 8.):
                q = np.array([[-.1, -.1, .56], [.1, .1, .63]])
                velocity = np.zeros_like(q)
                if time == 8.:
                    q[:, 0] += displacement
                    q[1, 2] += bulge
                    velocity[1, 2] = speed
                    q[0, 0] += shape_change
                    q[1, 0] -= shape_change
                bodies = {name: {"positions": q, "velocities": velocity} for name in names}
                metrics.update({"deformables": bodies, "time": time, "soft_contact_count": 0,
                                "peak_soft_penetration": 0., "max_soft_penetration": 0.,
                                "minimum_tet_volume_ratio": 1.}, "settle_after_release")
            return metrics.report()["acceptance"]["unattended_parcel_stability"]

        self.assertTrue(check())
        self.assertFalse(check(displacement=.01))
        self.assertFalse(check(bulge=.01))
        self.assertFalse(check(speed=.1))
        self.assertFalse(check(shape_change=.01))

    def test_newton_recovery_rejects_launch_and_local_jitter(self):
        from conveyor_newton_batch import TrialMetrics
        # Six upper-film points under the press pads, plus a lower layer.
        upper = np.array([(x, -.375, .64) for x in (.025, .070, .115, .205, .250, .295)])
        vertices = np.concatenate((upper - (0., 0., .08), upper))
        snapshot = {"deformables": [{"name": "soft_mailer_blue", "model": 1,
                    "vertices": vertices, "triangles": [], "tetrahedra": [],
                    "transform": {"matrix": np.eye(4)}}]}

        def report(*, drift=0., jitter=0., slip=False):
            metrics = TrialMetrics(snapshot)
            # This fixture measures the same explicit top vertices at each phase.
            metrics.press_region["soft_mailer_blue"] = np.arange(12) >= 6
            for phase, time, height in (("settle_and_carton_drop", .6, .08),
                                        ("press_hold", 2.4, .065), ("recover", 3.5, .08)):
                q = vertices.copy()
                q[6:, 2] = config.WORKTABLE_TOP_Z + height
                velocity = np.zeros_like(q)
                if phase == "recover":
                    q[:, 0] += drift
                    velocity[0, 2] = jitter
                bodies = {name: {"positions": q, "velocities": velocity}
                          for name in ("soft_mailer_blue", "soft_mailer_blue_fill")}
                metrics.update({"deformables": bodies, "time": time, "soft_contact_count": 0,
                                "peak_soft_penetration": 0., "max_soft_penetration": 0.,
                                "minimum_tet_volume_ratio": 1.}, phase)
            for time in (6., 7.2, 7.5):
                q = vertices + (0., 0., 0. if slip and time == 7.5 else .3)
                bodies = {name: {"positions": q, "velocities": np.zeros_like(q)}
                          for name in ("soft_mailer_blue", "soft_mailer_blue_fill")}
                metrics.update({"deformables": bodies, "time": time, "soft_contact_count": 0,
                                "peak_soft_penetration": 0., "max_soft_penetration": 0.,
                                "minimum_tet_volume_ratio": 1.}, "hold")
            return metrics.report()

        stable = report()
        self.assertTrue(stable["acceptance"]["press_compression"])
        self.assertTrue(stable["acceptance"]["height_recovery"])
        self.assertTrue(stable["acceptance"]["settled_recovery"])
        self.assertTrue(stable["acceptance"]["lift_and_hold"])
        slipped = report(slip=True)
        self.assertGreater(slipped["lift_hold_seconds"], 1.)
        self.assertFalse(slipped["acceptance"]["lift_and_hold"])
        for failed in (report(drift=.15), report(jitter=2.)):
            self.assertTrue(failed["acceptance"]["height_recovery"])
            self.assertFalse(failed["acceptance"]["settled_recovery"])

    def test_play_and_batch_apply_the_same_controls_and_forces(self):
        root, state = fixture()
        batch_context, play_context = Context(state), Context(state)
        control = ConveyorControl(batch_context, root)
        play_root, _ = fixture()
        script = play.Script()
        script.context = play_context
        script.get_root = lambda: play_root
        with patch.dict("os.environ", {"GOBOT_CONVEYOR_BACKEND": "superdex", "GOBOT_CONVEYOR_PREVIEW": "0", "GOBOT_CONVEYOR_DROP_ONLY": "0"}):
            script._ready()
        ticks = [0]
        end = 0
        for segment in profile.HAND_MOTION_SEGMENTS:
            end += segment.duration
            ticks.extend((end - 1, end))
        with patch.object(play, "clear_debug_arrows"):
            for tick in ticks:
                if tick >= profile.CYCLE_TICKS:
                    continue
                control.apply(tick, state)
                script.tick = tick
                # Keep the test focused on control, not periodic presentation.
                script._refresh_debug_arrows = Mock()
                if tick and tick % 300 == 0:
                    continue
                script._physics_process(profile.FIXED_DT)
                for left, right in zip(control.hand_joints, script.control.hand_joints):
                    for a, b in zip(left, right):
                        self.assertEqual(a.set_position_target.call_args, b.set_position_target.call_args)
                # Compare the last complete set of rigid and deformable commands.
                for a, b in zip(batch_context.set_deformable_external_forces.call_args_list[-4:],
                                play_context.set_deformable_external_forces.call_args_list[-4:]):
                    self.assertEqual(a.args[0], b.args[0])
                    np.testing.assert_array_equal(a.args[1], b.args[1])
                self.assertEqual(batch_context.set_link_external_force.call_args_list[-3:],
                                 play_context.set_link_external_force.call_args_list[-3:])
            script._reset_cycle()
        self.assertEqual(script.tick, 0)
        play_context.clear_external_forces.assert_called_once()
        self.assertEqual(play_context.state["simulation_time"], 0.)

    def test_missing_or_duplicate_joints_fail_before_applying_commands(self):
        root, state = fixture()
        root.children[0].children.pop()
        context = Context(state)
        with self.assertRaisesRegex(ValueError, "missing joints"):
            ConveyorControl(context, root)
        context.set_link_external_force.assert_not_called()
        root, state = fixture()
        root.children[0].children.append(root.children[0].children[0])
        with self.assertRaisesRegex(ValueError, "duplicate joint"):
            ConveyorControl(Context(state), root)

    def test_metrics_measure_flip_contacts_drift_and_phase_boundary(self):
        _, state = fixture()
        metrics = ConveyorMetrics(state, trace_force_flow=True, phase_diagnostics=True)
        moved = deepcopy(state)
        for body in moved["deformables"][:2]:
            body["world_vertices"][:, 0] += .25
        metrics.update(moved, tick=299, control_tick=298, rigid_drive=np.array([1., -2., 3.]),
                       soft_drive={"soft_mailer_blue": np.array([-.4, .2])})
        rotated = deepcopy(moved)
        rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        rotated["deformables"][0]["world_vertices"] = moved["deformables"][0]["world_vertices"] @ rotation.T
        rotated["contacts"] = [dict(robot_name="leap_left", link_name="th_ds", other_robot_name="",
                                    other_link_name="soft_mailer_blue", distance=-.003),
                               dict(robot_name="soft_mailer_blue", link_name="soft_mailer_blue",
                                    other_robot_name="leap_right", other_link_name="if_ds", distance=-.002)]
        metrics.update(rotated, tick=300, control_tick=299, rigid_drive=np.zeros(3), soft_drive={})
        report = metrics.report(rotated, 300)
        self.assertAlmostEqual(report["final_flip_degrees"]["blue_mailer"], 90.)
        self.assertAlmostEqual(report["precontact_horizontal_drift_meters"]["blue_mailer"], .25)
        self.assertEqual(report["first_hand_contact_tick"]["blue_mailer"], {"leap_left": 300, "leap_right": 300})
        self.assertEqual(report["hand_contact_frames"]["blue_mailer"], {"leap_left": 1, "leap_right": 1})
        self.assertEqual(report["maximum_simultaneous_hands"]["blue_mailer"], 2)
        self.assertTrue(report["simultaneous_hand_contact"]["blue_mailer"])
        self.assertEqual(report["peak_contact_penetration_meters"], .003)
        self.assertEqual(report["peak_penetration_contact"]["tick"], 300)
        self.assertEqual(report["peak_rigid_drive_force_newtons"], dict(zip(config.RIGID_BOX_NAMES, (1., 2., 3.))))
        self.assertEqual(report["peak_soft_nodal_drive_force_newtons"], .4)
        self.assertEqual(report["phase_diagnostics"][0]["tick"], 300)
        self.assertEqual(len(report["force_flow_trace"]), 2)

    def test_observer_stop_and_failures_preserve_tick_counts_and_cleanup(self):
        root, state = fixture()
        context = Context(state)
        context.load_scene = Mock(return_value=root)
        args = batch._parser().parse_args(["--backend", "superdex", "--steps", "10", "--warmup-steps", "2"])
        samples = []
        def observe(sample):
            samples.append(sample)
            return sample["tick"] < 2
        with patch.object(batch.gobot.app, "create_context", return_value=context):
            result = batch.run(args, tick_observer=observe)
        self.assertEqual(result["steps"], 2)
        self.assertEqual([s["tick"] for s in samples], [1, 2])
        self.assertEqual(samples[-1]["simulation_time_seconds"], 2 * profile.FIXED_DT)
        self.assertEqual(result["reset_max_error"], 0.)
        context.clear_world.assert_called_once()
        context.clear_scene.assert_called_once()
        for fail_setup in (False, True):
            context = Context(state)
            context.load_scene = Mock(return_value=root)
            failing = context.build_world if fail_setup else None
            if failing:
                failing.side_effect = RuntimeError("setup failed")
            else:
                context.step_once = Mock(side_effect=RuntimeError("step failed"))
            args.warmup_steps = 0
            failures = []
            with patch.object(batch.gobot.app, "create_context", return_value=context):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    batch.run(args, failure_observer=failures.append)
            self.assertEqual(len(failures), 0 if fail_setup else 1)
            if failures:
                self.assertEqual(failures[0]["tick"], 1)
            context.clear_world.assert_called_once()
            context.clear_scene.assert_called_once()


if __name__ == "__main__":
    unittest.main()
