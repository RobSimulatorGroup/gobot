"""Kinematic, scene and control contracts for the articulated sorting workcell."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from unittest.mock import Mock

import numpy as np

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/conveyor_packages"
sys.path.insert(0, str(EXAMPLE))
from conveyor_sharpa import (Kinematics, SIDES, ROBOT_NAMES, PALM_DOWN, BELT_SPEED,
                             ASSETS, ROBOT_POSITION, BASE_ROTATION, TABLE_TOP, PUSH_ROW_Y, transform)
from conveyor_sharpa_motion import motion_table, keyframes
from conveyor_sharpa_grasp import pinch_fingers, pinch_palm, push_fingers, push_palm


def test_openarm_conversion_matches_upstream_mujoco_frames_and_inertias():
    import mujoco
    from scipy.spatial.transform import Rotation
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "source/openarm_bimanual.xml"))
    data = mujoco.MjData(model)
    for side in SIDES:
        kin = Kinematics(side)
        for name, value in zip(kin.arm_names, kin.home_seed):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[joint]] = value
        mujoco.mj_forward(model, data)
        actual = kin.forward(dict(zip(kin.arm_names, kin.home_seed)))
        for name in kin.arm_names:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            np.testing.assert_allclose(kin.joints[name]["axis"], model.jnt_axis[joint], atol=1e-11)
            np.testing.assert_allclose(kin.joints[name]["limits"], model.jnt_range[joint], atol=1e-11)
            link = kin.joints[name]["child"]
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
            expected = kin.base @ transform(data.xpos[body], data.xmat[body].reshape(3, 3))
            np.testing.assert_allclose(actual[link], expected, atol=2e-9)
            assert kin.mass[link] == model.body_mass[body]
            np.testing.assert_allclose(kin.com[link], model.body_ipos[body], atol=1e-10)
            inertia = kin.xml.find(f"link[@name='{link}']/inertial")
            rotation = Rotation.from_euler("xyz", np.fromstring(inertia.find("origin").get("rpy"), sep=" ")).as_matrix()
            diag = [float(inertia.find("inertia").get(k)) for k in ("ixx", "iyy", "izz")]
            source_rotation = Rotation.from_quat(model.body_iquat[body][[1, 2, 3, 0]]).as_matrix()
            np.testing.assert_allclose(rotation @ np.diag(diag) @ rotation.T,
                source_rotation @ np.diag(model.body_inertia[body]) @ source_rotation.T, atol=1e-12)


def test_mirrored_sharpa_hands_preserve_geometry_and_joint_limits():
    left, right = (Kinematics(side) for side in SIDES)
    assert len(left.names) == len(right.names) == 29
    np.testing.assert_array_equal(left.limits[7:], right.limits[7:])
    q = .35*left.limits[7:, 0] + .65*left.limits[7:, 1]
    lp, rp = left.forward(dict(zip(left.finger_names, q))), right.forward(dict(zip(right.finger_names, q)))
    li, ri = np.linalg.inv(lp["left_hand_C_MC"]), np.linalg.inv(rp["right_hand_C_MC"])
    reflection = np.diag((1., -1., 1.))
    hand_links = {j["child"] for n, j in left.joints.items() if n in left.finger_names}
    for name, pose in lp.items():
        if name not in hand_links:
            continue
        pose, counterpart = li@pose, ri@rp[name.replace("left_", "right_")]
        np.testing.assert_allclose(counterpart[:3, 3], reflection@pose[:3, 3], atol=2e-7)
        np.testing.assert_allclose(counterpart[:3, :3], reflection@pose[:3, :3]@reflection, atol=2e-7)


def test_cartesian_sorting_path_is_continuous_and_respects_limits():
    phases, times, targets = motion_table()
    assert np.all(np.diff(times) > 0.)
    assert np.max(np.abs(np.diff(targets, axis=0))) < .35
    arm_speed = np.abs(np.diff(targets[:, :, :7], axis=0))/np.diff(times)[:, None, None]
    assert arm_speed.max() < 1.8  # Stay below the actual URDF drive velocity limit.
    hand_speed = np.abs(np.diff(targets[:, :, 7:], axis=0))/np.diff(times)[:, None, None]
    assert hand_speed.max() < 6.01
    np.testing.assert_allclose(sum(p[1] for p in phases), times[-1], atol=1e-10)
    for side_index, side in enumerate(SIDES):
        kin = Kinematics(side)
        assert np.all(targets[:, side_index] >= kin.limits[:, 0]-1e-7)
        assert np.all(targets[:, side_index] <= kin.limits[:, 1]+1e-7)
        # The physical joints must follow a continuous path above the worktop.
        for q in targets[::5, side_index]:
            pose = kin.forward(q, hand=False)[f"{side}_hand_C_MC"]
            assert pose[2, 3] > TABLE_TOP+.07
            assert pose[2, 0] < -.10  # Includes the entire approach/turn transition.
        for (_, _, q), (_, _, palms, _) in zip(phases, keyframes(), strict=True):
            actual = kin.forward(q[side_index], hand=False)[f"{side}_hand_C_MC"]
            # Motor commands include the statically predicted joint deflection.
            displacement, angle, motor = kin.gravity_deflection(q[side_index])
            static_q = q[side_index].copy(); static_q[:7] -= motor
            expected = kin.forward(static_q, hand=False)[f"{side}_hand_C_MC"]
            expected[:3, 3] += displacement
            np.testing.assert_allclose(expected[:3, 3], palms[side_index,:3,3], atol=.0003)


def test_ready_and_pushing_palms_face_the_worktop():
    frames = {name: palms for name, _, palms, _ in keyframes()}
    for name, hands in (("settle", (0, 1)), ("yellow_approach", (1,)),
                        ("yellow_push", (1,)), ("white_handoff", (0,)),
                        ("white_push", (1,)), ("return", (0, 1))):
        for hand in hands:
            # Sharpa's local +X is the palm surface normal.
            np.testing.assert_allclose(frames[name][hand, :3, 0], (0., 0., -1.), atol=1e-12)


def test_worktop_clearance_rejects_an_arm_below_the_table():
    kin = Kinematics("right")
    arm = np.zeros(7)
    values = dict(zip(kin.names[:7], arm))
    clearance, link = kin.arm_worktop_clearance(values)
    assert clearance < -.02
    assert link in kin.arm_collision_vertices


def test_mirrored_pinch_opposes_actual_pad_surfaces_and_opens_clear():
    for side in SIDES:
        kin = Kinematics(side)
        palm = pinch_palm(kin, (.0, -.14, .5895))
        sign = 1 if side == "left" else -1
        def points(opening, *, preload=False):
            q = pinch_fingers(kin, opening, preload=preload)
            poses = kin.forward(dict(zip(kin.finger_names, q)))
            basis = palm @ np.linalg.inv(poses[f"{side}_hand_C_MC"])
            index = basis @ poses[f"{side}_index_elastomer"] @ (0., -.019*sign, -.00488894737, 1.)
            thumb = basis @ poses[f"{side}_thumb_elastomer"] @ (0., -.022*sign, -.00391412, 1.)
            return index[:3], thumb[:3]
        index, thumb = points(0.)
        np.testing.assert_allclose((index+thumb)/2, (0., -.14, .5895), atol=1e-7)
        assert 0. < thumb[2]-index[2] < .001
        assert np.linalg.norm(thumb-index) < .001
        opened_index, opened_thumb = points(.08)
        assert np.linalg.norm(opened_thumb-opened_index) > .007
        assert opened_thumb[2]-thumb[2] > .020
        assert opened_thumb[1] < thumb[1]-.020
        assert abs(opened_thumb[0]-thumb[0]) > .080
        # Tip clearance alone missed the proximal thumb pushing the filled
        # bulge away before closure. Check the supplied collision hull too.
        from scipy.spatial.transform import Rotation
        poses = kin.forward(dict(zip(kin.finger_names, pinch_fingers(kin, .08))))
        collision = kin.xml.find(f"link[@name='{side}_thumb_PP']/collision")
        mesh = collision.find("geometry/mesh")
        vertices = np.array([np.fromstring(line[2:], sep=" ") for line in
            (ASSETS/mesh.get("filename")).read_text().splitlines() if line.startswith("v ")])
        vertices *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
        origin = collision.find("origin")
        local = transform(np.fromstring(origin.get("xyz", "0 0 0"), sep=" "),
            Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix())
        frame = palm @ np.linalg.inv(poses[f"{side}_hand_C_MC"]) @ poses[f"{side}_thumb_PP"] @ local
        world = vertices @ frame[:3, :3].T + frame[:3, 3]
        assert world[:, 1].max() < -.14-.01
        loaded_index, loaded_thumb = points(0., preload=True)
        # Motor targets close past the unloaded pad fit by a few millimetres,
        # providing squeeze through compliant physical contacts on a thin seal.
        assert .002 < np.linalg.norm(loaded_index-loaded_thumb) < .004
        assert loaded_thumb[2] < loaded_index[2]
        assert palm[2, 0] < -.2


def test_push_posture_keeps_thumb_above_the_four_finger_contact_row():
    for side in SIDES:
        kin = Kinematics(side)
        sign = 1 if side == "left" else -1
        center = np.array((-.1, -.5, .59))
        palm = push_palm(kin, center)
        fingers = push_fingers(kin)
        assert np.all(fingers >= kin.limits[7:, 0])
        assert np.all(fingers <= kin.limits[7:, 1])
        poses = kin.forward(dict(zip(kin.finger_names, fingers)))
        basis = palm @ np.linalg.inv(poses[f"{side}_hand_C_MC"])
        row = np.array([(basis @ poses[f"{side}_{finger}_elastomer"] @
                         (0., -.019*sign, -.00488894737, 1.))[:3]
                        for finger in ("index", "middle", "ring", "pinky")])
        thumb = basis @ poses[f"{side}_thumb_elastomer"] @ (0., -.022*sign, -.00391412, 1.)
        np.testing.assert_allclose(row.mean(0), center, atol=1e-10)
        assert np.ptp(row[:, 2]) < .02
        assert thumb[2] > row[:, 2].max()+.05


def test_gravity_feedforward_matches_urdf_potential_energy_gradient():
    for i, side in enumerate(SIDES):
        kin = Kinematics(side)
        q = motion_table()[2][0, i].copy()
        displacement, angle, motor = kin.gravity_deflection(q)
        assert -.05 < displacement[2] < -.0001
        assert np.linalg.norm(angle) < .01
        def potential(values):
            poses = kin.forward(values)
            return sum(9.81*kin.mass[name]*(pose[:3, :3]@kin.com[name]+pose[:3, 3])[2]
                       for name, pose in poses.items())
        for j in range(7):
            plus, minus = q.copy(), q.copy()
            plus[j] += 1e-5
            minus[j] -= 1e-5
            gradient = (potential(plus)-potential(minus))/2e-5
            np.testing.assert_allclose(motor[j]*24000., gradient, atol=1e-6)


def test_controller_only_commands_joints_and_running_belt():
    from conveyor_sharpa_runtime import SharpaController
    provider = Mock(tick=0)
    controller = SharpaController(provider)
    for time in (0., 2.4, 7.1, controller.duration+.5):
        provider.tick += 1
        controller.time = time
        controller.step(.002)
    assert provider.set_joint_position_targets.call_count == 8
    assert set(c.args[0] for c in provider.set_joint_position_targets.call_args_list) == set(ROBOT_NAMES)
    assert all(len(c.args[1]) == len(c.args[2]) == 29 for c in provider.set_joint_position_targets.call_args_list)
    assert provider.set_surface_velocity.call_args.args == ("conveyor", "belt_surface", (BELT_SPEED, 0., 0.))
    assert {c[0] for c in provider.mock_calls} == {"set_joint_position_targets", "set_surface_velocity"}
    controller.reset((0,))
    assert controller.time == 0.


def test_white_push_follows_observed_parcel_without_changing_the_scene_state():
    from conveyor_sharpa_runtime import SharpaController
    positions = np.array(((-.20, PUSH_ROW_Y-.06, TABLE_TOP+.04), (-.22, PUSH_ROW_Y-.04, TABLE_TOP+.05)))
    provider = Mock(tick=0)
    provider.read_state.return_value = {"deformables": {"soft_mailer_blue": {"positions": positions}}}
    controller = SharpaController(provider)
    nominal = controller.targets.copy()
    start = controller.phase_starts["white_push"]
    for tick in range(250):
        provider.tick = tick
        controller.time = start+.002*tick
        controller.step(.002)
    assert -.07 < controller.push_lateral_shift < -.04
    assert np.linalg.norm(controller.push_joint_correction) > .01
    np.testing.assert_array_equal(controller.targets, nominal)
    np.testing.assert_array_equal(positions, ((-.20, PUSH_ROW_Y-.06, TABLE_TOP+.04), (-.22, PUSH_ROW_Y-.04, TABLE_TOP+.05)))
    assert {call[0] for call in provider.mock_calls} == {
        "read_state", "set_joint_position_targets", "set_surface_velocity"}
    controller.reset((0,))
    assert controller.push_lateral_shift == 0.
    np.testing.assert_array_equal(controller.push_joint_correction, np.zeros(7))


def test_wrist_tracking_compares_the_issued_joint_targets():
    from scipy.spatial.transform import Rotation
    from conveyor_sharpa_runtime import SharpaMetrics
    metrics = SharpaMetrics({"deformables": [], "robots": []})
    state = dict(time=0., peak_soft_penetration=0., minimum_tet_volume_ratio=1.,
                 deformables={}, links={}, joints={}, max_soft_penetration=0., soft_contact_count=0)
    for i, kin in enumerate(metrics.kinematics):
        target = motion_table()[2][0, i].copy()
        target[0] += .2
        palm = kin.forward(target, hand=False)[f"{kin.side}_hand_C_MC"]
        state["links"][f"{ROBOT_NAMES[i]}/{kin.side}_hand_C_MC"] = np.r_[
            palm[:3, 3], Rotation.from_matrix(palm[:3, :3]).as_quat()]
        state["joints"].update({f"{ROBOT_NAMES[i]}/{name}": {"target": value}
                                for name, value in zip(kin.names, target, strict=True)})
    metrics.update(state, "white_push")
    assert metrics.maximum_wrist_target_error < 1e-12
    assert metrics.report()["acceptance"]["palms_face_down"]
    # Correct wrist position alone must not hide a palm-up orientation.
    pose = state["links"][f"{ROBOT_NAMES[0]}/left_hand_C_MC"]
    pose[3:] = Rotation.from_euler("y", -90., degrees=True).as_quat()
    metrics.update(state, "white_push")
    assert metrics.maximum_wrist_target_error < 1e-12
    assert metrics.samples[-1]["palm_normal_z"]["left"] > .99
    assert not metrics.report()["acceptance"]["palms_face_down"]


def test_parcel_following_saturates_before_switching_arm_branch():
    from conveyor_sharpa_runtime import SharpaController
    controller = SharpaController.__new__(SharpaController)
    controller.kinematics = tuple(Kinematics(side) for side in SIDES)
    kin = controller.kinematics[1]
    # Exercise a real configuration of the replacement arm, then request a
    # shift well beyond its workspace. Saturation must preserve the arm branch.
    target = next(q[1].copy() for name, _, q in motion_table()[0] if name == "white_push")
    original = target.copy()
    requested = kin.forward(target, hand=False)["right_hand_C_MC"].copy()
    requested[1, 3] -= .6
    try:
        unrestricted = kin.arm_pose(requested, target[:7])
    except ValueError:
        unrestricted = None
    assert unrestricted is None or np.max(np.abs(unrestricted-target[:7])) > .5
    arm, shift = controller._reachable_push_pose(target, -.6)
    assert -.6 < shift <= 0.
    assert np.max(np.abs(arm-target[:7])) <= .5
    clearance, _ = kin.arm_worktop_clearance(dict(zip(kin.names[:7], arm)))
    assert clearance >= .005
    before = kin.forward(target, hand=False)["right_hand_C_MC"]
    after = kin.forward(dict(zip(kin.names[:7], arm)), hand=False)["right_hand_C_MC"]
    np.testing.assert_allclose(after[:3, 3]-before[:3, 3], (0., shift, 0.), atol=.0002)
    np.testing.assert_array_equal(target, original)


def test_outfeed_acceptance_rejects_parcels_on_floor_or_past_belt_end():
    from copy import deepcopy
    from conveyor_sharpa import PARCEL_POSITIONS
    from conveyor_sharpa_runtime import SharpaMetrics
    metrics = SharpaMetrics.__new__(SharpaMetrics)
    metrics.peak_penetration = metrics.peak_carton_displacement = 0.
    metrics.maximum_wrist_target_error = 0.
    metrics.maximum_palm_normal_z = -1.
    metrics.minimum_volume_ratio = .9
    parcel = dict(center=[-1., -.25, TABLE_TOP+.04], min=[-1.2, -.4, TABLE_TOP+.001],
                  max=[-.8, -.1, TABLE_TOP+.1], face_normal_z=-1., vertical_rms_speed_m_s=0.)
    valid = {name: deepcopy(parcel) for name in PARCEL_POSITIONS}
    metrics.samples = [dict(time=30., phase="outfeed", parcels=valid)]
    assert metrics.report()["acceptance"]["both_parcels_on_outfeed"]
    for axis, distance in ((2, -.56), (0, -4.)):
        parcels = deepcopy(valid)
        fallen = parcels[next(iter(PARCEL_POSITIONS))]
        for field in ("center", "min", "max"):
            fallen[field][axis] += distance
        metrics.samples = [dict(time=30., phase="outfeed", parcels=parcels)]
        assert not metrics.report()["acceptance"]["both_parcels_on_outfeed"]


def test_authored_scene_compiles_with_nonzero_initial_arm_configuration():
    import gobot
    from conveyor_sharpa_motion import home_configuration
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(EXAMPLE))
        context.load_scene("res://conveyor_packages.jscn")
        snapshot = context.compile_physics_scene_snapshot()
        robots = {r["name"]: r for r in snapshot["robots"]}
        home = home_configuration()
        for side_index, side in enumerate(SIDES):
            robot = robots[ROBOT_NAMES[side_index]]
            assert len(robot["links"]) == len(ET.parse(ASSETS / "openarm_sharpa.urdf").getroot().findall("link"))
            assert not any("iiwa" in link["name"] or "bot_" in link["name"] for link in robot["links"])
            kin = Kinematics(side)
            poses = kin.forward(home[side_index])
            for link in robot["links"]:
                if link["name"] in poses:
                    np.testing.assert_allclose(link["global_transform"]["matrix"], poses[link["name"]], atol=3e-5)
            assert sum(j["type"] != 0 for j in robot["joints"]) == 58
        assert len(snapshot["deformables"]) == 4
        fills = [b for b in snapshot["deformables"] if b["name"].endswith("_fill")]
        np.testing.assert_allclose([b["poisson_ratio"] for b in fills], .49, atol=1e-7)
        # Changing work height must move the collision surfaces and parcels
        # together; a visually level table can otherwise hide a physical step.
        shapes = {shape["name"]: shape for robot in snapshot["robots"]
                  for link in robot["links"] for shape in link["shapes"]}
        table = shapes["worktable_surface_collision"]
        belt = shapes["moving_belt_collision"]
        tops = [s["transform"]["position"][2]+s["size"][2]/2 for s in (table, belt)]
        np.testing.assert_allclose(tops, TABLE_TOP, atol=1e-7)
        np.testing.assert_allclose(table["transform"]["position"][1],
                                   belt["transform"]["position"][1], atol=1e-7)
        np.testing.assert_allclose(table["size"][1], belt["size"][1], atol=1e-7)
        for parcel in snapshot["deformables"]:
            if parcel["name"].endswith("_fill"):
                continue  # Contents sit inside the film, above its bottom.
            matrix = parcel["transform"]["matrix"]
            world = np.asarray(parcel["vertices"]) @ matrix[:3, :3].T+matrix[:3, 3]
            assert -.0001 < world[:, 2].min()-TABLE_TOP < .005
    finally:
        context.clear_scene()
