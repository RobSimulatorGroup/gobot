"""OpenArm v2/Sharpa workcell and source-calibrated arm kinematics."""
from pathlib import Path
from functools import cached_property
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ASSETS = Path(__file__).resolve().parent / "assets/openarm_sharpa"
SIDES = ("left", "right")
ROBOT_NAMES = ("openarm_sharpa", "openarm_sharpa")
ARM_JOINTS = tuple(f"openarm_{side}_joint{i}" for side in SIDES for i in range(1, 8))
ROBOT_POSITION = (0., .48, 0.)
BASE_ROTATION = Rotation.from_euler("z", -90, degrees=True).as_matrix()
PALM_DOWN = np.array(((0., 1., 0.), (0., 0., -1.), (-1., 0., 0.)))
TABLE_TOP = .85
WORK_HEIGHT_OFFSET = TABLE_TOP-.56
WORK_ROW_OFFSET = .18
TABLE_LEFT = -.43
BELT_LENGTH = 3.8
BELT_CENTER = (TABLE_LEFT - .003 - BELT_LENGTH / 2, -.15, TABLE_TOP-.02)
BELT_WIDTH = 1.00
BELT_SPEED = -.14
PARCEL_POSITIONS = {"soft_mailer_blue": (.24, -.28+WORK_ROW_OFFSET),
                    "soft_pouch_yellow": (-.24, -.24+WORK_ROW_OFFSET)}
FILL_POISSON_RATIO = .49
PUSH_ROW_Y = -.10+WORK_ROW_OFFSET


def transform(position=(0., 0., 0.), rotation=None):
    result = np.eye(4)
    result[:3, 3] = position
    if rotation is not None:
        result[:3, :3] = rotation
    return result


class Kinematics:
    def __init__(self, side):
        self.side = side
        self.arm_names = tuple(f"openarm_{side}_joint{i}" for i in range(1, 8))
        self.root_link = "openarm_mount"
        self.xml = ET.parse(ASSETS / f"{side}_arm.urdf").getroot()
        self.joints = {}
        pending = list(self.xml.findall("joint"))
        ready = {self.root_link}
        while pending:
            before = len(pending)
            for joint in pending[:]:
                parent = joint.find("parent").get("link")
                if parent not in ready:
                    continue
                name = joint.get("name")
                child = joint.find("child").get("link")
                origin = joint.find("origin")
                xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                fixed = joint.get("type") == "fixed"
                limit = joint.find("limit")
                self.joints[name] = dict(parent=parent, child=child,
                    origin=transform(xyz, Rotation.from_euler("xyz", rpy).as_matrix()),
                    fixed=fixed, axis=np.zeros(3) if fixed else np.fromstring(joint.find("axis").get("xyz"), sep=" "),
                    limits=(0., 0.) if fixed else (float(limit.get("lower")), float(limit.get("upper"))))
                ready.add(child)
                pending.remove(joint)
            if len(pending) == before:
                raise ValueError("disconnected OpenArm/Sharpa arm asset")
        self.names = tuple(name for name, j in self.joints.items() if not j["fixed"])
        if self.names[:7] != self.arm_names:
            raise ValueError("OpenArm joint order does not match the calibrated chain")
        self.finger_names = tuple(name for name in self.names if name not in ARM_JOINTS)
        self.limits = np.array([self.joints[n]["limits"] for n in self.names])
        robot_xml = ET.parse(ASSETS / "openarm_sharpa.urdf").getroot()
        torso = robot_xml.find("joint[@name='openarm_mount_joint']/origin")
        self.base = transform(ROBOT_POSITION, BASE_ROTATION) @ transform(
            np.fromstring(torso.get("xyz"), sep=" "),
            Rotation.from_euler("xyz", np.fromstring(torso.get("rpy"), sep=" ")).as_matrix())
        self.mass = {}
        self.com = {}
        self.descendants = {}
        for link in self.xml.findall("link"):
            name = link.get("name")
            inertial = link.find("inertial")
            self.mass[name] = float(inertial.find("mass").get("value")) if inertial is not None else 0.
            self.com[name] = np.fromstring(inertial.find("origin").get("xyz", "0 0 0"), sep=" ") if inertial is not None else np.zeros(3)
            self.descendants[name] = [name]
        for joint in reversed(tuple(self.joints.values())):
            self.descendants[joint["parent"]].extend(self.descendants[joint["child"]])

    @cached_property
    def arm_collision_vertices(self):
        """Packaged convex OBJ hulls, expressed in their arm link frames."""
        result = {}
        for link in self.xml.findall("link"):
            name = link.get("name")
            if name not in {self.joints[n]["child"] for n in self.arm_names}:
                continue
            shapes = []
            for collision in link.findall("collision"):
                mesh = collision.find("geometry/mesh")
                path = ASSETS / mesh.get("filename")
                if path.suffix != ".obj":
                    raise ValueError(f"arm clearance requires the packaged OBJ hull: {path}")
                vertices = np.array([np.fromstring(line[2:], sep=" ")
                                     for line in path.read_text().splitlines() if line.startswith("v ")])
                vertices *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                origin = collision.find("origin")
                rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
                vertices = vertices @ rotation.T + np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                shapes.append(vertices)
            result[name] = np.concatenate(shapes)
        return result

    def arm_worktop_clearance(self, values):
        """Screen arm hull vertices against the table and adjoining belt.

        This catches low elbows that a wrist-only height check misses. Runtime
        collision detection still handles complete rigid and soft contacts.
        """
        poses = self.forward(values, hand=False)
        clearance, closest = float("inf"), ""
        for name, local in self.arm_collision_vertices.items():
            pose = poses[name]
            vertices = local @ pose[:3, :3].T + pose[:3, 3]
            above = ((vertices[:, 0] >= BELT_CENTER[0]-BELT_LENGTH/2)
                     & (vertices[:, 0] <= 1.30)
                     & (np.abs(vertices[:, 1]-BELT_CENTER[1]) <= BELT_WIDTH/2))
            if above.any():
                distance = float(vertices[above, 2].min()-TABLE_TOP)
                if distance < clearance:
                    clearance, closest = distance, name
        return clearance, closest

    def forward(self, values, *, world=True, hand=True):
        values = dict(zip(self.names, values, strict=True)) if not isinstance(values, dict) else values
        poses = {self.root_link: self.base.copy() if world else np.eye(4)}
        for name, joint in self.joints.items():
            pose = poses[joint["parent"]] @ joint["origin"]
            if not joint["fixed"]:
                pose = pose @ transform(rotation=Rotation.from_rotvec(joint["axis"] * values.get(name, 0.)).as_matrix())
            poses[joint["child"]] = pose
            if not hand and joint["child"] == f"{self.side}_hand_C_MC":
                break
        return poses

    def arm_pose(self, palm, previous=None):
        """Fit a reachable palm frame, keeping the elbow on its own side."""
        initial = self.home_seed if previous is None else np.asarray(previous)[:7]
        limits = self.limits[:7]
        def residual(q, posture_scale=1.):
            poses = self.forward(dict(zip(self.arm_names, q)), hand=False)
            actual = poses[f"{self.side}_hand_C_MC"]
            angle = Rotation.from_matrix(palm[:3, :3] @ actual[:3, :3].T).as_rotvec()
            elbow = poses[f"openarm_{self.side}_link4"][:3, 3]
            sign = 1 if self.side == "left" else -1
            # Cartesian accuracy takes precedence over the soft elbow and
            # joint-margin preferences, especially in low waiting poses.
            primary = np.r_[(actual[:3, 3] - palm[:3, 3])*600., angle*60.]
            posture = np.r_[max(0., .12-sign*elbow[0])*2., max(0., TABLE_TOP+.22-elbow[2])*10.,
                         max(0., elbow[2]-max(self.base[2, 3]-.03, palm[2, 3]))*10.,
                         np.maximum(.22-np.minimum(q-limits[:, 0], limits[:, 1]-q), 0.)*2.,
                         (q-initial)*.5]
            return np.r_[primary, posture_scale*posture]
        def reaches(q):
            error = residual(q)[:6]
            return np.linalg.norm(error[:3])/600. <= .0002 and np.linalg.norm(error[3:])/60. <= .002
        best = least_squares(residual, np.clip(initial, limits[:, 0]+1e-6, limits[:, 1]-1e-6),
                             bounds=(limits[:, 0]+1e-6, limits[:, 1]-1e-6), max_nfev=80, ftol=1e-5)
        if not reaches(best.x):
            # Refine the same arm branch before trying unrelated seeds. Elbow
            # style is a preference and must not displace the requested hand.
            best = least_squares(lambda q: residual(q, .1), best.x,
                                 bounds=(limits[:, 0]+1e-6, limits[:, 1]-1e-6), max_nfev=160)
        if not reaches(best.x):
            # At the narrower OpenArm wrist limits, even a small soft posture
            # penalty can bias a reachable Cartesian pose outside tolerance.
            # Refine that branch using only the requested pose before reseeding.
            best = least_squares(lambda q: residual(q, 0.), best.x,
                                 bounds=(limits[:, 0]+1e-6, limits[:, 1]-1e-6), max_nfev=160)
        if not reaches(best.x):
            for seed in np.random.default_rng(19).uniform(limits[:, 0], limits[:, 1], (12, 7)):
                candidate = least_squares(lambda q: residual(q, .1), seed,
                    bounds=(limits[:, 0]+1e-6, limits[:, 1]-1e-6), max_nfev=200)
                if np.sum(residual(candidate.x, .1)**2) < np.sum(residual(best.x, .1)**2):
                    best = candidate
                if reaches(best.x):
                    break
        if not reaches(best.x):
            raise ValueError(f"unreachable {self.side} palm {palm[:3, 3]}: {residual(best.x)[:6]}")
        return best.x

    @property
    def home_seed(self):
        return np.array((-.2 if self.side == "left" else .2,
                         -.4 if self.side == "left" else .4, 0., 1.4, 0., 0., 0.))

    def gravity_deflection(self, values, *, structural_stiffness=1.e5, drive_stiffness=24000.):
        """Static palm deflection and motor offsets for Newton's compliant joints.

        Each structural spring carries the weight of its descendant links.
        Rotational loads use the URDF centers of mass in the current pose.
        """
        poses = self.forward(values)
        palm = poses[f"{self.side}_hand_C_MC"]
        centers = {n: p[:3, :3]@self.com[n]+p[:3, 3] for n, p in poses.items()}
        displacement, angle, motor = np.zeros(3), np.zeros(3), np.zeros(7)
        for name, joint in self.joints.items():
            frame = poses[joint["parent"]] @ joint["origin"]
            force, torque = np.zeros(3), np.zeros(3)
            for child in self.descendants[joint["child"]]:
                load = np.array((0., 0., -9.81*self.mass[child]))
                force += load
                torque += np.cross(centers[child]-frame[:3, 3], load)
            if name in self.arm_names:
                axis = frame[:3, :3]@joint["axis"]
                axial = np.dot(torque, axis)
                motor[self.arm_names.index(name)] = -axial/drive_stiffness
                torque -= axial*axis
            rotation = torque/structural_stiffness
            displacement += force/structural_stiffness + np.cross(rotation, palm[:3, 3]-frame[:3, 3])
            angle += rotation
            if joint["child"] == f"{self.side}_hand_C_MC":
                break
        return displacement, angle, motor

    def compensated_arm_pose(self, palm, fingers, previous=None):
        ideal = self.arm_pose(palm, previous)
        displacement, angle, _ = self.gravity_deflection(np.r_[ideal, fingers])
        command = palm.copy()
        command[:3, 3] -= displacement
        command[:3, :3] = Rotation.from_rotvec(-angle).as_matrix() @ palm[:3, :3]
        target = self.arm_pose(command, ideal)
        _, _, motor = self.gravity_deflection(np.r_[target, fingers])
        # Feedforward offsets are actuator commands too; near a joint stop
        # they must not ask the drive to push beyond the URDF limits.
        return np.clip(target+motor, self.limits[:7, 0], self.limits[:7, 1])


def create_arms(root, home, *, builder):
    import gobot
    from conveyor_config import HAND_COLLISION_LAYER, HAND_COLLISION_MASK, HAND_TIP_COLLISION_LAYER
    scene = gobot.load_scene("res://assets/openarm_sharpa/openarm_sharpa.urdf")
    nodes = builder._nodes_by_name(scene.root)
    robot = gobot.create_node("Robot3D", "openarm_sharpa")
    nodes["base_link"].reparent(robot)
    robot.source_path = "res://assets/openarm_sharpa/openarm_sharpa.urdf"
    robot.mode = gobot.RobotMode.Assembly
    robot.position = ROBOT_POSITION
    robot.rotation_degrees = (0., 0., -90.)
    robot.semantic_label = "openarm_sharpa_bimanual"
    nodes = builder._nodes_by_name(robot)
    # The URDF importer assigns diagnostic surface colors. Restore the source
    # materials explicitly before saving the authored PBR scene.
    for link in ET.parse(ASSETS / "openarm_sharpa.urdf").getroot().findall("link"):
        for index, visual in enumerate(link.findall("visual")):
            color = visual.find("material/color")
            name = link.get("name")+"_visual"+(f"_{index+1}" if index else "")
            if color is not None and name in nodes:
                nodes[name].surface_color = tuple(np.fromstring(color.get("rgba"), sep=" "))
    for i, side in enumerate(SIDES):
        kin = Kinematics(side)
        for name, value in zip(kin.names, home[i], strict=True):
            joint = nodes[name]
            arm = name in ARM_JOINTS
            pinch = "thumb" in name or "index" in name
            joint.drive_mode = gobot.JointDriveMode.Position
            joint.drive_stiffness = 24000. if arm else 36. if pinch else 12.
            joint.drive_damping = 500. if arm else .45 if pinch else .25
            joint.armature = .1 if arm else .002
            joint.initial_position = float(value)
            joint.joint_position = float(value)
            # An unattached authored Robot3D has no editor motion session.
            # Bake its initial local link pose explicitly alongside joint state.
            rotation = Rotation.from_rotvec(kin.joints[name]["axis"]*value).as_matrix()
            nodes[kin.joints[name]["child"]].set_transform((0., 0., 0.), b_quat(rotation))
            joint.force_lower_limit = -joint.effort_limit
            joint.force_upper_limit = joint.effort_limit
    for node in nodes.values():
        if node.type_name == "MeshInstance3D" and any(part in node.name for part in
                ("_hand_", "_thumb_", "_index_", "_middle_", "_ring_", "_pinky_", "sharpa_mount")):
            node.surface_color = ((.065, .075, .085, 1.) if "elastomer" in node.name
                                  or "sharpa_mount" in node.name else (.70, .73, .76, 1.))
        elif node.type_name == "CollisionShape3D":
            pad = "elastomer" in node.parent.name
            node.visible = False
            node.collision_layer = HAND_COLLISION_LAYER | (HAND_TIP_COLLISION_LAYER if pad else 0)
            node.collision_mask = HAND_COLLISION_MASK | (HAND_TIP_COLLISION_LAYER if pad else 0)
            node.physics_material = {"sliding_friction": 2.0 if pad else .6,
                "torsional_friction": .006, "rolling_friction": .0002,
                "contact_compliance": 0., "contact_damping": 1.}
    robot.mode = gobot.RobotMode.Motion
    root.add_child(robot)


def b_quat(rotation):
    q = Rotation.from_matrix(rotation).as_quat()
    return tuple(q[[3, 0, 1, 2]])


def create_workcell(root, *, builder):
    import gobot
    b = builder
    frame_robot = gobot.create_node("Robot3D", "warehouse_frame")
    frame_robot.mode = gobot.RobotMode.Assembly
    frame = gobot.create_node("Link3D", "frame")
    b._set_box_inertia(frame, 300., (2., 1., .8))
    def box(name, size, position, color=(.28, .31, .34, 1.), friction=.3):
        b._add_box_geometry(frame, name, size, position, color, sliding_friction=friction)
    box("factory_floor", (10., 6., .08), (0., 0., -.04), (.26, .28, .30, 1.))
    near_y = BELT_CENTER[1]-BELT_WIDTH/2
    far_y = BELT_CENTER[1]+BELT_WIDTH/2
    table_length, table_center = 1.30-TABLE_LEFT, (1.30+TABLE_LEFT)/2
    box("worktable_surface", (table_length, BELT_WIDTH, .08), (table_center, BELT_CENTER[1], TABLE_TOP-.04), (.44, .47, .49, 1.))
    box("worktable_front_apron", (table_length, .06, .14), (table_center, near_y+.03, TABLE_TOP-.13))
    for x in (TABLE_LEFT+.13, 1.17):
        for y in (near_y+.10, far_y-.14):
            height = TABLE_TOP-.08
            box(f"table_leg_{x}_{y}", (.08, .08, height), (x, y, height/2))
    # A shallow incoming tray on the operator's left (viewer right).
    box("incoming_tray_back", (.50, .035, .26), (1.04, far_y-.02, TABLE_TOP+.13))
    box("incoming_tray_side", (.035, BELT_WIDTH-.06, .26), (1.30, BELT_CENTER[1], TABLE_TOP+.13))
    frame_robot.add_child(frame); root.add_child(frame_robot)
    conveyor = gobot.create_node("Robot3D", "conveyor")
    conveyor.mode = gobot.RobotMode.Assembly
    belt = gobot.create_node("Link3D", "belt_surface")
    belt.position = BELT_CENTER
    b._set_box_inertia(belt, 80., (BELT_LENGTH, BELT_WIDTH, .04))
    b._add_box_geometry(belt, "moving_belt", (BELT_LENGTH, BELT_WIDTH, .04), (0., 0., 0.),
                        (.045, .06, .07, 1.), sliding_friction=1e-5)
    for i, x in enumerate(np.arange(-BELT_LENGTH/2+.10, BELT_LENGTH/2, .22)):
        b._add_visual(belt, f"belt_marker_{i:02d}", (.012, BELT_WIDTH-.025, .001),
                      (float(x), 0., .0205), (.14, .16, .17, 1.))
    for sign in (-1, 1):
        b._add_visual(belt, f"belt_rail_{sign}", (BELT_LENGTH, .045, .10),
                      (0., sign*(BELT_WIDTH/2+.025), -.01), (.30, .33, .36, 1.))
    conveyor.add_child(belt); root.add_child(conveyor)
