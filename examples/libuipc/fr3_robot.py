"""Build the FR3 robot asset independently of any demonstration scene."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import gobot
import numpy as np

FR3_ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "franka_emika_panda"
FR3_URDF_PATH = FR3_ASSET_ROOT / "urdf" / "fr3_franka_hand.urdf"
FR3_URDF_RESOURCE = (
    "res://assets/franka_emika_panda/urdf/fr3_franka_hand.urdf"
)
FR3_INITIAL_ARM = {
    "fr3_joint1": -0.0036802115,
    "fr3_joint2": 0.023901723,
    "fr3_joint3": 0.003680411,
    "fr3_joint4": -2.3683236,
    "fr3_joint5": -0.00012918962,
    "fr3_joint6": 2.3922248,
    "fr3_joint7": 0.785492,
}
FR3_INITIAL_FINGER = 0.017


def _xml_vector(
    value: str | None,
    default: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    if value is None:
        return default
    values = tuple(float(component) for component in value.split())
    if len(values) != 3:
        raise ValueError(f"expected three vector components, got {value!r}")
    return values


def _origin_matrix(element: ET.Element) -> np.ndarray:
    origin = element.find("origin")
    xyz = _xml_vector(origin.get("xyz") if origin is not None else None)
    roll, pitch, yaw = _xml_vector(
        origin.get("rpy") if origin is not None else None
    )
    cosine_roll, sine_roll = math.cos(roll), math.sin(roll)
    cosine_pitch, sine_pitch = math.cos(pitch), math.sin(pitch)
    cosine_yaw, sine_yaw = math.cos(yaw), math.sin(yaw)
    rotation_x = np.array(
        (
            (1.0, 0.0, 0.0),
            (0.0, cosine_roll, -sine_roll),
            (0.0, sine_roll, cosine_roll),
        ),
        dtype=np.float64,
    )
    rotation_y = np.array(
        (
            (cosine_pitch, 0.0, sine_pitch),
            (0.0, 1.0, 0.0),
            (-sine_pitch, 0.0, cosine_pitch),
        ),
        dtype=np.float64,
    )
    rotation_z = np.array(
        (
            (cosine_yaw, -sine_yaw, 0.0),
            (sine_yaw, cosine_yaw, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_z @ rotation_y @ rotation_x
    transform[:3, 3] = xyz
    return transform


def _joint_motion_matrix(
    joint_type: str,
    axis: tuple[float, float, float],
    position: float,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    normalized = np.asarray(axis, dtype=np.float64)
    normalized /= np.linalg.norm(normalized)
    if joint_type == "prismatic":
        transform[:3, 3] = normalized * position
        return transform
    cosine, sine = math.cos(position), math.sin(position)
    x, y, z = normalized
    transform[:3, :3] = np.array(
        (
            (
                cosine + x * x * (1.0 - cosine),
                x * y * (1.0 - cosine) - z * sine,
                x * z * (1.0 - cosine) + y * sine,
            ),
            (
                y * x * (1.0 - cosine) + z * sine,
                cosine + y * y * (1.0 - cosine),
                y * z * (1.0 - cosine) - x * sine,
            ),
            (
                z * x * (1.0 - cosine) - y * sine,
                z * y * (1.0 - cosine) + x * sine,
                cosine + z * z * (1.0 - cosine),
            ),
        ),
        dtype=np.float64,
    )
    return transform


def _quaternion_wxyz(matrix: np.ndarray) -> tuple[float, float, float, float]:
    rotation = matrix[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
        w = 0.25 * scale
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(
                1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            ) * 2.0
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
            w = (rotation[2, 1] - rotation[1, 2]) / scale
        elif axis == 1:
            scale = math.sqrt(
                1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            ) * 2.0
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
            w = (rotation[0, 2] - rotation[2, 0]) / scale
        else:
            scale = math.sqrt(
                1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            ) * 2.0
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
            w = (rotation[1, 0] - rotation[0, 1]) / scale
    quaternion = np.asarray((w, x, y, z), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(component) for component in quaternion)


def set_transform_matrix(node: Any, matrix: np.ndarray) -> None:
    node.set_transform(
        tuple(float(component) for component in matrix[:3, 3]),
        _quaternion_wxyz(matrix),
    )


def _inertia_components(matrix: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (
        tuple(float(matrix[index, index]) for index in range(3)),
        (float(matrix[0, 1]), float(matrix[0, 2]), float(matrix[1, 2])),
    )


def _inertia_matrix(spec: dict[str, Any]) -> np.ndarray:
    diagonal = spec["inertia"]
    off_diagonal = spec.get("inertia_off_diagonal", (0.0, 0.0, 0.0))
    return np.asarray(
        (
            (diagonal[0], off_diagonal[0], off_diagonal[1]),
            (off_diagonal[0], diagonal[1], off_diagonal[2]),
            (off_diagonal[1], off_diagonal[2], diagonal[2]),
        ),
        dtype=np.float64,
    )


def _merge_fixed_link_inertia(
    parent: dict[str, Any],
    child: dict[str, Any],
    child_transform: np.ndarray,
) -> dict[str, Any]:
    """Return parent inertial data with one fixed child rigidly aggregated."""

    parent_mass = float(parent["mass"])
    child_mass = float(child["mass"])
    total_mass = parent_mass + child_mass
    if total_mass <= 0.0:
        raise ValueError("fixed-link inertia merge requires positive total mass")

    parent_center = np.asarray(parent["center_of_mass"], dtype=np.float64)
    child_center = (
        child_transform[:3, :3]
        @ np.asarray(child["center_of_mass"], dtype=np.float64)
        + child_transform[:3, 3]
    )
    center = (
        parent_mass * parent_center + child_mass * child_center
    ) / total_mass

    child_inertia = (
        child_transform[:3, :3]
        @ _inertia_matrix(child)
        @ child_transform[:3, :3].T
    )

    def shifted(inertia: np.ndarray, mass: float, body_center: np.ndarray) -> np.ndarray:
        offset = body_center - center
        return inertia + mass * (
            float(offset @ offset) * np.eye(3, dtype=np.float64)
            - np.outer(offset, offset)
        )

    inertia = shifted(
        _inertia_matrix(parent), parent_mass, parent_center
    ) + shifted(child_inertia, child_mass, child_center)
    diagonal, off_diagonal = _inertia_components(inertia)
    merged = dict(parent)
    merged.update(
        {
            "center_of_mass": tuple(float(value) for value in center),
            "inertia": diagonal,
            "inertia_off_diagonal": off_diagonal,
            "mass": total_mass,
        }
    )
    return merged


def _parse_fr3_urdf() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    robot = ET.parse(FR3_URDF_PATH).getroot()
    if robot.get("name") != "fr3":
        raise ValueError("expected the Newton brick_stacking FR3 asset")
    links: dict[str, dict[str, Any]] = {}
    for link_element in robot.findall("link"):
        name = link_element.get("name")
        if not name:
            raise ValueError("FR3 URDF contains an unnamed link")
        inertial = link_element.find("inertial")
        mass_element = inertial.find("mass") if inertial is not None else None
        inertia_element = inertial.find("inertia") if inertial is not None else None
        mass = float(mass_element.get("value", "0")) if mass_element is not None else 0.0
        inertial_transform = (
            _origin_matrix(inertial)
            if inertial is not None
            else np.eye(4, dtype=np.float64)
        )
        if inertia_element is not None:
            raw_inertia = np.asarray(
                (
                    (
                        float(inertia_element.get("ixx", "0")),
                        float(inertia_element.get("ixy", "0")),
                        float(inertia_element.get("ixz", "0")),
                    ),
                    (
                        float(inertia_element.get("ixy", "0")),
                        float(inertia_element.get("iyy", "0")),
                        float(inertia_element.get("iyz", "0")),
                    ),
                    (
                        float(inertia_element.get("ixz", "0")),
                        float(inertia_element.get("iyz", "0")),
                        float(inertia_element.get("izz", "0")),
                    ),
                ),
                dtype=np.float64,
            )
        else:
            raw_inertia = np.eye(3, dtype=np.float64) * 1.0e-6
        inertia_matrix = (
            inertial_transform[:3, :3]
            @ raw_inertia
            @ inertial_transform[:3, :3].T
        )
        inertia, inertia_off_diagonal = _inertia_components(inertia_matrix)
        center_of_mass = tuple(
            float(component) for component in inertial_transform[:3, 3]
        )
        collisions = []
        for index, collision_element in enumerate(link_element.findall("collision")):
            box = collision_element.find("./geometry/box")
            mesh = collision_element.find("./geometry/mesh")
            if box is not None:
                collisions.append(
                    {
                        "name": collision_element.get("name", f"box_{index + 1}"),
                        "type": "box",
                        "size": _xml_vector(box.get("size")),
                        "transform": _origin_matrix(collision_element),
                    }
                )
            elif mesh is not None:
                filename = mesh.get("filename", "")
                package_prefix = "package://franka_emika_panda/"
                if not filename.startswith(package_prefix):
                    raise ValueError(
                        f"FR3 collision mesh has an unexpected URI: {filename!r}"
                    )
                collisions.append(
                    {
                        "name": collision_element.get("name", f"mesh_{index + 1}"),
                        "type": "triangle_mesh",
                        "resource": "res://assets/franka_emika_panda/"
                        + filename.removeprefix(package_prefix),
                        "scale": _xml_vector(mesh.get("scale"), (1.0, 1.0, 1.0)),
                        "transform": _origin_matrix(collision_element),
                    }
                )
            else:
                raise ValueError(
                    f"FR3 collision {link_element.get('name')!r}/{index + 1} "
                    "uses an unsupported geometry"
                )
        links[name] = {
            "center_of_mass": center_of_mass,
            "collisions": collisions,
            "inertia": inertia,
            "inertia_off_diagonal": inertia_off_diagonal,
            "mass": mass,
        }

    joints: dict[str, dict[str, Any]] = {}
    for joint_element in robot.findall("joint"):
        name = joint_element.get("name")
        joint_type = joint_element.get("type")
        parent = joint_element.find("parent")
        child = joint_element.find("child")
        if not name or not joint_type or parent is None or child is None:
            raise ValueError("FR3 URDF contains an incomplete joint")
        axis_element = joint_element.find("axis")
        limit = joint_element.find("limit")
        dynamics = joint_element.find("dynamics")
        joints[name] = {
            "axis": _xml_vector(
                axis_element.get("xyz") if axis_element is not None else None,
                (1.0, 0.0, 0.0),
            ),
            "child": child.get("link"),
            "damping": float(dynamics.get("damping", "0")) if dynamics is not None else 0.0,
            "lower": float(limit.get("lower", "0")) if limit is not None else 0.0,
            "origin": _origin_matrix(joint_element),
            "parent": parent.get("link"),
            "type": joint_type,
            "upper": float(limit.get("upper", "0")) if limit is not None else 0.0,
        }

    required_links = {f"fr3_link{index}" for index in range(8)} | {
        "fr3_hand",
        "fr3_leftfinger",
        "fr3_rightfinger",
    }
    required_joints = {f"fr3_joint{index}" for index in range(1, 9)} | {
        "fr3_hand_joint",
        "fr3_finger_joint1",
        "fr3_finger_joint2",
    }
    if not required_links.issubset(links) or not required_joints.issubset(joints):
        raise ValueError("FR3 URDF is missing a required link or joint")
    return links, joints


def _load_fr3_assets() -> tuple[Any, dict[str, dict[str, Any]]]:
    scene = gobot.load_scene(FR3_URDF_RESOURCE)
    expected = {f"fr3_link{index}" for index in range(8)} | {
        "fr3_hand",
        "fr3_leftfinger",
        "fr3_rightfinger",
    }
    assets: dict[str, dict[str, Any]] = {}
    pending = [scene.root]
    while pending:
        source_link = pending.pop()
        pending.extend(source_link.children)
        if source_link.name not in expected or source_link.type_name != "Link3D":
            continue
        visuals = [
            child for child in source_link.children if child.type_name == "MeshInstance3D"
        ]
        collisions = [
            child for child in source_link.children if child.type_name == "CollisionShape3D"
        ]
        if len(visuals) != 1:
            raise ValueError(
                f"FR3 URDF link {source_link.name!r} has {len(visuals)} visual meshes"
            )
        assets[source_link.name] = {
            "collisions": collisions,
            "visual": visuals[0],
        }
    if set(assets) != expected:
        missing = ", ".join(sorted(expected - set(assets)))
        raise ValueError(f"FR3 URDF importer is missing links: {missing}")
    return scene, assets


def _attach_fr3_visual(
    parent: Any,
    visual: Any,
    transform: np.ndarray | None = None,
) -> None:
    visual.reparent(parent)
    set_transform_matrix(
        visual,
        np.eye(4, dtype=np.float64) if transform is None else transform,
    )


def _attach_fr3_collisions(
    link: Any,
    link_name: str,
    collisions: list[Any],
    spec: dict[str, Any],
    transform: np.ndarray | None = None,
) -> None:
    collision_specs = spec["collisions"]
    if len(collisions) != len(collision_specs) or not collision_specs:
        raise ValueError(
            f"FR3 URDF link {link_name!r} has inconsistent collision geometry"
        )
    sliding_friction = 1.25 if link_name.endswith("finger") else 0.25
    parent_transform = (
        np.eye(4, dtype=np.float64) if transform is None else transform
    )
    for collision, collision_spec in zip(collisions, collision_specs, strict=True):
        collision.reparent(link)
        collision.visible = False
        collision.physics_material = {
            "sliding_friction": sliding_friction,
            "torsional_friction": 0.005,
            "rolling_friction": 0.0001,
        }
        if link_name == "fr3_link0":
            # The fixed base is mounted through the workbench surface.
            collision.collision_layer = 2
            collision.collision_mask = 2
        set_transform_matrix(
            collision,
            parent_transform @ collision_spec["transform"],
        )
        collision.scale = collision_spec.get("scale", (1.0, 1.0, 1.0))


def _fr3_link(name: str, spec: dict[str, Any], asset: dict[str, Any]):
    link = gobot.create_node("Link3D", name)
    link.has_inertial = True
    link.mass = max(1.0e-4, float(spec["mass"]))
    link.center_of_mass = spec["center_of_mass"]
    link.inertia_diagonal = tuple(
        max(1.0e-7, float(component)) for component in spec["inertia"]
    )
    link.set("inertia_off_diagonal", spec["inertia_off_diagonal"])
    _attach_fr3_visual(link, asset["visual"])
    _attach_fr3_collisions(link, name, asset["collisions"], spec)
    return link


def _urdf_joint(parent: Any, child_name: str, name: str, spec: dict[str, Any], initial: float):
    joint = gobot.create_node("Joint3D", name)
    joint.joint_type = (
        gobot.JointType.Prismatic
        if spec["type"] == "prismatic"
        else gobot.JointType.Revolute
    )
    joint.parent_link = parent.name
    joint.child_link = child_name
    joint.axis = spec["axis"]
    joint.lower_limit = spec["lower"]
    joint.upper_limit = spec["upper"]
    joint.velocity_limit = 2.0 if spec["type"] == "revolute" else 0.2
    joint.damping = max(0.05, float(spec["damping"]))
    joint.initial_position = initial
    joint.joint_position = initial
    set_transform_matrix(joint, spec["origin"])
    parent.add_child(joint)
    return joint


def joint_limits() -> dict[str, tuple[float, float]]:
    """Return the source limits before float32 scene-property conversion."""
    _, joints = _parse_fr3_urdf()
    return {name: (spec["lower"], spec["upper"]) for name, spec in joints.items()}


def create_fr3_robot(
    name: str = "fr3_arm",
    *,
    arm_positions: Mapping[str, float] = FR3_INITIAL_ARM,
    finger_position: float = FR3_INITIAL_FINGER,
):
    """Return a detached Robot3D with the fixed hand merged into link7.

    The active Gobot project must contain the FR3 URDF and mesh resources.
    Joint poses and limits retain the original URDF coordinates.
    """
    links, joints = _parse_fr3_urdf()
    asset_scene, assets = _load_fr3_assets()
    hand_transform = (
        joints["fr3_joint8"]["origin"] @ joints["fr3_hand_joint"]["origin"]
    )
    links["fr3_link7"] = _merge_fixed_link_inertia(
        links["fr3_link7"], links["fr3_hand"], hand_transform
    )
    robot = gobot.create_node("Robot3D", name)
    robot.source_path = FR3_URDF_RESOURCE
    parent = _fr3_link("fr3_link0", links["fr3_link0"], assets["fr3_link0"])
    robot.add_child(parent)
    for index in range(1, 8):
        joint_name = f"fr3_joint{index}"
        child_name = f"fr3_link{index}"
        initial = arm_positions[joint_name]
        joint = _urdf_joint(parent, child_name, joint_name, joints[joint_name], initial)
        child = _fr3_link(child_name, links[child_name], assets[child_name])
        set_transform_matrix(
            child,
            _joint_motion_matrix(joints[joint_name]["type"], joints[joint_name]["axis"], initial),
        )
        joint.add_child(child)
        parent = child

    _attach_fr3_visual(parent, assets["fr3_hand"]["visual"], hand_transform)
    _attach_fr3_collisions(
        parent,
        "fr3_hand",
        assets["fr3_hand"]["collisions"],
        links["fr3_hand"],
        hand_transform,
    )
    for joint_name in ("fr3_finger_joint1", "fr3_finger_joint2"):
        spec = dict(joints[joint_name])
        spec["origin"] = hand_transform @ spec["origin"]
        child_name = str(spec["child"])
        joint = _urdf_joint(
            parent,
            child_name,
            joint_name,
            spec,
            finger_position,
        )
        child = _fr3_link(child_name, links[child_name], assets[child_name])
        set_transform_matrix(
            child,
            _joint_motion_matrix(spec["type"], spec["axis"], finger_position),
        )
        joint.add_child(child)
    del asset_scene
    return robot
