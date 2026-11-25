"""Mutable C++-backed PlanningWorld for kinematics, dynamic obstacles, attachments, and ACM."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
import warnings

import numpy as np
from scipy.spatial import ConvexHull

from .retime import check_trajectory_bounds
from .types import JointName, Link, PlanningGroup, PoseConstraint, Trajectory, _normalize_tol

try:
    from gobot import _coal_native as _native
except ImportError as error:
    raise ImportError("gobot.planning requires a Gobot build with GOB_BUILD_COAL=ON") from error


def _link_key(link: Link) -> str:
    return f"{link[0]}::{link[1]}"


def _rigid_transform(value: dict) -> tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(value["matrix"], dtype=np.float64)
    if (
        mat.shape != (4, 4)
        or not np.isfinite(mat).all()
        or not np.allclose(mat[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(mat[:3, :3].T @ mat[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(mat[:3, :3]), 1.0, atol=1e-5)
    ):
        raise ValueError("planning requires finite rigid 4x4 transforms")
    return mat[:3, 3].copy(), mat[:3, :3].copy()


def _frame_from_rt(position: Sequence[float] | np.ndarray,
                   rotation: Sequence[Sequence[float]] | np.ndarray | None = None) -> np.ndarray:
    p = np.asarray(position, dtype=np.float64)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("position must be a finite 3-vector")
    if rotation is None:
        r = np.eye(3, dtype=np.float64)
    else:
        r = np.asarray(rotation, dtype=np.float64)
        if (r.shape != (3, 3) or not np.isfinite(r).all()
                or not np.allclose(r.T @ r, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(r), 1.0, atol=1e-5)):
            raise ValueError("rotation must be a proper 3x3 rotation matrix")
    frame = np.empty((3, 4), dtype=np.float64)
    frame[:, :3] = r
    frame[:, 3] = p
    return frame


def _prepare_mesh_spec(spec: dict, label: str) -> dict:
    vertices = np.asarray(spec["vertices"], dtype=np.float64)
    indices = np.asarray(spec["indices"])
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"invalid vertices on {label}")
    if indices.size == 0 and len(vertices) % 3 == 0:
        indices = np.arange(len(vertices))
    if (not indices.size or indices.dtype.kind not in "iu" or indices.size % 3
            or (indices < 0).any() or (indices >= len(vertices)).any()):
        raise ValueError(f"invalid triangle indices on {label}")
    points = np.unique(vertices[indices.reshape(-1)], axis=0)
    if len(points) < 3:
        raise ValueError(f"degenerate mesh on {label}")
    centered = points - points.mean(axis=0)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    tolerance = singular[0] * max(centered.shape) * np.finfo(np.float64).eps
    if singular[1] <= tolerance:
        raise ValueError(f"collinear mesh on {label}")
    if singular[2] <= tolerance:
        polygon = ConvexHull(centered @ axes[:2].T)
        hull_points = points[polygon.vertices]
        triangles = [(0, i, i + 1) for i in range(1, len(hull_points) - 1)]
    else:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        triangles = np.searchsorted(hull.vertices, hull.simplices)
        a, b, c = (hull_points[triangles[:, i]] for i in range(3))
        reverse = (np.cross(b - a, c - a) * hull.equations[:, :3]).sum(axis=1) < 0
        triangles[reverse] = triangles[reverse, ::-1]
    return dict(spec, indices=indices.reshape(-1), hull_points=hull_points, hull_faces=triangles)


def _make_native_shape(spec: dict, label: str):
    kind = int(spec["type"])
    for field in {1: ("size",), 2: ("radius",), 3: ("radius", "height"), 4: ("radius", "height")}.get(kind, ()):
        val = np.asarray(spec[field], dtype=np.float64)
        if val.shape != ((3,) if field == "size" else ()) or not np.isfinite(val).all() or (val < 0).any():
            raise ValueError(f"invalid {field} on {label}")
    if kind == 5:
        spec = _prepare_mesh_spec(spec, label)
    return _native.Shape(label, spec)


class PlanningWorld:
    """Mutable C++-backed planning scene with kinematics, dynamic obstacles, attach/detach, and ACM."""

    def __init__(self, native_world, *, joint_names: tuple[JointName, ...], link_map: dict[Link, int],
                 robot_branches: dict[str, list[int]], lower: np.ndarray, upper: np.ndarray,
                 velocity_limits: np.ndarray, acceleration_limits: np.ndarray,
                 initial_positions: np.ndarray) -> None:
        self._native = native_world
        self.joint_names = joint_names
        self._joint_index = {name: i for i, name in enumerate(joint_names)}
        self.links = link_map
        self._robot_branches = robot_branches
        self.lower_limits = lower
        self.upper_limits = upper
        self.velocity_limits = velocity_limits
        self.acceleration_limits = acceleration_limits
        self.initial_positions = initial_positions
        self._groups: dict[str, PlanningGroup] = {}

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict,
        robots: Mapping[str, str | Sequence[str]],
        *,
        ignored_link_pairs: Sequence[tuple[Link, Link]] = (),
        default_acceleration_limit: float = 10.0,
    ) -> PlanningWorld:
        if snapshot.get("schema_version") != 1:
            raise ValueError("unsupported physics snapshot schema")
        if not robots:
            raise ValueError("select at least one robot and base link")
        if snapshot.get("deformables"):
            raise ValueError("planning collision does not support deformable bodies")

        by_name = defaultdict(list)
        for robot in snapshot["robots"]:
            by_name[robot["name"]].append(robot)

        parent: list[int] = []
        kinds: list[int] = []
        joint_idx: list[int] = []
        parent_joint: list[np.ndarray] = []
        joint_child: list[np.ndarray] = []
        axes: list[np.ndarray] = []
        branches_in: list[tuple[str, int, np.ndarray, np.ndarray]] = []
        robot_branches: dict[str, list[int]] = defaultdict(list)
        body_names: list[str] = []
        joint_names: list[JointName] = []
        lower_list: list[float] = []
        upper_list: list[float] = []
        vel_list: list[float] = []
        acc_list: list[float] = []
        defaulted_acc_joints: list[str] = []
        init_list: list[float] = []
        link_map: dict[Link, int] = {}
        link_index_str: dict[str, int] = {}
        adjacent: set[tuple[Link, Link]] = set()

        for robot_name, bases in robots.items():
            if len(by_name[robot_name]) != 1:
                raise ValueError(f"expected one robot {robot_name!r} in snapshot")
            robot = by_name[robot_name][0]
            roots = (bases,) if isinstance(bases, str) else tuple(bases)
            if not roots:
                raise ValueError(f"select at least one base link for {robot_name!r}")
            links_by_name = {l["name"]: l for l in robot["links"]}
            children = defaultdict(list)
            parent_joints = {}
            for j in robot["joints"]:
                p_name, c_name = j["parent_link"], j["child_link"]
                children[p_name].append(c_name)
                if c_name in parent_joints:
                    raise ValueError(f"multiple parent joints on link {c_name!r}")
                parent_joints[c_name] = j
                adjacent.add(((robot_name, p_name), (robot_name, c_name)))

            for base in roots:
                if base not in links_by_name:
                    raise ValueError(f"unknown base link {(robot_name, base)!r}")
                # Topological BFS from base link.
                order = [base]
                seen = {base}
                head = 0
                while head < len(order):
                    cur = order[head]
                    head += 1
                    for ch in children[cur]:
                        if ch in links_by_name and ch not in seen:
                            seen.add(ch)
                            order.append(ch)
                if any((robot_name, nm) in link_map for nm in order):
                    raise ValueError(f"overlapping base links for robot {robot_name!r}")

                root_body_id = len(parent)
                root_p, root_r = _rigid_transform(links_by_name[base]["global_transform"])
                branch_id = len(branches_in)
                branches_in.append((robot_name, root_body_id, root_p, root_r))
                robot_branches[robot_name].append(branch_id)

                # Preserve the robot's authored joint order for movable joints in this branch.
                branch_movable = [j for j in robot["joints"]
                                  if j["child_link"] in seen and int(j["type"]) in (1, 2, 3)]
                joint_base_idx = len(joint_names)
                movable_idx_map = {j["name"]: joint_base_idx + k for k, j in enumerate(branch_movable)}
                for j in branch_movable:
                    kind = int(j["type"])
                    joint_names.append((robot_name, j["name"]))
                    if kind == 2:
                        lower_list.append(-math.inf)
                        upper_list.append(math.inf)
                    elif j["upper_limit"] > j["lower_limit"]:
                        lower_list.append(float(j["lower_limit"]))
                        upper_list.append(float(j["upper_limit"]))
                    else:
                        lower_list.append(-math.inf)
                        upper_list.append(math.inf)
                    speed = float(j.get("velocity_limit", 0.0))
                    vel_list.append(speed if speed > 0 else math.inf)
                    acc = float(j.get("acceleration_limit", 0.0))
                    if acc > 0:
                        acc_list.append(acc)
                    else:
                        acc_list.append(float(default_acceleration_limit))
                        defaulted_acc_joints.append(f"{robot_name}::{j['name']}")
                    init_list.append(float(j.get("initial_position", 0.0)))

                local_body_id: dict[str, int] = {}
                for link_name in order:
                    bid = len(parent)
                    local_body_id[link_name] = bid
                    key = (robot_name, link_name)
                    link_map[key] = bid
                    link_index_str[_link_key(key)] = bid
                    body_names.append(_link_key(key))
                    if link_name == base:
                        parent.append(-1)
                        kinds.append(4)
                        joint_idx.append(-1)
                        parent_joint.append(np.eye(4, dtype=np.float64))
                        joint_child.append(np.eye(4, dtype=np.float64))
                        axes.append(np.array([1.0, 0.0, 0.0], dtype=np.float64))
                        continue

                    j = parent_joints[link_name]
                    kind = int(j["type"])
                    if kind not in (0, 1, 2, 3):
                        raise ValueError(f"unsupported joint type {kind} on {j['name']!r}")
                    p_bid = local_body_id[j["parent_link"]]
                    parent.append(p_bid)
                    kinds.append(kind)
                    joint_idx.append(-1 if kind == 0 else movable_idx_map[j["name"]])

                    p_tf = np.asarray(links_by_name[j["parent_link"]]["global_transform"]["matrix"], dtype=np.float64)
                    j_tf = np.asarray(j["global_transform"]["matrix"], dtype=np.float64)
                    c_tf = np.asarray(links_by_name[link_name]["global_transform"]["matrix"], dtype=np.float64)
                    parent_joint.append(np.linalg.inv(p_tf) @ j_tf)
                    joint_child.append(np.linalg.inv(j_tf) @ c_tf)
                    axis = np.asarray(j["axis"], dtype=np.float64)
                    norm = float(np.linalg.norm(axis))
                    if axis.shape != (3,) or not np.isfinite(axis).all() or norm < 1e-10:
                        raise ValueError(f"invalid joint axis on {j['name']!r}")
                    axes.append(axis / norm)

        if not joint_names:
            raise ValueError("selected robots have no movable joints")
        if defaulted_acc_joints:
            warnings.warn(
                f"Joints missing acceleration_limit ({len(defaulted_acc_joints)} joints, e.g. "
                f"{defaulted_acc_joints[0]}): defaulting to {float(default_acceleration_limit):.1f} rad/s^2",
                UserWarning,
                stacklevel=2,
            )
        lower = np.asarray(lower_list, dtype=np.float64)
        upper = np.asarray(upper_list, dtype=np.float64)
        vel = np.asarray(vel_list, dtype=np.float64)
        acc = np.asarray(acc_list, dtype=np.float64)
        init_q = np.clip(np.asarray(init_list, dtype=np.float64), lower, upper)

        native_world = _native.PlanningWorld()
        native_world.init_kinematics(
            parent, kinds, joint_idx,
            np.ascontiguousarray(parent_joint, dtype=np.float64),
            np.ascontiguousarray(joint_child, dtype=np.float64),
            np.ascontiguousarray(axes, dtype=np.float64),
            branches_in, body_names, [f"{r}::{j}" for r, j in joint_names],
            lower, upper, vel, acc, init_q, link_index_str,
        )

        # Populate initial ACM (adjacent links + explicit ignored_link_pairs).
        for a, b in adjacent:
            native_world.set_collision_enabled(_link_key(a), _link_key(b), False)
        for a, b in ignored_link_pairs:
            native_world.set_collision_enabled(_link_key(a), _link_key(b), False)

        # Register robot and environment collision shapes.
        used_names: dict[str, int] = defaultdict(int)

        def unique_name(base_name: str) -> str:
            idx = used_names[base_name]
            used_names[base_name] = idx + 1
            return base_name if idx == 0 else f"{base_name}#{idx}"

        def register_shape(spec: dict, link: Link, body_id: int, rest: tuple[np.ndarray, np.ndarray] | None) -> None:
            p, r = _rigid_transform(spec["transform"])
            if rest is not None:
                origin, frame = rest
                p = frame.T @ (p - origin)
                r = frame.T @ r
            raw_name = spec.get("path") or spec.get("name") or f"shape_{spec['type']}"
            full_name = unique_name(raw_name if not link[0] else f"{link[0]}/{link[1]}/{raw_name}")
            sh = _make_native_shape(spec, full_name)
            native_world.add_shape(
                full_name, _link_key(link), body_id, _frame_from_rt(p, r),
                int(spec.get("collision_layer", 1)), int(spec.get("collision_mask", 1)), sh,
            )

        for robot in snapshot["robots"]:
            for link in robot["links"]:
                if link.get("role") == "virtual_root":
                    continue
                key = (robot["name"], link["name"])
                bid = link_map.get(key, -1)
                rest = _rigid_transform(link["global_transform"]) if bid >= 0 else None
                for sh in link.get("shapes", ()):
                    if not sh.get("disabled", False):
                        register_shape(sh, key, bid, rest)

        for sh in snapshot.get("loose_shapes", ()):
            if not sh.get("disabled", False):
                key = ("", sh.get("path", sh["name"]))
                register_shape(sh, key, -1, None)

        for terrain in snapshot.get("terrains", ()):
            if terrain.get("heightfields"):
                raise ValueError("planning collision does not support terrain heightfields")
            key = ("", terrain["name"])
            for geom in terrain.get("boxes", ()) + terrain.get("meshes", ()):
                val = dict(terrain, **geom, type=1 if "size" in geom else 5)
                register_shape(val, key, -1, None)

        return cls(
            native_world, joint_names=tuple(joint_names), link_map=link_map,
            robot_branches=dict(robot_branches), lower=lower, upper=upper,
            velocity_limits=vel, acceleration_limits=acc, initial_positions=init_q,
        )

    @property
    def version(self) -> int:
        return int(self._native.version)

    @property
    def topology_version(self) -> int:
        return int(self._native.topology_version)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def num_pairs(self) -> int:
        return int(self._native.num_pairs)

    @property
    def self_pair_count(self) -> int:
        return int(self._native.self_pair_count)

    @property
    def has_mesh(self) -> bool:
        return bool(self._native.has_mesh)

    @property
    def collision_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._native.pair_shape_names())

    # --- Mutable World Operations ---

    def add_box(self, name: str, size: Sequence[float], position: Sequence[float],
                rotation: Sequence[Sequence[float]] | None = None, *,
                collision_layer: int = 1, collision_mask: int = 1) -> None:
        spec = {"type": 1, "size": np.asarray(size, dtype=np.float64)}
        sh = _make_native_shape(spec, name)
        self._native.add_shape(name, f"::{name}", -1, _frame_from_rt(position, rotation),
                               collision_layer, collision_mask, sh)

    def add_sphere(self, name: str, radius: float, position: Sequence[float], *,
                   collision_layer: int = 1, collision_mask: int = 1) -> None:
        spec = {"type": 2, "radius": float(radius)}
        sh = _make_native_shape(spec, name)
        self._native.add_shape(name, f"::{name}", -1, _frame_from_rt(position, None),
                               collision_layer, collision_mask, sh)

    def add_capsule(self, name: str, radius: float, height: float, position: Sequence[float],
                    rotation: Sequence[Sequence[float]] | None = None, *,
                    collision_layer: int = 1, collision_mask: int = 1) -> None:
        spec = {"type": 4, "radius": float(radius), "height": float(height)}
        sh = _make_native_shape(spec, name)
        self._native.add_shape(name, f"::{name}", -1, _frame_from_rt(position, rotation),
                               collision_layer, collision_mask, sh)

    def update_obstacle_pose(self, name: str, position: Sequence[float],
                             rotation: Sequence[Sequence[float]] | None = None) -> None:
        """O(1) pose update for a static world obstacle without rebuilding BVHs or pairs."""
        self._native.update_obstacle_pose(name, _frame_from_rt(position, rotation))

    def remove_obstacle(self, name: str) -> None:
        self._native.remove_obstacle(name)

    def attach_object(self, name: str, link: Link,
                      local_position: Sequence[float] = (0.0, 0.0, 0.0),
                      local_rotation: Sequence[Sequence[float]] | None = None,
                      *, touch_links: Sequence[Link] = ()) -> None:
        """Attach a world obstacle to a robot link (e.g., grasping a payload)."""
        self._native.attach_object(
            name, _link_key(link), _frame_from_rt(local_position, local_rotation),
            [_link_key(tl) for tl in touch_links],
        )

    def detach_object(self, name: str, q: Sequence[float] | np.ndarray) -> None:
        """Detach an attached object at configuration q back into the static world."""
        self._native.detach_object(name, self.configuration(q))

    def set_collision_enabled(self, link_a: Link | str, link_b: Link | str, enabled: bool) -> None:
        ka = _link_key(link_a) if isinstance(link_a, tuple) else f"::{link_a}"
        kb = _link_key(link_b) if isinstance(link_b, tuple) else f"::{link_b}"
        self._native.set_collision_enabled(ka, kb, bool(enabled))

    def set_robot_base_pose(self, robot: str, position: Sequence[float],
                            rotation: Sequence[Sequence[float]] | None = None,
                            *, branch_index: int = 0) -> None:
        branches = self._robot_branches.get(robot)
        if not branches or not (0 <= branch_index < len(branches)):
            raise ValueError(f"unknown robot/branch {robot!r}[{branch_index}]")
        frame = _frame_from_rt(position, rotation)
        self._native.set_robot_base_pose(branches[branch_index], frame[:, 3], frame[:, :3])

    # --- Planning Groups ---

    def define_group(self, name: str, active_joints: Sequence[JointName]) -> PlanningGroup:
        for j in active_joints:
            if j not in self._joint_index:
                raise ValueError(f"unknown joint {j!r} in group {name!r}")
        group = PlanningGroup(name, tuple(active_joints))
        self._groups[name] = group
        return group

    def group_mask(self, group: str | PlanningGroup | Sequence[JointName] | None) -> np.ndarray:
        if group is None:
            return np.ones(self.num_joints, dtype=np.bool_)
        if isinstance(group, str):
            if group not in self._groups:
                raise ValueError(f"unknown planning group {group!r}")
            joints = self._groups[group].active_joints
        elif isinstance(group, PlanningGroup):
            joints = group.active_joints
        else:
            joints = tuple(group)
        mask = np.zeros(self.num_joints, dtype=np.bool_)
        for j in joints:
            if j not in self._joint_index:
                raise ValueError(f"unknown joint {j!r}")
            mask[self._joint_index[j]] = True
        if not mask.any():
            raise ValueError("planning group must contain at least one active joint")
        return mask

    # --- Kinematics & Queries ---

    def configuration(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(q, dtype=np.float64)
        if arr.shape != (self.num_joints,) or not np.isfinite(arr).all():
            raise ValueError(f"expected finite configuration of length {self.num_joints}")
        return np.ascontiguousarray(arr)

    def configurations(self, q: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        arr = np.asarray(q, dtype=np.float64)
        if arr.ndim == 1:
            return self.configuration(arr)[None, :]
        if arr.shape[-1] != self.num_joints or not np.isfinite(arr).all():
            raise ValueError(f"expected finite configurations with trailing size {self.num_joints}")
        return np.ascontiguousarray(arr)

    def forward_kinematics(self, q: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(q, dtype=np.float64)
        single = arr.ndim == 1
        batch_shape = arr.shape[:-1]
        flat = np.ascontiguousarray(arr.reshape(-1, self.num_joints))
        pos, rot = self._native.forward_kinematics(flat)
        if single:
            return pos[0], rot[0]
        return pos.reshape(*batch_shape, -1, 3), rot.reshape(*batch_shape, -1, 3, 3)

    def link_pose(self, q: Sequence[float] | np.ndarray, link: Link) -> tuple[np.ndarray, np.ndarray]:
        if link not in self.links:
            raise ValueError(f"unknown link {link!r}")
        idx = self.links[link]
        pos, rot = self.forward_kinematics(q)
        return pos[..., idx, :], rot[..., idx, :, :]

    def pair_margins(self, margin: float, world_margin: float | None = None) -> np.ndarray:
        wm = margin if world_margin is None else world_margin
        out = np.full(self.num_pairs, float(wm), dtype=np.float64)
        out[:self.self_pair_count] = float(margin)
        return out

    def collision_distances(self, q: Sequence[float] | np.ndarray, *, verify_original: bool = False) -> np.ndarray:
        arr = np.asarray(q, dtype=np.float64)
        single = arr.ndim == 1
        batch_shape = arr.shape[:-1]
        flat = np.ascontiguousarray(arr.reshape(-1, self.num_joints))
        d = self._native.query_all_distances(flat, verify_original)
        if single:
            return d[0]
        return d.reshape(*batch_shape, -1)

    def compile_pose_constraint(self, c: PoseConstraint):
        if c.link not in self.links:
            raise ValueError(f"unknown target link {c.link!r}")
        spec = _native.PoseConstraintSpec()
        spec.link_id = self.links[c.link]
        spec.weight = float(c.weight)
        if c.position is not None:
            p = np.asarray(c.position, dtype=np.float64)
            if p.shape != (3,) or not np.isfinite(p).all():
                raise ValueError("target position must be a finite 3-vector")
            spec.has_position = True
            spec.position = p
            spec.pos_tol = _normalize_tol(c.position_tolerance)
        if c.rotation is not None:
            r = np.asarray(c.rotation, dtype=np.float64)
            if (r.shape != (3, 3) or not np.isfinite(r).all()
                    or not np.allclose(r.T @ r, np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(r), 1.0, atol=1e-5)):
                raise ValueError("target rotation must be a proper 3x3 rotation matrix")
            spec.has_rotation = True
            spec.rotation = r
            spec.rot_tol = _normalize_tol(c.rotation_tolerance)
        return spec

    def validate_trajectory(
        self,
        positions_or_trajectory: np.ndarray | Trajectory,
        *,
        dt: float | None = None,
        margin: float = 0.002,
        world_margin: float | None = None,
        collision_substeps: int = 16,
        max_segment_length: float | None = None,
        verify_original: bool = True,
        velocity_scale: float = 1.0,
        acceleration_scale: float = 1.0,
        check_acceleration: bool = True,
    ) -> tuple[bool, str | None]:
        """Validate either a waypoint polyline ``[T, J]`` or a continuous ``Trajectory`` curve."""
        if isinstance(positions_or_trajectory, Trajectory):
            traj = positions_or_trajectory
            q = np.asarray(traj.positions, dtype=np.float64)
            if q.ndim != 2 or q.shape[0] < 2 or q.shape[1] != self.num_joints:
                return False, "trajectory must have shape [T>=2, J]"
            ok_bounds, bound_err, _ = check_trajectory_bounds(
                traj,
                self.velocity_limits * velocity_scale,
                self.acceleration_limits * acceleration_scale,
                check_velocity=True,
                check_acceleration=check_acceleration,
            )
            if not ok_bounds:
                return False, bound_err
            substeps = int(collision_substeps)
            if max_segment_length is not None:
                longest = float(np.linalg.norm(np.diff(q, axis=0), axis=-1).max())
                substeps = max(substeps, math.ceil(longest / max_segment_length))
            substeps = max(substeps, 1)
            # Sample along the actual continuous cubic Hermite curve per segment.
            u = np.arange(substeps, dtype=np.float64) / substeps
            t0 = traj.times[:-1, None]
            t1 = traj.times[1:, None]
            ts = np.concatenate(((t0 + u[None, :] * (t1 - t0)).reshape(-1), traj.times[-1:]))
            samples, _, _ = traj.sample(ts)
            if (not np.isfinite(samples).all()
                    or (samples < self.lower_limits - 1e-6).any()
                    or (samples > self.upper_limits + 1e-6).any()):
                return False, "trajectory violates joint limits"
        else:
            q = np.asarray(positions_or_trajectory, dtype=np.float64)
            if q.ndim != 2 or q.shape[0] < 2 or q.shape[1] != self.num_joints:
                return False, "trajectory must have shape [T>=2, J]"
            if not np.isfinite(q).all() or (q < self.lower_limits).any() or (q > self.upper_limits).any():
                return False, "trajectory violates joint limits"
            if dt is not None:
                speed = np.abs(np.diff(q, axis=0)) / dt
                if (speed > self.velocity_limits * velocity_scale * (1.0 + 1e-6)).any():
                    return False, "trajectory violates joint velocity limits"
            substeps = int(collision_substeps)
            if max_segment_length is not None:
                longest = float(np.linalg.norm(np.diff(q, axis=0), axis=-1).max())
                substeps = max(substeps, math.ceil(longest / max_segment_length))
            substeps = max(substeps, 1)
            u = (np.arange(substeps, dtype=np.float64) / substeps)[:, None]
            between = (1.0 - u)[None, :, :] * q[:-1, None, :] + u[None, :, :] * q[1:, None, :]
            samples = np.concatenate((between.reshape(-1, self.num_joints), q[-1:]), axis=0)

        if self.num_pairs == 0:
            return True, None
        (frames,) = self._native.compute_frames(np.ascontiguousarray(samples), False)
        margins = self.pair_margins(margin, world_margin)
        is_traj = isinstance(positions_or_trajectory, Trajectory)
        t_arg = np.ascontiguousarray(positions_or_trajectory.times, dtype=np.float64) if is_traj else None
        v_arg = np.ascontiguousarray(positions_or_trajectory.velocities, dtype=np.float64) if is_traj else None
        q_c = np.ascontiguousarray(q, dtype=np.float64)

        rows, pids = self._native.screen_aabbs(frames, margins, False, substeps)
        if len(rows):
            hit = self._native.first_violation_indexed(frames, rows, pids, margins, False)
            if hit is not None:
                seg = min(int(hit[0]) // substeps, len(q) - 2)
                return False, f"convex clearance failed on segment {seg}, pair {self.collision_pairs[int(hit[1])]}"
        ccd = self._native.first_violation_continuous(q_c, substeps, frames, margins, False, 5e-4, t_arg, v_arg)
        if ccd is not None:
            return False, f"continuous convex clearance failed on segment {int(ccd[0])}, pair {self.collision_pairs[int(ccd[2])]}"
        if verify_original and self.has_mesh:
            rows, pids = self._native.screen_aabbs(frames, margins, True, substeps)
            if len(rows):
                hit = self._native.first_violation_indexed(frames, rows, pids, margins, True)
                if hit is not None:
                    seg = min(int(hit[0]) // substeps, len(q) - 2)
                    return False, f"mesh clearance failed on segment {seg}, pair {self.collision_pairs[int(hit[1])]}"
            ccd_m = self._native.first_violation_continuous(q_c, substeps, frames, margins, True, 5e-4, t_arg, v_arg)
            if ccd_m is not None:
                return False, f"continuous mesh clearance failed on segment {int(ccd_m[0])}, pair {self.collision_pairs[int(ccd_m[2])]}"
        return True, None
