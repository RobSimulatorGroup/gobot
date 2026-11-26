"""Mutable C++-backed PlanningWorld for kinematics, dynamic obstacles, attachments, and ACM."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import threading
from time import perf_counter
import warnings

import numpy as np
from scipy.spatial import ConvexHull

from .retime import check_trajectory_bounds
from .types import (
    JointName,
    Link,
    LinkSphere,
    PlanningGroup,
    PoseConstraint,
    SyncResult,
    Trajectory,
    _normalize_tol,
    _validate_manifold_constraints,
)

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


def _shape_geom_signature(spec: Mapping, label: str) -> tuple:
    """Compute a deterministic, hashable geometry signature without building convex hulls or BVHs."""
    kind = int(spec["type"])
    convex = bool(spec.get("convex", True))
    if kind == 1:
        size = np.asarray(spec["size"], dtype=np.float64)
        if size.shape != (3,) or not np.isfinite(size).all() or (size < 0).any():
            raise ValueError(f"invalid size on {label}")
        return (1, tuple(float(x) for x in size), convex)
    if kind == 2:
        radius = np.asarray(spec["radius"], dtype=np.float64)
        if radius.shape != () or not np.isfinite(radius) or float(radius) < 0.0:
            raise ValueError(f"invalid radius on {label}")
        return (2, float(radius), convex)
    if kind in (3, 4):
        radius = np.asarray(spec["radius"], dtype=np.float64)
        height = np.asarray(spec["height"], dtype=np.float64)
        if (radius.shape != () or not np.isfinite(radius) or float(radius) < 0.0
                or height.shape != () or not np.isfinite(height) or float(height) < 0.0):
            raise ValueError(f"invalid radius/height on {label}")
        return (kind, float(radius), float(height), convex)
    if kind == 5:
        geom_ver = spec.get("geometry_version")
        if geom_ver is not None:
            return (5, "version", geom_ver, convex)
        vertices = np.ascontiguousarray(spec["vertices"], dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError(f"invalid vertices on {label}")
        indices = np.ascontiguousarray(spec["indices"])
        if indices.size == 0 and len(vertices) % 3 == 0:
            indices = np.arange(len(vertices), dtype=np.int64)
        if (not indices.size or indices.dtype.kind not in "iu" or indices.size % 3
                or (indices < 0).any() or (indices >= len(vertices)).any()):
            raise ValueError(f"invalid triangle indices on {label}")
        idx_i64 = np.ascontiguousarray(indices.reshape(-1), dtype=np.int64)
        h = hashlib.blake2b(digest_size=16)
        h.update(vertices.tobytes())
        h.update(idx_i64.tobytes())
        return (5, "digest", vertices.shape, int(idx_i64.size), h.digest(), convex)
    if kind == 6:
        return (6,)
    raise ValueError(f"unsupported shape type {kind} on {label}")


@dataclass
class _TrackedObstacle:
    sync_id: tuple
    node_key: tuple | None
    stable_key: tuple | None
    path_key: tuple
    shape_name: str
    link_key: str
    world_frame: np.ndarray | None
    layer: int
    mask: int
    geom_sig: tuple


@dataclass(frozen=True)
class _IncomingObstacle:
    sync_id: tuple
    node_key: tuple | None
    stable_key: tuple | None
    path_key: tuple
    default_name: str
    link_key: str
    world_frame: np.ndarray
    layer: int
    mask: int
    geom_sig: tuple
    spec: dict


def _extract_robot_model(
    snapshot: Mapping,
    robots_spec: Mapping[str, tuple[str, ...]],
    default_acceleration_limit: float,
) -> dict:
    """Parse selected robots' kinematic trees, joint specs, base poses, and link-local shape signatures."""
    if snapshot.get("schema_version") != 1:
        raise ValueError("unsupported physics snapshot schema")
    if not robots_spec:
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
    branch_poses: list[tuple[np.ndarray, np.ndarray]] = []
    robot_branches: dict[str, list[int]] = defaultdict(list)
    body_names: list[str] = []
    joint_names: list[JointName] = []
    lower_list: list[float] = []
    upper_list: list[float] = []
    vel_list: list[float] = []
    acc_list: list[float] = []
    jerk_list: list[float] = []
    defaulted_acc_joints: list[str] = []
    init_list: list[float] = []
    link_map: dict[Link, int] = {}
    link_index_str: dict[str, int] = {}
    adjacent: set[tuple[Link, Link]] = set()
    link_local_shapes: dict[Link, list[tuple[np.ndarray, tuple, int, int]]] = {}
    robot_shapes_to_register: list[tuple[dict, Link, int, tuple[np.ndarray, np.ndarray]]] = []

    for robot_name, roots in robots_spec.items():
        if len(by_name[robot_name]) != 1:
            raise ValueError(f"expected one robot {robot_name!r} in snapshot")
        robot = by_name[robot_name][0]
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
            branch_poses.append((root_p.copy(), root_r.copy()))
            robot_branches[robot_name].append(branch_id)

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
                jerk = float(j.get("jerk_limit", 0.0))
                jerk_list.append(jerk if jerk > 0 else math.inf)
                init_list.append(float(j.get("initial_position", 0.0)))

            local_body_id: dict[str, int] = {}
            for link_name in order:
                bid = len(parent)
                local_body_id[link_name] = bid
                key = (robot_name, link_name)
                link_map[key] = bid
                link_index_str[_link_key(key)] = bid
                body_names.append(_link_key(key))

                link_dict = links_by_name[link_name]
                rest = _rigid_transform(link_dict["global_transform"])
                origin, frame = rest
                lshapes: list[tuple[np.ndarray, tuple, int, int]] = []
                if link_dict.get("role") != "virtual_root":
                    for sh in link_dict.get("shapes", ()):
                        if sh.get("disabled", False):
                            continue
                        sp, sr = _rigid_transform(sh["transform"])
                        lp = frame.T @ (sp - origin)
                        lr = frame.T @ sr
                        lf = _frame_from_rt(lp, lr)
                        raw_name = sh.get("path") or sh.get("name") or f"shape_{sh['type']}"
                        gsig = _shape_geom_signature(sh, f"{robot_name}/{link_name}/{raw_name}")
                        layer = int(sh.get("collision_layer", 1))
                        mask = int(sh.get("collision_mask", 1))
                        lshapes.append((lf, gsig, layer, mask))
                        robot_shapes_to_register.append((sh, key, bid, rest))
                link_local_shapes[key] = lshapes

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
                c_tf = np.asarray(link_dict["global_transform"]["matrix"], dtype=np.float64)
                parent_joint.append(np.linalg.inv(p_tf) @ j_tf)
                joint_child.append(np.linalg.inv(j_tf) @ c_tf)
                axis = np.asarray(j["axis"], dtype=np.float64)
                norm = float(np.linalg.norm(axis))
                if axis.shape != (3,) or not np.isfinite(axis).all() or norm < 1e-10:
                    raise ValueError(f"invalid joint axis on {j['name']!r}")
                axes.append(axis / norm)

    if not joint_names:
        raise ValueError("selected robots have no movable joints")

    lower = np.asarray(lower_list, dtype=np.float64)
    upper = np.asarray(upper_list, dtype=np.float64)
    vel = np.asarray(vel_list, dtype=np.float64)
    acc = np.asarray(acc_list, dtype=np.float64)
    jerk = np.asarray(jerk_list, dtype=np.float64)
    init_q = np.clip(np.asarray(init_list, dtype=np.float64), lower, upper)

    topo_sig = {
        "body_names": tuple(body_names),
        "parent": tuple(parent),
        "kinds": tuple(kinds),
        "joint_idx": tuple(joint_idx),
        "joint_names": tuple(joint_names),
        "lower": lower.copy(),
        "upper": upper.copy(),
        "vel": vel.copy(),
        "acc": acc.copy(),
        "jerk": jerk.copy(),
        "parent_joint": np.ascontiguousarray(parent_joint, dtype=np.float64),
        "joint_child": np.ascontiguousarray(joint_child, dtype=np.float64),
        "axes": np.ascontiguousarray(axes, dtype=np.float64),
        "link_local_shapes": link_local_shapes,
    }
    return {
        "parent": parent,
        "kinds": kinds,
        "joint_idx": joint_idx,
        "parent_joint": np.ascontiguousarray(parent_joint, dtype=np.float64),
        "joint_child": np.ascontiguousarray(joint_child, dtype=np.float64),
        "axes": np.ascontiguousarray(axes, dtype=np.float64),
        "branches_in": branches_in,
        "branch_poses": branch_poses,
        "robot_branches": dict(robot_branches),
        "body_names": body_names,
        "joint_names": tuple(joint_names),
        "lower": lower,
        "upper": upper,
        "vel": vel,
        "acc": acc,
        "jerk": jerk,
        "defaulted_acc_joints": defaulted_acc_joints,
        "init_q": init_q,
        "link_map": link_map,
        "link_index_str": link_index_str,
        "adjacent": adjacent,
        "robot_shapes": robot_shapes_to_register,
        "topo_sig": topo_sig,
    }


def _check_robot_topology_unchanged(expected: dict, actual: dict) -> None:
    """Verify robot topology, kinematics, limits, and link-local collision shapes have not changed."""
    for field in ("body_names", "parent", "kinds", "joint_idx", "joint_names"):
        if expected[field] != actual[field]:
            raise ValueError(
                f"robot topology changed ({field}); rebuild PlanningWorld via PlanningWorld.from_snapshot()"
            )
    for field in ("lower", "upper", "vel", "acc", "jerk", "parent_joint", "joint_child", "axes"):
        if not np.allclose(expected[field], actual[field], atol=1e-5, rtol=1e-5, equal_nan=True):
            raise ValueError(
                f"robot kinematics or limits changed ({field}); rebuild PlanningWorld via PlanningWorld.from_snapshot()"
            )
    exp_shapes = expected["link_local_shapes"]
    act_shapes = actual["link_local_shapes"]
    if exp_shapes.keys() != act_shapes.keys():
        raise ValueError("robot link set changed; rebuild PlanningWorld via PlanningWorld.from_snapshot()")
    for link_key, exp_list in exp_shapes.items():
        act_list = act_shapes[link_key]
        if len(exp_list) != len(act_list):
            raise ValueError(
                f"robot link {link_key!r} collision shapes changed; rebuild PlanningWorld via PlanningWorld.from_snapshot()"
            )
        for (ef, eg, el, em), (af, ag, al, am) in zip(exp_list, act_list):
            if eg != ag or el != al or em != am or not np.allclose(ef, af, atol=1e-5, rtol=1e-5):
                raise ValueError(
                    f"robot link {link_key!r} collision shape changed; rebuild PlanningWorld via PlanningWorld.from_snapshot()"
                )


def _extract_world_obstacles(snapshot: Mapping, link_map: Mapping[Link, int]) -> list[_IncomingObstacle]:
    """Extract all active static world obstacles (non-selected links, loose shapes, terrains) with stable IDs."""
    obstacles: list[_IncomingObstacle] = []
    node_counts: dict[tuple, int] = defaultdict(int)
    stable_counts: dict[tuple, int] = defaultdict(int)
    path_counts: dict[tuple, int] = defaultdict(int)

    def unique_key(counts: dict[tuple, int], base: tuple) -> tuple:
        occ = counts[base]
        counts[base] = occ + 1
        return base if occ == 0 else (*base, occ)

    def add_entry(
        spec: dict,
        link: Link,
        *,
        terrain_node_key: tuple | None = None,
        terrain_stable_key: tuple | None = None,
        terrain_path_key: tuple | None = None,
    ) -> None:
        p, r = _rigid_transform(spec["transform"])
        wf = _frame_from_rt(p, r)
        raw_name = spec.get("path") or spec.get("name") or f"shape_{spec['type']}"
        default_name = raw_name if not link[0] else f"{link[0]}/{link[1]}/{raw_name}"
        if terrain_path_key is not None:
            node_key = unique_key(node_counts, terrain_node_key) if terrain_node_key is not None else None
            stable_key = unique_key(stable_counts, terrain_stable_key) if terrain_stable_key is not None else None
            path_key = unique_key(path_counts, terrain_path_key)
        else:
            nid = spec.get("node_id")
            sid = spec.get("stable_id")
            node_key = unique_key(node_counts, ("node", str(nid))) if (nid is not None and nid != "" and nid != 0) else None
            stable_key = unique_key(stable_counts, ("shape", str(sid))) if (sid is not None and sid != "" and sid != 0) else None
            path_key = unique_key(path_counts, ("path", default_name))
        sync_id = node_key if node_key is not None else (stable_key if stable_key is not None else path_key)
        gsig = _shape_geom_signature(spec, default_name)
        obstacles.append(_IncomingObstacle(
            sync_id=sync_id,
            node_key=node_key,
            stable_key=stable_key,
            path_key=path_key,
            default_name=default_name,
            link_key=_link_key(link),
            world_frame=wf,
            layer=int(spec.get("collision_layer", 1)),
            mask=int(spec.get("collision_mask", 1)),
            geom_sig=gsig,
            spec=spec,
        ))

    for robot in snapshot["robots"]:
        for link in robot["links"]:
            if link.get("role") == "virtual_root":
                continue
            key = (robot["name"], link["name"])
            if key in link_map:
                continue
            for sh in link.get("shapes", ()):
                if not sh.get("disabled", False):
                    add_entry(sh, key)

    for sh in snapshot.get("loose_shapes", ()):
        if not sh.get("disabled", False):
            key = ("", sh.get("path", sh["name"]))
            add_entry(sh, key)

    for terrain in snapshot.get("terrains", ()):
        if terrain.get("heightfields"):
            raise ValueError("planning collision does not support terrain heightfields")
        key = ("", terrain["name"])
        t_nid = terrain.get("node_id")
        t_sid = terrain.get("stable_id")
        t_path = str(terrain.get("path") or terrain["name"])
        for idx, geom in enumerate(terrain.get("boxes", ())):
            val = dict(terrain, **geom, type=1)
            add_entry(
                val, key,
                terrain_node_key=("terrain_box_node", str(t_nid), idx) if (t_nid is not None and t_nid != "" and t_nid != 0) else None,
                terrain_stable_key=("terrain_box", str(t_sid), idx) if (t_sid is not None and t_sid != "" and t_sid != 0) else None,
                terrain_path_key=("terrain_box_path", t_path, idx),
            )
        for idx, geom in enumerate(terrain.get("meshes", ())):
            val = dict(terrain, **geom, type=5)
            add_entry(
                val, key,
                terrain_node_key=("terrain_mesh_node", str(t_nid), idx) if (t_nid is not None and t_nid != "" and t_nid != 0) else None,
                terrain_stable_key=("terrain_mesh", str(t_sid), idx) if (t_sid is not None and t_sid != "" and t_sid != 0) else None,
                terrain_path_key=("terrain_mesh_path", t_path, idx),
            )

    return obstacles


def _collect_scene_node_ids(root) -> dict[str, int]:
    """Build a mapping from canonical scene node path to persistent runtime ObjectID."""
    if root is None:
        return {}
    path_to_id: dict[str, int] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        p = getattr(node, "path", None)
        nid = getattr(node, "id", None)
        if p and nid:
            path_to_id[str(p)] = int(nid)
        children = getattr(node, "children", None)
        if children:
            stack.extend(children)
    return path_to_id


def _annotate_snapshot_node_ids(snapshot: dict, path_to_id: Mapping[str, int]) -> dict:
    """Attach persistent runtime node_id to shape/terrain dicts without mutating engine snapshot contracts."""
    if not path_to_id:
        return snapshot
    out = dict(snapshot)
    robots = []
    for rob in snapshot.get("robots", ()):
        links = []
        for lk in rob.get("links", ()):
            shapes = []
            for sh in lk.get("shapes", ()):
                nid = path_to_id.get(sh.get("path", ""))
                shapes.append(dict(sh, node_id=nid) if nid else sh)
            links.append(dict(lk, shapes=shapes))
        robots.append(dict(rob, links=links))
    out["robots"] = robots

    loose = []
    for sh in snapshot.get("loose_shapes", ()):
        nid = path_to_id.get(sh.get("path", ""))
        loose.append(dict(sh, node_id=nid) if nid else sh)
    out["loose_shapes"] = loose

    terrains = []
    for terr in snapshot.get("terrains", ()):
        nid = path_to_id.get(terr.get("path", ""))
        terrains.append(dict(terr, node_id=nid) if nid else terr)
    out["terrains"] = terrains
    return out


def _export_scene_snapshot(scene_or_context) -> dict:
    """Compile a physics scene snapshot through the stable engine context interface."""
    if hasattr(scene_or_context, "compile_physics_scene_snapshot"):
        ctx = scene_or_context
    else:
        ctx = getattr(scene_or_context, "context", None)
        if ctx is None or not hasattr(ctx, "compile_physics_scene_snapshot"):
            raise TypeError(
                "expected an EngineContext or Scene/App object exposing compile_physics_scene_snapshot()"
            )
    snapshot = ctx.compile_physics_scene_snapshot()
    root = getattr(ctx, "root", None)
    if root is not None:
        return _annotate_snapshot_node_ids(snapshot, _collect_scene_node_ids(root))
    return snapshot


class PlanningWorld:
    """Mutable C++-backed planning scene with kinematics, dynamic obstacles, attach/detach, and ACM."""

    def __init__(
        self,
        native_world,
        *,
        joint_names: tuple[JointName, ...],
        link_map: dict[Link, int],
        robot_branches: dict[str, list[int]],
        lower: np.ndarray,
        upper: np.ndarray,
        velocity_limits: np.ndarray,
        acceleration_limits: np.ndarray,
        initial_positions: np.ndarray,
        jerk_limits: np.ndarray | None = None,
        robots_spec: dict[str, tuple[str, ...]] | None = None,
        robot_topology_sig: dict | None = None,
        branch_poses: list[tuple[np.ndarray, np.ndarray]] | None = None,
        synced_obstacles: dict[tuple, _TrackedObstacle] | None = None,
        default_acceleration_limit: float = 10.0,
    ) -> None:
        self._native = native_world
        self.joint_names = joint_names
        self._joint_index = {name: i for i, name in enumerate(joint_names)}
        self.links = link_map
        self._robot_branches = robot_branches
        self.lower_limits = lower
        self.upper_limits = upper
        self.velocity_limits = velocity_limits
        self.acceleration_limits = acceleration_limits
        self.jerk_limits = (
            np.full(len(joint_names), math.inf, dtype=np.float64)
            if jerk_limits is None else np.asarray(jerk_limits, dtype=np.float64).copy()
        )
        self.initial_positions = initial_positions
        self._groups: dict[str, PlanningGroup] = {}
        self._robots_spec: dict[str, tuple[str, ...]] = dict(robots_spec or {})
        self._robot_topology_sig: dict | None = robot_topology_sig
        self._branch_poses: list[tuple[np.ndarray, np.ndarray]] = list(branch_poses or [])
        self._synced_obstacles: dict[tuple, _TrackedObstacle] = dict(synced_obstacles or {})
        self._default_acceleration_limit = float(default_acceleration_limit)
        self._write_lock = threading.RLock()

    @classmethod
    def from_scene(
        cls,
        scene_or_context,
        robots: Mapping[str, str | Sequence[str]],
        *,
        ignored_link_pairs: Sequence[tuple[Link, Link]] = (),
        default_acceleration_limit: float = 10.0,
    ) -> PlanningWorld:
        """Build a PlanningWorld from a live Gobot EngineContext/Scene snapshot."""
        snapshot = _export_scene_snapshot(scene_or_context)
        return cls.from_snapshot(
            snapshot,
            robots,
            ignored_link_pairs=ignored_link_pairs,
            default_acceleration_limit=default_acceleration_limit,
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict,
        robots: Mapping[str, str | Sequence[str]],
        *,
        ignored_link_pairs: Sequence[tuple[Link, Link]] = (),
        default_acceleration_limit: float = 10.0,
    ) -> PlanningWorld:
        robots_spec = {
            rname: ((bases,) if isinstance(bases, str) else tuple(bases))
            for rname, bases in robots.items()
        }
        model = _extract_robot_model(snapshot, robots_spec, default_acceleration_limit)
        if model["defaulted_acc_joints"]:
            warnings.warn(
                f"Joints missing acceleration_limit ({len(model['defaulted_acc_joints'])} joints, e.g. "
                f"{model['defaulted_acc_joints'][0]}): defaulting to {float(default_acceleration_limit):.1f} rad/s^2",
                UserWarning,
                stacklevel=2,
            )

        obstacles = _extract_world_obstacles(snapshot, model["link_map"])

        native_world = _native.PlanningWorld()
        native_world.init_kinematics(
            model["parent"], model["kinds"], model["joint_idx"],
            model["parent_joint"], model["joint_child"], model["axes"],
            model["branches_in"], model["body_names"],
            [f"{r}::{j}" for r, j in model["joint_names"]],
            model["lower"], model["upper"], model["vel"], model["acc"],
            model["init_q"], model["link_index_str"],
        )
        if np.isfinite(model["jerk"]).any():
            native_world.set_jerk_limits(model["jerk"])

        # Populate initial auto ACM separately from user overrides.
        for a, b in model["adjacent"]:
            native_world.set_auto_collision_disabled(_link_key(a), _link_key(b))
        for a, b in ignored_link_pairs:
            native_world.set_auto_collision_disabled(_link_key(a), _link_key(b))

        used_names: dict[str, int] = defaultdict(int)

        def unique_name(base_name: str) -> str:
            idx = used_names[base_name]
            used_names[base_name] = idx + 1
            return base_name if idx == 0 else f"{base_name}#{idx}"

        for spec, link, body_id, rest in model["robot_shapes"]:
            p, r = _rigid_transform(spec["transform"])
            origin, frame = rest
            p = frame.T @ (p - origin)
            r = frame.T @ r
            raw_name = spec.get("path") or spec.get("name") or f"shape_{spec['type']}"
            full_name = unique_name(f"{link[0]}/{link[1]}/{raw_name}")
            sh = _make_native_shape(spec, full_name)
            native_world.add_shape(
                full_name, _link_key(link), body_id, _frame_from_rt(p, r),
                int(spec.get("collision_layer", 1)), int(spec.get("collision_mask", 1)), sh,
            )

        synced_obstacles: dict[tuple, _TrackedObstacle] = {}
        for obs in obstacles:
            full_name = unique_name(obs.default_name)
            sh = _make_native_shape(obs.spec, full_name)
            native_world.add_shape(
                full_name, obs.link_key, -1, obs.world_frame, obs.layer, obs.mask, sh,
            )
            synced_obstacles[obs.sync_id] = _TrackedObstacle(
                sync_id=obs.sync_id,
                node_key=obs.node_key,
                stable_key=obs.stable_key,
                path_key=obs.path_key,
                shape_name=full_name,
                link_key=obs.link_key,
                world_frame=obs.world_frame.copy(),
                layer=obs.layer,
                mask=obs.mask,
                geom_sig=obs.geom_sig,
            )

        return cls(
            native_world,
            joint_names=model["joint_names"],
            link_map=model["link_map"],
            robot_branches=model["robot_branches"],
            lower=model["lower"],
            upper=model["upper"],
            velocity_limits=model["vel"],
            acceleration_limits=model["acc"],
            initial_positions=model["init_q"],
            jerk_limits=model["jerk"],
            robots_spec=robots_spec,
            robot_topology_sig=model["topo_sig"],
            branch_poses=model["branch_poses"],
            synced_obstacles=synced_obstacles,
            default_acceleration_limit=default_acceleration_limit,
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
    def has_esdf(self) -> bool:
        return bool(self._native.has_esdf)

    @property
    def esdf_info(self) -> dict | None:
        info = self._native.esdf_info()
        if info is None:
            return None
        return {
            "origin": np.asarray(info["origin"], dtype=np.float64),
            "resolution": float(info["resolution"]),
            "shape": tuple(int(v) for v in info["shape"]),
            "bounds_min": np.asarray(info["bounds_min"], dtype=np.float64),
            "bounds_max": np.asarray(info["bounds_max"], dtype=np.float64),
            "lipschitz_bound": float(info["lipschitz_bound"]),
            "min_value": float(info["min_value"]),
            "is_empty": bool(info["is_empty"]),
            "collision_layer": int(info["collision_layer"]),
            "collision_mask": int(info["collision_mask"]),
        }

    @property
    def collision_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._native.pair_shape_names())

    # --- Mutable World Operations ---

    def add_box(self, name: str, size: Sequence[float], position: Sequence[float],
                rotation: Sequence[Sequence[float]] | None = None, *,
                collision_layer: int = 1, collision_mask: int = 1) -> None:
        spec = {"type": 1, "size": np.asarray(size, dtype=np.float64)}
        sh = _make_native_shape(spec, name)
        frame = _frame_from_rt(position, rotation)
        with self._write_lock:
            self._native.add_shape(name, f"::{name}", -1, frame, collision_layer, collision_mask, sh)

    def add_sphere(self, name: str, radius: float, position: Sequence[float], *,
                   collision_layer: int = 1, collision_mask: int = 1) -> None:
        spec = {"type": 2, "radius": float(radius)}
        sh = _make_native_shape(spec, name)
        frame = _frame_from_rt(position, None)
        with self._write_lock:
            self._native.add_shape(name, f"::{name}", -1, frame, collision_layer, collision_mask, sh)

    def add_capsule(self, name: str, radius: float, height: float, position: Sequence[float],
                    rotation: Sequence[Sequence[float]] | None = None, *,
                    collision_layer: int = 1, collision_mask: int = 1) -> None:
        spec = {"type": 4, "radius": float(radius), "height": float(height)}
        sh = _make_native_shape(spec, name)
        frame = _frame_from_rt(position, rotation)
        with self._write_lock:
            self._native.add_shape(name, f"::{name}", -1, frame, collision_layer, collision_mask, sh)

    def update_obstacle_pose(self, name: str, position: Sequence[float],
                             rotation: Sequence[Sequence[float]] | None = None) -> None:
        """Update pose for a single static world obstacle without rebuilding BVHs or pairs."""
        frame = _frame_from_rt(position, rotation)
        with self._write_lock:
            self._native.update_obstacle_pose(name, frame)
            for tracked in self._synced_obstacles.values():
                if tracked.shape_name == name:
                    tracked.world_frame = frame.copy()
                    break

    def remove_obstacle(self, name: str) -> None:
        with self._write_lock:
            self._native.remove_obstacle(name)
            for sid, tracked in list(self._synced_obstacles.items()):
                if tracked.shape_name == name:
                    del self._synced_obstacles[sid]
                    break

    def attach_object(self, name: str, link: Link,
                      local_position: Sequence[float] = (0.0, 0.0, 0.0),
                      local_rotation: Sequence[Sequence[float]] | None = None,
                      *, touch_links: Sequence[Link] = ()) -> None:
        """Attach a world obstacle to a robot link (e.g., grasping a payload)."""
        frame = _frame_from_rt(local_position, local_rotation)
        tl_keys = [_link_key(tl) for tl in touch_links]
        with self._write_lock:
            self._native.attach_object(name, _link_key(link), frame, tl_keys)

    def detach_object(self, name: str, q: Sequence[float] | np.ndarray) -> None:
        """Detach an attached object at configuration q back into the static world."""
        q_arr = self.configuration(q)
        with self._write_lock:
            self._native.detach_object(name, q_arr)
            for tracked in self._synced_obstacles.values():
                if tracked.shape_name == name:
                    # Pose now comes from FK(q); force reconciliation on next scene sync.
                    tracked.world_frame = None
                    break

    def set_collision_enabled(self, link_a: Link | str, link_b: Link | str, enabled: bool) -> None:
        ka = _link_key(link_a) if isinstance(link_a, tuple) else f"::{link_a}"
        kb = _link_key(link_b) if isinstance(link_b, tuple) else f"::{link_b}"
        with self._write_lock:
            self._native.set_collision_enabled(ka, kb, bool(enabled))

    def set_robot_base_pose(self, robot: str, position: Sequence[float],
                            rotation: Sequence[Sequence[float]] | None = None,
                            *, branch_index: int = 0) -> None:
        branches = self._robot_branches.get(robot)
        if not branches or not (0 <= branch_index < len(branches)):
            raise ValueError(f"unknown robot/branch {robot!r}[{branch_index}]")
        frame = _frame_from_rt(position, rotation)
        bid = branches[branch_index]
        with self._write_lock:
            self._native.set_robot_base_pose(bid, frame[:, 3], frame[:, :3])
            if 0 <= bid < len(self._branch_poses):
                self._branch_poses[bid] = (frame[:, 3].copy(), frame[:, :3].copy())

    def set_jerk_limits(self, jerk_limits: float | Sequence[float] | np.ndarray) -> None:
        """Set per-joint (or scalar) jerk limits [rad/s^3 or m/s^3] used by C^2 quintic retiming & validation."""
        arr = np.asarray(jerk_limits, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(self.num_joints, float(arr), dtype=np.float64)
        if arr.shape != (self.num_joints,) or np.isnan(arr).any() or (arr <= 0.0).any():
            raise ValueError(f"jerk_limits must be a positive scalar or length-{self.num_joints} vector (inf = unlimited)")
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        with self._write_lock:
            self._native.set_jerk_limits(arr)
            self.jerk_limits = arr.copy()

    # --- Voxel / 3D ESDF & Link Sphere-Cover Obstacles ---

    def set_occupancy_grid(
        self,
        occupied: np.ndarray,
        origin: Sequence[float] | np.ndarray,
        resolution: float,
        *,
        max_distance: float = math.inf,
        inflation_radius: float = 0.0,
        collision_layer: int = 1,
        collision_mask: int = 1,
    ) -> None:
        """Build a voxel-center signed distance field from a binary occupancy grid ``[nx, ny, nz]``.

        Voxel ``(ix, iy, iz)`` spans ``[origin + (ix, iy, iz) * resolution, origin + (ix+1, iy+1, iz+1) * resolution]``
        with voxel center at ``origin + (ix + 0.5, iy + 0.5, iz + 0.5) * resolution``.
        Atomically publishes a new immutable snapshot and increments ``world.version``.
        """
        arr = np.asarray(occupied)
        if arr.ndim != 3:
            raise ValueError("occupied must be a 3D array with shape [nx, ny, nz]")
        occ_u8 = np.ascontiguousarray(arr != 0, dtype=np.uint8)
        org = np.asarray(origin, dtype=np.float64)
        if org.shape != (3,) or not np.isfinite(org).all():
            raise ValueError("origin must be a finite 3-vector")
        with self._write_lock:
            self._native.set_occupancy_grid(
                occ_u8, org, float(resolution), float(max_distance), float(inflation_radius),
                int(collision_layer), int(collision_mask),
            )

    def set_point_cloud(
        self,
        points: Sequence[Sequence[float]] | np.ndarray,
        resolution: float,
        *,
        origin: Sequence[float] | np.ndarray | None = None,
        shape: Sequence[int] | None = None,
        padding: float = 0.2,
        point_radius: float = 0.0,
        max_distance: float = math.inf,
        collision_layer: int = 1,
        collision_mask: int = 1,
    ) -> None:
        """Voxelize a 3D point cloud ``[N, 3]`` and build a voxel-center signed distance field in C++.

        If ``origin`` / ``shape`` are omitted, a tight axis-aligned grid padded by
        ``max(padding, 2 * resolution)`` is fitted around the point cloud automatically.
        Passing fixed ``origin`` and ``shape`` is recommended for high-rate live sensor updates.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape == (0,):
            pts = np.empty((0, 3), dtype=np.float64)
        elif pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points must have shape [N, 3]")
        pts = np.ascontiguousarray(pts, dtype=np.float64)
        org_arg = None
        if origin is not None:
            org_arg = np.asarray(origin, dtype=np.float64)
            if org_arg.shape != (3,) or not np.isfinite(org_arg).all():
                raise ValueError("origin must be a finite 3-vector")
        shape_arg = None
        if shape is not None:
            if len(shape) != 3:
                raise ValueError("shape must be a 3-tuple (nx, ny, nz)")
            shape_arg = (int(shape[0]), int(shape[1]), int(shape[2]))
        with self._write_lock:
            self._native.set_point_cloud(
                pts, float(resolution), org_arg, shape_arg, float(padding),
                float(point_radius), float(max_distance), int(collision_layer), int(collision_mask),
            )

    def set_esdf(
        self,
        values: np.ndarray,
        origin: Sequence[float] | np.ndarray,
        resolution: float,
        *,
        collision_layer: int = 1,
        collision_mask: int = 1,
    ) -> None:
        """Install a precomputed 3D signed distance grid ``[nx, ny, nz]`` (meters; positive = free).

        Grid cell ``(ix, iy, iz)`` holds the signed distance at voxel center
        ``origin + (ix + 0.5, iy + 0.5, iz + 0.5) * resolution``.
        """
        arr = np.ascontiguousarray(values, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError("values must be a 3D array with shape [nx, ny, nz]")
        org = np.asarray(origin, dtype=np.float64)
        if org.shape != (3,) or not np.isfinite(org).all():
            raise ValueError("origin must be a finite 3-vector")
        with self._write_lock:
            self._native.set_esdf(arr, org, float(resolution), int(collision_layer), int(collision_mask))

    def clear_esdf(self) -> None:
        """Remove any active voxel / 3D ESDF field from this world."""
        with self._write_lock:
            self._native.clear_esdf()

    def query_esdf(
        self,
        points: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
        *,
        return_gradient: bool = False,
    ) -> np.ndarray | float | tuple[np.ndarray | float, np.ndarray]:
        """Query the interpolated center-sampled field and its optional world-space gradient.

        Use ``esdf_distances(q)`` for the sphere clearances used by planning and validation.
        """
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim == 0 or not np.isfinite(arr).all():
            raise ValueError("expected finite 3D points")
        single = arr.ndim == 1
        if single:
            if arr.shape != (3,):
                raise ValueError("point must have shape (3,)")
            flat = np.ascontiguousarray(arr[None, :])
            batch_shape: tuple[int, ...] = ()
        else:
            if arr.shape[-1] != 3:
                raise ValueError("points must have trailing dimension 3")
            batch_shape = arr.shape[:-1]
            flat = np.ascontiguousarray(arr.reshape(-1, 3))
        if return_gradient:
            dist, grad = self._native.query_esdf(flat, True)
            if single:
                return float(dist[0]), grad[0]
            return dist.reshape(batch_shape), grad.reshape(*batch_shape, 3)
        dist = self._native.query_esdf(flat, False)
        if single:
            return float(dist[0])
        return dist.reshape(batch_shape)

    def esdf_distances(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return the planner's conservative clearance for each active sphere, in meters.

        Occupancy fields allow for up to ``(sqrt(1.5) - 0.5) * resolution`` of voxel
        interpolation error. Supplied fields use their Lipschitz bound to measure
        clearance to the interpolated zero surface. Both account for the sphere radius.
        """
        arr = np.asarray(q, dtype=np.float64)
        if arr.ndim == 0 or arr.shape[-1] != self.num_joints or not np.isfinite(arr).all():
            raise ValueError("expected finite joint configurations with trailing dimension J")
        batch_shape = arr.shape[:-1]
        flat = np.ascontiguousarray(arr.reshape(math.prod(batch_shape), self.num_joints))
        d = self._native.query_esdf_sphere_distances(flat)
        return d.reshape(*batch_shape, d.shape[-1])

    def link_spheres(self, link: Link | str | None = None, *, esdf_only: bool = False) -> tuple[LinkSphere, ...]:
        """Return the active conservative link sphere-cover (optionally filtered to one link or ESDF-active spheres)."""
        raw = self._native.esdf_spheres() if esdf_only else self._native.link_spheres()
        filter_key = None
        if link is not None:
            filter_key = _link_key(link) if isinstance(link, tuple) else f"::{link}"
        out: list[LinkSphere] = []
        for item in raw:
            lk = str(item["link_key"])
            if filter_key is not None and lk != filter_key:
                continue
            if lk.startswith("::"):
                parsed_link: Link | str = lk[2:]
            elif "::" in lk:
                rname, lname = lk.split("::", 1)
                parsed_link = (rname, lname)
            else:
                parsed_link = lk
            out.append(LinkSphere(
                link=parsed_link,
                shape_name=str(item["shape_name"]),
                center=np.asarray(item["center"], dtype=np.float64).copy(),
                radius=float(item["radius"]),
                collision_layer=int(item["layer"]),
                collision_mask=int(item["mask"]),
            ))
        return tuple(out)

    def set_link_spheres(
        self,
        link: Link | str,
        spheres: Sequence[tuple[Sequence[float] | np.ndarray, float]],
        *,
        collision_layer: int = 1,
        collision_mask: int = 1,
    ) -> None:
        """Override the sphere cover with centers in the named link or attached object frame."""
        lk = _link_key(link) if isinstance(link, tuple) else f"::{link}"
        compiled: list[tuple[np.ndarray, float]] = []
        for center, radius in spheres:
            c = np.asarray(center, dtype=np.float64)
            r = float(radius)
            if c.shape != (3,) or not np.isfinite(c).all() or not math.isfinite(r) or r < 0.0:
                raise ValueError("each sphere must be (finite 3-vector center, nonnegative finite radius)")
            compiled.append((c, r))
        with self._write_lock:
            self._native.set_link_spheres(lk, compiled, int(collision_layer), int(collision_mask))

    def clear_link_spheres(self, link: Link | str | None = None) -> None:
        """Reset custom link sphere override(s) back to automatic conservative sphere-cover generation."""
        lk = "" if link is None else (_link_key(link) if isinstance(link, tuple) else f"::{link}")
        with self._write_lock:
            self._native.clear_link_spheres(lk)

    # --- Incremental Scene Synchronization ---

    def sync_from_snapshot(self, snapshot: Mapping) -> SyncResult:
        """Atomically synchronize obstacle poses, add/remove/replace, and robot base poses from a snapshot.

        Guarantees:
        - Serializes concurrent writers via ``_write_lock`` while readers/planners remain lock-free.
        - Publishes at most one new immutable WorldSnapshot per call (zero version bump on no-op).
        - Reuses convex hulls and BVHs whenever geometry signatures are unchanged.
        - Preserves PlanningGroups, user ACM overrides, manually added obstacles, and attached objects.
        - Rejects robot topology changes (requiring PlanningWorld.from_snapshot rebuild) atomically
          without leaving a half-updated world.

        Note on rename/reparent persistence: ``compile_physics_scene_snapshot()`` hashes ``scene_path``
        into ``stable_id``, so renaming or reparenting live Gobot scene nodes should be synchronized
        through ``sync_from_scene(ctx)`` (which attaches persistent runtime ``node_id``s) or snapshots
        carrying persistent ``node_id`` / ``stable_id`` values. ``sync_from_snapshot()`` intentionally
        does not guess object identity by pose/geometry, as that could drop newly spawned identical
        workpieces in pick-and-place cells.
        """
        if self._robot_topology_sig is None or not self._robots_spec:
            raise RuntimeError("PlanningWorld was not initialized from a scene snapshot; cannot sync incrementally")

        with self._write_lock:
            t0 = perf_counter()
            model = _extract_robot_model(snapshot, self._robots_spec, self._default_acceleration_limit)
            _check_robot_topology_unchanged(self._robot_topology_sig, model["topo_sig"])
            incoming = _extract_world_obstacles(snapshot, self.links)

            # Diff robot base poses.
            base_updates: list[tuple[int, np.ndarray, np.ndarray]] = []
            next_branch_poses: list[tuple[np.ndarray, np.ndarray]] = []
            for bid, (new_p, new_r) in enumerate(model["branch_poses"]):
                old_p, old_r = self._branch_poses[bid]
                if not (np.allclose(new_p, old_p, atol=1e-12, rtol=0.0)
                        and np.allclose(new_r, old_r, atol=1e-12, rtol=0.0)):
                    base_updates.append((bid, new_p, new_r))
                next_branch_poses.append((new_p.copy(), new_r.copy()))

            attached_names = set(self._native.attached_shape_names())
            occupied_names = set(self._native.shape_names())

            # Match existing tracked obstacles against incoming obstacles across node_key -> stable_key -> path_key
            # so mixing sync_from_scene (with node_id) and sync_from_snapshot (without node_id) never duplicates shapes.
            existing_list = list(self._synced_obstacles.values())
            matched_inc_for_old: dict[int, int] = {}
            claimed_inc: set[int] = set()

            node_to_inc = {obs.node_key: idx for idx, obs in enumerate(incoming) if obs.node_key is not None}
            stable_to_inc = {obs.stable_key: idx for idx, obs in enumerate(incoming) if obs.stable_key is not None}
            path_to_inc = {obs.path_key: idx for idx, obs in enumerate(incoming)}

            # Pass 1: exact runtime node_id match (survives renames and reparents in live Scenes).
            for oi, tr in enumerate(existing_list):
                if tr.node_key is not None and tr.node_key in node_to_inc:
                    ii = node_to_inc[tr.node_key]
                    if ii not in claimed_inc:
                        matched_inc_for_old[oi] = ii
                        claimed_inc.add(ii)

            # Pass 2: exact stable_id match (bridges from_scene <-> sync_from_snapshot and synthetic stable_id renames).
            for oi, tr in enumerate(existing_list):
                if oi in matched_inc_for_old:
                    continue
                if tr.stable_key is not None and tr.stable_key in stable_to_inc:
                    ii = stable_to_inc[tr.stable_key]
                    if ii not in claimed_inc:
                        # Do not bridge two distinct live runtime nodes that happen to reuse a path-hashed stable_id.
                        inc_nk = incoming[ii].node_key
                        if tr.node_key is not None and inc_nk is not None and tr.node_key != inc_nk:
                            continue
                        matched_inc_for_old[oi] = ii
                        claimed_inc.add(ii)

            # Pass 3: path_key fallback (when IDs are absent on one or both sides).
            for oi, tr in enumerate(existing_list):
                if oi in matched_inc_for_old:
                    continue
                if tr.path_key in path_to_inc:
                    ii = path_to_inc[tr.path_key]
                    if ii not in claimed_inc:
                        inc_obs = incoming[ii]
                        if tr.node_key is not None and inc_obs.node_key is not None and tr.node_key != inc_obs.node_key:
                            continue
                        if tr.stable_key is not None and inc_obs.stable_key is not None and tr.stable_key != inc_obs.stable_key:
                            continue
                        matched_inc_for_old[oi] = ii
                        claimed_inc.add(ii)

            remove_names: list[str] = []
            replace_shapes: list[tuple[str, str, np.ndarray, int, int, object]] = []
            add_shapes: list[tuple[str, str, np.ndarray, int, int, object]] = []
            pose_updates: list[tuple[str, np.ndarray]] = []
            next_tracked: dict[tuple, _TrackedObstacle] = {}

            # 1. Process removals and updates on existing scene-managed obstacles.
            for oi, tracked in enumerate(existing_list):
                ii = matched_inc_for_old.get(oi)
                inc = None if ii is None else incoming[ii]
                if tracked.shape_name in attached_names:
                    # Never overwrite or remove an attached payload from stale scene world transforms,
                    # but refresh identity aliases when matched so subsequent scene/snapshot syncs stay linked.
                    if inc is not None:
                        merged_node = inc.node_key if inc.node_key is not None else tracked.node_key
                        merged_stable = inc.stable_key if inc.stable_key is not None else tracked.stable_key
                        merged_path = inc.path_key
                        new_sid = merged_node or merged_stable or merged_path
                        next_tracked[new_sid] = _TrackedObstacle(
                            sync_id=new_sid, node_key=merged_node, stable_key=merged_stable, path_key=merged_path,
                            shape_name=tracked.shape_name, link_key=tracked.link_key,
                            world_frame=tracked.world_frame, layer=tracked.layer, mask=tracked.mask,
                            geom_sig=tracked.geom_sig,
                        )
                    else:
                        next_tracked[tracked.sync_id] = tracked
                    continue
                if inc is None:
                    remove_names.append(tracked.shape_name)
                    occupied_names.discard(tracked.shape_name)
                    continue

                merged_node = inc.node_key if inc.node_key is not None else tracked.node_key
                merged_stable = inc.stable_key if inc.stable_key is not None else tracked.stable_key
                merged_path = inc.path_key
                new_sid = merged_node or merged_stable or merged_path

                if inc.geom_sig != tracked.geom_sig:
                    sh = _make_native_shape(inc.spec, tracked.shape_name)
                    replace_shapes.append((
                        tracked.shape_name, inc.link_key, inc.world_frame, inc.layer, inc.mask, sh,
                    ))
                    next_tracked[new_sid] = _TrackedObstacle(
                        sync_id=new_sid, node_key=merged_node, stable_key=merged_stable, path_key=merged_path,
                        shape_name=tracked.shape_name, link_key=inc.link_key,
                        world_frame=inc.world_frame.copy(), layer=inc.layer, mask=inc.mask, geom_sig=inc.geom_sig,
                    )
                elif inc.layer != tracked.layer or inc.mask != tracked.mask or inc.link_key != tracked.link_key:
                    # Filter or link_key change: pass shape=None so C++ reuses existing convex hull & BVH
                    # while rebuilding active collision pairs for the new link/filter rules.
                    replace_shapes.append((
                        tracked.shape_name, inc.link_key, inc.world_frame, inc.layer, inc.mask, None,
                    ))
                    next_tracked[new_sid] = _TrackedObstacle(
                        sync_id=new_sid, node_key=merged_node, stable_key=merged_stable, path_key=merged_path,
                        shape_name=tracked.shape_name, link_key=inc.link_key,
                        world_frame=inc.world_frame.copy(), layer=inc.layer, mask=inc.mask, geom_sig=tracked.geom_sig,
                    )
                elif tracked.world_frame is None or not np.allclose(inc.world_frame, tracked.world_frame, atol=1e-12, rtol=0.0):
                    pose_updates.append((tracked.shape_name, inc.world_frame))
                    next_tracked[new_sid] = _TrackedObstacle(
                        sync_id=new_sid, node_key=merged_node, stable_key=merged_stable, path_key=merged_path,
                        shape_name=tracked.shape_name, link_key=tracked.link_key,
                        world_frame=inc.world_frame.copy(), layer=tracked.layer, mask=tracked.mask, geom_sig=tracked.geom_sig,
                    )
                else:
                    next_tracked[new_sid] = _TrackedObstacle(
                        sync_id=new_sid, node_key=merged_node, stable_key=merged_stable, path_key=merged_path,
                        shape_name=tracked.shape_name, link_key=tracked.link_key,
                        world_frame=tracked.world_frame, layer=tracked.layer, mask=tracked.mask, geom_sig=tracked.geom_sig,
                    )

            # 2. Process newly added obstacles.
            def alloc_unique_name(base_name: str) -> str:
                if base_name not in occupied_names:
                    occupied_names.add(base_name)
                    return base_name
                idx = 1
                while f"{base_name}#{idx}" in occupied_names:
                    idx += 1
                chosen = f"{base_name}#{idx}"
                occupied_names.add(chosen)
                return chosen

            for ii, obs in enumerate(incoming):
                if ii in claimed_inc:
                    continue
                shape_name = alloc_unique_name(obs.default_name)
                sh = _make_native_shape(obs.spec, shape_name)
                add_shapes.append((shape_name, obs.link_key, obs.world_frame, obs.layer, obs.mask, sh))
                next_tracked[obs.sync_id] = _TrackedObstacle(
                    sync_id=obs.sync_id, node_key=obs.node_key, stable_key=obs.stable_key, path_key=obs.path_key,
                    shape_name=shape_name, link_key=obs.link_key,
                    world_frame=obs.world_frame.copy(), layer=obs.layer, mask=obs.mask, geom_sig=obs.geom_sig,
                )

            t_diff = perf_counter()
            raw = self._native.apply_sync_batch(
                base_updates, pose_updates, remove_names, replace_shapes, add_shapes,
            )
            t_commit = perf_counter()

            # Commit Python-side tracking only after atomic C++ commit succeeds.
            self._branch_poses = next_branch_poses
            self._synced_obstacles = next_tracked

            return SyncResult(
                version=int(raw["version"]),
                topology_version=int(raw["topology_version"]),
                changed=bool(raw["changed"]),
                poses_updated=int(raw["poses_updated"]),
                shapes_added=int(raw["shapes_added"]),
                shapes_removed=int(raw["shapes_removed"]),
                shapes_replaced=int(raw["shapes_replaced"]),
                robot_bases_updated=int(raw["robot_bases_updated"]),
                export_ms=0.0,
                diff_ms=(t_diff - t0) * 1000.0,
                commit_ms=(t_commit - t_diff) * 1000.0,
            )

    def sync_from_scene(self, scene_or_context) -> SyncResult:
        """Compile a physics snapshot from a Gobot EngineContext/Scene and incrementally sync this world.

        Note: this reads the current Scene state via ``compile_physics_scene_snapshot()``. When running
        a live physics simulation, ensure simulation transforms have been synced back to the Scene, or
        pass a runtime snapshot directly to ``sync_from_snapshot()``.
        """
        t0 = perf_counter()
        snapshot = _export_scene_snapshot(scene_or_context)
        export_ms = (perf_counter() - t0) * 1000.0
        res = self.sync_from_snapshot(snapshot)
        return SyncResult(
            version=res.version,
            topology_version=res.topology_version,
            changed=res.changed,
            poses_updated=res.poses_updated,
            shapes_added=res.shapes_added,
            shapes_removed=res.shapes_removed,
            shapes_replaced=res.shapes_replaced,
            robot_bases_updated=res.robot_bases_updated,
            export_ms=export_ms,
            diff_ms=res.diff_ms,
            commit_ms=res.commit_ms,
        )

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
        jerk_scale: float = 1.0,
        check_acceleration: bool = True,
        check_jerk: bool = True,
        disp_tol: float = 5e-4,
        strict_ccd: bool = False,
        manifold_constraints: Sequence[PoseConstraint] = (),
    ) -> tuple[bool, str | None]:
        """Validate either a waypoint polyline ``[T, J]`` or a continuous ``Trajectory`` curve.

        All collision checks (FK, AABB screening, discrete distance, and continuous CCD for convex
        and original triangle meshes) execute atomically against a single immutable C++ WorldSnapshot.
        Any ``manifold_constraints`` must specify strictly positive tolerances (``> 0`` or ``inf``);
        zero tolerances are rejected with ``ValueError``.
        """
        if dt is not None and (not math.isfinite(dt) or dt <= 0.0):
            raise ValueError("dt must be finite and positive")
        if not math.isfinite(velocity_scale) or velocity_scale <= 0.0:
            raise ValueError("velocity_scale must be finite and positive")
        if not math.isfinite(acceleration_scale) or acceleration_scale <= 0.0:
            raise ValueError("acceleration_scale must be finite and positive")
        if not math.isfinite(jerk_scale) or jerk_scale <= 0.0:
            raise ValueError("jerk_scale must be finite and positive")
        if manifold_constraints:
            _validate_manifold_constraints(manifold_constraints)

        # Pin ONE immutable C++ WorldSnapshot so joint/velocity/acc/jerk bounds and all collision checks
        # evaluate atomically against the exact same world version.
        snap = self._native.snapshot()
        lo_lim = np.asarray(snap.lower_limits, dtype=np.float64)
        hi_lim = np.asarray(snap.upper_limits, dtype=np.float64)
        v_lim = np.asarray(snap.velocity_limits, dtype=np.float64) * velocity_scale
        a_lim = np.asarray(snap.acceleration_limits, dtype=np.float64) * acceleration_scale
        j_lim = np.asarray(snap.jerk_limits, dtype=np.float64) * jerk_scale

        if isinstance(positions_or_trajectory, Trajectory):
            traj = positions_or_trajectory
            q = np.asarray(traj.positions, dtype=np.float64)
            if q.ndim != 2 or q.shape[0] < 2 or q.shape[1] != self.num_joints:
                return False, "trajectory must have shape [T>=2, J]"
            ok_bounds, bound_err, _ = check_trajectory_bounds(
                traj,
                v_lim,
                a_lim,
                check_velocity=True,
                check_acceleration=check_acceleration,
                lower_limits=lo_lim,
                upper_limits=hi_lim,
                jerk_limits=j_lim,
                check_jerk=check_jerk,
            )
            if not ok_bounds:
                return False, bound_err
            substeps = int(collision_substeps)
            if max_segment_length is not None:
                longest = float(np.linalg.norm(np.diff(q, axis=0), axis=-1).max())
                substeps = max(substeps, math.ceil(longest / max_segment_length))
            substeps = max(substeps, 1)
            # Sample along the actual continuous spline curve (cubic or C^2 quintic) per segment.
            u = np.arange(substeps, dtype=np.float64) / substeps
            t0 = traj.times[:-1, None]
            t1 = traj.times[1:, None]
            ts = np.concatenate(((t0 + u[None, :] * (t1 - t0)).reshape(-1), traj.times[-1:]))
            samples, _, _ = traj.sample(ts)
            if (not np.isfinite(samples).all()
                    or (samples < lo_lim - 1e-6).any()
                    or (samples > hi_lim + 1e-6).any()):
                return False, "trajectory violates joint limits"
        else:
            q = np.asarray(positions_or_trajectory, dtype=np.float64)
            if q.ndim != 2 or q.shape[0] < 2 or q.shape[1] != self.num_joints:
                return False, "trajectory must have shape [T>=2, J]"
            if not np.isfinite(q).all() or (q < lo_lim).any() or (q > hi_lim).any():
                return False, "trajectory violates joint limits"
            if dt is not None:
                speed = np.abs(np.diff(q, axis=0)) / dt
                if (speed > v_lim * (1.0 + 1e-6)).any():
                    return False, "trajectory violates joint velocity limits"
            substeps = int(collision_substeps)
            if max_segment_length is not None:
                longest = float(np.linalg.norm(np.diff(q, axis=0), axis=-1).max())
                substeps = max(substeps, math.ceil(longest / max_segment_length))
            substeps = max(substeps, 1)
            u = (np.arange(substeps, dtype=np.float64) / substeps)[:, None]
            between = (1.0 - u)[None, :, :] * q[:-1, None, :] + u[None, :, :] * q[1:, None, :]
            samples = np.concatenate((between.reshape(-1, self.num_joints), q[-1:]), axis=0)

        is_traj = isinstance(positions_or_trajectory, Trajectory)
        t_arg = np.ascontiguousarray(positions_or_trajectory.times, dtype=np.float64) if is_traj else None
        v_arg = np.ascontiguousarray(positions_or_trajectory.velocities, dtype=np.float64) if is_traj else None
        a_arg = (np.ascontiguousarray(positions_or_trajectory.accelerations, dtype=np.float64)
                 if (is_traj and positions_or_trajectory.spline_order == 5) else None)
        order_arg = int(positions_or_trajectory.spline_order) if is_traj else 3
        wm = margin if world_margin is None else world_margin
        ok, err = snap.validate_trajectory_collision(
            np.ascontiguousarray(q, dtype=np.float64),
            np.ascontiguousarray(samples, dtype=np.float64),
            int(substeps),
            float(margin),
            float(wm),
            bool(verify_original),
            float(disp_tol),
            t_arg,
            v_arg,
            bool(strict_ccd),
            a_arg,
            order_arg,
        )
        if not ok:
            return False, err
        if manifold_constraints:
            compiled_mc = [self.compile_pose_constraint(c) for c in manifold_constraints]
            ok_m, err_m = snap.validate_trajectory_manifold(
                np.ascontiguousarray(q, dtype=np.float64),
                compiled_mc,
                t_arg,
                v_arg,
                a_arg,
                order_arg,
            )
            if not ok_m:
                return False, err_m
        return True, None
