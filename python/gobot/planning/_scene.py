"""Compile the owned physics snapshot into fixed-base Torch planning data."""

from collections import defaultdict
from collections.abc import Sequence
from copy import copy, deepcopy
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
import torch
from torch import Tensor

from .._kinematics import SceneKinematics
from ._geometry import box_distance, capsule_distance, clearance_mask, plane_distance

# PhysicsShapeType in the version 1 snapshot.
_BOX, _SPHERE, _CYLINDER, _CAPSULE, _MESH, _PLANE = range(1, 7)


@dataclass(frozen=True)
class Capsule:
    """Collision capsule in link-local metres. Equal endpoints define a sphere."""

    start: Sequence[float] | Tensor
    end: Sequence[float] | Tensor
    radius: float
    collision_layer: int = 1
    collision_mask: int = 1


@dataclass(frozen=True)
class _Shape:
    kind: int
    link: tuple[str, str]
    body: int | None
    data: tuple[Tensor, ...]
    layer: int
    mask: int


def _transform(value, tensor):
    matrix = tensor(value["matrix"])
    if (
        matrix.shape != (4, 4)
        or not torch.isfinite(matrix).all()
        or not torch.allclose(matrix[3], tensor([0, 0, 0, 1]), atol=1e-6)
        or not torch.allclose(matrix[:3, :3].T @ matrix[:3, :3], tensor(torch.eye(3)), atol=1e-5)
        or not torch.isclose(torch.linalg.det(matrix[:3, :3]), tensor(1), atol=1e-5)
    ):
        raise ValueError("planning requires finite rigid transforms")
    return matrix[:3, 3], matrix[:3, :3]


def _shape(value, link, body, rest, tensor) -> _Shape:
    kind = value["type"]
    center, rotation = _transform(value["transform"], tensor)
    size = None
    if kind == _BOX:
        size = tensor(value["size"]) * 0.5
    elif kind == _MESH:
        vertices = tensor(np.asarray(value["vertices"]))
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not torch.isfinite(vertices).all():
            raise ValueError(f"invalid mesh vertices on {link!r}")
        lo, hi = vertices.amin(0), vertices.amax(0)
        center = center + rotation @ ((lo + hi) * 0.5)
        size = (hi - lo) * 0.5

    if size is not None:
        if size.shape != (3,) or not torch.isfinite(size).all() or (size < 0).any():
            raise ValueError(f"invalid box dimensions on {link!r}")
        if body is None:
            kind, data = _BOX, (center, rotation, size)
        else:
            # Enclose the whole box in a capsule along its longest local axis.
            axis = int(size.argmax())
            offset = rotation[:, axis] * size[axis]
            radius = torch.linalg.vector_norm(size[[i for i in range(3) if i != axis]])
            kind, data = _CAPSULE, (center - offset, center + offset, radius)
    elif kind in (_SPHERE, _CYLINDER, _CAPSULE):
        radius = tensor(value["radius"])
        height = tensor(0 if kind == _SPHERE else value["height"])
        if not torch.isfinite(torch.stack((radius, height))).all() or radius < 0 or height < 0:
            raise ValueError(f"invalid sphere, cylinder or capsule dimensions on {link!r}")
        # The same-radius capsule encloses a cylinder of this axis length.
        offset = rotation[:, 2] * (height * 0.5)
        kind, data = _CAPSULE, (center - offset, center + offset, radius)
    elif kind == _PLANE and body is None:
        data = (center, rotation[:, 2])
    else:
        raise ValueError(f"unsupported planning collision shape type {kind} on {link!r}")

    if body is not None:
        origin, frame = rest
        a, b, radius = data
        data = (frame.T @ (a - origin), frame.T @ (b - origin), radius)
    return _Shape(kind, link, body, data, value["collision_layer"], value["collision_mask"])


class PlanningScene:
    def __init__(self, snapshot, robots, ignored_link_pairs, link_capsules, dtype, device, collision_backend="torch"):
        if snapshot.get("schema_version") != 1:
            raise ValueError("unsupported physics snapshot schema")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("planning requires float32 or float64")
        if not robots:
            raise ValueError("select at least one robot and its base link")
        if snapshot.get("deformables"):
            raise ValueError("planning collision does not support deformable bodies")

        if collision_backend not in ("torch", "coal", "mujoco_warp"):
            raise ValueError("collision_backend must be 'torch', 'coal' or 'mujoco_warp'")
        use_convex = collision_backend != "torch"
        self.distance_models = ()
        self.validation_models = ()
        if use_convex:
            if device.type != "cpu":
                raise ValueError("Coal validation and CPU SQP require CPU planner tensors")
            from . import _coal

        self.dtype, self.device = dtype, device
        self.kinematics = []
        self.links = {}
        self.joint_names = []
        lower, upper, speeds, initial, continuous = [], [], [], [], []
        adjacent = set()
        by_name = defaultdict(list)
        for robot in snapshot["robots"]:
            by_name[robot["name"]].append(robot)
        for name, bases in robots.items():
            if len(by_name[name]) != 1:
                raise ValueError(f"expected one robot {name!r} in the snapshot")
            robot = by_name[name][0]
            roots = (bases,) if isinstance(bases, str) else tuple(bases)
            if not roots:
                raise ValueError(f"select at least one base link for robot {name!r}")
            links = {link["name"]: link for link in robot["links"]}
            children = defaultdict(list)
            for joint in robot["joints"]:
                parent, child = joint["parent_link"], joint["child_link"]
                children[parent].append(child)
                adjacent.add(frozenset(((name, parent), (name, child))))
            for base in roots:
                if base not in links:
                    raise ValueError(f"unknown base link {(name, base)!r}")
                descendants = {base}
                pending = [base]
                while pending:
                    for child in children[pending.pop()]:
                        if child not in descendants:
                            descendants.add(child)
                            pending.append(child)
                if any((name, link) in self.links for link in descendants):
                    raise ValueError(f"overlapping base links for robot {name!r}")
                branch_joints = [joint for joint in robot["joints"] if joint["child_link"] in descendants]
                movable = [joint for joint in branch_joints if int(joint["type"]) in (1, 2, 3)]
                names = [joint["name"] for joint in movable]
                branch = dict(robot, links=[link for link in robot["links"] if link["name"] in descendants],
                              joints=branch_joints)
                kin = SceneKinematics({"schema_version": 1, "robots": [branch]}, robot_name=name,
                                      base_link=base, joint_names=names, dtype=dtype, device=device,
                                      include_virtual_base=True)
                root = _transform(links[base]["global_transform"], self.tensor)
                first = len(self.joint_names)
                self.kinematics.append((kin, root, slice(first, first + len(names))))
                offset = len(self.links)
                self.links.update({(name, link): offset + i for i, link in enumerate(kin.body_names)})
                self.joint_names.extend((name, key) for key in names)
                lower.append(kin.q_lo)
                upper.append(kin.q_hi)
                for joint in movable:
                    speed = float(joint.get("velocity_limit", 0))
                    if math.isnan(speed):
                        raise ValueError(f"invalid velocity limit on joint {(name, joint['name'])!r}")
                    speeds.append(speed if speed > 0 else float("inf"))
                    initial.append(joint.get("initial_position", 0))
                    continuous.append(int(joint["type"]) == 2)
        if not self.joint_names:
            raise ValueError("the selected robots have no movable joints")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint names must be unique within each robot")
        self.joint_names = tuple(self.joint_names)
        self.lower, self.upper = torch.cat(lower), torch.cat(upper)
        self.speed = self.tensor(speeds)
        # Continuous joints are unbounded regardless of imported placeholder ranges.
        continuous = torch.tensor(continuous, dtype=torch.bool, device=device)
        self.lower = self.lower.masked_fill(continuous, -torch.inf)
        self.upper = self.upper.masked_fill(continuous, torch.inf)
        self.initial = self.tensor(initial).clamp(self.lower, self.upper)
        if not torch.isfinite(self.initial).all():
            raise ValueError("the snapshot contains a nonfinite initial joint position")

        universe = {(robot["name"], link["name"]) for robot in snapshot["robots"]
                    for link in robot["links"] if link.get("role") != "virtual_root"}
        if link_capsules.keys() - (self.links.keys() & universe):
            raise ValueError("link_capsules must refer to selected physical links")
        for a, b in ignored_link_pairs:
            if a not in universe or b not in universe:
                raise ValueError(f"unknown ignored collision link pair {(a, b)!r}")
            adjacent.add(frozenset((a, b)))

        shapes, convex_shapes, authored_shapes = [], [], []

        def add_shape(value, key, body, rest, *, validation_only=False):
            if use_convex:
                p, r = _transform(value["transform"], self.tensor)
                if rest is not None:
                    origin, frame = rest
                    p, r = frame.T @ (p - origin), frame.T @ r
                item = _coal.shape(value, key, body, p, r)
                authored_shapes.append(item)
                if validation_only:
                    return
                convex_shapes.append(item)
                # The existing chord sweep applies to moving spheres/capsules
                # against spheres, capsules, boxes and halfspaces.
                kinds = (_SPHERE, _CAPSULE) if body is not None else (_SPHERE, _CAPSULE, _BOX, _PLANE)
                if value["type"] not in kinds:
                    return
            shapes.append(_shape(value, key, body, rest, self.tensor))

        for robot in snapshot["robots"]:
            for link in robot["links"]:
                if link.get("role") == "virtual_root":
                    continue
                key = (robot["name"], link["name"])
                index = self.links.get(key)
                rest = _transform(link["global_transform"], self.tensor) if index is not None else None
                if key in link_capsules:
                    authored = [v for v in link.get("shapes", ()) if not v["disabled"]]
                    if use_convex:
                        for value in authored:
                            add_shape(value, key, index, rest, validation_only=True)
                    for capsule in link_capsules[key]:
                        a, b, radius = map(self.tensor, (capsule.start, capsule.end, capsule.radius))
                        if (
                            a.shape != (3,) or b.shape != (3,) or radius.ndim != 0
                            or not torch.isfinite(torch.cat((a, b, radius[None]))).all() or radius < 0
                        ):
                            raise ValueError(f"invalid collision capsule on {key!r}")
                        if use_convex:
                            item = _coal.capsule_shape(key, index, a, b, radius, capsule.collision_layer,
                                                       capsule.collision_mask)
                            convex_shapes.append(item)
                            if not authored:
                                authored_shapes.append(item)
                        shapes.append(_Shape(_CAPSULE, key, index, (a, b, radius),
                                             capsule.collision_layer, capsule.collision_mask))
                    continue
                for shape in link.get("shapes", ()):
                    if not shape["disabled"]:
                        add_shape(shape, key, index, rest)
        for shape in snapshot.get("loose_shapes", ()):
            if not shape["disabled"]:
                key = ("", shape.get("path", shape["name"]))
                add_shape(shape, key, None, None)
        for terrain in snapshot.get("terrains", ()):
            if terrain["heightfields"]:
                raise ValueError("planning collision does not support terrain heightfields")
            key = ("", terrain["name"])
            for geometry in terrain["boxes"] + terrain["meshes"]:
                value = dict(terrain, **geometry, type=_BOX if "size" in geometry else _MESH)
                add_shape(value, key, None, None)

        mobile = [shape for shape in shapes if shape.body is not None]
        fixed = [shape for shape in shapes if shape.body is None]
        self._body_indices = torch.tensor([shape.body for shape in mobile], dtype=torch.long, device=device)
        if mobile:
            self._end_a, self._end_b, self._radius = (
                torch.stack(values) for values in zip(*(shape.data for shape in mobile))
            )
            sphere_count = 1 if torch.equal(self._end_a, self._end_b) else 5
            self._sweep_fractions = torch.linspace(0, 1, sphere_count, dtype=dtype, device=device)[:, None, None]

        def enabled(a, b):
            return (a.link != b.link and frozenset((a.link, b.link)) not in adjacent
                    and ((a.layer & b.mask) or (b.layer & a.mask)))

        same = [(i, j) for i, j in combinations(range(len(mobile)), 2) if enabled(mobile[i], mobile[j])]
        self._mobile_pairs = torch.tensor(same, dtype=torch.long, device=device).reshape(-1, 2)
        labels = [(mobile[i].link, mobile[j].link) for i, j in same]
        self._fixed_groups = []
        for kind, kernel in ((_CAPSULE, capsule_distance), (_BOX, box_distance), (_PLANE, plane_distance)):
            pairs = [(i, shape) for i, part in enumerate(mobile) for shape in fixed
                     if shape.kind == kind and enabled(part, shape)]
            if pairs:
                indices, obstacles = zip(*pairs)
                data = tuple(torch.stack(values) for values in zip(*(shape.data for shape in obstacles)))
                self._fixed_groups.append((kernel, torch.tensor(indices, device=device), data))
                labels.extend((mobile[i].link, shape.link) for i, shape in pairs)
        self.swept_pairs = tuple(labels[len(same):])
        self.collision_pairs = tuple(labels)
        if use_convex:
            query_factory = None
            if collision_backend == "mujoco_warp":
                from ._mujoco_warp import WarpPairs
                query_factory = WarpPairs
            model = _coal.CollisionModel(convex_shapes, enabled, query_factory)
            self.distance_models = (model,)
            self.collision_pairs = model.labels
            if link_capsules or model.has_mesh or query_factory is not None:
                validation = _coal.CollisionModel(authored_shapes, enabled) if link_capsules or query_factory else model
                self.validation_models = (validation,)

    @property
    def has_collision_pairs(self):
        return bool(self.collision_pairs)

    def tensor(self, value):
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    @staticmethod
    def stack(scenes):
        """Stack numerical model data while sharing link, joint and collision indices."""
        first = scenes[0]

        def topology(scene):
            trees = [(kin.body_names, kin.parent, kin.kinds, kin.joint_idx, kin.joint_names)
                     for kin, _, _ in scene.kinematics]
            return (scene.links, scene.joint_names, trees, scene.collision_pairs,
                    [kernel for kernel, _, _ in scene._fixed_groups],
                    tuple(model.topology for model in scene.distance_models))

        def indices(scene):
            return (scene._body_indices, scene._mobile_pairs, *(i for _, i, _ in scene._fixed_groups))

        reference = topology(first)
        for scene in scenes[1:]:
            if scene.dtype != first.dtype or scene.device != first.device:
                raise ValueError("batched planners must share a dtype and device")
            if topology(scene) != reference or any(
                not torch.equal(a, b) for a, b in zip(indices(first), indices(scene))
            ):
                raise ValueError("batched planners must share joint and collision topology")

        batch = copy(first)
        batch.distance_models = tuple(m for scene in scenes for m in scene.distance_models)
        if any(scene.validation_models for scene in scenes):
            batch.validation_models = tuple((s.validation_models or s.distance_models)[0] for s in scenes)
        batch.kinematics = []
        for branches in zip(*(scene.kinematics for scene in scenes)):
            models, roots, slices = zip(*branches)
            kin = models[0]
            # Preserve the link axis first. FK selects a link, then operates
            # on its independent model frames along the batch axis.
            if any(model.digest != kin.digest for model in models[1:]):
                kin = deepcopy(kin)
                for name in ("parent_joint", "joint_child", "axes"):
                    setattr(kin, name, torch.stack([getattr(model, name) for model in models], dim=1))
            batch.kinematics.append((kin, tuple(torch.stack(x) for x in zip(*roots)), slices[0]))
        fields = ["initial", "lower", "upper", "speed"]
        if first._body_indices.numel():
            fields.extend(("_end_a", "_end_b", "_radius"))
        for name in fields:
            setattr(batch, name, torch.stack([getattr(scene, name) for scene in scenes]))
        batch._fixed_groups = []
        for groups in zip(*(scene._fixed_groups for scene in scenes)):
            kernel, i, _ = groups[0]
            data = tuple(torch.stack(x) for x in zip(*(group[2] for group in groups)))
            batch._fixed_groups.append((kernel, i, data))
        return batch

    def poses(self, q):
        all_positions, all_rotations = [], []
        for kin, (position, rotation), joint_slice in self.kinematics:
            p, r = kin.poses(position.expand(q.shape[:-1] + (3,)),
                             rotation.expand(q.shape[:-1] + (3, 3)), q[..., joint_slice])
            all_positions.append(p)
            all_rotations.append(r)
        return torch.cat(all_positions, dim=-2), torch.cat(all_rotations, dim=-3)

    def distances(self, q):
        if not self.has_collision_pairs:
            return q[..., :0]
        return self.pose_distances(*self.poses(q))

    @property
    def self_pair_count(self):
        return self.distance_models[0].self_count if self.distance_models else len(self._mobile_pairs)

    def pair_margins(self, margin, world_margin):
        margins = self.initial.new_full((len(self.collision_pairs),), world_margin)
        margins[:self.self_pair_count] = margin
        return margins

    def _capsule_ends(self, positions, rotations):
        rotation = rotations.index_select(-3, self._body_indices)
        position = positions.index_select(-2, self._body_indices)
        a = (rotation @ self._end_a[..., None]).squeeze(-1) + position
        b = (rotation @ self._end_b[..., None]).squeeze(-1) + position
        return a, b

    def _world_distances(self, a, b):
        return [kernel(a.index_select(-2, i), b.index_select(-2, i), self._radius[..., i], *data)
                for kernel, i, data in self._fixed_groups]

    def swept_world_distances(self, start, end):
        """Distances to world obstacles along chords connecting corresponding proxy sphere centers."""
        if not self._fixed_groups:
            return start[..., :0]
        a, b = self._capsule_ends(*self.poses(torch.stack((start, end), dim=-2)))
        centers = torch.lerp(a.unsqueeze(-3), b.unsqueeze(-3), self._sweep_fractions)
        prev, curr = centers.unbind(-4)
        return torch.cat(self._world_distances(prev, curr), dim=-1).amin(dim=-2)

    def pose_distances(self, positions, rotations):
        if not self.has_collision_pairs:
            return positions.new_empty(positions.shape[:-2] + (0,))
        if self.distance_models:
            return self._model_distances(self.distance_models, positions, rotations)
        a, b = self._capsule_ends(positions, rotations)
        distances = []
        if len(self._mobile_pairs):
            i, j = self._mobile_pairs.unbind(-1)
            distances.append(capsule_distance(a.index_select(-2, i), b.index_select(-2, i), self._radius[..., i],
                                              a.index_select(-2, j), b.index_select(-2, j), self._radius[..., j]))
        distances.extend(self._world_distances(a, b))
        return torch.cat(distances, dim=-1)

    @staticmethod
    def _model_distances(models, positions, rotations):
        if len(models) == 1:
            return models[0].distances(positions, rotations)
        return torch.stack([model.distances(p, r) for model, p, r in zip(models, positions, rotations, strict=True)])

    def convex_clearance(self, positions, rotations, margins):
        if not self.distance_models:
            return clearance_mask(self.pose_distances(positions, rotations), margins)
        if len(self.distance_models) == 1:
            return self.distance_models[0].clearance_mask(positions, rotations, margins)
        return torch.stack([model.clearance_mask(p, r, margins)
                            for model, p, r in zip(self.distance_models, positions, rotations, strict=True)])

    def original_checks(self, positions, rotations, margin, world_margin):
        """Per-scene authored geometries can have different pair counts."""
        for i, model in enumerate(self.validation_models):
            rows = slice(None) if len(self.validation_models) == 1 else slice(i, i + 1)
            margins = positions.new_full((len(model.labels),), world_margin)
            margins[:model.self_count] = margin
            yield rows, model.clearance_mask(positions[rows], rotations[rows], margins, original=True), model.names
