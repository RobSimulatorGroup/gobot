"""CPU Coal narrow phase and its first-order rigid-transform derivative."""

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
import torch
from torch.func import jacfwd, vmap
from scipy.spatial import ConvexHull

from ._geometry import clearance_mask

try:
    from gobot import _coal_native as native
except ImportError as error:
    raise ImportError("Coal planning requires a Gobot build with GOB_BUILD_COAL=ON") from error


def _query(frames, pairs, verify, clearance=math.inf):
    # The native batch always has one flat pose axis, independently of FK's batch shape.
    batch = math.prod(frames.shape[:-4])
    array = frames.detach().reshape(batch, *frames.shape[-4:]).contiguous().numpy()
    values = torch.from_numpy(pairs.query(array, verify, clearance)).to(frames.dtype)
    return values.reshape(*frames.shape[:-4], frames.shape[-4], values.size(-1))


def witness_coefficients(data):
    a, b, normal = data[..., 1:].split(3, dim=-1)
    points = torch.stack((a, b), dim=-2)
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    directions = torch.stack((normal, -normal), dim=-2)
    # For a rigid frame [R|t], dp = d[R|t] @ [witness_local, 1].
    return directions.unsqueeze(-1) * homogeneous.unsqueeze(-2)


class _Distance(torch.autograd.Function):
    @staticmethod
    def forward(frames, pairs):
        data = _query(frames, pairs, False)
        return data[..., 0], witness_coefficients(data)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, coefficients = output
        ctx.mark_non_differentiable(coefficients)
        ctx.save_for_forward(coefficients)
        ctx.save_for_backward(coefficients)

    @staticmethod
    def jvp(ctx, frame_tangent, _pairs):
        coefficients, = ctx.saved_tensors
        return (coefficients * frame_tangent).sum((-3, -2, -1)), None

    @staticmethod
    def backward(ctx, adjoint, _coefficients):
        coefficients, = ctx.saved_tensors
        return adjoint[..., None, None, None] * coefficients, None

    @staticmethod
    def vmap(info, in_dims, frames, pairs):
        return _Distance.apply(frames.movedim(in_dims[0], 0), pairs), (0, 0)


@dataclass
class Shape:
    link: tuple
    body: int | None
    kind: int
    frame: torch.Tensor
    geometry: object
    label: str
    layer: int
    mask: int
    spec: dict


def _mesh_geometry(spec, label):
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
        # A triangle or planar polygon has an exact 2-D hull without any artificial thickness.
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


def shape(spec, link, body, position, rotation):
    kind = spec['type']
    label = str((link, spec.get('path') or spec.get('name') or kind))
    if kind == 6 and body is not None:
        raise ValueError(f"unsupported moving plane on {label}")
    for field in {1: ('size',), 2: ('radius',), 3: ('radius', 'height'), 4: ('radius', 'height')}.get(kind, ()):
        value = np.asarray(spec[field])
        if value.shape != ((3,) if field == 'size' else ()) or not np.isfinite(value).all() or (value < 0).any():
            raise ValueError(f"invalid {field} on {label}")
    try:
        if kind == 5:
            spec = _mesh_geometry(spec, label)
        geometry = native.Shape(label, spec)
    except (ValueError, RuntimeError) as error:
        raise ValueError(f"invalid collision geometry {label}: {error}") from error
    frame = torch.cat((rotation, position[:, None]), -1)
    return Shape(link, body, kind, frame, geometry, label, spec['collision_layer'], spec['collision_mask'], spec)


def capsule_shape(link, body, a, b, radius, layer, mask):
    axis = b - a
    height = axis.norm().item()
    rotation = torch.eye(3, dtype=a.dtype)
    if height:
        z = axis / height
        x = torch.linalg.cross(rotation[z.abs().argmin()], z)
        x = x / x.norm()
        rotation = torch.stack((x, torch.linalg.cross(z, x), z), dim=-1)
    spec = dict(type=4 if height else 2, radius=radius.item(), height=height,
                collision_layer=layer, collision_mask=mask)
    return shape(spec, link, body, (a + b) / 2, rotation)


class _Bounds:
    """Cache fixed world bounds and transform each moving local AABB once per pose."""
    def __init__(self, shapes, frames, moving, original):
        # Halfspaces are always candidates. Avoid transforming their infinite bounds.
        limits = np.array([s.geometry.bounds(original) if s.kind != 6 else np.zeros((2, 3))
                           for s in shapes], dtype=np.float64)
        self.center = (limits[:, 1] + limits[:, 0]) * 0.5
        self.half = (limits[:, 1] - limits[:, 0]) * 0.5
        self.fixed_bounds = self.transform(frames.double().numpy(), self.center, self.half)
        self.center, self.half = self.center[moving], self.half[moving]

    @staticmethod
    def transform(frames, centers, half):
        rotation = frames[..., :3]
        center = (rotation @ centers[..., None])[..., 0] + frames[..., 3]
        extent = (np.abs(rotation) @ half[..., None])[..., 0]
        # Round outwards, including cancellation in an offset local center.
        scale = np.abs(frames[..., 3]) + (np.abs(rotation) @ np.abs(centers[..., None]))[..., 0] + extent
        extent += 16 * np.finfo(np.float64).eps * (1 + scale)
        return center - extent, center + extent

    @staticmethod
    def overlaps(lo, hi, endpoints, planes, thresholds):
        i, j = endpoints.T
        gap = np.maximum(lo[:, i] - hi[:, j], lo[:, j] - hi[:, i])
        return planes | ~(gap > thresholds[:, None]).any(axis=-1)

    def candidates(self, frames, moving, endpoints, planes, thresholds, group_size=1):
        shape = (len(frames), len(self.fixed_bounds[0]), 3)
        lo, hi = (np.broadcast_to(x, shape).copy() for x in self.fixed_bounds)
        lo[:, moving], hi[:, moving] = self.transform(frames[:, moving], self.center, self.half)
        if group_size == 1 or len(frames) < 2:
            return np.nonzero(self.overlaps(lo, hi, endpoints, planes, thresholds))

        # Include every sampled pose in its segment's bounds. When validation
        # appends the final waypoint, fold it into the last segment.
        starts = np.arange(0, len(frames) - 1, group_size)
        possible = self.overlaps(np.minimum.reduceat(lo, starts), np.maximum.reduceat(hi, starts),
                                 endpoints, planes, thresholds)
        rows, pairs = [], []
        for start, stop, mask in zip(starts, [*starts[1:], len(frames)], possible):
            selected = np.flatnonzero(mask)
            hit, col = np.nonzero(self.overlaps(lo[start:stop], hi[start:stop], endpoints[selected],
                                                planes[selected], thresholds[selected]))
            rows.append(start + hit)
            pairs.append(selected[col])
        return np.concatenate(rows), np.concatenate(pairs)


class CollisionModel:
    def __init__(self, shapes, enabled, query_factory=None):
        mobile = [i for i, s in enumerate(shapes) if s.body is not None]
        fixed = [i for i, s in enumerate(shapes) if s.body is None]

        def active(i, j):
            return enabled(shapes[i], shapes[j])

        pairs = [(i, j) for i, j in combinations(mobile, 2) if active(i, j)]
        self.self_count = len(pairs)
        # Match the existing public order of environment pairs (capsules, boxes, planes).
        for kinds in ((2, 3, 4), (1, 5), (6,)):
            pairs.extend((i, j) for i in mobile for j in fixed if shapes[j].kind in kinds and active(i, j))
        self.labels = tuple((shapes[i].link, shapes[j].link) for i, j in pairs)
        self.names = tuple((shapes[i].label, shapes[j].label) for i, j in pairs)
        self.topology = (tuple((s.body, s.kind) for s in shapes), tuple(pairs))
        self.has_mesh = any(shapes[k].kind == 5 for pair in pairs for k in pair)
        if query_factory is None:
            self.pairs = native.Pairs([(shapes[i].geometry, shapes[j].geometry) for i, j in pairs])
        elif pairs:
            self.pairs = query_factory(shapes, pairs)
        if pairs:
            self.endpoints = torch.tensor(pairs)
            self.indices = self.endpoints.flatten()
            self.moving = np.asarray(mobile, dtype=np.intp)
            self.planes = np.array([shapes[i].kind == 6 or shapes[j].kind == 6 for i, j in pairs])
            self.frames = torch.stack([s.frame for s in shapes])
            self.mobile_ids = torch.from_numpy(self.moving)
            self.bodies = torch.tensor([shapes[i].body for i in mobile])
            self.mobile_frames = self.frames[self.mobile_ids]
            # The final entry in the frame Jacobian is zero for static obstacles.
            mobile_index = torch.full((len(shapes),), len(mobile))
            mobile_index[self.mobile_ids] = torch.arange(len(mobile))
            self.pair_mobile_indices = mobile_index[self.endpoints]

            self.bounds = {original: _Bounds(shapes, self.frames, self.moving, original)
                           for original in (False, True)}

    def moving_frames(self, positions, rotations):
        rotation = rotations.index_select(-3, self.bodies)
        origin = positions.index_select(-2, self.bodies)
        transformed = rotation @ self.mobile_frames
        return torch.cat((transformed[..., :3], transformed[..., 3:] + origin.unsqueeze(-1)), -1)

    def world_frames(self, positions, rotations):
        moving = self.moving_frames(positions, rotations)
        world = self.frames.expand(*positions.shape[:-2], *self.frames.shape)
        return world.index_copy(-3, self.mobile_ids, moving)

    def _selected_query(self, frames, rows, pairs, original):
        packed = frames[rows[:, None], self.endpoints.numpy()[pairs]]
        return self.pairs.query(packed[:, None], original, math.inf, pairs)[:, 0]

    def nearby(self, frames, thresholds, original=False, group_size=1):
        """Return packed (sample, pair, query result) for conservative AABB candidates."""
        array = frames.detach().numpy()
        rows, pairs = self.bounds[original].candidates(array, self.moving, self.endpoints.numpy(),
                                                       self.planes, thresholds.detach().numpy(), group_size)
        if len(rows):
            data = self._selected_query(array, rows, pairs, original)
            values = torch.from_numpy(data).to(frames.dtype)
        else:
            values = frames.new_empty((0, 1 if original else 10))
        return torch.from_numpy(rows), torch.from_numpy(pairs), values

    def constraint_rows(self, poses, samples, margins, buffer, linearize, group_size, segment_minimum=False):
        """Sparse d_safe - d and joint derivatives, in sample then pair order."""
        frames = self.world_frames(*poses(samples))
        rows, ids, data = self.nearby(frames, margins + buffer, group_size=group_size)
        residual = margins[ids] - data[:, 0]
        near = (residual > -buffer).nonzero().flatten()
        if segment_minimum and len(near):
            # Keep the worst sampled clearance per segment and shape pair. Stable
            # ties pick the earliest sample. Restore sample order for FK reuse.
            group = ((rows[near] // group_size) * len(margins) + ids[near]).numpy()
            order = np.lexsort((-residual[near].numpy(), group))
            first = np.r_[True, np.diff(group[order]) != 0]
            near = near[np.sort(order[first])]
        rows, ids, residual = rows[near], ids[near], residual[near]
        gradient = None
        if linearize and len(rows):
            # Share one FK Jacobian per active configuration. Contract the selected
            # frames with the witnesses from the same narrow-phase query.
            unique, inverse = torch.unique_consecutive(rows, return_inverse=True)
            def moving_frames(q):
                return self.moving_frames(*poses(q))

            pose_jac = vmap(jacfwd(moving_frames))(samples[unique])
            pose_jac = torch.cat((pose_jac, torch.zeros_like(pose_jac[:, :1])), dim=1)
            pair_jac = pose_jac[inverse[:, None], self.pair_mobile_indices[ids]]
            gradient = -torch.einsum("nkij,nkijq->nq", witness_coefficients(data[near]), pair_jac)
        return rows, residual, gradient

    def trajectory_check(self, margins, original=False):
        """Make a sampled clearance check for one trajectory and geometry model."""
        previous = None

        def check(positions, rotations, substeps):
            nonlocal previous
            if not self.labels:
                return None
            frames = self.world_frames(positions, rotations)
            if previous is not None:
                sample, pair, last_substeps = previous
                if last_substeps == substeps and sample < len(frames):
                    data = self._selected_query(frames.detach().numpy(), np.array([sample]), np.array([pair]), original)
                    distance = frames.new_tensor(data[0, 0])
                    if not clearance_mask(distance, margins[pair]):
                        return sample, self.names[pair]
                previous = None

            rows, pairs, data = self.nearby(frames, margins, original, substeps)
            failed = (~clearance_mask(data[:, 0], margins[pairs])).nonzero()
            if len(failed):
                index = failed[0, 0]
                sample, pair = int(rows[index]), int(pairs[index])
                previous = sample, pair, substeps
                return sample, self.names[pair]
            return None

        return check

    def clearance_mask(self, positions, rotations, margins, original=False):
        ok = torch.ones((*positions.shape[:-2], len(self.labels)), dtype=torch.bool)
        if self.labels:
            frames = self.world_frames(positions, rotations).reshape(-1, len(self.frames), 3, 4)
            rows, pairs, data = self.nearby(frames, margins, original)
            ok.view(-1, len(self.labels))[rows, pairs] = clearance_mask(data[:, 0], margins[pairs])
        return ok

    def distances(self, positions, rotations, verify=False, clearance=math.inf):
        if not self.labels:
            return positions[..., 0, :0]
        frames = self.world_frames(positions, rotations)
        paired = frames.index_select(-3, self.indices).unflatten(-3, (-1, 2))
        if verify:
            return _query(paired, self.pairs, True, clearance)[..., 0]
        return _Distance.apply(paired, self.pairs)[0]
