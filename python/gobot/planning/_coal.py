"""CPU Coal narrow phase and its first-order rigid-transform derivative."""

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
import torch
from scipy.spatial import ConvexHull

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


class _Distance(torch.autograd.Function):
    @staticmethod
    def forward(frames, pairs):
        data = _query(frames, pairs, False)
        a, b, normal = data[..., 1:].split(3, dim=-1)
        points = torch.stack((a, b), dim=-2)
        homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
        directions = torch.stack((normal, -normal), dim=-2)
        # For a rigid frame [R|t], dp = d[R|t] @ [witness_local, 1].
        coefficients = directions.unsqueeze(-1) * homogeneous.unsqueeze(-2)
        return data[..., 0], coefficients

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
    return Shape(link, body, kind, frame, geometry, label, spec['collision_layer'], spec['collision_mask'])


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


class CollisionModel:
    def __init__(self, shapes, enabled):
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
        self.pairs = native.Pairs([(shapes[i].geometry, shapes[j].geometry) for i, j in pairs])
        if pairs:
            self.indices = torch.tensor(pairs).flatten()
            self.frames = torch.stack([s.frame for s in shapes])
            # The final, identity FK frame represents the static world.
            self.bodies = torch.tensor([s.body if s.body is not None else -1 for s in shapes])

    def distances(self, positions, rotations, verify=False, clearance=math.inf):
        if not self.labels:
            return positions[..., 0, :0]
        world = torch.eye(3, dtype=positions.dtype).expand(*rotations.shape[:-3], 1, 3, 3)
        rotation = torch.cat((rotations, world), -3)[..., self.bodies, :, :]
        origin = torch.cat((positions, torch.zeros_like(positions[..., :1, :])), -2)[..., self.bodies, :]
        transformed = rotation @ self.frames
        frames = torch.cat((transformed[..., :3], transformed[..., 3:] + origin.unsqueeze(-1)), -1)
        paired = frames.index_select(-3, self.indices).unflatten(-3, (-1, 2))
        if verify:
            return _query(paired, self.pairs, True, clearance)[..., 0]
        return _Distance.apply(paired, self.pairs)[0]
