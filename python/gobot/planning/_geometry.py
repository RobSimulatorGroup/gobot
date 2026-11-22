"""Batched Torch distance kernels for the planner's collision primitives."""

import torch
from torch import Tensor


def clearance_mask(distances, margin):
    minimum = (torch.as_tensor(margin, dtype=distances.dtype, device=distances.device) - 1e-6).clamp_min(0)
    return torch.isfinite(distances) & (distances >= minimum)


def _dot(left: Tensor, right: Tensor) -> Tensor:
    return (left * right).sum(dim=-1)


def _nearest_point(start: Tensor, end: Tensor, point: Tensor) -> Tensor:
    direction = end - start
    length_squared = _dot(direction, direction)
    fraction = _dot(point - start, direction) / length_squared.clamp_min(torch.finfo(start.dtype).tiny)
    return start + fraction.clamp(0, 1)[..., None] * direction


def capsule_distance(a: Tensor, b: Tensor, ra: Tensor, c: Tensor, d: Tensor, rb: Tensor) -> Tensor:
    # A sphere is a capsule whose endpoints coincide. The minimum is either
    # on the boundary of the segment parameters or at the closest line pair.
    u, v, offset = b - a, d - c, c - a
    normal = torch.linalg.cross(u.expand_as(offset), v.expand_as(offset))
    denom = _dot(normal, normal)
    safe_denom = torch.where(denom > 0, denom, 1)
    s = _dot(torch.linalg.cross(offset, v.expand_as(offset)), normal) / safe_denom
    t = _dot(torch.linalg.cross(offset, u.expand_as(offset)), normal) / safe_denom
    interior = (denom > 0) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    line_delta = a + s.clamp(0, 1)[..., None] * u - c - t.clamp(0, 1)[..., None] * v
    edge_deltas = torch.stack((
        a - _nearest_point(c, d, a), b - _nearest_point(c, d, b),
        c - _nearest_point(a, b, c), d - _nearest_point(a, b, d),
    ), dim=-2)
    edge_distance = torch.linalg.vector_norm(edge_deltas, dim=-1).amin(dim=-1)
    line_distance = torch.linalg.vector_norm(line_delta, dim=-1)
    return torch.where(interior, torch.minimum(edge_distance, line_distance), edge_distance) - ra - rb


def plane_distance(a: Tensor, b: Tensor, radius: Tensor, point: Tensor, normal: Tensor) -> Tensor:
    return torch.minimum(_dot(a - point, normal), _dot(b - point, normal)) - radius


def box_distance(
    a: Tensor, b: Tensor, radius: Tensor, center: Tensor, rotation: Tensor, half_size: Tensor,
) -> Tensor:
    inverse = rotation.transpose(-1, -2)
    start = (inverse @ (a - center)[..., None]).squeeze(-1)
    direction = (inverse @ (b - a)[..., None]).squeeze(-1)

    # Between crossings of the six AABB face planes, squared exterior distance
    # is quadratic in the segment parameter. Minimize it on each interval.
    moving = direction != 0
    safe_direction = torch.where(moving, direction, 1)
    crossings = torch.cat(((-half_size - start) / safe_direction, (half_size - start) / safe_direction), -1)
    crossings = torch.where(torch.cat((moving, moving), -1), crossings.clamp(0, 1), 0)
    zero = torch.zeros_like(start[..., :1])
    breaks = torch.cat((zero, crossings, zero + 1), -1).sort(dim=-1).values
    lo, hi = breaks[..., :-1], breaks[..., 1:]
    midpoint = (lo + hi) * 0.5
    delta = start[..., None, :] + midpoint[..., None] * direction[..., None, :]
    exterior = delta - delta.clamp(-half_size[..., None, :], half_size[..., None, :])
    slope = torch.where(exterior != 0, direction[..., None, :], 0)
    denom = _dot(slope, slope)
    step = _dot(exterior, slope) / torch.where(denom > 0, denom, 1)
    closest = (midpoint - step).clamp(lo, hi)

    # An interior midpoint gives a negative distance and a useful escape
    # gradient even when the capsule segment crosses through the box.
    points = start[..., None, :] + closest[..., None] * direction[..., None, :]
    faces = points.abs() - half_size[..., None, :]
    outside = torch.linalg.vector_norm(faces.clamp_min(0), dim=-1)
    inside = faces.amax(dim=-1).clamp_max(0)
    return (outside + inside).amin(dim=-1) - radius


def rotation_log(matrix: Tensor) -> Tensor:
    """Shortest rotation vector, including well-conditioned angles near pi."""
    xx, xy, xz = matrix[..., 0, :].unbind(-1)
    yx, yy, yz = matrix[..., 1, :].unbind(-1)
    zx, zy, zz = matrix[..., 2, :].unbind(-1)
    w = 1 + xx + yy + zz
    x = 1 + xx - yy - zz
    y = 1 - xx + yy - zz
    z = 1 - xx - yy + zz
    candidates = torch.stack((
        torch.stack((w, zy - yz, xz - zx, yx - xy), -1),
        torch.stack((zy - yz, x, xy + yx, xz + zx), -1),
        torch.stack((xz - zx, xy + yx, y, yz + zy), -1),
        torch.stack((yx - xy, xz + zx, yz + zy, z), -1),
    ), -2)
    largest = torch.stack((w, x, y, z), -1).argmax(-1, keepdim=True)
    quaternion = candidates.gather(-2, largest[..., None].expand(*largest.shape, 4)).squeeze(-2)
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    scalar, vector = quaternion[..., :1], quaternion[..., 1:]
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    # The selected quaternion row is proportional to a unit quaternion. The
    # small-angle branch accounts for that proportionality without dividing by zero.
    epsilon = torch.finfo(matrix.dtype).eps
    exact = 2 * torch.atan2(norm, scalar) / norm.clamp_min(epsilon)
    scale = torch.where(norm > epsilon, exact, 2 / scalar.clamp_min(epsilon))
    return vector * scale
