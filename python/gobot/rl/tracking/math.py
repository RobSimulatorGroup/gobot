# Adapted from mjlab (Apache-2.0); modified for Gobot's tensor-only task boundary.
# See LICENSE-mjlab. Quaternions in this module are wxyz.
import torch


def conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def multiply(a, b):
    w, v = a[..., :1], a[..., 1:]
    s, u = b[..., :1], b[..., 1:]
    return torch.cat(
        (
            w * s - (v * u).sum(-1, keepdim=True),
            w * u + s * v + torch.linalg.cross(v, u),
        ),
        -1,
    )


def apply(q, v):
    xyz, vector = torch.broadcast_tensors(q[..., 1:], v)
    t = 2 * torch.linalg.cross(xyz, vector)
    return vector + q[..., :1] * t + torch.linalg.cross(xyz, t)


def inverse_apply(q, v):
    return apply(conjugate(q), v)


def matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(q.shape[:-1] + (3, 3))


def angle_error(a, b):
    q = multiply(a, conjugate(b))
    return 2 * torch.atan2(q[..., 1:].norm(dim=-1), q[..., 0].abs())


def yaw(q):
    w, x, y, z = q.unbind(-1)
    a = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) / 2
    zero = torch.zeros_like(a)
    return torch.stack((a.cos(), zero, zero, a.sin()), -1)


def from_euler(xyz):
    r, p, y = (xyz / 2).unbind(-1)
    cr, sr, cp, sp, cy, sy = r.cos(), r.sin(), p.cos(), p.sin(), y.cos(), y.sin()
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        -1,
    )


def quaternion(m):
    """Stable wxyz conversion, selecting the largest diagonal component."""
    a, b, c = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    magnitudes = (
        torch.stack((1 + a + b + c, 1 + a - b - c, 1 - a + b - c, 1 - a - b + c), -1)
        .clamp_min(0)
        .sqrt()
    )
    rows = torch.stack(
        (
            torch.stack(
                (
                    magnitudes[..., 0].square(),
                    m[..., 2, 1] - m[..., 1, 2],
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] - m[..., 0, 1],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 2, 1] - m[..., 1, 2],
                    magnitudes[..., 1].square(),
                    m[..., 0, 1] + m[..., 1, 0],
                    m[..., 0, 2] + m[..., 2, 0],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 0, 1] + m[..., 1, 0],
                    magnitudes[..., 2].square(),
                    m[..., 1, 2] + m[..., 2, 1],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 1, 0] - m[..., 0, 1],
                    m[..., 0, 2] + m[..., 2, 0],
                    m[..., 1, 2] + m[..., 2, 1],
                    magnitudes[..., 3].square(),
                ),
                -1,
            ),
        ),
        -2,
    )
    rows = rows / (2 * magnitudes.clamp_min(0.1)[..., None])
    index = magnitudes.argmax(-1)[..., None, None].expand(*m.shape[:-2], 1, 4)
    q = rows.gather(-2, index).squeeze(-2)
    return q / q.norm(dim=-1, keepdim=True)
