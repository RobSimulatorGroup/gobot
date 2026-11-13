# Adapted from HybridRobotics/supermimic, abc124c1bd5976341034668af3d12c087f2f1f55.
# Copyright 2026 The SuperMimic Authors. SPDX-License-Identifier: Apache-2.0
# See LICENSE-SuperMimic. Gobot adaptation: backend-neutral tensor interfaces.
"""Differentiable rotation helpers used by the official core."""

from __future__ import annotations
import torch
from gobot.rl.supermimic.rotation_primitives import (
    matrix_from_quat,
    quat_apply,
    quat_inv,
    quat_mul,
)


def small_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` without TF32 rounding, fusable by ``torch.compile``."""
    return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)


def rotation_error_magnitude(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """Angular distance between two rotation matrices, in radians.

    The clamp keeps ``cos_angle`` away from the ±1 acos singularity so the
    gradient stays finite for identical or 180°-apart inputs.

    Args:
      mat1: (..., 3, 3) rotation matrix
      mat2: (..., 3, 3) rotation matrix

    Returns:
      (...,) angle in radians
    """
    # trace(R1^T R2) via element-wise product (avoids materializing R1^T R2)
    tr = torch.sum(mat1 * mat2, dim=(-2, -1))
    cos_angle = torch.clamp((tr - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_angle)


def quat_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to 6D rotation representation (first two columns of rotation matrix)."""
    mat = matrix_from_quat(quat)  # (..., 3, 3)
    return mat[..., :2].flatten(-2, -1)  # (..., 6)


def quat_integrate(
    quat: torch.Tensor, ang_vel: torch.Tensor, dt: float
) -> torch.Tensor:
    """Integrate quaternion by angular velocity over dt (axis-angle method).

    Args:
      quat: (..., 4) current quaternion (w, x, y, z)
      ang_vel: (..., 3) angular velocity in world frame
      dt: time step

    Returns:
      (..., 4) next quaternion, normalized
    """
    omega = ang_vel * dt
    norm_sq = (omega * omega).sum(dim=-1, keepdim=True)
    norm_safe = torch.sqrt(norm_sq + 1e-8)
    half_angle = norm_safe / 2
    axis = omega / norm_safe
    delta_q = torch.cat([torch.cos(half_angle), torch.sin(half_angle) * axis], dim=-1)
    q_next = quat_mul(delta_q, quat)
    return q_next / torch.norm(q_next, dim=-1, keepdim=True)


def subtract_frame_transforms(root_pos, root_quat, target_pos, target_quat):
    inverse = quat_inv(root_quat)
    return quat_apply(inverse, target_pos - root_pos), quat_mul(inverse, target_quat)


def euler_quat(angles):
    """XYZ Euler angles to a wxyz quaternion for reset perturbations."""
    half = angles * 0.5
    c, s = half.cos(), half.sin()
    return torch.stack(
        (
            c[:, 0] * c[:, 1] * c[:, 2] + s[:, 0] * s[:, 1] * s[:, 2],
            s[:, 0] * c[:, 1] * c[:, 2] - c[:, 0] * s[:, 1] * s[:, 2],
            c[:, 0] * s[:, 1] * c[:, 2] + s[:, 0] * c[:, 1] * s[:, 2],
            c[:, 0] * c[:, 1] * s[:, 2] - s[:, 0] * s[:, 1] * c[:, 2],
        ),
        -1,
    )
