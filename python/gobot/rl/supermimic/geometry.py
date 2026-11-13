# Adapted from Motphys/Uni-SuperMimic, faf7818e8abac3ec6abb8f00e6e904d3577f800b.
# SPDX-License-Identifier: Apache-2.0
"""Matrix-to-quaternion conversion for scene FK."""

import torch
import torch.nn.functional as F


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    qw = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-10)) / 2
    qx = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-10)) / 2
    qy = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=1e-10)) / 2
    qz = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=1e-10)) / 2
    candidates = torch.stack(
        (
            torch.stack(
                (
                    qw,
                    (m21 - m12) / (4 * qw),
                    (m02 - m20) / (4 * qw),
                    (m10 - m01) / (4 * qw),
                ),
                -1,
            ),
            torch.stack(
                (
                    (m21 - m12) / (4 * qx),
                    qx,
                    (m01 + m10) / (4 * qx),
                    (m02 + m20) / (4 * qx),
                ),
                -1,
            ),
            torch.stack(
                (
                    (m02 - m20) / (4 * qy),
                    (m01 + m10) / (4 * qy),
                    qy,
                    (m12 + m21) / (4 * qy),
                ),
                -1,
            ),
            torch.stack(
                (
                    (m10 - m01) / (4 * qz),
                    (m02 + m20) / (4 * qz),
                    (m12 + m21) / (4 * qz),
                    qz,
                ),
                -1,
            ),
        ),
        dim=-2,
    )
    indices = torch.stack((qw, qx, qy, qz), -1).argmax(-1)
    q = torch.gather(
        candidates, -2, indices[..., None, None].expand(indices.shape + (1, 4))
    ).squeeze(-2)
    q = F.normalize(q, dim=-1)
    return q * torch.where(q[..., :1] < 0, -1.0, 1.0)
