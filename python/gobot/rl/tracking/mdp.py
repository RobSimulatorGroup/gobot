# Adapted from mujocolab/mjlab dcdf6ce5, Apache-2.0. See LICENSE-mjlab.
# Modified: pure tensor functions over Gobot state, independent of mjlab managers.
"""Observations, rewards and termination predicates for whole-body tracking."""

import torch
from .math import apply, inverse_apply, conjugate, multiply, matrix, angle_error, yaw


def reanchor(reference, actual, anchor):
    p, q, *_ = reference
    ap, aq, *_ = actual
    delta = multiply(aq[:, anchor], conjugate(q[:, anchor]))
    delta = yaw(delta)[:, None]
    origin = torch.cat((ap[:, anchor, :2], p[:, anchor, 2:]), -1)
    return origin[:, None] + apply(delta, p - p[:, anchor, None]), multiply(delta, q)


def observations(
    reference,
    actual,
    joint_reference,
    q,
    qd,
    action,
    default,
    bias,
    *,
    anchor,
    imu,
    imu_position,
    imu_orientation,
    state_estimation=True,
):
    p, r, v, w = actual
    refp, refr, *_ = reference
    aq = r[:, anchor]
    fields = {
        "command": joint_reference,
        "motion_anchor_pos_b": inverse_apply(aq, refp[:, anchor] - p[:, anchor]),
        "motion_anchor_ori_b": matrix(multiply(conjugate(aq), refr[:, anchor]))[
            ..., :2
        ].flatten(1),
    }
    body = {
        "body_pos": inverse_apply(aq[:, None], p - p[:, anchor, None]).flatten(1),
        "body_ori": matrix(multiply(conjugate(aq[:, None]), r))[..., :2].flatten(1),
    }
    orientation = multiply(r[:, imu], imu_orientation.expand(q.shape[0], -1))
    sensor_offset = apply(r[:, imu], imu_position)
    tail = {
        "base_lin_vel": inverse_apply(
            orientation, v[:, imu] + torch.linalg.cross(w[:, imu], sensor_offset)
        ),
        "base_ang_vel": inverse_apply(orientation, w[:, imu]),
        "joint_pos": q - default,
        "joint_vel": qd,
        "actions": action,
    }
    critic = {**fields, **body, **tail}
    actor = {**fields, **tail, "joint_pos": tail["joint_pos"] + bias}
    if not state_estimation:
        actor.pop("motion_anchor_pos_b")
        actor.pop("base_lin_vel")
    return actor, critic


def rewards(
    reference,
    actual,
    relative,
    q,
    soft_limits,
    action,
    previous,
    self_hits,
    *,
    anchor,
    weights,
    dt,
):
    p, r, v, w = actual
    rp, rr, rv, rw = reference
    lp, lr = relative
    terms = dict(
        motion_global_root_pos=torch.exp(
            -(rp[:, anchor] - p[:, anchor]).square().sum(-1) / 0.3**2
        ),
        motion_global_root_ori=torch.exp(
            -angle_error(rr[:, anchor], r[:, anchor]).square() / 0.4**2
        ),
        motion_body_pos=torch.exp(-(lp - p).square().sum(-1).mean(-1) / 0.3**2),
        motion_body_ori=torch.exp(-angle_error(lr, r).square().mean(-1) / 0.4**2),
        motion_body_lin_vel=torch.exp(-(rv - v).square().sum(-1).mean(-1)),
        motion_body_ang_vel=torch.exp(-(rw - w).square().sum(-1).mean(-1) / 3.14**2),
        action_rate_l2=(action - previous).square().sum(-1),
        joint_limit=(
            (soft_limits[:, 0] - q).clamp_min(0) + (q - soft_limits[:, 1]).clamp_min(0)
        ).sum(-1),
        self_collisions=self_hits,
    )
    return {k: v * weights[k] * dt for k, v in terms.items()}


def terminations(reference, actual, relative, *, anchor, end_effectors):
    p, r, *_ = actual
    rp, rr, *_ = reference
    down = torch.zeros_like(p[:, anchor])
    down[:, 2] = -1
    return dict(
        anchor_pos=(rp[:, anchor, 2] - p[:, anchor, 2]).abs() > 0.25,
        anchor_ori=(
            inverse_apply(rr[:, anchor], down)[:, 2]
            - inverse_apply(r[:, anchor], down)[:, 2]
        ).abs()
        > 0.8,
        ee_body_pos=(relative[0][:, end_effectors, 2] - p[:, end_effectors, 2])
        .abs()
        .amax(-1)
        > 0.25,
    )
