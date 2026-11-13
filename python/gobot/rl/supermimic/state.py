# Adapted from HybridRobotics/supermimic, abc124c1bd5976341034668af3d12c087f2f1f55.
# Copyright 2026 The SuperMimic Authors. SPDX-License-Identifier: Apache-2.0
# See LICENSE-SuperMimic. Gobot adaptation: backend-neutral tensor interfaces.
"""GenCoord tensor contract: world-frame full-state velocities, wxyz quaternions.

G1: full=71, representation=68, prediction=35. The FK callable, when needed,
accepts (root_position, root_quaternion, joints, world_linear_velocity,
world_angular_velocity, joint_velocity) and returns body poses and velocities.
It is injected by the caller; no simulator or editor objects are owned here.
"""

from __future__ import annotations
import math
from typing import Protocol
import torch
from torch import nn
from .rotation_primitives import matrix_from_quat, quat_apply, quat_apply_inverse
from .rotations import quat_integrate, quat_to_rot6d, subtract_frame_transforms


class MotionWindow(Protocol):
    motion_ends: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor


NUM_GOAL_FRAMES = 10
GOAL_FRAME_STRIDE = 5
GOAL_REL_POS_DIM = 3 * NUM_GOAL_FRAMES
GOAL_REL_ORI_DIM = 6 * NUM_GOAL_FRAMES
PROPRIO_NOISE_HALF_WIDTHS = {
    "projected_gravity": 0.05,
    "base_ang_vel": 0.2,
    "joint_pos": 0.01,
    "joint_vel": 0.5,
    "actions": 0.0,
}


def infer_num_joints_from_state_dim(state_dim: int) -> int:
    """Infers number of joints from full-state dimension.

    Full-state layout is:
    [root_pos(3), root_quat(4), qj(N), root_lin_vel(3), root_ang_vel(3), vj(N)]
    """
    if state_dim < 13 or (state_dim - 13) % 2 != 0:
        raise ValueError(
            f"Invalid full-state dim {state_dim}: expected 13 + 2 * num_joints"
        )
    return (state_dim - 13) // 2


def rep_state_dim_from_num_joints(num_joints: int) -> int:
    return 10 + 2 * num_joints


def full_state_offsets(num_joints: int) -> tuple[int, int, int, int]:
    """Returns start offsets for [qj, root_lin_vel, root_ang_vel, vj] in full state."""
    idx_qj = 7
    idx_lin = idx_qj + num_joints
    idx_ang = idx_lin + 3
    idx_vj = idx_ang + 3
    return idx_qj, idx_lin, idx_ang, idx_vj


def rep_state_offsets(num_joints: int) -> tuple[int, int]:
    """Returns start offsets for [qj, vj] in rep state."""
    idx_qj = 10
    idx_vj = idx_qj + num_joints
    return idx_qj, idx_vj


def decompose_state(
    full_state: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Decomposes full state into canonical components."""
    num_joints = infer_num_joints_from_state_dim(full_state.shape[-1])

    idx_qj, idx_lin, idx_ang, idx_vj = full_state_offsets(num_joints)

    root_pos = full_state[..., 0:3]
    root_quat = full_state[..., 3:7]
    qj = full_state[..., idx_qj:idx_lin]
    root_lin_vel = full_state[..., idx_lin:idx_ang]
    root_ang_vel = full_state[..., idx_ang:idx_vj]
    vj = full_state[..., idx_vj : idx_vj + num_joints]
    return root_pos, root_quat, qj, root_lin_vel, root_ang_vel, vj


def decompose_rep_state(
    rep_state: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Decomposes gencoord rep state into canonical components."""
    # rep state layout: [z(1), g_b(3), v_b(3), w_b(3), qj(N), vj(N)]
    if rep_state.shape[-1] < 10 or (rep_state.shape[-1] - 10) % 2 != 0:
        raise ValueError(
            f"Invalid rep-state dim {rep_state.shape[-1]}: expected 10 + 2 * num_joints"
        )
    num_joints = (rep_state.shape[-1] - 10) // 2

    idx_qj, idx_vj = rep_state_offsets(num_joints)

    z = rep_state[..., 0:1]
    g_b = rep_state[..., 1:4]
    v_b = rep_state[..., 4:7]
    w_b = rep_state[..., 7:10]
    qj = rep_state[..., idx_qj:idx_vj]
    vj = rep_state[..., idx_vj : idx_vj + num_joints]
    return z, g_b, v_b, w_b, qj, vj


def compute_gencoord_state(full_state: torch.Tensor) -> torch.Tensor:
    """Converts raw full state to gencoord representation.

    Uses root body frame for velocities (no heading alignment).
    This avoids all singularities since we're just rotating to body frame.
    """
    (
        root_pos,
        root_quat,
        qj,
        root_lin_vel,
        root_ang_vel,
        vj,
    ) = decompose_state(full_state)

    z = root_pos[..., 2:3]
    world_down = torch.zeros_like(root_lin_vel)
    world_down[..., 2] = -1.0
    g_b = quat_apply_inverse(root_quat, world_down)
    v_b = quat_apply_inverse(root_quat, root_lin_vel)
    w_b = quat_apply_inverse(root_quat, root_ang_vel)

    return torch.cat([z, g_b, v_b, w_b, qj, vj], dim=-1)


def actor_goal_dim(num_goals: int) -> int:
    """Number of goal dimensions seen by the actor (wm_goal minus rel_pos_b)."""
    return num_goals - GOAL_REL_POS_DIM


def term_major_to_temporal(
    flat: torch.Tensor, term_widths: list[int], history_length: int
) -> torch.Tensor:
    """Convert mjlab's term-major flat obs to a ``(B, H, sum(w_i))`` frame-major window.

    Input layout ``[term0_t0..tH-1, term1_t0..tH-1, ...]`` is what the deployed
    actor obs group emits; the decoder consumes frame-major. Applied at the ONNX
    export boundary so deploy and training see the same window.
    """
    parts = []
    off = 0
    for w in term_widths:
        chunk = flat[..., off : off + history_length * w]
        parts.append(chunk.unflatten(-1, (history_length, w)))
        off += history_length * w
    return torch.cat(parts, dim=-1)


def extract_actor_goal(wm_goal: torch.Tensor) -> torch.Tensor:
    """Extract actor goal from full WM goal by removing rel_pos_b dims."""
    tail = GOAL_REL_POS_DIM + GOAL_REL_ORI_DIM
    return torch.cat([wm_goal[..., :-tail], wm_goal[..., -GOAL_REL_ORI_DIM:]], dim=-1)


def _root_loss_blocks(aux_state: torch.Tensor) -> dict[str, torch.Tensor]:
    """World root position (3) and rotation (9) as squared-error channels.

    The rotation is scaled by sqrt(1/2) so a plain squared difference is
    0.5*||dR||_F^2 = 2(1 - cos theta) = theta^2 + O(theta^4): a nominal in radians.
    """
    rot = matrix_from_quat(aux_state[..., 3:7]).flatten(-2, -1)
    return {"root_pos": aux_state[..., :3], "root_ori": rot * math.sqrt(0.5)}


def _assemble_goal(
    content: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble ``(wm_goal, actor_goal)``; only wm_goal carries the rel_pos dims."""
    rel_pos, rel_ori = assemble_anchor_window(
        root_pos, root_quat, target_pos, target_quat
    )
    wm_goal = torch.cat([content, rel_pos, rel_ori], dim=-1)
    actor_goal = torch.cat([content, rel_ori], dim=-1)
    return wm_goal, actor_goal


def goal_window_indices(
    motion: MotionWindow, motion_times: torch.Tensor
) -> torch.Tensor:
    """Clamped windowed frame indices ``[t, t+stride, ..., t+(N-1)*stride]``, ``(E, N)``.

    Future offsets are clamped to the last frame of each env's current clip so the
    window never bleeds into the next concatenated clip — matching how the env
    advances ``motion_times``.
    """
    clip = torch.searchsorted(motion.motion_ends, motion_times, right=True)
    # Past-the-end guard: searchsorted returns numel() at the final frame.
    clip = clip.clamp(max=motion.motion_ends.numel() - 1)
    last_valid = motion.motion_ends[clip] - 1
    offsets = (
        torch.arange(NUM_GOAL_FRAMES, device=motion_times.device) * GOAL_FRAME_STRIDE
    )
    return torch.minimum(motion_times[:, None] + offsets, last_valid[:, None])


def motion_goal_window(
    motion: MotionWindow, motion_times: torch.Tensor
) -> torch.Tensor:
    """Windowed joint targets as the goal content, ``(E, N * 2J)`` — per frame
    ``[joint_pos(J), joint_vel(J)]``.
    """
    idx = goal_window_indices(motion, motion_times)
    frames = torch.cat([motion.joint_pos[idx], motion.joint_vel[idx]], dim=-1)
    return frames.reshape(frames.shape[0], -1)


def motion_anchor_window(
    motion: MotionWindow, motion_times: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Motion anchor (the root) world pose ``(pos (E, N, 3), quat (E, N, 4))`` at
    the same windowed indices as :func:`motion_goal_window`.
    """
    idx = goal_window_indices(motion, motion_times)
    return motion.root_pos_w[idx], motion.root_quat_w[idx]


def assemble_anchor_window(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame motion anchor expressed relative to the robot's current root.

    ``root_*``: ``(E, 3)`` / ``(E, 4)``; ``target_*``: ``(E, N, 3)`` / ``(E, N, 4)``.
    Returns flattened ``(rel_pos (E, N*3), rel_ori_rot6d (E, N*6))``. The world-frame
    origin offset cancels in the relative transform, so callers only need the root
    and target anchors expressed in a common frame.
    """
    num_envs, num_frames = target_pos.shape[:2]
    root_pos = root_pos[:, None, :].expand(num_envs, num_frames, 3).reshape(-1, 3)
    root_quat = root_quat[:, None, :].expand(num_envs, num_frames, 4).reshape(-1, 4)
    rel_pos, rel_ori_w = subtract_frame_transforms(
        root_pos, root_quat, target_pos.reshape(-1, 3), target_quat.reshape(-1, 4)
    )
    rel_pos = rel_pos.reshape(num_envs, -1)
    rel_ori = quat_to_rot6d(rel_ori_w).reshape(num_envs, -1)
    return rel_pos, rel_ori


class GenCoordRepresentation(nn.Module):
    """Generalized-coordinate (joint-space) representation.

    Full state: [root_pos(3), root_quat(4), qj(N), root_lin_vel(3), root_ang_vel(3), vj(N)]
    Rep state:  [z(1), g_b(3), v_b(3), w_b(3), qj(N), vj(N)]
    Prediction: [Δv_b(3), Δw_b(3), Δvj(N)] — accel-level deltas in the current
    root frame; poses and derived channels are integrated analytically.
    """

    LOSS_WEIGHT_CONFIG = {
        "root_pos": "nominal_error_root_pose_pos",
        "root_ori": "nominal_error_root_pose_ori",
        "joint_pos": "nominal_error_joint_pos",
        "joint_vel": "nominal_error_joint_vel",
    }

    def __init__(
        self,
        num_joints: int,
        default_joint_pos: torch.Tensor,
        *,
        dt: float = 0.02,
        decimation: int = 4,
        actor_obs_mode: str = "rep_state",
    ):
        super().__init__()
        if (
            isinstance(num_joints, bool)
            or not isinstance(num_joints, int)
            or num_joints < 1
        ):
            raise ValueError("num_joints must be a positive integer")
        if default_joint_pos.shape != (num_joints,):
            raise ValueError("default_joint_pos must match num_joints")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if (
            isinstance(decimation, bool)
            or not isinstance(decimation, int)
            or decimation < 1
        ):
            raise ValueError("decimation must be a positive integer")
        if actor_obs_mode not in ("rep_state", "proprio"):
            raise ValueError("actor_obs_mode must be rep_state or proprio")
        self.dt = dt
        self.substep_coeff = (decimation + 1) / (2 * decimation)
        self.actor_obs_mode = actor_obs_mode
        self.num_joints = num_joints
        self.full_state_dim = 13 + 2 * num_joints
        self.rep_state_dim = rep_state_dim_from_num_joints(num_joints)
        self.prediction_dim = 6 + num_joints  # [Δv_b(3), Δw_b(3), Δvj(N)]
        self.register_buffer(
            "_default_joint_pos", default_joint_pos.clone(), persistent=False
        )
        self.register_buffer(
            "_proprio_noise_bounds",
            self.proprio_noise_bounds(num_joints, default_joint_pos.device),
            persistent=False,
        )
        self.fk = None

    def bind_kinematics(self, fk) -> None:
        """Use a scene-owned differentiable FK callable with world-frame velocities."""
        self.fk = fk

    def represent(self, full_state: torch.Tensor) -> torch.Tensor:
        flat = full_state.reshape(-1, self.full_state_dim)
        rep = compute_gencoord_state(flat)
        return rep.reshape(*full_state.shape[:-1], -1)

    def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
        return full_state[..., :7]

    def apply_prediction(
        self,
        aux_state: torch.Tensor,
        curr_rep_state: torch.Tensor,
        prediction: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Semi-implicit joint-space integration of accel-level predictions.

        The network output is ``[Δv_b, Δw_b, Δvj]`` in the CURRENT root
        frame; ``qj`` integrates the joint velocities, the root pose
        integrates the twist, and ``z``/``g_b`` are derived from the
        integrated pose, so every prediction maps to a kinematically
        consistent state. Pose updates weight the delta by ``substep_coeff``
        to match how the sim composes its substeps; velocity states advance
        by the full delta.
        """
        N = self.num_joints
        c = self.substep_coeff
        _, _, v_b, w_b, qj, vj = decompose_rep_state(curr_rep_state)
        dv_b = prediction[..., 0:3]
        dw_b = prediction[..., 3:6]
        dvj = prediction[..., 6 : 6 + N]

        next_v_cur = v_b + dv_b  # next twist, expressed in the current root frame
        next_w_cur = w_b + dw_b
        next_vj = vj + dvj
        next_qj = qj + (vj + c * dvj) * dt

        root_pos = aux_state[..., :3]
        root_quat = aux_state[..., 3:7]
        v_global = quat_apply(root_quat, next_v_cur)
        w_global = quat_apply(root_quat, next_w_cur)
        v_pose = quat_apply(root_quat, v_b + c * dv_b)
        w_pose = quat_apply(root_quat, w_b + c * dw_b)
        next_root_pos = root_pos + v_pose * dt
        next_root_quat = quat_integrate(root_quat, w_pose, dt)

        next_v_b = quat_apply_inverse(next_root_quat, v_global)
        next_w_b = quat_apply_inverse(next_root_quat, w_global)
        world_down = torch.zeros_like(v_global)
        world_down[..., 2] = -1.0
        next_g_b = quat_apply_inverse(next_root_quat, world_down)

        next_aux = torch.cat([next_root_pos, next_root_quat], dim=-1)
        next_rep = torch.cat(
            [next_root_pos[..., 2:3], next_g_b, next_v_b, next_w_b, next_qj, next_vj],
            dim=-1,
        )
        return next_aux, next_rep

    def reconstruct_full_state(
        self, aux_state: torch.Tensor, rep_state: torch.Tensor
    ) -> torch.Tensor:
        z, _, v_b, w_b, qj, vj = decompose_rep_state(rep_state)
        root_pos = aux_state[..., :3]
        root_quat = aux_state[..., 3:7]
        # Override root Z with rep_state's z
        root_pos = torch.cat([root_pos[..., :2], z], dim=-1)
        v_global = quat_apply(root_quat, v_b)
        w_global = quat_apply(root_quat, w_b)
        parts = [root_pos, root_quat, qj, v_global, w_global, vj]
        return torch.cat(parts, dim=-1)

    def loss_blocks(
        self, rep_state: torch.Tensor, aux_state: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # z, g_b and the root twist all follow from the integrated root pose.
        _, _, _, _, joint_pos, joint_vel = decompose_rep_state(rep_state)
        return {
            **_root_loss_blocks(aux_state),
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
        }

    def world_frame_bodies(
        self, aux_state: torch.Tensor, rep_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, v_b, w_b, qj, vj = decompose_rep_state(rep_state)
        root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
        if self.fk is None:
            raise RuntimeError("bind scene kinematics before computing body state")
        pos, quat, lin_vel, ang_vel = self.fk(
            root_pos,
            root_quat,
            qj,
            quat_apply(root_quat, v_b),
            quat_apply(root_quat, w_b),
            vj,
        )
        return pos, matrix_from_quat(quat), lin_vel, ang_vel

    def joint_state(self, rep_state: torch.Tensor) -> torch.Tensor:
        _, _, _, _, qj, vj = decompose_rep_state(rep_state)
        return torch.cat([qj, vj], dim=-1)

    def compute_wm_obs(
        self,
        rep_state: torch.Tensor,
        aux_state: torch.Tensor,
        joint_state: torch.Tensor,
        sim_ctx: torch.Tensor,
    ) -> torch.Tensor:
        del aux_state, sim_ctx
        _, g_b, _, w_b, _, _ = decompose_rep_state(rep_state)
        qj, vj = joint_state.chunk(2, dim=-1)
        return torch.cat([g_b, w_b, qj - self._default_joint_pos, vj], dim=-1)

    def proprio_obs_dim(self, num_actions: int) -> int:
        """Dim of the deploy-shape sensor proprio."""
        return sum(self.proprio_term_widths(num_actions))

    def proprio_term_widths(self, num_actions: int) -> list[int]:
        """Widths of ``[projected_gravity, base_ang_vel, joint_pos, joint_vel, actions]``,
        the sensor proprio in the order ``_sensor_actor_group`` concatenates it."""
        return [3, 3, self.num_joints, self.num_joints, num_actions]

    def proprio_noise_bounds(
        self, num_actions: int, device: torch.device | str
    ) -> torch.Tensor:
        """Per-dim uniform-noise half-widths matching the sensor proprio layout."""
        widths = self.proprio_term_widths(num_actions)
        halves = PROPRIO_NOISE_HALF_WIDTHS.values()
        return torch.cat(
            [
                torch.full((w,), h, device=device)
                for w, h in zip(widths, halves, strict=True)
            ]
        )

    def actor_obs_dim(
        self, num_actions: int, num_goals: int, history_length: int
    ) -> int:
        """Dim of the actor input — ``history_length`` sensor frames (proprio or
        rep_state) cat'd with the goal."""
        sensor_or_state = (
            self.rep_state_dim
            if self.actor_obs_mode == "rep_state"
            else self.proprio_obs_dim(num_actions)
        )
        return history_length * sensor_or_state + actor_goal_dim(num_goals)

    def add_proprio_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Add per-dim uniform proprio jitter — the single noise site shared by the
        synthesized sensor and the real-sim / imagination history rings."""
        return (
            obs + ((torch.rand_like(obs) * 2 - 1) * self._proprio_noise_bounds).detach()
        )

    def compute_actor_sensor(
        self,
        rep_state: torch.Tensor,
        sim_ctx: torch.Tensor,
        aux_state: torch.Tensor | None = None,
        joint_state: torch.Tensor | None = None,
        last_action: torch.Tensor | None = None,
        sim_actor_obs: torch.Tensor | None = None,
        noisy: bool = False,
    ) -> torch.Tensor:
        """Sensor-or-state half of the actor input (no goal).

        Returns ``rep_state`` in rep_state mode. In proprio mode, returns
        ``sim_actor_obs`` if provided (mjlab's ``obs["actor"]``) else
        synthesizes via :meth:`compute_wm_obs`; ``noisy=True`` adds per-dim
        uniform jitter scaled by the precomputed ``_proprio_noise_bounds``. ``aux_state``,
        ``joint_state`` and ``last_action`` are required only when synthesizing
        (proprio mode with no ``sim_actor_obs``).
        """
        if self.actor_obs_mode == "rep_state":
            return rep_state
        if sim_actor_obs is not None:
            obs = sim_actor_obs
        else:
            if aux_state is None or joint_state is None or last_action is None:
                raise ValueError(
                    "aux_state, joint_state and last_action are required to synthesize "
                    "proprio obs (no sim_actor_obs provided)"
                )
            obs = self.compute_wm_obs(rep_state, aux_state, joint_state, sim_ctx)
            obs = torch.cat([obs, last_action], dim=-1)
        if noisy:
            obs = self.add_proprio_noise(obs)
        return obs

    def build_actor_obs(
        self, sensor_history: torch.Tensor, actor_goal: torch.Tensor
    ) -> torch.Tensor:
        """Flatten the ``(B, history_length, sensor_dim)`` window and cat the goal."""
        return torch.cat([sensor_history.flatten(start_dim=1), actor_goal], dim=-1)

    def compute_wm_goal(
        self,
        motion: MotionWindow,
        motion_times: torch.Tensor,
        aux_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``(wm_goal, actor_goal)`` from the motion window at ``motion_times``.

        The single goal-build path shared by the WM env, the task goal extractor, and
        the viewer so every site produces an identical layout.
        """
        content = motion_goal_window(motion, motion_times)
        target_pos, target_quat = motion_anchor_window(motion, motion_times)
        root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
        return _assemble_goal(content, root_pos, root_quat, target_pos, target_quat)
