# SPDX-License-Identifier: Apache-2.0
# Option/scratch handling adapted from Newton 1.7.0, Copyright (c) 2025 The Newton Developers.
"""Local Newton 1.7 solver adaptation for Gobot's MuJoCo/Warp 3.17 stack.

MuJoCo Warp 3.17 removed solver mean-inertia statistics. Keep Newton's
parameter update and graph storage contracts without manufacturing that field.
No installed module or global SolverMuJoCo class is modified.
"""
from __future__ import annotations

from functools import lru_cache


def solver_type(newton, mujoco, mujoco_warp):
    base = newton.solvers.SolverMuJoCo
    if str(getattr(mujoco_warp, "__version__", "")) != "3.17.0":
        return base
    if str(getattr(newton, "__version__", "")) != "1.7.0" or mujoco.mj_versionString() != "3.17.0":
        raise RuntimeError("Gobot's MuJoCo 3.17 Newton adapter requires Newton 1.7.0 and matching MuJoCo/Warp 3.17.0")
    return _adapt(base)


@lru_cache(maxsize=1)
def _adapt(base):
    import warp as wp
    from newton._src.solvers.mujoco.kernels import repeat_array_kernel

    class MuJoCo317Solver(base):
        # This class has the explicit version check above; the upstream check
        # only knows Newton's older optional [sim] dependency set.
        _versions_checked = True

        def __init__(self, model, **kwargs):
            super().__init__(model, **kwargs)
            # Newton 1.7 adds a final reset-mask entry for global entities.
            # Gobot selects authored worlds only; keep the global entry false.
            self._gobot_world_mask = wp.zeros(model.world_count + 1, dtype=wp.bool, device=model.device)

        def reset(self, state, world_mask=None):
            if world_mask is not None and world_mask.shape == (self.model.world_count,):
                wp.copy(self._gobot_world_mask, world_mask, count=self.model.world_count)
                world_mask = self._gobot_world_mask
            return super().reset(state, world_mask)

        def _expand_option_fields(self, model, nworld):
            if nworld == 1:
                return
            fields = ("impratio_invsqrt", "tolerance", "ls_tolerance", "ccd_tolerance",
                      "sleep_tolerance", "density", "viscosity", "gravity", "wind", "magnetic")
            for name in fields:
                source = getattr(model.opt, name)
                shape = (nworld, *source.shape[1:])
                target = wp.empty(shape=shape, dtype=source.dtype, device=source.device)
                src, dst = source.flatten(), target.flatten()
                wp.launch(repeat_array_kernel, dim=dst.shape[0],
                          inputs=[src, dst.shape[0] // nworld], outputs=[dst], device=source.device)
                setattr(model.opt, name, target)

        def _allocate_notify_model_changed_scratch(self):
            self._notify_body_flags = wp.clone(self.model.body_flags)
            self._notify_joint_armature = wp.clone(self.model.joint_armature)
            solref = getattr(getattr(self.model, "mujoco", None), "solreflimit", None)
            self._joint_limit_solref_snapshot = wp.clone(solref) if solref is not None else None
            if not self.use_mujoco_cpu:
                self._notify_qpos_saved = wp.empty_like(self.mjw_data.qpos)
                for name in ("_joint_limit_ke_snapshot", "_joint_limit_kd_snapshot", "_solreflimit_mode_snapshot"):
                    snapshot = getattr(self, name)
                    if snapshot is not None:
                        setattr(self, name, wp.array(snapshot, device=self.model.device))
            if not (self.has_connect_constraints or self.has_jnt_connect_constraints):
                return
            device = self.model.device
            self._notify_ref_q = wp.zeros(self.model.joint_coord_count, dtype=wp.float32, device=device)
            self._notify_ref_qd = wp.zeros(self.model.joint_dof_count, dtype=wp.float32, device=device)
            self._notify_ref_body_q = wp.zeros(self.model.body_count, dtype=wp.transform, device=device)
            self._notify_ref_body_qd = wp.zeros(self.model.body_count, dtype=wp.spatial_vector, device=device)
            if self.has_connect_constraints:
                neq = self.model.mujoco.equality_constraint_count
                self.connect_constraint_q_rel = wp.zeros(neq, dtype=wp.quat, device=device)
                self.connect_constraint_t_rel = wp.zeros(neq, dtype=wp.vec3, device=device)
            if self.has_jnt_connect_constraints:
                shape = (self.mjc_eq_to_newton_jnt.shape[0], self.mjw_model.neq)
                self.jnt_connect_constraint_q_rel = wp.zeros(shape, dtype=wp.quat, device=device)
                self.jnt_connect_constraint_t_rel = wp.zeros(shape, dtype=wp.vec3, device=device)

        def _set_const_0_with_physical_meaninertia(self):
            if self.use_mujoco_cpu:
                return super()._set_const_0_with_physical_meaninertia()
            # 3.17 no longer scales convergence by stat.meaninertia. The
            # physical/kinematic statistic pre-pass is therefore unnecessary.
            self._update_body_properties(apply_kinematic_armature=True)
            self._mujoco_warp.set_const_0(self.mjw_model, self.mjw_data)

    return MuJoCo317Solver
