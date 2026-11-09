"""Collision-checked placement for MuJoCo evaluation resets, without stepping."""
from __future__ import annotations

import numpy as np


class MuJoCoSpawnPlacement:
    """Lift a robot out of scene geometry while preserving its authored pose.

    CPU collision queries are shared by CPU and Warp evaluation. This helper is
    constructed lazily for offset spawns; normal training resets do not use it.
    """

    def __init__(self, artifact, *, robot_name, base_link, joint_names):
        import mujoco

        self._mj = mujoco
        self._model = mujoco.MjModel.from_xml_string(artifact.content)
        self._data = mujoco.MjData(self._model)
        prefix = artifact.robot_prefix(robot_name)
        root = self._model.body(prefix + base_link).id
        joint = int(self._model.body_jntadr[root])
        if self._model.body_jntnum[root] != 1 or self._model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("spawn placement requires a free robot root")
        self._root_qpos = int(self._model.jnt_qposadr[joint])
        self._joint_qpos = [int(self._model.jnt_qposadr[self._model.joint(prefix + name).id])
                            for name in joint_names]
        robot_bodies = {root}
        for body in range(root + 1, self._model.nbody):
            if int(self._model.body_parentid[body]) in robot_bodies:
                robot_bodies.add(body)
        self._robot_geoms = np.isin(self._model.geom_bodyid, list(robot_bodies))

    def _intersects_scene(self, height):
        self._data.qpos[self._root_qpos + 2] = height
        self._mj.mj_fwdPosition(self._model, self._data)
        return any(contact.dist < -1.e-6 and
                   self._robot_geoms[contact.geom[0]] != self._robot_geoms[contact.geom[1]]
                   for contact in self._data.contact)

    def positions(self, positions, orientations, joint_positions):
        """Return corrected positions; reject spawns needing more than 0.5 m lift."""
        result = np.asarray(positions, dtype=np.float32).copy()
        for row, (position, orientation, joints) in enumerate(zip(
                result, orientations, joint_positions, strict=True)):
            self._data.qpos[:] = self._model.qpos0
            address = self._root_qpos
            self._data.qpos[address:address + 3] = position
            self._data.qpos[address + 3:address + 7] = orientation
            self._data.qpos[self._joint_qpos] = joints
            initial = float(position[2])
            if not self._intersects_scene(initial):
                continue
            lift = .01
            while self._intersects_scene(initial + lift):
                if lift >= .5:
                    raise RuntimeError(f"evaluation spawn {row} cannot clear scene geometry within 0.5 m")
                lift = min(2 * lift, .5)
            low, high = initial, initial + lift
            for _ in range(14):
                mid = (low + high) * .5
                if self._intersects_scene(mid):
                    low = mid
                else:
                    high = mid
            result[row, 2] = high + .002
            if self._intersects_scene(float(result[row, 2])):
                raise RuntimeError(f"evaluation spawn {row} still intersects scene geometry")
        return result
