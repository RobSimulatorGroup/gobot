"""MuJoCo Warp SDK contact export for the engine's typed physics snapshots.

This adapter owns device buffers and copies the solver's completed contact
pairs. Sign conventions, per-body forces and all visualization belong to C++.
"""
from __future__ import annotations

import numpy as np


class MuJoCoWarpContactSource:
    def __init__(self, model, wp_model, wp_data, *, bodies, sample_dt, device):
        import warp as wp

        self.model, self.wp_model, self.wp_data = model, wp_model, wp_data
        self.sample_dt, self.device = float(sample_dt), device
        if not np.isfinite(self.sample_dt) or self.sample_dt <= 0:
            raise ValueError("contact sample timestep must be positive")
        self.capacity = wp_data.naconmax
        self.geometries = []
        for index in range(model.ngeom):
            body = model.body(int(model.geom_bodyid[index])).name
            robot, link = bodies.get(body, ("", body))
            self.geometries.append((robot, link, model.geom(index).name))
        with wp.ScopedDevice(device):
            self.ids = wp.array(np.arange(self.capacity, dtype=np.int32), dtype=wp.int32)
            self.force = wp.zeros(self.capacity, dtype=wp.spatial_vector)
        self.clear()

    def clear(self, environments=None):
        if environments is None:
            self.pairs = np.empty((0, 16), dtype=np.float64)
        else:
            self.pairs = self.pairs[~np.isin(self.pairs[:, 0], environments)]

    def capture(self):
        """Call after a physics substep, before any forward/reset overwrites it."""
        import warp as wp
        import mujoco_warp as mjw

        data = self.wp_data
        with wp.ScopedDevice(self.device):
            count = int(data.nacon.numpy()[0])
            if count < 0 or count > self.capacity:
                raise OverflowError("MuJoCo Warp contact capacity exceeded")
            if count == 0:
                self.clear()
                return
            # SDK returns force:torque in N and N.m, on geom B, in world frame.
            mjw.contact_force(self.wp_model, data, self.ids, True, self.force)
            wrench = self.force.numpy()[:count]
            geom = data.contact.geom.numpy()[:count]
            active = (data.contact.efc_address.numpy()[:count, 0] >= 0)
            active &= (geom >= 0).all(axis=1)
            self.pairs = np.column_stack((
                data.contact.worldid.numpy()[:count], geom,
                data.contact.pos.numpy()[:count], data.contact.frame.numpy()[:count, 0, :],
                wrench, data.contact.dist.numpy()[:count],
            ))[active].astype(np.float64, copy=False)

    def snapshot(self, environments):
        # Selection and strict validation are performed by the native channel.
        return dict(geometries=self.geometries, pairs=self.pairs, sample_dt=self.sample_dt)
