"""Selected provider output and scene-owner application of completed snapshots."""
from __future__ import annotations

import threading


class SceneStateOutput:
    """Worker-side exporter. Inputs are provider views and immutable authored data.

    Robot specs use Gobot robot/link/joint names. Deformable entries come from
    the compiled authoring artifact; source entries describe runtime offsets.
    Snapshot vertices are local to their authored DeformableBody3D.
    """
    def __init__(self, provider, *, robots=(), deformables=(), source_bodies=(),
                 positions_field="positions", refresh_positions=None):
        import numpy as np

        self.provider = provider
        self.robot_views = {}
        for spec in robots:
            options = dict(spec)
            field = options.pop("field")
            if field in self.robot_views:
                raise ValueError("duplicate robot snapshot field")
            self.robot_views[field] = provider.create_robot_view(**options)
        sources = {entry["path"]: entry for entry in source_bodies}
        self.deformables = []
        self.positions_field = positions_field
        self.refresh_positions = refresh_positions
        for entry in deformables:
            source = sources[entry["path"]]
            count = int(entry["vertex_count"])
            if count != int(source["element_count"]):
                raise ValueError("runtime deformable topology differs from the compiled artifact")
            transform = np.asarray(entry["transform"]["matrix_row_major"], dtype=np.float64).reshape(4, 4)
            self.deformables.append((int(source["element_offset"]), count, np.linalg.inv(transform)))
        self.max_vertices = max((count for _, count, _ in self.deformables), default=0)

    def snapshot(self, fields, environments):
        import numpy as np

        requested = set(fields)
        unknown = requested - self.robot_views.keys() - {"deformable.local_vertices"}
        if unknown:
            raise KeyError(f"unknown scene snapshot fields: {sorted(unknown)}")
        result = {}
        indices = list(environments)
        for field in requested & self.robot_views.keys():
            # View reads retain device arrays. Only selected environment poses
            # are copied for presentation, at the worker publication cadence.
            poses = self.robot_views[field].read_state().link_pose[indices]
            if hasattr(poses, "detach"):
                poses = poses.detach().cpu().numpy()
            result[field] = np.asarray(poses)
        if "deformable.local_vertices" in requested:
            if self.refresh_positions is not None:
                self.refresh_positions()
            positions = self.provider.arrays[self.positions_field]
            if positions.ndim == 2 and self.provider.num_envs == 1:
                positions = positions[None]
            positions = positions[indices]
            if hasattr(positions, "detach"):
                positions = positions.detach().cpu().numpy()
            positions = np.asarray(positions)
            output = np.zeros((len(indices), len(self.deformables), self.max_vertices, 3), dtype=np.float64)
            for body, (offset, count, inverse) in enumerate(self.deformables):
                output[:, body, :count] = positions[:, offset:offset + count] @ inverse[:3, :3].T + inverse[:3, 3]
            result["deformable.local_vertices"] = output
        return result


class SceneSnapshotSync:
    """Scene-owner consumer; never reads a provider or synchronizes a device.

    Bindings map each field to one sequence of links per environment. Soft
    bindings similarly contain one sequence of bodies per environment. Display
    offsets affect only rendered link poses; local soft vertices need no offset.
    """
    def __init__(self, context, *, links=None, deformables=(), vertex_counts=(), display_offsets=None):
        import numpy as np

        self._owner = threading.get_ident()
        self.context = context
        self.links = {name: tuple(tuple(row) for row in rows) for name, rows in (links or {}).items()}
        self.deformables = tuple(tuple(row) for row in deformables)
        self.vertex_counts = tuple(int(count) for count in vertex_counts)
        if any(len(row) != len(self.vertex_counts) for row in self.deformables):
            raise ValueError("deformable display bindings and vertex counts differ")
        self.offsets = None if display_offsets is None else np.asarray(display_offsets, dtype=np.float64).copy()
        if self.offsets is not None and (self.offsets.ndim != 2 or self.offsets.shape[1] != 3 or
                                         not np.isfinite(self.offsets).all()):
            raise ValueError("display_offsets must be a finite [environment,3] array")

    def __call__(self, snapshot):
        import numpy as np

        if threading.get_ident() != self._owner:
            raise RuntimeError("SceneSnapshotSync must run on the scene owner thread")
        environments = tuple(snapshot.environments)
        fields = set(snapshot.fields)
        links_to_apply = []
        poses_to_apply = []
        for field, rows in self.links.items():
            if field not in fields:
                continue
            poses = np.asarray(snapshot.buffer(field)).copy()
            if poses.ndim != 3 or poses.shape[0] != len(environments) or poses.shape[-1] != 7:
                raise ValueError("robot snapshot poses must have shape [environment,link,7]")
            for index, env in enumerate(environments):
                if len(rows[env]) != poses.shape[1]:
                    raise ValueError("robot snapshot layout differs from the display bindings")
                if self.offsets is not None:
                    poses[index, :, :3] += self.offsets[env]
                links_to_apply.extend(rows[env])
            poses_to_apply.append(poses.reshape(-1, 7))
        bodies = []
        local = None
        if self.deformables and "deformable.local_vertices" in fields:
            local = np.asarray(snapshot.buffer("deformable.local_vertices"))
            if local.ndim != 4 or local.shape[0] != len(environments) or local.shape[1] != len(self.vertex_counts) or local.shape[-1] != 3:
                raise ValueError("deformable snapshot layout differs from the display bindings")
            if max(self.vertex_counts, default=0) > local.shape[2]:
                raise ValueError("deformable snapshot is narrower than its vertex counts")
            for env in environments:
                bodies.extend(self.deformables[env])
        if local is not None:
            self.context.apply_deformable_vertices(bodies, local.reshape(len(bodies), local.shape[2], 3),
                    self.vertex_counts * len(environments), space="local")
        if links_to_apply:
            self.context.apply_link_poses(links_to_apply, np.concatenate(poses_to_apply))
