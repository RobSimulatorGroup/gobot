"""Worker-side translation of native libuipc output to selected scene snapshots."""
from __future__ import annotations

from ..sim.scene_snapshot import SceneStateOutput
from ._libuipc_provider import _affine_transforms_to_poses


class LibuipcSceneOutput:
    def __init__(self, provider, *, affine_paths):
        self.provider = provider
        offsets = {entry["path"]: int(entry["element_offset"]) for entry in provider.affine_bodies}
        if len(set(affine_paths)) != len(affine_paths) or set(affine_paths) != set(offsets):
            raise ValueError("affine output paths must match the compiled solver layout")
        self.affine_indices = [offsets[path] for path in affine_paths]
        self.deformable = SceneStateOutput(provider,
            deformables=provider.artifact.deformable_bodies,
            source_bodies=provider.deformable_bodies)

    def snapshot(self, fields, environments):
        import numpy as np

        requested = set(fields)
        if requested - {"affine.link_pose", "deformable.local_vertices", "positions", "contact_forces"}:
            raise KeyError("unsupported native libuipc snapshot field")
        if any(env != 0 for env in environments) or len(environments) > 1:
            raise ValueError("native libuipc output has one environment")
        result = {}
        if "deformable.local_vertices" in requested:
            result.update(self.deformable.snapshot(["deformable.local_vertices"], environments))
        if "affine.link_pose" in requested:
            transforms = self.provider.arrays["affine_transforms"][self.affine_indices]
            result["affine.link_pose"] = _affine_transforms_to_poses(transforms)[None][list(environments)]
        for field in requested & {"positions", "contact_forces"}:
            result[field] = np.asarray(self.provider.arrays[field])[None][list(environments)]
        return result
