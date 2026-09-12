from types import SimpleNamespace

import numpy as np
import pytest

from gobot.sim import SceneSnapshotSync, SceneStateOutput


class Snapshot:
    def __init__(self, environments, buffers):
        self.environments = environments
        self.fields = tuple(buffers)
        self.buffers = buffers

    def buffer(self, name):
        return self.buffers[name]


def test_selected_output_converts_world_vertices_to_authored_local_space():
    transform = np.eye(4)
    transform[:3, :3] = np.diag([2, 3, 4])
    transform[:3, 3] = [10, 20, 30]
    local = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    world = local @ transform[:3, :3].T + transform[:3, 3]
    positions = np.zeros((4, 3, 3))
    positions[2, 1:] = world
    provider = SimpleNamespace(num_envs=4, arrays={"positions": positions})
    refreshes = []
    output = SceneStateOutput(provider,
        deformables=[{"path": "soft", "vertex_count": 2,
                      "transform": {"matrix_row_major": transform.ravel().tolist()}}],
        source_bodies=[{"path": "soft", "element_count": 2, "element_offset": 1}],
        refresh_positions=lambda: refreshes.append(True))
    assert output.snapshot([], [2]) == {}
    assert not refreshes
    buffers = output.snapshot(["deformable.local_vertices"], [2])
    np.testing.assert_allclose(buffers["deformable.local_vertices"], local[None, None])
    assert len(refreshes) == 1
    with pytest.raises(KeyError):
        output.snapshot(["unknown"], [2])


def test_selected_scene_sync_offsets_links_once_and_keeps_soft_vertices_local():
    calls = []
    context = SimpleNamespace(
        apply_link_poses=lambda *args: calls.append(("links", args)),
        apply_deformable_vertices=lambda *args, **kw: calls.append(("soft", args, kw)))
    sync = SceneSnapshotSync(context, links={"robot": [("link0",), ("link1",)]},
        deformables=[("body0",), ("body1",)], vertex_counts=[2],
        display_offsets=[[0, 0, 0], [10, 20, 30]])
    local = np.ones((1, 1, 2, 3))
    poses = np.array([[[1, 2, 3, 0, 0, 0, 1]]], dtype=float)
    sync(Snapshot([1], {"robot": poses, "deformable.local_vertices": local}))
    assert calls[0][1][0] == ["body1"]
    assert calls[0][2] == {"space": "local"}
    np.testing.assert_array_equal(calls[0][1][1], local.reshape(1, 2, 3))
    assert calls[1][1][0] == ["link1"]
    np.testing.assert_array_equal(calls[1][1][1][0, :3], [11, 22, 33])
    np.testing.assert_array_equal(poses[0, 0, :3], [1, 2, 3])
    calls.clear()
    with pytest.raises(ValueError):
        sync(Snapshot([1], {"robot": poses, "deformable.local_vertices": np.ones((1, 1, 1, 3))}))
    assert not calls
