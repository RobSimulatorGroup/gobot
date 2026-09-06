import gc
import json

import gobot
import numpy as np
import pytest


def _write_scene(path, name):
    path.write_text(json.dumps({
        "__VERSION__": 3, "__META_TYPE__": "SCENE", "__TYPE__": "PackedScene",
        "__EXT_RESOURCES__": [], "__SUB_RESOURCES__": [],
        "__NODES__": [{"type": "Node3D", "name": name, "parent": -1, "properties": {}}],
    }))


def test_independent_contexts_resolve_same_authored_path(tmp_path):
    contexts = []
    for name in ("first", "second"):
        project = tmp_path / name
        project.mkdir()
        _write_scene(project / "world.jscn", name)
        context = gobot.app.create_context()
        context.set_project_path(str(project))
        context.load_scene("res://world.jscn")
        contexts.append(context)
    for _ in range(3):
        for context, name in zip(contexts, ("first", "second")):
            assert context.project_path == str(tmp_path / name)
            context.load_scene("res://world.jscn")
            assert context.root.name == name
    for context in contexts:
        context.clear_scene()


def test_native_arrays_retain_an_immutable_frame_after_reset_and_close(tmp_path):
    author = gobot.app.context()
    author.set_project_path(str(tmp_path))
    root = gobot.create_node("Node3D", "world")
    root.add_child(gobot.scene.create_cartpole_scene())
    body = gobot.create_node("DeformableBody3D", "soft")
    mesh = gobot.TetrahedralMesh()
    mesh.vertices = [(0, 0, 0), (0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.1)]
    mesh.tetrahedra = [(0, 1, 2, 3)]
    body.mesh = mesh
    root.add_child(body)
    gobot.save_scene(root, "res://frame.jscn")
    context = gobot.app.create_context()
    context.set_project_path(str(tmp_path))
    context.load_scene("res://frame.jscn")
    context.build_world(gobot.PhysicsBackendType.Null)
    view = context.get_physics_state_view()
    old = context.get_physics_state()
    vertices = view["deformables"][0]["local_vertices"]
    joints = view["robots"][0]["joint_position"]
    np.testing.assert_allclose(vertices, old["deformables"][0]["local_vertices"])
    assert vertices.shape == (4, 3)
    assert view["deformables"][0]["contact_forces_world"].shape == (0, 3)
    assert vertices.flags.writeable is False
    assert joints.flags.writeable is False
    with pytest.raises(ValueError):
        vertices[0, 0] = 10
    with pytest.raises(ValueError):
        vertices.setflags(write=True)
    retained = vertices.copy()
    context.step(2)
    advanced = context.get_physics_state_view()
    assert advanced["tick"] == view["tick"] + 2
    context.reset_simulation()
    assert context.get_physics_state_view()["epoch"] != view["epoch"]
    context.clear_scene()
    del context, view, advanced
    gc.collect()
    np.testing.assert_array_equal(vertices, retained)
