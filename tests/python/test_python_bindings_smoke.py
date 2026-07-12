import json
import os
from pathlib import Path
import textwrap

import gobot
import numpy as np


def assert_close_tuple(actual, expected):
    if hasattr(actual, "tolist"):
        actual = actual.tolist()
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        assert abs(left - right) < 1e-9


def main():
    expected_build_dir = os.environ.get("GOBOT_TEST_BUILD_PYTHON_DIR")
    if expected_build_dir:
        core_path = Path(gobot._core.__file__).resolve()
        assert core_path.is_relative_to(Path(expected_build_dir).resolve()), core_path
    assert gobot.__version__ == gobot._core.__version__
    assert gobot.version() == gobot.__version__
    version_parts = [int(part) for part in gobot.__version__.split(".")[:3]]
    version_info = gobot.version_info()
    assert version_info["major"] == version_parts[0]
    assert version_info["minor"] == version_parts[1]
    assert version_info["patch"] == version_parts[2]
    assert isinstance(version_info["commit"], str)

    infos = gobot.backend_infos()
    assert infos
    assert any(info["name"] == "Null" and info["available"] for info in infos)
    assert any(info["name"] == "MuJoCo CPU" and info["robotics_focused"] for info in infos)
    mujoco_available = any(info["name"] == "MuJoCo CPU" and info["available"] for info in infos)
    assert gobot.PhysicsBackendType.Null == gobot.physics.PhysicsBackendType.Null
    assert gobot.sim.JointControllerGains is gobot.JointControllerGains
    assert gobot.scene.Node is gobot.Node
    assert gobot.scene.Terrain3D is gobot.Terrain3D
    assert "create_terrain_node" in gobot.terrain.__all__
    assert "BatchEnvState" in gobot.rl.__all__
    assert "BatchSimulationRuntime" in gobot.rl.__all__
    assert "CpuBatchEnv" in gobot.rl.__all__
    assert "ManagerBasedEnv" not in gobot.rl.__all__
    assert "VectorEnv" not in gobot.rl.__all__
    assert "velocity_actor_observation_schema" in gobot.rl.locomotion.__all__
    assert not hasattr(gobot._core, "NativeVectorEnv")

    action_spec = gobot.rl.ActionSpec(
        "smoke_action_v1",
        (gobot.rl.SpecField("left", 1), gobot.rl.SpecField("right", 1)),
    )
    assert action_spec.names == ("left", "right")
    assert action_spec.dim == 2
    assert action_spec.metadata()["dim"] == 2
    assert np.allclose(action_spec.clip([2.0, -2.0]), [1.0, -1.0])
    gobot.rl.validate_spec_metadata({"version": "smoke_action_v1", "dim": 2}, action_spec, kind="action")
    try:
        gobot.rl.validate_spec_metadata({"version": "wrong", "dim": 2}, action_spec, kind="action")
        raise AssertionError("spec metadata mismatch should fail")
    except RuntimeError as error:
        assert "version mismatch" in str(error)

    state = gobot.rl.BatchEnvState(
        obs={"actor": np.zeros((2, 3), dtype=np.float32)},
        reward=np.zeros(2, dtype=np.float32),
        terminated=np.asarray([False, True]),
        truncated=np.asarray([False, False]),
    )
    assert state.done.tolist() == [False, True]

    context = gobot.app.context()
    assert context.backend_type == gobot.PhysicsBackendType.Null
    assert context.has_scene is False
    assert context.has_world is False
    assert context.frame_count == 0
    assert not hasattr(gobot, "set_editor_tick_callback")
    assert not hasattr(gobot, "set_editor_physics_callback")
    assert not hasattr(gobot.app, "set_editor_tick_callback")
    assert not hasattr(gobot.app, "set_editor_physics_callback")

    try:
        context.build_world()
        raise AssertionError("build_world should fail without a loaded scene")
    except RuntimeError as error:
        assert "loaded scene" in str(error)

    reflected_gains = gobot.JointControllerGains()
    reflected_gains.position_stiffness = 12.0
    reflected_gains.velocity_damping = 1.5
    gains_dict = reflected_gains.to_dict()
    assert gains_dict["position_stiffness"] == 12.0
    reflected_gains_from_dict = gobot.JointControllerGains.from_dict(
        {"position_stiffness": 13.0, "velocity_damping": 2.5}
    )
    assert reflected_gains_from_dict.position_stiffness == 13.0
    assert reflected_gains_from_dict.velocity_damping == 2.5

    scene = gobot.create_test_scene()
    root = scene.root
    assert root.name == "robot"
    assert root.type == "Robot3D"
    assert root.type_name == "Robot3D"
    assert root.valid is True
    assert isinstance(root.id, int)
    assert root.child_count == 2
    assert [child.name for child in root.children] == ["base", "joint"]

    base = root.child(0)
    assert base.type == "Link3D"
    assert isinstance(base.position, np.ndarray)
    assert base.position.shape == (3,)
    assert_close_tuple(base.get("position"), (0.0, 0.0, 1.0))
    base.set("position", np.array([1.0, 2.0, 3.0]))
    assert_close_tuple(base.get("position"), (1.0, 2.0, 3.0))
    assert gobot.undo() is True
    assert_close_tuple(base.get("position"), (0.0, 0.0, 1.0))
    assert gobot.redo() is True
    assert_close_tuple(base.get("position"), (1.0, 2.0, 3.0))
    assert "position" in base.property_names()

    with gobot.transaction("Batch transform"):
        base.name = "renamed_base"
        base.position = (4.0, 5.0, 6.0)
        base.scale = (2.0, 2.0, 2.0)
    assert base.name == "renamed_base"
    assert_close_tuple(base.position, (4.0, 5.0, 6.0))
    assert_close_tuple(base.scale, (2.0, 2.0, 2.0))
    assert gobot.undo() is True
    assert base.name == "base"
    assert_close_tuple(base.position, (1.0, 2.0, 3.0))
    assert_close_tuple(base.scale, (1.0, 1.0, 1.0))
    assert gobot.redo() is True
    assert base.name == "renamed_base"

    authored = gobot.create_node("Robot3D", "authored")
    link = gobot.create_node("Link3D", "link")
    authored.add_child(link)
    collision = gobot.create_box_collision("collision", (0.2, 0.3, 0.4))
    link.add_child(collision)
    visual = gobot.create_box_visual("visual", (0.2, 0.3, 0.4))
    link.add_child(visual)
    imu = gobot.create_node("IMUSensor3D", "imu")
    imu.sensor_period = 0.01
    imu.noise_stddev = 0.02
    link.add_child(imu)
    contact = gobot.create_node("ContactSensor3D", "foot_contact")
    contact.radius = 0.05
    contact.min_threshold = 0.1
    contact.max_threshold = 10.0
    link.add_child(contact)
    angular_momentum = gobot.create_node("AngularMomentumSensor3D", "root_angmom")
    link.add_child(angular_momentum)
    assert authored.find("link/collision").name == "collision"
    assert authored.find("link/imu").type == "IMUSensor3D"
    assert authored.find("link/root_angmom").type == "AngularMomentumSensor3D"
    assert abs(authored.find("link/foot_contact").radius - 0.05) < 1e-6
    assert visual.type == "MeshInstance3D"

    context.clear_scene()
    assert context.scene_epoch > 0
    assert context.root is None
    assert base.valid is False
    try:
        _ = base.name
        raise AssertionError("old scene handle should raise ReferenceError after scene clear")
    except ReferenceError as error:
        assert "inactive scene epoch" in str(error)

    cartpole_root = gobot.scene.create_cartpole_scene()
    assert cartpole_root.name == "cartpole"
    assert cartpole_root.find("rail/slider/cart/hinge/pole").name == "pole"
    context.set_project_path("/tmp")
    context.clear_scene()
    gobot.save_scene(cartpole_root, "res://gobot_python_binding_cartpole.jscn")
    context.load_scene("res://gobot_python_binding_cartpole.jscn")
    if mujoco_available:
        compiled_artifact = context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
        assert compiled_artifact["format"] == "mjcf"
        assert compiled_artifact["robot_names"] == ["cartpole"]
        assert compiled_artifact["dimensions"]["nq"] == 2
        assert context.has_world is False
    context.build_world(gobot.PhysicsBackendType.Null)
    try:
        context.compiled_scene_artifact()
        raise AssertionError("Null backend must not expose a compiled scene artifact")
    except RuntimeError as error:
        assert "does not expose" in str(error)
    runtime_cartpole = context.root
    assert runtime_cartpole is not None
    slider = runtime_cartpole.find("rail/slider")
    hinge = runtime_cartpole.find("rail/slider/cart/hinge")
    assert slider.get_runtime_state()["name"] == "slider"
    assert hinge.get_runtime_state()["name"] == "hinge"
    assert not hasattr(context, "get_runtime_state")
    assert not hasattr(context, "get_runtime_name_map")
    assert not hasattr(context, "set_joint_position_target")
    assert not hasattr(runtime_cartpole, "get_runtime_state")
    assert not hasattr(runtime_cartpole, "set_joint_position_target")
    slider.set_position_target(0.0)
    slider.set_velocity_target(0.0)
    slider.set_effort_target(0.0)
    slider.set_passive()
    slider.reset_runtime_state(0.0, 0.0)

    eval_context = gobot.app.create_context()
    assert eval_context is not context
    eval_context.set_project_path("/tmp")
    eval_context.load_scene("res://gobot_python_binding_cartpole.jscn")
    eval_root = eval_context.root
    main_root = context.root
    assert eval_root is not None
    assert main_root is not None
    assert eval_root.id != main_root.id
    assert eval_root.name == main_root.name == "cartpole"
    eval_root.name = "eval_cartpole"
    assert eval_context.root.name == "eval_cartpole"
    assert context.root.name == "cartpole"
    eval_child = eval_context.root.find("rail")
    assert eval_child is not None
    assert eval_child.name == "rail"
    eval_context.build_world(gobot.PhysicsBackendType.Null)
    assert eval_context.has_world is True
    assert context.has_world is True
    stale_eval_root = eval_context.root
    eval_context.clear_scene()
    assert eval_context.root is None
    try:
        _ = stale_eval_root.name
        raise AssertionError("old eval context handle should raise ReferenceError after scene clear")
    except ReferenceError as error:
        assert "inactive scene epoch" in str(error)
    assert context.root.name == "cartpole"

    sensor_root = gobot.create_node("Robot3D", "sensor_bot")
    sensor_link = gobot.create_node("Link3D", "base")
    sensor_root.add_child(sensor_link)
    sensor_imu = gobot.create_node("IMUSensor3D", "imu")
    sensor_link.add_child(sensor_imu)
    sensor_angular_momentum = gobot.create_node("AngularMomentumSensor3D", "root_angmom")
    sensor_link.add_child(sensor_angular_momentum)
    sensor_contact = gobot.create_node("ContactSensor3D", "contact")
    sensor_link.add_child(sensor_contact)
    sensor_height = gobot.create_node("HeightScanner3D", "terrain_scan")
    sensor_height.position = (0.0, 0.0, 1.0)
    sensor_height.sample_offsets = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]
    sensor_height.ray_direction = (0.0, 0.0, -1.0)
    sensor_height.ray_direction_world_space = True
    sensor_height.max_distance = 2.0
    sensor_link.add_child(sensor_height)
    terrain_for_sensor = gobot.create_node("Terrain3D", "sensor_terrain")
    terrain_for_sensor.add_box((0.0, 0.0, -0.05), (2.0, 2.0, 0.1))
    sensor_world = gobot.create_node("Node3D", "sensor_world")
    sensor_world.add_child(sensor_root)
    sensor_world.add_child(terrain_for_sensor)
    context.clear_scene()
    gobot.save_scene(sensor_world, "res://gobot_python_binding_sensor_scene.jscn")
    context.load_scene("res://gobot_python_binding_sensor_scene.jscn")
    context.build_world(gobot.PhysicsBackendType.Null)
    context.configure_batch_world(1)
    assert context.resolved_batch_workers(0) == 1
    assert context.resolved_batch_workers(8) == 1
    context.step_batch(2, workers=2)
    context.set_batch_joint_position_targets("sensor_bot", [], np.zeros((1, 0), dtype=np.float64))
    batch_state = context.get_batch_robot_state(
        "sensor_bot",
        "base",
        [],
        ["base"],
        ["terrain_scan"],
    )
    assert batch_state["env_count"] == 1
    assert batch_state["base_position"].shape == (1, 3)
    assert batch_state["base_quaternion"].shape == (1, 4)
    assert batch_state["link_position"].shape == (1, 1, 3)
    assert batch_state["sensor_values"].shape == (1, 1, 2)
    assert batch_state["sensor_hit"].shape == (1, 1, 2)
    assert batch_state["sensor_hit_point"].shape == (1, 1, 2, 3)
    assert batch_state["contact_link_index"].shape[0] == 1
    assert tuple(batch_state["sensor_names"]) == ("terrain_scan",)
    assert tuple(batch_state["link_names"]) == ("base",)
    runtime_sensor_bot = context.root.find("sensor_bot")
    assert runtime_sensor_bot is not None
    terrain_scan_node = runtime_sensor_bot.find("base/terrain_scan")
    assert terrain_scan_node is not None
    assert len(terrain_scan_node.sample_offsets) == 2
    assert np.asarray(terrain_scan_node.ray_direction).shape == (3,)
    assert tuple(terrain_scan_node.ray_direction) == (0.0, 0.0, -1.0)
    assert terrain_scan_node.ray_direction_world_space is True
    assert abs(float(terrain_scan_node.max_distance) - 2.0) < 1e-6
    sensors = [
        runtime_sensor_bot.find("base/imu").get_runtime_state(),
        runtime_sensor_bot.find("base/root_angmom").get_runtime_state(),
        runtime_sensor_bot.find("base/contact").get_runtime_state(),
        terrain_scan_node.get_runtime_state(),
    ]
    assert sensors[0]["type"] == "imu"
    assert len(sensors[0]["values"]) == 13
    assert sensors[0]["channel_names"][7:10] == [
        "linear_velocity_x",
        "linear_velocity_y",
        "linear_velocity_z",
    ]
    assert sensors[1]["type"] == "angular_momentum"
    assert len(sensors[1]["values"]) == 3
    assert sensors[2]["type"] == "contact"
    assert len(sensors[2]["values"]) == 1
    assert sensors[3]["type"] == "height_scanner"
    assert sensors[3]["channel_names"] == ["distance_0", "distance_1"]
    assert len(sensors[3]["values"]) == 2
    assert all(abs(float(value) - 1.0) < 1e-6 for value in sensors[3]["values"])
    assert len(sensors[3]["hits"]) == 2
    assert all(hit["hit"] for hit in sensors[3]["hits"])
    assert all(abs(float(hit["distance"]) - 1.0) < 1e-6 for hit in sensors[3]["hits"])
    assert tuple(sensors[3]["global_transform"]["position"]) == (0.0, 0.0, 1.0)

    terrain_cfg = gobot.terrain.TerrainGeneratorCfg(
        size=(2.0, 2.0),
        num_rows=1,
        num_cols=1,
        seed=42,
        sub_terrains={"rough": gobot.terrain.random_rough(noise_range=(-0.02, 0.02))},
        horizontal_scale=0.5,
    )
    terrain_a = gobot.terrain.create_terrain_node(terrain_cfg, "terrain_a")
    terrain_b = gobot.terrain.create_terrain_node(terrain_cfg, "terrain_b")
    assert isinstance(terrain_a, gobot.Terrain3D)
    assert terrain_a.heightfield_count == 1
    assert terrain_a.color_mode == gobot.TerrainColorMode.Palette
    assert terrain_a.get_heightfield_heights(0) == terrain_b.get_heightfield_heights(0)
    assert len(terrain_a.spawn_origins) == 1
    terrain_root = gobot.create_node("Node3D", "terrain_world")
    terrain_root.add_child(terrain_a)
    gobot.save_scene(terrain_root, "res://gobot_python_binding_terrain_scene.jscn")
    context.load_scene("res://gobot_python_binding_terrain_scene.jscn")
    loaded_terrain = context.root.find("terrain_a")
    assert loaded_terrain.type == "Terrain3D"
    assert loaded_terrain.color_mode == gobot.TerrainColorMode.Palette
    context.build_world(gobot.PhysicsBackendType.Null)

    script_path = "/tmp/gobot_python_binding_smoke.py"
    with open(script_path, "w", encoding="utf-8") as script_file:
        script_file.write("class Script(gobot.NodeScript):\\n    pass\\n")
    script = gobot.load_resource(script_path, "PythonScript")
    assert script["type"] == "PythonScript"
    assert "class Script" in script["source_code"]

    mujoco_available = any(
        info["name"] == "MuJoCo CPU" and info["available"]
        for info in infos
    )
    if mujoco_available:
        split_project = "/tmp/gobot_python_mjcf_split"
        os.makedirs(split_project, exist_ok=True)
        with open(os.path.join(split_project, "robot.xml"), "w", encoding="utf-8") as robot_file:
            robot_file.write(
                textwrap.dedent(
                    """
                    <mujoco model="test_bot">
                      <worldbody>
                        <body name="base" pos="0 0 0.2">
                          <freejoint name="floating_base_joint"/>
                          <geom name="base_collision" type="box" size="0.1 0.1 0.1"/>
                          <body name="arm" pos="0 0 0.2">
                            <joint name="hinge" type="hinge" axis="0 1 0"/>
                            <geom name="arm_collision" type="box" size="0.05 0.05 0.05"/>
                          </body>
                        </body>
                      </worldbody>
                    </mujoco>
                    """
                ).strip()
            )
        with open(os.path.join(split_project, "world.xml"), "w", encoding="utf-8") as world_file:
            world_file.write(
                textwrap.dedent(
                    """
                    <mujoco model="test_world">
                      <default>
                        <joint damping="3"/>
                      </default>
                      <include file="robot.xml"/>
                      <worldbody>
                        <geom name="ground" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
                      </worldbody>
                    </mujoco>
                    """
                ).strip()
            )
        with open(os.path.join(split_project, "script.py"), "w", encoding="utf-8") as split_script_file:
            split_script_file.write("class Script(gobot.NodeScript):\\n    pass\\n")

        context.set_project_path(split_project)
        gobot.import_mjcf_scene(
            "res://world.xml",
            "res://world.jscn",
            name="world",
            script="res://script.py",
        )

        with open(os.path.join(split_project, "world.jscn"), encoding="utf-8") as world_scene_file:
            world_scene = json.load(world_scene_file)
        ext_paths = {resource["__PATH__"] for resource in world_scene["__EXT_RESOURCES__"]}
        assert "res://robot.jscn" in ext_paths
        assert "res://script.py" in ext_paths
        assert world_scene["__NODES__"][0]["name"] == "world"
        assert world_scene["__NODES__"][0]["type"] == "Node3D"
        assert str(world_scene["__NODES__"][0]["properties"]["script"]).startswith("ExtResource(")
        assert any(node["name"] == "ground" and node["type"] == "Node3D" for node in world_scene["__NODES__"])
        robot_instance_nodes = [node for node in world_scene["__NODES__"] if node["name"] == "test_bot"]
        assert len(robot_instance_nodes) == 1
        assert str(robot_instance_nodes[0]["instance"]).startswith("ExtResource(")
        assert robot_instance_nodes[0]["properties"]["source_path"] == "res://robot.xml"

        with open(os.path.join(split_project, "robot.jscn"), encoding="utf-8") as robot_scene_file:
            robot_scene = json.load(robot_scene_file)
        assert robot_scene["__NODES__"][0]["name"] == "test_bot"
        assert robot_scene["__NODES__"][0]["type"] == "Robot3D"
        assert robot_scene["__NODES__"][0]["properties"]["source_path"] == "res://robot.xml"
        hinge_nodes = [node for node in robot_scene["__NODES__"] if node["name"] == "hinge"]
        assert len(hinge_nodes) == 1
        assert hinge_nodes[0]["properties"]["damping"] == 3.0


if __name__ == "__main__":
    main()
