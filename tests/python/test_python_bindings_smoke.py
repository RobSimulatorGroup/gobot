import gobot
import numpy as np


def assert_close_tuple(actual, expected):
    if hasattr(actual, "tolist"):
        actual = actual.tolist()
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        assert abs(left - right) < 1e-9


def main():
    infos = gobot.backend_infos()
    assert infos
    assert any(info["name"] == "Null" and info["available"] for info in infos)
    assert any(info["name"] == "MuJoCo CPU" and info["robotics_focused"] for info in infos)
    assert any(info["name"] == "MuJoCo Warp" and info["gpu"] and not info["available"] for info in infos)
    assert gobot.PhysicsBackendType.Null == gobot.physics.PhysicsBackendType.Null
    assert gobot.PhysicsBackendType.MuJoCoWarp == gobot.physics.PhysicsBackendType.MuJoCoWarp
    assert gobot.sim.JointControllerGains is gobot.JointControllerGains
    assert gobot.scene.Node is gobot.Node
    assert "ManagerBasedEnv" in gobot.rl.__all__
    assert "VectorEnv" not in gobot.rl.__all__
    assert not hasattr(gobot._core, "NativeVectorEnv")

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
    assert authored.find("link/collision").name == "collision"
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
    context.build_world(gobot.PhysicsBackendType.Null)
    name_map = context.get_runtime_name_map()
    assert name_map["robots"][0]["name"] == "cartpole"
    assert name_map["robots"][0]["controllable_joint_names"] == ["slider", "hinge"]
    state = context.get_runtime_state()
    assert state["robots"][0]["joints"][0]["name"] == "slider"

    env = gobot.rl.ManagerBasedEnv(
        {
            "backend": "null",
            "num_envs": 1,
            "physics_dt": 1.0 / 240.0,
            "decimation": 4,
            "episode_length_s": 1.0 / 30.0,
            "robot": "cartpole",
            "controlled_joints": ["slider"],
            "observations": {},
            "rewards": {},
            "terminations": {},
            "events": {},
        }
    )
    observation, info = env.reset(seed=5)
    assert observation.shape == (1, env.observation_spec.size)
    assert env.action_spec.names == ("slider/target_position_normalized",)
    observation, reward, terminated, truncated, info = env.step([[2.0]])
    assert observation.shape == (1, env.observation_spec.size)
    assert reward.shape == (1,)
    assert terminated.shape == (1,)
    assert truncated.shape == (1,)
    assert reward[0] == env.env_dt
    assert not bool(truncated[0])
    observation, reward, terminated, truncated, info = env.step([[0.0]])
    assert "terminal_observation" in info
    assert info["terminal_observation"].shape == (1, env.observation_spec.size)
    gym_env = gobot.rl.GymWrapper(env)
    gym_obs, gym_info = gym_env.reset(seed=6)
    assert gym_obs.shape == (env.observation_spec.size,)
    gym_obs, gym_reward, gym_terminated, gym_truncated, gym_info = gym_env.step([0.0])
    assert isinstance(gym_reward, float)
    assert isinstance(gym_terminated, bool)
    assert isinstance(gym_truncated, bool)
    try:
        gobot.rl.ManagerBasedEnv({"backend": "null", "num_envs": 2})
        raise AssertionError("num_envs > 1 should require a vector backend")
    except NotImplementedError as error:
        assert "num_envs=1" in str(error)

    script_path = "/tmp/gobot_python_binding_smoke.py"
    with open(script_path, "w", encoding="utf-8") as script_file:
        script_file.write("class Script(gobot.NodeScript):\\n    pass\\n")
    script = gobot.load_resource(script_path, "PythonScript")
    assert script["type"] == "PythonScript"
    assert "class Script" in script["source_code"]


if __name__ == "__main__":
    main()
