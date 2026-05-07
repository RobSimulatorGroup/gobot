import gobot
import numpy as np
from gobot.gym_adapter import GobotBox, GobotGymEnv, space_from_spec
from gobot_gym_adapter import GobotGymEnv as LegacyGobotGymEnv


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
    assert gobot.PhysicsBackendType.Null == gobot.physics.PhysicsBackendType.Null
    assert any(info["type"] == gobot.PhysicsBackendType.Null for info in infos)

    context = gobot.app.context()
    assert gobot.rl.RLEnvironment is gobot.RLEnvironment
    assert gobot.sim.JointControllerGains is gobot.JointControllerGains
    assert gobot.scene.Node is gobot.Node
    assert context.backend_type == gobot.PhysicsBackendType.Null
    assert context.has_scene is False
    assert context.has_world is False
    assert context.frame_count == 0
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

    reward_settings = gobot.RLEnvironmentRewardSettings.from_dict(
        {
            "alive_reward": 2.0,
            "terminate_on_fall": False,
            "target_forward_velocity": 1.0,
            "action_rate_penalty_scale": 0.1,
        }
    )
    assert reward_settings.to_dict()["alive_reward"] == 2.0
    assert reward_settings.terminate_on_fall is False
    assert reward_settings.target_forward_velocity == 1.0
    assert abs(reward_settings.action_rate_penalty_scale - 0.1) < 1e-6

    spec = gobot.RLVectorSpec()
    spec.names = ["joint"]
    spec.lower_bounds = [-1.0]
    spec.upper_bounds = [1.0]
    spec.units = ["normalized"]
    assert spec.to_dict()["names"] == ["joint"]
    box = GobotBox(spec.lower_bounds, spec.upper_bounds, names=spec.names, units=spec.units)
    assert box.shape == (1,)
    assert len(box.sample()) == 1
    converted_space = space_from_spec(spec.to_dict())
    assert converted_space.shape == (1,)

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
    assert_close_tuple(base.position, (4.0, 5.0, 6.0))
    assert_close_tuple(base.scale, (2.0, 2.0, 2.0))

    try:
        with gobot.transaction("Cancelled transform"):
            base.name = "cancelled"
            raise RuntimeError("cancel transaction")
    except RuntimeError:
        pass
    assert base.name == "renamed_base"

    authored = gobot.create_node("Robot3D", "authored")
    assert authored.name == "authored"
    link = gobot.create_node("Link3D", "link")
    authored.add_child(link)
    assert authored.child_count == 1
    assert link.parent.id == authored.id
    collision = gobot.create_box_collision("collision", (0.2, 0.3, 0.4))
    link.add_child(collision)
    assert collision.type == "CollisionShape3D"
    visual = gobot.create_box_visual("visual", (0.2, 0.3, 0.4))
    link.add_child(visual)
    assert visual.type == "MeshInstance3D"
    assert authored.find("link/collision").name == "collision"
    detached_visual = link.remove_child(visual, delete=False)
    assert detached_visual.valid is True
    link.add_child(detached_visual)
    link.remove_child(visual, delete=True)
    assert link.child_count == 1
    assert visual.valid is False
    try:
        _ = visual.name
        raise AssertionError("deleted Gobot node handle should raise ReferenceError")
    except ReferenceError as error:
        assert "no longer resolves" in str(error) or "invalid" in str(error)
    assert gobot.undo() is True
    restored_visual = link.find("visual")
    assert restored_visual is not None
    assert restored_visual.valid is True
    assert visual.valid is False

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

    script_path = "/tmp/gobot_python_binding_smoke.py"
    with open(script_path, "w", encoding="utf-8") as script_file:
        script_file.write("class Agent:\\n    pass\\n")
    script = gobot.load_resource(script_path, "PythonScript")
    assert script["type"] == "PythonScript"
    assert "class Agent" in script["source_code"]

    env = gobot.RLEnvironment()
    observation, info = env.reset(seed=7)
    assert info["ok"] is True
    assert info["seed"] == 7
    assert len(observation) == env.get_observation_size()
    assert env.get_action_size() == 1
    assert env.get_observation_size() == 17

    observation, reward, terminated, truncated, info = env.step([0.0])
    assert len(observation) == env.get_observation_size()
    assert reward == 1.0
    assert terminated is False
    assert truncated is False
    assert info["frame_count"] == 1

    gains = env.get_default_joint_gains()
    assert gains["position_stiffness"] == 100.0
    env.set_default_joint_gains({"position_stiffness": 20.0, "velocity_damping": 2.0})
    gains = env.get_default_joint_gains()
    assert gains["position_stiffness"] == 20.0
    assert gains["velocity_damping"] == 2.0

    config = env.get_controller_config()
    assert config.controlled_joints == ["joint"]
    assert config.default_action == [0.0]
    config.joint_gains = {"position_stiffness": 30.0, "velocity_damping": 3.0}
    env.apply_controller_config(config.to_dict())
    gains = env.get_default_joint_gains()
    assert gains["position_stiffness"] == 30.0
    assert gains["velocity_damping"] == 3.0

    config_from_dict = gobot.RLControllerConfig.from_dict(
        {
            "controlled_joints": ["joint"],
            "default_action": [0.0],
            "joint_gains": {"position_stiffness": 40.0, "velocity_damping": 4.0},
        }
    )
    assert config_from_dict.to_dict()["joint_gains"]["position_stiffness"] == 40.0

    env.apply_controller_config({"controlled_joints": ["joint"], "default_action": [0.0]})
    observation, info = env.reset(seed=8)
    assert info["ok"] is True
    assert env.get_action_size() == 1
    assert env.get_controller_config().default_action == [0.0]
    assert env.get_action_spec()["names"] == ["joint/target_position_normalized"]
    assert "joint/previous_action" in env.get_observation_spec()["names"]
    assert env.step_result([0.0])["error"] == ""

    env.apply_controller_config({"controlled_joints": ["missing"], "default_action": [0.0]})
    reset_result = env.reset_result(seed=9)
    assert reset_result["ok"] is False
    assert "missing" in reset_result["error"]

    gym_env = GobotGymEnv()
    observation, info = gym_env.reset(seed=3)
    assert info["ok"] is True
    observation, reward, terminated, truncated, info = gym_env.step([0.0])
    assert reward == 1.0
    assert terminated is False
    assert truncated is False
    assert LegacyGobotGymEnv is GobotGymEnv


if __name__ == "__main__":
    main()
