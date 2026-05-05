import gobot
from gobot_gym_adapter import GobotGymEnv


def assert_close_tuple(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        assert abs(left - right) < 1e-9


def main():
    infos = gobot.backend_infos()
    assert infos
    assert any(info["name"] == "Null" and info["available"] for info in infos)

    scene = gobot.create_test_scene()
    root = scene.root
    assert root.name == "robot"
    assert root.type == "Robot3D"
    assert root.child_count == 2
    assert [child.name for child in root.children] == ["base", "joint"]

    base = root.child(0)
    assert base.type == "Link3D"
    assert_close_tuple(base.get("position"), (0.0, 0.0, 1.0))
    base.set("position", (1.0, 2.0, 3.0))
    assert_close_tuple(base.get("position"), (1.0, 2.0, 3.0))
    assert "position" in base.property_names()

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

    gym_env = GobotGymEnv()
    observation, info = gym_env.reset(seed=3)
    assert info["ok"] is True
    observation, reward, terminated, truncated, info = gym_env.step([0.0])
    assert reward == 1.0
    assert terminated is False
    assert truncated is False


if __name__ == "__main__":
    main()
