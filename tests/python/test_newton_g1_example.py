from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import runpy
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "newton_g1"
SCENE_PATH = "res://newton_g1.jscn"


def test_project_hook_and_initial_camera_are_valid() -> None:
    project = json.loads((EXAMPLE / "project.gobot").read_text(encoding="utf-8"))
    assert project["main_scene"] == SCENE_PATH
    assert project["project_load_hook"] == "res://download_assets.py"
    view = project["editor_scene_views"][SCENE_PATH]
    eye = view["eye"]
    at = view["at"]
    up = view["up"]
    forward = [target - origin for origin, target in zip(eye, at)]
    distance = math.sqrt(sum(value * value for value in forward))
    up_length = math.sqrt(sum(value * value for value in up))
    assert 3.0 < distance < 6.0
    assert math.isclose(up_length, 1.0, abs_tol=2.0e-3)
    assert abs(sum(a * b for a, b in zip(forward, up))) < 2.0e-2
    assert 35.0 <= float(view["fov_y"]) <= 60.0


def test_scene_instances_one_generated_gobot_robot() -> None:
    scene = json.loads((EXAMPLE / "newton_g1.jscn").read_text(encoding="utf-8"))
    contract = runpy.run_path(str(EXAMPLE / "scripts" / "g1_policy_contract.py"))
    resources = {entry["__ID__"]: entry for entry in scene["__EXT_RESOURCES__"]}
    assert resources["2_robot"]["__PATH__"] == "res://assets/generated/g1_29dof.jscn"
    assert all("mjcf" not in resource["__PATH__"].lower() for resource in resources.values())

    nodes = {node["name"]: node for node in scene["__NODES__"]}
    robot_nodes = [node for node in scene["__NODES__"] if node["type"] == "Robot3D"]
    assert len(robot_nodes) == 1
    assert nodes["g1"]["instance"] == "ExtResource(2_robot)"
    assert nodes["g1"]["properties"]["visible"] is True
    assert nodes["g1"]["properties"]["mode"] == "Motion"
    preview_position = nodes["g1"]["properties"]["position"]["matrix_data"]["storage"]
    assert preview_position == list(contract["BASE_POSE_XYZW"][:3])
    assert nodes["ground_collision"]["type"] == "CollisionShape3D"
    assert nodes["ground_visual"]["type"] == "MeshInstance3D"


def test_policy_contract_matches_newton_g1_29dof() -> None:
    contract = runpy.run_path(str(EXAMPLE / "scripts" / "g1_policy_contract.py"))
    joint_names = contract["JOINT_NAMES"]
    assert len(joint_names) == 43
    assert len(set(joint_names)) == 43
    assert contract["ACTION_DIM"] == 43
    assert contract["OBSERVATION_DIM"] == 141
    assert contract["PHYSICS_DT"] == 0.005
    assert contract["POLICY_DECIMATION"] == 4
    assert contract["POLICY_DT"] == 0.02
    assert contract["ACTION_SCALE"] == 0.5
    assert joint_names[:6] == (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    )
    assert joint_names[-3:] == (
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
    )
    for name in (
        "DEFAULT_JOINT_POSITION",
        "JOINT_STIFFNESS",
        "JOINT_DAMPING",
        "JOINT_ARMATURE",
    ):
        assert len(contract[name]) == 43
    assert len(contract["LINK_NAMES"]) == 44
    assert contract["BASE_LINK"] == "pelvis"


def test_native_policy_yaml_reader_preserves_order_and_values() -> None:
    contract = runpy.run_path(str(EXAMPLE / "scripts" / "g1_policy_contract.py"))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "contract.yaml"
        path.write_text(
            """num_dofs: 2
action_scale: 0.5
mjw_joint_names:
  - "joint_a"
  - "joint_b"
mjw_joint_pos: [0.1, -0.2]
mjw_joint_stiffness: [10.0, 20.0]
mjw_joint_damping: [1.0, 2.0]
mjw_joint_armature: [0.1, 0.1]
""",
            encoding="utf-8",
        )
        loaded = contract["load_native_policy_contract"](path)
    assert loaded["mjw_joint_names"] == ("joint_a", "joint_b")
    assert loaded["mjw_joint_pos"] == (0.1, -0.2)
    assert loaded["mjw_joint_stiffness"] == (10.0, 20.0)


def test_playback_uses_stable_provider_and_warp_nn_without_newton_viewer() -> None:
    path = EXAMPLE / "scripts" / "g1_policy.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert "gobot.rl.providers" in imports
    assert "warp_nn.runtime" in imports
    assert not any(name.startswith("newton.viewer") for name in imports)
    assert "onnxruntime" not in imports
    assert "resolve_robot_layout" in source
    assert "set_joint_position_targets" in source
    assert "reset_robot_state" in source
    assert "ProviderPlaySession" in source
    assert "self.play_session.reset()" in source
    assert "_apply_link_pose_batch" in source
    assert "self.provider.step" not in source
    assert "self.context.backend_type" not in source
    assert 'type_name="Link3D"' in source
    assert "load_native_policy_contract" in source
    assert "_configure_native_g1_scene" in source
    assert "NewtonModelConfig" in source
    assert "joint_limit_stiffness=1.0e2" in source
    assert "contact_stiffness=5.0e4" in source
    assert "contact_friction_override=0.75" in source
    assert "model_config=NEWTON_MODEL_CONFIG" in source
    assert "joint.friction_loss = 0.0" in source
    assert "joint.effort_limit = 0.0" in source
    assert "joint.force_lower_limit = 0.0" in source
    assert "joint.force_upper_limit = 0.0" in source
    assert 'node.set("contype", 0)' in source
    assert 'int(artifact["dimensions"]["nu"]) != ACTION_DIM' in source
    assert 'artifact["dimensions"]["nu"]' in source
    assert "g1_visual" not in source
    assert "g1_physics" not in source
    assert "use_mujoco_contacts=False" not in source
    assert "use_mujoco_contacts=True" in source
    assert "provider.use_mujoco_contacts" in source
    assert "synchronize_device" in source
    assert "compiling the Gobot scene artifact" in source
    assert "initializing the Newton provider" in source
    assert "loading the Warp ONNX policy" in source
    assert "first CUDA simulation frame ready" in source
    assert "flush=True" in source


def test_playback_module_imports_from_the_project_root() -> None:
    sys.path.insert(0, str(EXAMPLE))
    try:
        namespace = runpy.run_path(str(EXAMPLE / "scripts" / "g1_policy.py"))
    finally:
        sys.path.remove(str(EXAMPLE))
    assert namespace["OBSERVATION_DIM"] == 141


def test_default_dependency_installs_warp_nn_without_newton_examples() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dependency = '"newton[onnx,sim]==1.4.0; sys_platform == \'linux\''
    assert dependency in project
    assert '"newton[examples]' not in project


def main() -> None:
    test_project_hook_and_initial_camera_are_valid()
    test_scene_instances_one_generated_gobot_robot()
    test_policy_contract_matches_newton_g1_29dof()
    test_native_policy_yaml_reader_preserves_order_and_values()
    test_playback_uses_stable_provider_and_warp_nn_without_newton_viewer()
    test_playback_module_imports_from_the_project_root()
    test_default_dependency_installs_warp_nn_without_newton_examples()


if __name__ == "__main__":
    main()
