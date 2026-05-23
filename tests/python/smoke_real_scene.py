import argparse
import importlib.util
import json
import os
import pathlib
import gobot


GO1_RESET_MIN_CONTACT_DISTANCE = -0.025


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--scene", default="res://world.jscn")
    parser.add_argument("--backend", default="mujoco", choices=["null", "mujoco"])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--expect-go1-stand", action="store_true")
    parser.add_argument("--expect-empty-robot-source-path", action="store_true")
    args = parser.parse_args()

    context = gobot.app.context()
    context.set_project_path(args.project)
    scene_path = pathlib.Path(args.scene.replace("res://", args.project + "/", 1))

    if args.expect_empty_robot_source_path:
        scene_json = json.loads(scene_path.read_text())
        robot_nodes = [
            node for node in scene_json.get("__NODES__", [])
            if node.get("type") == "Robot3D"
        ]
        if not robot_nodes:
            raise AssertionError(f"No Robot3D nodes found in {scene_path}")
        for node in robot_nodes:
            source_path = node.get("properties", {}).get("source_path", "")
            if source_path:
                raise AssertionError(
                    f"Robot3D '{node.get('name', '<unnamed>')}' still depends on source_path={source_path!r}"
                )

    root = context.load_scene(args.scene)
    print(f"loaded scene root={root.name} type={root.type} children={root.child_count}")
    context.build_world(gobot.PhysicsBackendType.MuJoCoCpu if args.backend == "mujoco"
                        else gobot.PhysicsBackendType.Null)
    context.reset_simulation()
    for index in range(args.steps):
        context.step_once()
        print(f"step={index + 1} time={context.simulation_time:.6f} frame={context.frame_count}")

    if args.expect_go1_stand:
        script_path = pathlib.Path(args.project) / "scripts" / "go1.py"
        spec = importlib.util.spec_from_file_location("go1_scene_script", script_path)
        script_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script_module)
        os.environ["GOBOT_GO1_POLICY"] = ""

        context.reset_link_state("go1", "trunk", script_module.RESET_BASE_POSITION,
                                 [1.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0])
        for name, target in zip(script_module.JOINT_NAMES, script_module.DEFAULT_POS):
            context.reset_joint_state("go1", name, target, 0.0)
            context.set_joint_position_target("go1", name, target)

        context.step_once()
        state = context.get_runtime_state()
        contact_distances = [
            float(contact["distance"])
            for contact in state.get("contacts", [])
            if contact.get("robot_name") == "go1"
        ]
        if contact_distances:
            min_contact_distance = min(contact_distances)
            print(f"go1_reset_min_contact_distance={min_contact_distance:.6f}")
            if min_contact_distance < GO1_RESET_MIN_CONTACT_DISTANCE:
                raise AssertionError(
                    f"Go1 reset pose starts too far inside the ground: {min_contact_distance:.6f}"
                )

        for _ in range(200):
            for name, target in zip(script_module.JOINT_NAMES, script_module.DEFAULT_POS):
                context.set_joint_position_target("go1", name, target)
            context.step_once()

        state = context.get_runtime_state()
        robot = next(robot for robot in state["robots"] if robot["name"] == "go1")
        base = next(link for link in robot["links"] if link["link_name"] == "trunk")
        base_z = base["global_transform"]["position"][2]
        print(f"go1_stand_base_z={base_z:.6f}")
        if base_z <= 0.15:
            raise AssertionError(f"Go1 default stand base height is too low: {base_z:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
