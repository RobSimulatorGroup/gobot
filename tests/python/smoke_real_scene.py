import argparse
import importlib.util
import pathlib
import gobot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--scene", default="res://world.jscn")
    parser.add_argument("--backend", default="mujoco", choices=["null", "mujoco"])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--expect-go1-stand", action="store_true")
    args = parser.parse_args()

    context = gobot.app.context()
    context.set_project_path(args.project)
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

        if script_module.DEFAULT_POLICY_PATH:
            raise AssertionError("Go1 example must not auto-load a local policy by default")

        context.reset_link_state("go1", "trunk", script_module.RESET_BASE_POSITION,
                                 [1.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0])
        for name, target in zip(script_module.JOINT_NAMES, script_module.DEFAULT_POS):
            context.reset_joint_state("go1", name, target, 0.0)
            context.set_joint_position_target("go1", name, target)

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
