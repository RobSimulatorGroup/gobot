import argparse
import gobot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--scene", default="res://world.jscn")
    parser.add_argument("--backend", default="mujoco", choices=["null", "mujoco"])
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()

    gobot.set_project_path(args.project)
    scene = gobot.load_scene(args.scene)
    print(f"loaded scene root={scene.root.name} type={scene.root.type} children={scene.root.child_count}")

    context = gobot.app.context()
    context.build_world(gobot.PhysicsBackendType.MuJoCoCpu if args.backend == "mujoco"
                        else gobot.PhysicsBackendType.Null)
    context.reset_simulation()
    for index in range(args.steps):
        context.step_once()
        print(f"step={index + 1} time={context.simulation_time:.6f} frame={context.frame_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
