import argparse
import sys

import gobot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--scene", default="res://world.jscn")
    parser.add_argument("--robot", default="H2")
    parser.add_argument("--backend", default="mujoco")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()

    gobot.set_project_path(args.project)
    scene = gobot.load_scene(args.scene)
    print(f"loaded scene root={scene.root.name} type={scene.root.type} children={scene.root.child_count}")

    env = gobot.RLEnvironment(args.scene, robot=args.robot, backend=args.backend)
    observation, info = env.reset(seed=1)
    print(f"reset ok={info['ok']} observation_size={len(observation)} action_size={env.get_action_size()}")
    if not info["ok"]:
        print(info["error"], file=sys.stderr)
        return 1

    action = [0.0] * env.get_action_size()
    for index in range(args.steps):
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"step={index + 1} reward={reward} terminated={terminated} truncated={truncated} frame={info['frame_count']}")
        if terminated or truncated:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
