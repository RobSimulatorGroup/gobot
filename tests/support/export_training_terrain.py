"""Compile the authored locomotion terrain once for both training frameworks."""
import argparse
import json
from pathlib import Path

from gobot.rl.locomotion import make_velocity_env
from gobot.rl.locomotion.experiment import Experiment, task_config
from gobot.rl.locomotion.terrain_reference import from_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', choices=('g1','go1'), default='g1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output} already exists; choose a new path to keep prior runs reproducible')
    cfg = task_config(Experiment(robot=args.robot, stage='rough'))
    env = make_velocity_env(cfg, backend='mujoco-cpu', num_envs=1, collect_step_extras=False)
    try:
        data = from_environment(env)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2) + '\n')
        print(json.dumps({key: data[key] for key in ('sha256','seed','grid_shape','names')}))
    finally:
        env.close()


if __name__ == '__main__':
    main()
