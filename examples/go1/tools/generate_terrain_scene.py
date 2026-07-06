"""Regenerate the Go1 rough terrain scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

from _repo_imports import prefer_repo_gobot

prefer_repo_gobot()

import gobot


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("unilab", "mjlab"), default="unilab")
    parser.add_argument("--out", default=None)
    parser.add_argument("--curriculum", action="store_true", default=None)
    parser.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    project_path = Path(__file__).resolve().parents[1]
    gobot.set_project_path(str(project_path))

    root = gobot.create_node("Node3D", "terrain_world")
    if args.profile == "mjlab":
        curriculum = True if args.curriculum is None else bool(args.curriculum)
        cfg = gobot.terrain.go1_mjlab_rough_terrain_cfg(curriculum=curriculum)
        out = args.out or "res://mjlab_terrain_scene.jscn"
    else:
        curriculum = False if args.curriculum is None else bool(args.curriculum)
        cfg = gobot.terrain.go1_rough_terrain_cfg(curriculum=curriculum)
        out = args.out or "res://terrain_scene.jscn"
    terrain = gobot.terrain.create_terrain_node(cfg, "terrain")
    root.add_child(terrain)
    gobot.save_scene(root, out)


if __name__ == "__main__":
    main()
