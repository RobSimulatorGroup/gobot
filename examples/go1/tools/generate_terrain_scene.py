"""Regenerate the Go1 rough terrain scene."""

from __future__ import annotations

from pathlib import Path

import gobot


def main() -> None:
    project_path = Path(__file__).resolve().parents[1]
    gobot.set_project_path(str(project_path))

    root = gobot.create_node("Node3D", "terrain_world")
    terrain = gobot.terrain.create_terrain_node(gobot.terrain.go1_rough_terrain_cfg(), "terrain")
    root.add_child(terrain)
    gobot.save_scene(root, "res://terrain_scene.jscn")


if __name__ == "__main__":
    main()
