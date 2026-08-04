"""Generate a deterministic Gobot tactile sensor scene without GUI dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gobot
from gobot.ipc._fabrication import create_fabricated_sensor_scene, fabricate_box_gel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("fabricated_sensor.jscn"))
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    gobot.app.context().set_project_path(str(output.parent))
    resolution = (args.height, args.width)
    root = create_fabricated_sensor_scene(resolution=resolution)
    gobot.save_scene(root, "res://" + output.name)

    fabricated = fabricate_box_gel(resolution=resolution)
    print(
        json.dumps(
            {
                "coat_vertices": len(fabricated.coat_vertex_indices),
                "markers": len(fabricated.marker_positions),
                "output": str(output),
                "stick_vertices": len(fabricated.stick_vertex_indices),
                "tetrahedra": len(fabricated.tetrahedra),
                "vertices": len(fabricated.vertices),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
