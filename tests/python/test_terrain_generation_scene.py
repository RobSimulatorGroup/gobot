import json
import os

import gobot


def main():
    project = "/tmp/gobot_terrain_generation_scene"
    os.makedirs(project, exist_ok=True)
    gobot.set_project_path(project)

    cfg = gobot.terrain.TerrainGeneratorCfg(
        size=(3.0, 3.0),
        num_rows=1,
        num_cols=2,
        seed=7,
        sub_terrains={
            "rough": gobot.terrain.random_rough(noise_range=(-0.03, 0.03)),
            "stairs": gobot.terrain.pyramid_stairs(step_height=0.05),
            "tilted": gobot.terrain.tilted_grid(proportion=0.5),
        },
        horizontal_scale=0.5,
    )

    root = gobot.create_node("Node3D", "world")
    terrain = gobot.terrain.create_terrain_node(cfg, "generated_terrain")
    box_count = terrain.box_count
    heightfield_count = terrain.heightfield_count
    mesh_patch_count = terrain.mesh_patch_count
    spawn_count = len(terrain.spawn_origins)
    assert box_count + heightfield_count + mesh_patch_count > 0
    assert spawn_count == cfg.num_rows * cfg.num_cols
    assert terrain.color_mode == gobot.TerrainColorMode.MjLab

    root.add_child(terrain)
    gobot.save_scene(root, "res://generated_terrain.jscn")
    assert os.path.exists(os.path.join(project, "generated_terrain.jscn"))

    scene = gobot.load_scene("res://generated_terrain.jscn")
    loaded_terrain = scene.root.find("generated_terrain")
    assert loaded_terrain.type == "Terrain3D"
    assert loaded_terrain.color_mode == gobot.TerrainColorMode.MjLab

    context = gobot.app.context()
    context.set_project_path(project)
    context.load_scene("res://generated_terrain.jscn")
    context_terrain = context.root.find("generated_terrain")
    assert context_terrain.type == "Terrain3D"
    assert context_terrain.color_mode == gobot.TerrainColorMode.MjLab
    context.build_world(gobot.PhysicsBackendType.Null)

    print(
        json.dumps(
            {
                "box_count": box_count,
                "heightfield_count": heightfield_count,
                "mesh_patch_count": mesh_patch_count,
                "scene_path": os.path.join(project, "generated_terrain.jscn"),
                "spawn_origins": spawn_count,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
