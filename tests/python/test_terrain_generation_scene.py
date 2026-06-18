import json
import os

from _gobot_test_import import prefer_build_gobot

prefer_build_gobot()

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
    assert terrain.color_mode == gobot.TerrainColorMode.Palette

    root.add_child(terrain)
    gobot.save_scene(root, "res://generated_terrain.jscn")
    assert os.path.exists(os.path.join(project, "generated_terrain.jscn"))

    scene = gobot.load_scene("res://generated_terrain.jscn")
    loaded_terrain = scene.root.find("generated_terrain")
    assert loaded_terrain.type == "Terrain3D"
    assert loaded_terrain.color_mode == gobot.TerrainColorMode.Palette

    context = gobot.app.context()
    context.set_project_path(project)
    context.load_scene("res://generated_terrain.jscn")
    context_terrain = context.root.find("generated_terrain")
    assert context_terrain.type == "Terrain3D"
    assert context_terrain.color_mode == gobot.TerrainColorMode.Palette
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


def test_go1_rough_terrain_cfg_matches_mjlab_schema():
    cfg = gobot.terrain.go1_rough_terrain_cfg()

    assert cfg.size == (8.0, 8.0)
    assert cfg.border_width == 20.0
    assert cfg.num_rows == 10
    assert cfg.num_cols == 20
    assert cfg.curriculum
    assert cfg.origin_row == 5
    assert cfg.origin_sub_terrain == "flat"
    assert cfg.horizontal_scale == 0.1

    terrains = cfg.sub_terrains
    assert tuple(terrains) == (
        "flat",
        "pyramid_stairs",
        "pyramid_stairs_inv",
        "hf_pyramid_slope",
        "hf_pyramid_slope_inv",
        "random_rough",
        "wave_terrain",
    )
    assert [terrains[name].proportion for name in terrains] == [
        0.2,
        0.2,
        0.2,
        0.1,
        0.1,
        0.1,
        0.1,
    ]

    assert terrains["pyramid_stairs"].kwargs["step_height_range"] == (0.0, 0.1)
    assert terrains["pyramid_stairs"].kwargs["step_width"] == 0.30
    assert terrains["pyramid_stairs"].kwargs["platform_width"] == 3.0
    assert terrains["pyramid_stairs"].kwargs["border_width"] == 1.0
    assert terrains["pyramid_stairs_inv"].kwargs["inverted"]
    assert terrains["hf_pyramid_slope"].kwargs["slope_range"] == (0.0, 1.0)
    assert terrains["hf_pyramid_slope"].kwargs["platform_width"] == 2.0
    assert terrains["hf_pyramid_slope"].kwargs["border_width"] == 0.25
    assert terrains["hf_pyramid_slope_inv"].kwargs["inverted"]
    assert terrains["random_rough"].kwargs["noise_min"] == 0.02
    assert terrains["random_rough"].kwargs["noise_max"] == 0.10
    assert terrains["random_rough"].kwargs["noise_step"] == 0.02
    assert terrains["random_rough"].kwargs["downsampled_scale"] is None
    assert not terrains["random_rough"].kwargs["scale_by_difficulty"]
    assert terrains["wave_terrain"].kwargs["amplitude_range"] == (0.0, 0.2)
    assert terrains["wave_terrain"].kwargs["num_waves"] == 4.0


def test_go1_rough_terrain_generation_shape():
    cfg = gobot.terrain.go1_rough_terrain_cfg(seed=11)
    terrain = gobot.terrain.create_terrain_node(cfg, "go1_rough_terrain")

    assert len(terrain.spawn_origins) == cfg.num_rows * len(cfg.sub_terrains)
    spawn_origins = [tuple(float(value) for value in origin) for origin in terrain.spawn_origins]
    closest_origin = min(spawn_origins, key=lambda origin: origin[0] * origin[0] + origin[1] * origin[1])
    assert closest_origin == (0.0, 0.0, 0.0)
    assert terrain.box_count > 0
    assert terrain.heightfield_count == 40
    assert terrain.mesh_patch_count == 0


if __name__ == "__main__":
    test_go1_rough_terrain_cfg_matches_mjlab_schema()
    test_go1_rough_terrain_generation_shape()
    main()
