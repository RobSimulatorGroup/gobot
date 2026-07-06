import json
import os

from _gobot_test_import import prefer_build_gobot

prefer_build_gobot()

import gobot
import numpy as np


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


def test_go1_rough_terrain_cfg_matches_unilab_go1_ppo_schema():
    cfg = gobot.terrain.go1_rough_terrain_cfg()

    assert cfg.size == (8.0, 8.0)
    assert cfg.border_width == 20.0
    assert cfg.num_rows == 6
    assert cfg.num_cols == 6
    assert not cfg.curriculum
    assert cfg.origin_row is None
    assert cfg.origin_sub_terrain is None
    assert cfg.horizontal_scale == 0.2
    assert cfg.seed == 42
    assert cfg.merged_heightfield

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
        0.0,
        0.1,
        0.1,
        0.2,
        0.2,
        0.3,
        0.3,
    ]

    assert terrains["pyramid_stairs"].kwargs["step_height_range"] == (0.025, 0.10)
    assert terrains["pyramid_stairs"].kwargs["step_width"] == 0.40
    assert terrains["pyramid_stairs"].kwargs["platform_width"] == 3.0
    assert terrains["pyramid_stairs"].kwargs["border_width"] == 0.2
    assert terrains["pyramid_stairs_inv"].kwargs["inverted"]
    assert terrains["hf_pyramid_slope"].kwargs["slope_range"] == (0.0, 0.3)
    assert terrains["hf_pyramid_slope"].kwargs["platform_width"] == 2.0
    assert terrains["hf_pyramid_slope"].kwargs["border_width"] == 0.2
    assert terrains["hf_pyramid_slope_inv"].kwargs["inverted"]
    assert terrains["random_rough"].kwargs["noise_min"] == 0.01
    assert terrains["random_rough"].kwargs["noise_max"] == 0.06
    assert terrains["random_rough"].kwargs["noise_step"] == 0.01
    assert not terrains["random_rough"].kwargs["scale_by_difficulty"]
    assert terrains["wave_terrain"].kwargs["amplitude_range"] == (0.0, 0.12)
    assert terrains["wave_terrain"].kwargs["num_waves"] == 4.0


def test_go1_mjlab_rough_terrain_cfg_matches_velocity_go1_schema():
    cfg = gobot.terrain.go1_mjlab_rough_terrain_cfg()

    assert cfg.size == (8.0, 8.0)
    assert cfg.border_width == 20.0
    assert cfg.num_rows == 10
    assert cfg.num_cols == 20
    assert cfg.curriculum
    assert cfg.horizontal_scale == 0.1
    assert cfg.seed == 42
    assert cfg.merged_heightfield

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
    assert terrains["pyramid_stairs"].kwargs["step_width"] == 0.3
    assert terrains["pyramid_stairs"].kwargs["border_width"] == 1.0
    assert terrains["hf_pyramid_slope"].kwargs["slope_range"] == (0.0, 1.0)
    assert terrains["hf_pyramid_slope"].kwargs["border_width"] == 0.25
    assert terrains["random_rough"].kwargs["noise_min"] == 0.02
    assert terrains["random_rough"].kwargs["noise_max"] == 0.10
    assert terrains["random_rough"].kwargs["noise_step"] == 0.02
    assert terrains["random_rough"].kwargs["scale_by_difficulty"]
    assert terrains["wave_terrain"].kwargs["amplitude_range"] == (0.0, 0.2)
    assert terrains["wave_terrain"].kwargs["num_waves"] == 4.0


def test_go1_rough_terrain_generation_shape():
    cfg = gobot.terrain.go1_rough_terrain_cfg(seed=42)
    terrain = gobot.terrain.create_terrain_node(cfg, "go1_rough_terrain")

    assert len(terrain.spawn_origins) == cfg.num_rows * cfg.num_cols
    spawn_origins = np.asarray(terrain.spawn_origins, dtype=np.float64)
    assert min(origin[0] for origin in spawn_origins) >= -24.0
    assert max(origin[0] for origin in spawn_origins) <= 24.0
    assert min(origin[1] for origin in spawn_origins) >= -24.0
    assert max(origin[1] for origin in spawn_origins) <= 24.0
    np.testing.assert_allclose(
        spawn_origins[:6],
        np.asarray(
            [
                (-20.0, -20.0, 0.0),
                (-20.0, -12.0, 0.0),
                (-20.0, -4.0, -0.6),
                (-20.0, 4.0, 0.0),
                (-20.0, 12.0, -0.36),
                (-20.0, 20.0, -0.575),
            ],
            dtype=np.float64,
        ),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray([spawn_origins[:, 2].min(), spawn_origins[:, 2].mean(), spawn_origins[:, 2].max()]),
        np.asarray([-0.6, 0.0148611111, 0.56]),
        atol=1.0e-6,
    )
    assert terrain.box_count == 0
    assert terrain.heightfield_count == 1
    rows = cfg.num_cols * int(round(cfg.size[1] / cfg.horizontal_scale)) + 2 * int(round(cfg.border_width / cfg.horizontal_scale))
    cols = cfg.num_rows * int(round(cfg.size[0] / cfg.horizontal_scale)) + 2 * int(round(cfg.border_width / cfg.horizontal_scale))
    assert rows == 440
    assert cols == 440
    assert len(terrain.get_heightfield_heights(0)) == rows * cols
    assert terrain.mesh_patch_count == 0


if __name__ == "__main__":
    test_go1_rough_terrain_cfg_matches_unilab_go1_ppo_schema()
    test_go1_mjlab_rough_terrain_cfg_matches_velocity_go1_schema()
    test_go1_rough_terrain_generation_shape()
    main()
