"""Terrain3D generation helpers for Gobot scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
import colorsys
from typing import Any, Literal

import numpy as np

try:
    from scipy import ndimage
except Exception:  # pragma: no cover - exercised when optional build env is incomplete.
    ndimage = None

from .. import _core

Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]
Color4 = tuple[float, float, float, float]
TerrainKind = Literal[
    "flat",
    "pyramid_stairs",
    "random_grid",
    "hf_pyramid_slope",
    "random_rough",
    "wave",
    "perlin_noise",
    "discrete_obstacles",
    "box_random_grid",
    "random_spread_boxes",
    "open_stairs",
    "random_stairs",
    "stepping_stones",
    "narrow_beams",
    "nested_rings",
    "tilted_grid",
    "radial_mound",
    "radial_pit",
]

_MUJOCO_BLUE: Color4 = (0.25, 0.35, 0.55, 1.0)
_MUJOCO_RED: Color4 = (0.65, 0.28, 0.22, 1.0)
_MUJOCO_GREEN: Color4 = (0.34, 0.52, 0.36, 1.0)
_PLATFORM_COLOR: Color4 = (0.44, 0.46, 0.42, 1.0)
_UNILAB_VERTICAL_SCALE = 0.005


@dataclass(slots=True)
class SubTerrainCfg:
    kind: TerrainKind
    proportion: float = 1.0
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TerrainGeneratorCfg:
    size: Vector2 = (8.0, 8.0)
    num_rows: int = 1
    num_cols: int = 1
    border_width: float = 0.0
    seed: int | None = None
    curriculum: bool = False
    difficulty_range: Vector2 = (0.0, 1.0)
    origin_row: int | None = None
    origin_sub_terrain: str | None = None
    sub_terrains: dict[str, SubTerrainCfg] = field(default_factory=dict)
    horizontal_scale: float = 0.1
    base_thickness: float = 0.1
    surface_color: Color4 = (1.0, 1.0, 1.0, 1.0)
    color_mode: _core.TerrainColorMode = _core.TerrainColorMode.Palette
    height_low_color: Color4 = (0.10, 0.34, 0.30, 1.0)
    height_high_color: Color4 = (0.62, 0.44, 0.25, 1.0)
    height_range_min: float = 0.0
    height_range_max: float = 0.0
    merged_heightfield: bool = False


def darken_rgba(color: Color4, amount: float = 0.72) -> Color4:
    return (
        max(0.0, min(1.0, color[0] * amount)),
        max(0.0, min(1.0, color[1] * amount)),
        max(0.0, min(1.0, color[2] * amount)),
        color[3],
    )


def brand_ramp(value: float) -> Color4:
    t = max(0.0, min(1.0, value))
    anchors = (_MUJOCO_BLUE, _MUJOCO_GREEN, _MUJOCO_RED)
    scaled = t * (len(anchors) - 1)
    index = min(int(scaled), len(anchors) - 2)
    local = scaled - index
    a = anchors[index]
    b = anchors[index + 1]
    return tuple(a[i] + (b[i] - a[i]) * local for i in range(4))  # type: ignore[return-value]


def _heightfield_hsv_color(elevation: float) -> Color4:
    e = max(0.0, min(1.0, elevation))
    rgb = colorsys.hsv_to_rgb(0.5 - e * 0.45, 0.6 - e * 0.2, 0.4 + e * 0.3)
    return (rgb[0], rgb[1], rgb[2], 1.0)


def _get_platform_color() -> Color4:
    return _PLATFORM_COLOR


def flat(proportion: float = 1.0) -> SubTerrainCfg:
    return SubTerrainCfg("flat", proportion)


def pyramid_stairs(
    *,
    proportion: float = 1.0,
    step_height: float = 0.08,
    step_height_range: Vector2 | None = None,
    step_width: float = 0.35,
    platform_width: float = 1.0,
    border_width: float = 0.0,
    inverted: bool = False,
) -> SubTerrainCfg:
    kwargs: dict[str, Any] = {
        "step_height": step_height,
        "step_width": step_width,
        "platform_width": platform_width,
        "border_width": border_width,
        "inverted": inverted,
    }
    if step_height_range is not None:
        kwargs["step_height_range"] = step_height_range
    return SubTerrainCfg("pyramid_stairs", proportion, kwargs)


def pyramid_stairs_inv(**kwargs) -> SubTerrainCfg:
    kwargs["inverted"] = True
    return pyramid_stairs(**kwargs)


def random_grid(
    *,
    proportion: float = 1.0,
    grid_width: float = 0.5,
    height_range: Vector2 = (-0.05, 0.18),
    platform_width: float = 1.0,
) -> SubTerrainCfg:
    return box_random_grid(
        proportion=proportion,
        grid_width=grid_width,
        height_range=height_range,
        platform_width=platform_width,
    )


def box_random_grid(
    *,
    proportion: float = 1.0,
    grid_width: float = 0.5,
    height_range: Vector2 = (-0.05, 0.18),
    platform_width: float = 1.0,
) -> SubTerrainCfg:
    return SubTerrainCfg(
        "box_random_grid",
        proportion,
        {
            "grid_width": grid_width,
            "height_min": height_range[0],
            "height_max": height_range[1],
            "platform_width": platform_width,
        },
    )


def hf_pyramid_slope(
    *,
    proportion: float = 1.0,
    slope: float = 0.65,
    slope_range: Vector2 | None = None,
    platform_width: float = 1.0,
    border_width: float = 0.0,
    inverted: bool = False,
) -> SubTerrainCfg:
    kwargs: dict[str, Any] = {
        "slope": slope,
        "platform_width": platform_width,
        "border_width": border_width,
        "inverted": inverted,
    }
    if slope_range is not None:
        kwargs["slope_range"] = slope_range
    return SubTerrainCfg("hf_pyramid_slope", proportion, kwargs)


def hf_pyramid_slope_inv(**kwargs) -> SubTerrainCfg:
    kwargs["inverted"] = True
    return hf_pyramid_slope(**kwargs)


def random_rough(
    *,
    proportion: float = 1.0,
    noise_range: Vector2 = (-0.05, 0.05),
    noise_step: float = 0.01,
    downsampled_scale: float | None = 0.3,
    border_width: float = 0.0,
    scale_by_difficulty: bool = True,
) -> SubTerrainCfg:
    return SubTerrainCfg(
        "random_rough",
        proportion,
        {
            "noise_min": noise_range[0],
            "noise_max": noise_range[1],
            "noise_step": noise_step,
            "downsampled_scale": downsampled_scale,
            "border_width": border_width,
            "scale_by_difficulty": scale_by_difficulty,
        },
    )


def wave_terrain(
    *,
    proportion: float = 1.0,
    amplitude: float = 0.08,
    amplitude_range: Vector2 | None = None,
    num_waves: float = 2.0,
    border_width: float = 0.0,
) -> SubTerrainCfg:
    kwargs: dict[str, Any] = {
        "amplitude": amplitude,
        "num_waves": num_waves,
        "border_width": border_width,
    }
    if amplitude_range is not None:
        kwargs["amplitude_range"] = amplitude_range
    return SubTerrainCfg("wave", proportion, kwargs)


def wave(**kwargs) -> SubTerrainCfg:
    return wave_terrain(**kwargs)


def perlin_noise(
    *,
    proportion: float = 1.0,
    amplitude: float = 0.08,
    frequency: float = 2.0,
    octaves: int = 3,
) -> SubTerrainCfg:
    return SubTerrainCfg(
        "perlin_noise",
        proportion,
        {"amplitude": amplitude, "frequency": frequency, "octaves": octaves},
    )


def discrete_obstacles(
    *,
    proportion: float = 1.0,
    obstacle_count: int = 20,
    obstacle_size: Vector2 = (0.35, 0.9),
    height_range: Vector2 = (0.05, 0.35),
    platform_width: float = 1.0,
) -> SubTerrainCfg:
    return SubTerrainCfg(
        "discrete_obstacles",
        proportion,
        {
            "obstacle_count": obstacle_count,
            "size_min": obstacle_size[0],
            "size_max": obstacle_size[1],
            "height_min": height_range[0],
            "height_max": height_range[1],
            "platform_width": platform_width,
        },
    )


def random_spread_boxes(**kwargs) -> SubTerrainCfg:
    return discrete_obstacles(**kwargs)


def open_stairs(
    *,
    proportion: float = 1.0,
    step_width: float = 0.4,
    step_height: float = 0.08,
    platform_width: float = 1.0,
) -> SubTerrainCfg:
    return SubTerrainCfg(
        "open_stairs",
        proportion,
        {"step_width": step_width, "step_height": step_height, "platform_width": platform_width},
    )


def random_stairs(
    *,
    proportion: float = 1.0,
    step_width: float = 0.4,
    height_range: Vector2 = (-0.08, 0.16),
    platform_width: float = 1.0,
) -> SubTerrainCfg:
    return SubTerrainCfg(
        "random_stairs",
        proportion,
        {
            "step_width": step_width,
            "height_min": height_range[0],
            "height_max": height_range[1],
            "platform_width": platform_width,
        },
    )


def stepping_stones(
    *,
    proportion: float = 1.0,
    stone_size: float = 0.55,
    stone_distance: float = 0.25,
    height_range: Vector2 = (-0.04, 0.08),
    platform_width: float = 1.0,
) -> SubTerrainCfg:
    return SubTerrainCfg(
        "stepping_stones",
        proportion,
        {
            "stone_size": stone_size,
            "stone_distance": stone_distance,
            "height_min": height_range[0],
            "height_max": height_range[1],
            "platform_width": platform_width,
        },
    )


def narrow_beams(
    *,
    proportion: float = 1.0,
    beam_width: float = 0.18,
    gap: float = 0.45,
    height: float = 0.08,
) -> SubTerrainCfg:
    return SubTerrainCfg(
        "narrow_beams",
        proportion,
        {"beam_width": beam_width, "gap": gap, "height": height},
    )


def nested_rings(
    *,
    proportion: float = 1.0,
    ring_width: float = 0.35,
    step_height: float = 0.08,
    platform_width: float = 1.0,
) -> SubTerrainCfg:
    return SubTerrainCfg(
        "nested_rings",
        proportion,
        {"ring_width": ring_width, "step_height": step_height, "platform_width": platform_width},
    )


def tilted_grid(
    *,
    proportion: float = 1.0,
    tile_size: float = 0.65,
    gap: float = 0.08,
    tilt_range: Vector2 = (-12.0, 12.0),
    height_range: Vector2 = (-0.04, 0.08),
    platform_width: float = 1.0,
) -> SubTerrainCfg:
    return SubTerrainCfg(
        "tilted_grid",
        proportion,
        {
            "tile_size": tile_size,
            "gap": gap,
            "tilt_min": tilt_range[0],
            "tilt_max": tilt_range[1],
            "height_min": height_range[0],
            "height_max": height_range[1],
            "platform_width": platform_width,
        },
    )


def radial_mound(
    *,
    proportion: float = 1.0,
    height: float = 0.55,
    radius: float = 1.15,
    flat_radius: float = 0.25,
) -> SubTerrainCfg:
    return SubTerrainCfg(
        "radial_mound",
        proportion,
        {"height": height, "radius": radius, "flat_radius": flat_radius},
    )


def radial_pit(
    *,
    proportion: float = 1.0,
    depth: float = 0.55,
    radius: float = 1.15,
    flat_radius: float = 0.25,
) -> SubTerrainCfg:
    return SubTerrainCfg(
        "radial_pit",
        proportion,
        {"depth": depth, "radius": radius, "flat_radius": flat_radius},
    )


def go1_training_terrains() -> dict[str, SubTerrainCfg]:
    """UniLab Go1 rough terrain presets used by the PPO training profile."""

    return {
        "flat": flat(proportion=0.0),
        "pyramid_stairs": pyramid_stairs(
            proportion=0.1,
            step_height_range=(0.025, 0.10),
            step_width=0.40,
            platform_width=3.0,
            border_width=0.2,
        ),
        "pyramid_stairs_inv": pyramid_stairs_inv(
            proportion=0.1,
            step_height_range=(0.025, 0.10),
            step_width=0.40,
            platform_width=3.0,
            border_width=0.2,
        ),
        "hf_pyramid_slope": hf_pyramid_slope(
            proportion=0.2,
            slope_range=(0.0, 0.3),
            platform_width=2.0,
            border_width=0.2,
        ),
        "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
            proportion=0.2,
            slope_range=(0.0, 0.3),
            platform_width=2.0,
            border_width=0.2,
        ),
        "random_rough": random_rough(
            proportion=0.3,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            downsampled_scale=None,
            border_width=0.2,
            scale_by_difficulty=False,
        ),
        "wave_terrain": wave_terrain(
            proportion=0.3,
            amplitude_range=(0.0, 0.12),
            num_waves=4.0,
            border_width=0.2,
        ),
    }


def go1_rough_terrain_cfg(*, seed: int = 42, curriculum: bool = False) -> TerrainGeneratorCfg:
    """Go1 rough terrain grid aligned with UniLab's Go1 PPO MuJoCo profile."""

    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=6,
        num_cols=6,
        seed=seed,
        curriculum=curriculum,
        sub_terrains=go1_training_terrains(),
        horizontal_scale=0.2,
        base_thickness=0.6,
        color_mode=_core.TerrainColorMode.Palette,
        merged_heightfield=True,
    )


def showcase_terrains() -> dict[str, SubTerrainCfg]:
    return {
        "blue_stairs": pyramid_stairs(step_height=0.10, step_width=0.28, platform_width=0.75),
        "pit": radial_pit(depth=0.65, radius=1.20, flat_radius=0.28),
        "flat": flat(),
        "bumps": wave_terrain(amplitude=0.16, num_waves=5.0),
        "rough": random_rough(noise_range=(-0.08, 0.10), noise_step=0.01, downsampled_scale=0.18),
        "red_stairs_inv": pyramid_stairs_inv(step_height=0.10, step_width=0.28, platform_width=0.75),
        "mound": radial_mound(height=0.65, radius=1.20, flat_radius=0.28),
        "obstacles": discrete_obstacles(obstacle_count=26, obstacle_size=(0.18, 0.38), height_range=(0.04, 0.18)),
        "grid": box_random_grid(grid_width=0.38, height_range=(-0.03, 0.16), platform_width=0.65),
        "waves": wave_terrain(amplitude=0.12, num_waves=2.5),
    }


def create_terrain_node(cfg: TerrainGeneratorCfg, name: str = "terrain") -> _core.Terrain3D:
    terrain = _core.create_node("Terrain3D", name)
    if not isinstance(terrain, _core.Terrain3D):
        raise TypeError("Gobot core did not create a Terrain3D node")

    terrain.surface_color = cfg.surface_color
    terrain.color_mode = cfg.color_mode
    terrain.height_low_color = cfg.height_low_color
    terrain.height_high_color = cfg.height_high_color
    terrain.height_range_min = cfg.height_range_min
    terrain.height_range_max = cfg.height_range_max
    if cfg.merged_heightfield:
        _populate_merged_heightfield_terrain(terrain, cfg)
        return terrain

    terrains = cfg.sub_terrains or {"flat": flat()}
    choices = _normalized_choices(terrains)
    curriculum_items = list(terrains.items()) or [("flat", flat())]
    curriculum_choices = _ordered_curriculum_choices(curriculum_items, cfg)
    rng = np.random.default_rng(cfg.seed)
    origins: list[Vector3] = []

    column_count = len(curriculum_choices) if cfg.curriculum else max(cfg.num_cols, 0)
    patch_span_x = max(cfg.num_rows, 0) * cfg.size[0]
    patch_span_y = column_count * cfg.size[1]
    total_x = patch_span_x + 2.0 * cfg.border_width
    total_y = patch_span_y + 2.0 * cfg.border_width

    if cfg.origin_row is not None and cfg.num_rows > 0:
        origin_x = -float(np.clip(cfg.origin_row, 0, cfg.num_rows - 1)) * cfg.size[0]
    else:
        origin_x = -patch_span_x * 0.5 + cfg.size[0] * 0.5
    if cfg.origin_sub_terrain and cfg.curriculum and column_count > 0:
        origin_y = -float(column_count // 2) * cfg.size[1]
    else:
        origin_y = -patch_span_y * 0.5 + cfg.size[1] * 0.5

    patch_min_x = origin_x - cfg.size[0] * 0.5
    patch_max_x = origin_x + (max(cfg.num_rows, 1) - 1) * cfg.size[0] + cfg.size[0] * 0.5
    patch_min_y = origin_y - cfg.size[1] * 0.5
    patch_max_y = origin_y + (max(column_count, 1) - 1) * cfg.size[1] + cfg.size[1] * 0.5
    terrain_center = ((patch_min_x + patch_max_x) * 0.5, (patch_min_y + patch_max_y) * 0.5, -0.55)

    if cfg.border_width > 0.0:
        terrain.add_box(terrain_center, (total_x, total_y, 0.9), color=darken_rgba(_get_platform_color(), 0.55))

    for row in range(max(cfg.num_rows, 0)):
        for col in range(column_count):
            patch_center = (origin_x + row * cfg.size[0], origin_y + col * cfg.size[1], 0.0)
            if cfg.curriculum:
                lower, upper = cfg.difficulty_range
                row_fraction = (row + float(rng.uniform())) / max(cfg.num_rows, 1)
                difficulty = lower + (upper - lower) * row_fraction
                sub_cfg = curriculum_choices[col % len(curriculum_choices)]
            else:
                difficulty = float(rng.uniform(cfg.difficulty_range[0], cfg.difficulty_range[1]))
                sub_cfg = choices[int(rng.integers(0, len(choices)))]
            origins.append((patch_center[0], patch_center[1], _add_patch(terrain, cfg, sub_cfg, patch_center, difficulty, rng)))

    terrain.spawn_origins = origins
    return terrain


def _populate_merged_heightfield_terrain(terrain: _core.Terrain3D, cfg: TerrainGeneratorCfg) -> None:
    terrains = cfg.sub_terrains or {"flat": flat()}
    rng = np.random.default_rng(cfg.seed)
    sub_terrain_items = list(terrains.items())
    sub_terrain_cfgs = [item[1] for item in sub_terrain_items]
    proportions = np.asarray([max(float(sub_cfg.proportion), 0.0) for sub_cfg in sub_terrain_cfgs], dtype=np.float64)
    if not np.any(proportions > 0.0):
        proportions[:] = 1.0
    proportions /= np.sum(proportions)

    tile_x_px = int(round(cfg.size[0] / cfg.horizontal_scale))
    tile_y_px = int(round(cfg.size[1] / cfg.horizontal_scale))
    border_px = int(round(cfg.border_width / cfg.horizontal_scale))
    num_cols = len(sub_terrain_cfgs) if cfg.curriculum else max(cfg.num_cols, 0)
    rows_y = num_cols * tile_y_px + 2 * border_px
    cols_x = max(cfg.num_rows, 0) * tile_x_px + 2 * border_px
    if rows_y < 2 or cols_x < 2:
        terrain.spawn_origins = []
        return

    heights_yx = np.zeros((rows_y, cols_x), dtype=np.float64)
    spawn_origins: list[Vector3] = []
    curriculum_choices = [item[1] for item in sub_terrain_items]
    for row in range(max(cfg.num_rows, 0)):
        for col in range(num_cols):
            lower, upper = cfg.difficulty_range
            if cfg.curriculum:
                difficulty = (row + float(rng.uniform())) / max(cfg.num_rows, 1)
                difficulty = float(lower) + (float(upper) - float(lower)) * difficulty
                sub_cfg = curriculum_choices[col % len(curriculum_choices)]
            else:
                sub_cfg = sub_terrain_cfgs[int(rng.choice(len(sub_terrain_cfgs), p=proportions))]
                difficulty = float(rng.uniform(float(lower), float(upper)))

            heights_xy, origin_local = _unilab_patch_heightfield(cfg, sub_cfg, difficulty, rng)
            if heights_xy.shape != (tile_x_px, tile_y_px):
                raise ValueError(
                    "merged terrain patch shape does not match TerrainGeneratorCfg.size: "
                    f"{heights_xy.shape} != {(tile_x_px, tile_y_px)}"
                )
            x0 = border_px + row * tile_x_px
            y0 = border_px + (num_cols - 1 - col) * tile_y_px
            heights_yx[y0 : y0 + tile_y_px, x0 : x0 + tile_x_px] = heights_xy.T
            world_position = _unilab_sub_terrain_position(cfg, row, col, num_cols)
            spawn = origin_local + world_position
            spawn_origins.append((float(spawn[0]), float(spawn[1]), float(spawn[2])))

    z_min = float(np.min(heights_yx))
    physical_heights = heights_yx - z_min
    center = (0.0, 0.0, 0.0)
    size = (cols_x * cfg.horizontal_scale, rows_y * cfg.horizontal_scale)
    terrain.add_heightfield(
        center,
        size,
        rows_y,
        cols_x,
        physical_heights.astype(float).reshape(-1).tolist(),
        cfg.base_thickness,
        _normalize_elevation(heights_yx).astype(float).reshape(-1).tolist(),
        z_min,
    )
    terrain.spawn_origins = spawn_origins


def _ordered_curriculum_choices(
    items: list[tuple[str, SubTerrainCfg]],
    cfg: TerrainGeneratorCfg,
) -> list[SubTerrainCfg]:
    if not items:
        return [flat()]
    if not cfg.curriculum or not cfg.origin_sub_terrain:
        return [sub_cfg for _, sub_cfg in items]

    origin_index = next(
        (index for index, (name, _) in enumerate(items) if name == cfg.origin_sub_terrain),
        None,
    )
    if origin_index is None:
        return [sub_cfg for _, sub_cfg in items]

    target_index = len(items) // 2
    shift = target_index - origin_index
    return [items[(column - shift) % len(items)][1] for column in range(len(items))]


def _normalized_choices(sub_terrains: dict[str, SubTerrainCfg]) -> list[SubTerrainCfg]:
    weighted: list[SubTerrainCfg] = []
    for cfg in sub_terrains.values():
        count = int(round(max(cfg.proportion, 0.0) * 100.0))
        if count > 0:
            weighted.extend([cfg] * count)
    return weighted or [flat()]


def _add_patch(
    terrain: _core.Terrain3D,
    cfg: TerrainGeneratorCfg,
    sub_cfg: SubTerrainCfg,
    center: Vector3,
    difficulty: float,
    rng: np.random.Generator,
) -> float:
    if sub_cfg.kind == "flat":
        return _add_flat(terrain, cfg, center)
    if sub_cfg.kind == "pyramid_stairs":
        return _add_pyramid_stairs(terrain, cfg, sub_cfg, center, difficulty)
    if sub_cfg.kind in {"random_grid", "box_random_grid"}:
        return _add_random_grid(terrain, cfg, sub_cfg, center, difficulty, rng)
    if sub_cfg.kind in {"discrete_obstacles", "random_spread_boxes"}:
        return _add_discrete_obstacles_heightfield(terrain, cfg, sub_cfg, center, difficulty, rng)
    if sub_cfg.kind == "open_stairs":
        return _add_open_stairs(terrain, cfg, sub_cfg, center, difficulty)
    if sub_cfg.kind == "random_stairs":
        return _add_random_stairs(terrain, cfg, sub_cfg, center, difficulty, rng)
    if sub_cfg.kind == "stepping_stones":
        return _add_stepping_stones_heightfield(terrain, cfg, sub_cfg, center, difficulty, rng)
    if sub_cfg.kind == "narrow_beams":
        return _add_narrow_beams_heightfield(terrain, cfg, sub_cfg, center, difficulty)
    if sub_cfg.kind == "nested_rings":
        return _add_pyramid_stairs(terrain, cfg, SubTerrainCfg("pyramid_stairs", kwargs=sub_cfg.kwargs), center, difficulty)
    if sub_cfg.kind == "tilted_grid":
        return _add_tilted_grid(terrain, cfg, sub_cfg, center, difficulty, rng)
    if sub_cfg.kind == "radial_mound":
        return _add_radial_heightfield(terrain, cfg, sub_cfg, center, difficulty, inverted=False)
    if sub_cfg.kind == "radial_pit":
        return _add_radial_heightfield(terrain, cfg, sub_cfg, center, difficulty, inverted=True)

    heights = _heightfield(cfg, sub_cfg, difficulty, rng)
    return _add_heightfield_surface(terrain, cfg, center, heights)


def _add_radial_heightfield(
    terrain: _core.Terrain3D,
    cfg: TerrainGeneratorCfg,
    sub_cfg: SubTerrainCfg,
    center: Vector3,
    difficulty: float,
    *,
    inverted: bool,
) -> float:
    rows, cols = _heightfield_shape(cfg)
    x = np.linspace(-cfg.size[0] * 0.5, cfg.size[0] * 0.5, cols)
    y = np.linspace(-cfg.size[1] * 0.5, cfg.size[1] * 0.5, rows)
    xx, yy = np.meshgrid(x, y)
    radius = max(float(sub_cfg.kwargs.get("radius", 1.15)), 1.0e-6)
    flat_radius = max(float(sub_cfg.kwargs.get("flat_radius", 0.25)), 0.0)
    amplitude_key = "depth" if inverted else "height"
    amplitude = float(sub_cfg.kwargs.get(amplitude_key, 0.55)) * (0.75 + difficulty * 0.35)
    distance = np.sqrt(xx * xx + yy * yy)
    falloff = 1.0 - np.clip((distance - flat_radius) / max(radius - flat_radius, 1.0e-6), 0.0, 1.0)
    falloff = falloff * falloff * (3.0 - 2.0 * falloff)
    heights = amplitude * falloff
    if inverted:
        heights = -heights
    return _add_heightfield_surface(terrain, cfg, center, heights)


def _unilab_patch_heightfield(
    cfg: TerrainGeneratorCfg,
    sub_cfg: SubTerrainCfg,
    difficulty: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if sub_cfg.kind == "flat":
        width_px, length_px = _unilab_patch_shape(cfg)
        return np.zeros((width_px, length_px), dtype=np.float64), np.asarray(
            [cfg.size[0] * 0.5, cfg.size[1] * 0.5, 0.0],
            dtype=np.float64,
        )
    if sub_cfg.kind == "pyramid_stairs":
        return _unilab_pyramid_stairs_heightfield(cfg, sub_cfg, difficulty)
    if sub_cfg.kind == "hf_pyramid_slope":
        return _unilab_pyramid_slope_heightfield(cfg, sub_cfg, difficulty)
    if sub_cfg.kind == "random_rough":
        return _unilab_random_rough_heightfield(cfg, sub_cfg, rng)
    if sub_cfg.kind == "wave":
        return _unilab_wave_heightfield(cfg, sub_cfg, difficulty)
    return np.zeros(_unilab_patch_shape(cfg), dtype=np.float64), np.asarray(
        [cfg.size[0] * 0.5, cfg.size[1] * 0.5, 0.0],
        dtype=np.float64,
    )


def _unilab_patch_shape(cfg: TerrainGeneratorCfg) -> tuple[int, int]:
    return (
        max(2, int(round(cfg.size[0] / cfg.horizontal_scale))),
        max(2, int(round(cfg.size[1] / cfg.horizontal_scale))),
    )


def _unilab_sub_terrain_position(
    cfg: TerrainGeneratorCfg,
    row: int,
    col: int,
    num_cols: int,
) -> np.ndarray:
    return np.asarray(
        [
            -cfg.num_rows * cfg.size[0] * 0.5 + row * cfg.size[0],
            -num_cols * cfg.size[1] * 0.5 + col * cfg.size[1],
            0.0,
        ],
        dtype=np.float64,
    )


def _unilab_noise_to_physical(
    noise: np.ndarray,
    *,
    z_offset: float = 0.0,
) -> np.ndarray:
    return (noise.astype(np.float64) - int(np.min(noise))) * _UNILAB_VERTICAL_SCALE + float(z_offset)


def _unilab_pyramid_stairs_heightfield(
    cfg: TerrainGeneratorCfg,
    sub_cfg: SubTerrainCfg,
    difficulty: float,
) -> tuple[np.ndarray, np.ndarray]:
    step_height = _difficulty_value(sub_cfg.kwargs, "step_height", "step_height_range", difficulty, 0.08)
    width_px, length_px = _unilab_patch_shape(cfg)
    step_px = int(round(float(sub_cfg.kwargs.get("step_width", 0.4)) / cfg.horizontal_scale))
    platform_px = int(round(float(sub_cfg.kwargs.get("platform_width", 3.0)) / cfg.horizontal_scale))
    border_px = int(round(float(sub_cfg.kwargs.get("border_width", 0.0)) / cfg.horizontal_scale))
    step_units = int(round(step_height / _UNILAB_VERTICAL_SCALE))
    noise = np.zeros((width_px, length_px), dtype=np.int16)
    inner = min(width_px, length_px) - 2 * border_px
    n_steps = max(0, (inner - platform_px) // (2 * step_px)) if step_px > 0 else 0
    inverted = bool(sub_cfg.kwargs.get("inverted", False))
    sign = -1 if inverted else 1

    for step in range(n_steps):
        lo_x = border_px + step * step_px
        hi_x = width_px - border_px - step * step_px
        lo_y = border_px + step * step_px
        hi_y = length_px - border_px - step * step_px
        noise[lo_x:hi_x, lo_y:hi_y] = sign * (step + 1) * step_units

    plat_lo_x = border_px + n_steps * step_px
    plat_hi_x = width_px - border_px - n_steps * step_px
    plat_lo_y = border_px + n_steps * step_px
    plat_hi_y = length_px - border_px - n_steps * step_px
    noise[plat_lo_x:plat_hi_x, plat_lo_y:plat_hi_y] = sign * (n_steps + 1) * step_units

    z_offset = -float(np.max(noise) - np.min(noise)) * _UNILAB_VERTICAL_SCALE if inverted else 0.0
    spawn_z = sign * (n_steps + 1) * step_units * _UNILAB_VERTICAL_SCALE
    origin = np.asarray([cfg.size[0] * 0.5, cfg.size[1] * 0.5, spawn_z], dtype=np.float64)
    return _unilab_noise_to_physical(noise, z_offset=z_offset), origin


def _unilab_pyramid_slope_heightfield(
    cfg: TerrainGeneratorCfg,
    sub_cfg: SubTerrainCfg,
    difficulty: float,
) -> tuple[np.ndarray, np.ndarray]:
    inverted = bool(sub_cfg.kwargs.get("inverted", False))
    slope = _difficulty_value(sub_cfg.kwargs, "slope", "slope_range", difficulty, 0.35)
    if inverted:
        slope = -slope

    width_px, length_px = _unilab_patch_shape(cfg)
    border_px = int(float(sub_cfg.kwargs.get("border_width", 0.0)) / cfg.horizontal_scale)
    inner_width_px = width_px - 2 * border_px
    inner_length_px = length_px - 2 * border_px
    noise = np.zeros((width_px, length_px), dtype=np.int16)

    target_width_px = inner_width_px if border_px > 0 else width_px
    target_length_px = inner_length_px if border_px > 0 else length_px
    height_max = int(slope * (target_width_px * cfg.horizontal_scale) * 0.5 / _UNILAB_VERTICAL_SCALE)
    center_x = int(target_width_px * 0.5)
    center_y = int(target_length_px * 0.5)
    x = np.arange(0, target_width_px)
    y = np.arange(0, target_length_px)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = ((center_x - np.abs(center_x - xx)) / max(center_x, 1)).reshape(target_width_px, 1)
    yy = ((center_y - np.abs(center_y - yy)) / max(center_y, 1)).reshape(1, target_length_px)
    hf_raw = height_max * xx * yy

    platform_px = int(float(sub_cfg.kwargs.get("platform_width", 2.0)) / cfg.horizontal_scale / 2)
    x_pf = target_width_px // 2 - platform_px
    y_pf = target_length_px // 2 - platform_px
    z_pf = hf_raw[x_pf, y_pf] if x_pf >= 0 and y_pf >= 0 else 0
    hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))
    if border_px > 0:
        noise[border_px : width_px - border_px, border_px : length_px - border_px] = np.rint(hf_raw).astype(np.int16)
    else:
        noise = np.rint(hf_raw).astype(np.int16)

    max_h = float(np.max(noise) - np.min(noise)) * _UNILAB_VERTICAL_SCALE
    z_offset = -max_h if inverted else 0.0
    spawn_height = z_offset if inverted else max_h
    origin = np.asarray([cfg.size[0] * 0.5, cfg.size[1] * 0.5, spawn_height], dtype=np.float64)
    return _unilab_noise_to_physical(noise, z_offset=z_offset), origin


def _unilab_random_rough_heightfield(
    cfg: TerrainGeneratorCfg,
    sub_cfg: SubTerrainCfg,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    width_px, length_px = _unilab_patch_shape(cfg)
    border_px = int(float(sub_cfg.kwargs.get("border_width", 0.0)) / cfg.horizontal_scale)
    noise = np.zeros((width_px, length_px), dtype=np.int16)
    downsampled_scale = sub_cfg.kwargs.get("downsampled_scale")
    if downsampled_scale is None:
        scale = cfg.horizontal_scale
    else:
        scale = max(float(downsampled_scale), cfg.horizontal_scale)

    target_width_px = width_px - 2 * border_px if border_px > 0 else width_px
    target_length_px = length_px - 2 * border_px if border_px > 0 else length_px
    target_size = (target_width_px * cfg.horizontal_scale, target_length_px * cfg.horizontal_scale)
    width_downsampled = max(2, int(target_size[0] / scale))
    length_downsampled = max(2, int(target_size[1] / scale))
    height_min = int(float(sub_cfg.kwargs.get("noise_min", -0.05)) / _UNILAB_VERTICAL_SCALE)
    height_max = int(float(sub_cfg.kwargs.get("noise_max", 0.05)) / _UNILAB_VERTICAL_SCALE)
    height_step = max(1, int(float(sub_cfg.kwargs.get("noise_step", 0.01)) / _UNILAB_VERTICAL_SCALE))
    height_range = np.arange(height_min, height_max + height_step, height_step)
    downsampled = rng.choice(height_range, size=(width_downsampled, length_downsampled))
    upsampled = _bilinear_resample_grid(
        np.linspace(0.0, target_size[0], width_downsampled),
        np.linspace(0.0, target_size[1], length_downsampled),
        downsampled.astype(np.float64),
        np.linspace(0.0, target_size[0], target_width_px),
        np.linspace(0.0, target_size[1], target_length_px),
    )
    if border_px > 0:
        noise[border_px : width_px - border_px, border_px : length_px - border_px] = np.rint(upsampled).astype(np.int16)
    else:
        noise = np.rint(upsampled).astype(np.int16)
    spawn_height = (float(sub_cfg.kwargs.get("noise_min", -0.05)) + float(sub_cfg.kwargs.get("noise_max", 0.05))) * 0.5
    origin = np.asarray([cfg.size[0] * 0.5, cfg.size[1] * 0.5, spawn_height], dtype=np.float64)
    return _unilab_noise_to_physical(noise), origin


def _unilab_wave_heightfield(
    cfg: TerrainGeneratorCfg,
    sub_cfg: SubTerrainCfg,
    difficulty: float,
) -> tuple[np.ndarray, np.ndarray]:
    width_px, length_px = _unilab_patch_shape(cfg)
    border_px = int(float(sub_cfg.kwargs.get("border_width", 0.0)) / cfg.horizontal_scale)
    target_width_px = width_px - 2 * border_px if border_px > 0 else width_px
    target_length_px = length_px - 2 * border_px if border_px > 0 else length_px
    amplitude = _difficulty_value(sub_cfg.kwargs, "amplitude", "amplitude_range", difficulty, 0.08)
    amplitude_units = int(0.5 * amplitude / _UNILAB_VERTICAL_SCALE)
    num_waves = max(float(sub_cfg.kwargs.get("num_waves", 1.0)), 1.0e-6)
    wave_length = target_length_px / num_waves
    wave_number = 2.0 * np.pi / max(wave_length, 1.0e-6)
    x = np.arange(0, target_width_px)
    y = np.arange(0, target_length_px)
    xx, yy = np.meshgrid(x, y, sparse=True)
    hf_raw = amplitude_units * (np.cos(yy.reshape(1, target_length_px) * wave_number) + np.sin(xx.reshape(target_width_px, 1) * wave_number))
    noise = np.zeros((width_px, length_px), dtype=np.int16)
    if border_px > 0:
        noise[border_px : width_px - border_px, border_px : length_px - border_px] = np.rint(hf_raw).astype(np.int16)
    else:
        noise = np.rint(hf_raw).astype(np.int16)
    max_h = float(np.max(noise) - np.min(noise)) * _UNILAB_VERTICAL_SCALE
    origin = np.asarray([cfg.size[0] * 0.5, cfg.size[1] * 0.5, 0.0], dtype=np.float64)
    return _unilab_noise_to_physical(noise, z_offset=-max_h * 0.5), origin


def _add_heightfield_surface(terrain: _core.Terrain3D, cfg: TerrainGeneratorCfg, center: Vector3, heights: np.ndarray) -> float:
    normalized = _normalize_elevation(heights)
    rows, cols = heights.shape
    z_offset = float(np.min(heights))
    physical_range = max(float(np.max(heights) - z_offset), 1.0e-6)
    terrain.add_heightfield(
        center,
        cfg.size,
        rows,
        cols,
        ((normalized * physical_range)).astype(float).reshape(-1).tolist(),
        cfg.base_thickness,
        normalized.astype(float).reshape(-1).tolist(),
        z_offset,
    )
    return float(np.max(heights))


def _add_base_box(terrain: _core.Terrain3D, cfg: TerrainGeneratorCfg, center: Vector3, color: Color4) -> None:
    terrain.add_box((center[0], center[1], -0.5), (cfg.size[0], cfg.size[1], 1.0), color=color)


def _add_flat(terrain: _core.Terrain3D, cfg: TerrainGeneratorCfg, center: Vector3) -> float:
    _add_base_box(terrain, cfg, center, _get_platform_color())
    return 0.0


def _add_box_top(terrain: _core.Terrain3D, x: float, y: float, height: float, sx: float, sy: float, color: Color4) -> None:
    base_z = -1.0
    size_z = max(height - base_z, 0.01)
    center_z = (height + base_z) * 0.5
    terrain.add_box((x, y, center_z), (sx, sy, size_z), color=color)


def _add_pyramid_stairs(terrain: _core.Terrain3D, cfg: TerrainGeneratorCfg, sub_cfg: SubTerrainCfg, center: Vector3, difficulty: float) -> float:
    step_width = float(sub_cfg.kwargs.get("step_width", 0.35))
    step_height = _difficulty_value(sub_cfg.kwargs, "step_height", "step_height_range", difficulty, 0.08)
    platform_width = float(sub_cfg.kwargs.get("platform_width", 1.0))
    border_width = max(0.0, float(sub_cfg.kwargs.get("border_width", 0.0)))
    inverted = bool(sub_cfg.kwargs.get("inverted", False))
    if inverted:
        return _add_inverted_pyramid_stairs(terrain, cfg, step_width, step_height, platform_width, border_width, center)

    half_x = max(1.0e-6, (cfg.size[0] - 2.0 * border_width) * 0.5)
    half_y = max(1.0e-6, (cfg.size[1] - 2.0 * border_width) * 0.5)
    layer = 0
    extrema = 0.0
    if border_width > 0.0:
        _add_flat(terrain, cfg, center)
    while half_x > platform_width * 0.5 and half_y > platform_width * 0.5:
        height = step_height * layer * (-1.0 if inverted else 1.0)
        shade = 0.55 + 0.45 * min(layer / 8.0, 1.0)
        base_color = _MUJOCO_RED if inverted else _MUJOCO_BLUE
        color = (
            min(base_color[0] * shade, 1.0),
            min(base_color[1] * shade, 1.0),
            min(base_color[2] * shade, 1.0),
            base_color[3],
        )
        _add_box_top(terrain, center[0], center[1], height, half_x * 2.0, half_y * 2.0, color)
        extrema = max(extrema, height)
        half_x -= step_width
        half_y -= step_width
        layer += 1
    _add_box_top(terrain, center[0], center[1], 0.0, platform_width, platform_width, _get_platform_color())
    return extrema


def _add_inverted_pyramid_stairs(
    terrain: _core.Terrain3D,
    cfg: TerrainGeneratorCfg,
    step_width: float,
    step_height: float,
    platform_width: float,
    border_width: float,
    center: Vector3,
) -> float:
    half_x = max(1.0e-6, (cfg.size[0] - 2.0 * border_width) * 0.5)
    half_y = max(1.0e-6, (cfg.size[1] - 2.0 * border_width) * 0.5)
    target_half = platform_width * 0.5
    layer = 0
    min_height = 0.0

    if border_width > 0.0:
        _add_flat(terrain, cfg, center)

    while half_x > target_half and half_y > target_half:
        inner_x = max(half_x - step_width, target_half)
        inner_y = max(half_y - step_width, target_half)
        height = -step_height * layer
        shade = 0.55 + 0.45 * min(layer / 8.0, 1.0)
        color = (
            min(_MUJOCO_RED[0] * shade, 1.0),
            min(_MUJOCO_RED[1] * shade, 1.0),
            min(_MUJOCO_RED[2] * shade, 1.0),
            _MUJOCO_RED[3],
        )

        strip_y = half_y - inner_y
        if strip_y > 1.0e-6:
            _add_box_top(terrain, center[0], center[1] + (inner_y + half_y) * 0.5, height, half_x * 2.0, strip_y, color)
            _add_box_top(terrain, center[0], center[1] - (inner_y + half_y) * 0.5, height, half_x * 2.0, strip_y, color)
        strip_x = half_x - inner_x
        if strip_x > 1.0e-6 and inner_y > 1.0e-6:
            _add_box_top(terrain, center[0] + (inner_x + half_x) * 0.5, center[1], height, strip_x, inner_y * 2.0, color)
            _add_box_top(terrain, center[0] - (inner_x + half_x) * 0.5, center[1], height, strip_x, inner_y * 2.0, color)

        min_height = height
        half_x = inner_x
        half_y = inner_y
        layer += 1

    center_height = -step_height * layer
    _add_box_top(terrain, center[0], center[1], center_height, platform_width, platform_width, _get_platform_color())
    return min_height


def _add_random_grid(terrain: _core.Terrain3D, cfg: TerrainGeneratorCfg, sub_cfg: SubTerrainCfg, center: Vector3, difficulty: float, rng: np.random.Generator) -> float:
    grid_width = float(sub_cfg.kwargs.get("grid_width", 0.5))
    height_min = float(sub_cfg.kwargs.get("height_min", -0.05)) * (0.5 + difficulty)
    height_max = float(sub_cfg.kwargs.get("height_max", 0.18)) * (0.5 + difficulty)
    platform_width = float(sub_cfg.kwargs.get("platform_width", 1.0))
    nx = max(1, int(round(cfg.size[0] / grid_width)))
    ny = max(1, int(round(cfg.size[1] / grid_width)))
    cell_x = cfg.size[0] / nx
    cell_y = cfg.size[1] / ny
    max_height = 0.0
    for ix in range(nx):
        for iy in range(ny):
            x = center[0] - cfg.size[0] * 0.5 + (ix + 0.5) * cell_x
            y = center[1] - cfg.size[1] * 0.5 + (iy + 0.5) * cell_y
            if abs(x - center[0]) <= platform_width * 0.5 and abs(y - center[1]) <= platform_width * 0.5:
                height = 0.0
                color = _get_platform_color()
            else:
                height = float(rng.uniform(height_min, height_max))
                color = brand_ramp((height - height_min) / max(height_max - height_min, 1.0e-6))
            _add_box_top(terrain, x, y, height, cell_x, cell_y, color)
            max_height = max(max_height, height)
    return max_height


def _add_discrete_obstacles(terrain: _core.Terrain3D, cfg: TerrainGeneratorCfg, sub_cfg: SubTerrainCfg, center: Vector3, difficulty: float, rng: np.random.Generator) -> float:
    count = max(0, int(sub_cfg.kwargs.get("obstacle_count", 20) * (0.5 + difficulty)))
    size_min = float(sub_cfg.kwargs.get("size_min", 0.35))
    size_max = float(sub_cfg.kwargs.get("size_max", 0.9))
    height_min = float(sub_cfg.kwargs.get("height_min", 0.05))
    height_max = float(sub_cfg.kwargs.get("height_max", 0.35)) * (0.5 + difficulty)
    platform_width = float(sub_cfg.kwargs.get("platform_width", 1.0))
    max_height = 0.0
    _add_flat(terrain, cfg, center)
    for _ in range(count):
        x = float(rng.uniform(center[0] - cfg.size[0] * 0.5, center[0] + cfg.size[0] * 0.5))
        y = float(rng.uniform(center[1] - cfg.size[1] * 0.5, center[1] + cfg.size[1] * 0.5))
        if abs(x - center[0]) <= platform_width * 0.5 and abs(y - center[1]) <= platform_width * 0.5:
            continue
        sx = float(rng.uniform(size_min, size_max))
        sy = float(rng.uniform(size_min, size_max))
        height = float(rng.uniform(height_min, height_max))
        _add_box_top(terrain, x, y, height, sx, sy, brand_ramp(height / max(height_max, 1.0e-6)))
        max_height = max(max_height, height)
    return max_height


def _add_discrete_obstacles_heightfield(
    terrain: _core.Terrain3D,
    cfg: TerrainGeneratorCfg,
    sub_cfg: SubTerrainCfg,
    center: Vector3,
    difficulty: float,
    rng: np.random.Generator,
) -> float:
    count = max(0, int(sub_cfg.kwargs.get("obstacle_count", 20) * (0.5 + difficulty)))
    size_min = float(sub_cfg.kwargs.get("size_min", 0.35))
    size_max = float(sub_cfg.kwargs.get("size_max", 0.9))
    height_min = float(sub_cfg.kwargs.get("height_min", 0.05))
    height_max = float(sub_cfg.kwargs.get("height_max", 0.35)) * (0.5 + difficulty)
    platform_width = float(sub_cfg.kwargs.get("platform_width", 1.0))
    rows, cols = _heightfield_shape(cfg)
    heights = np.zeros((rows, cols), dtype=float)
    xs = np.linspace(-cfg.size[0] * 0.5, cfg.size[0] * 0.5, cols)
    ys = np.linspace(-cfg.size[1] * 0.5, cfg.size[1] * 0.5, rows)
    xx, yy = np.meshgrid(xs, ys)
    for _ in range(count):
        x = float(rng.uniform(-cfg.size[0] * 0.5, cfg.size[0] * 0.5))
        y = float(rng.uniform(-cfg.size[1] * 0.5, cfg.size[1] * 0.5))
        if abs(x) <= platform_width * 0.5 and abs(y) <= platform_width * 0.5:
            continue
        sx = float(rng.uniform(size_min, size_max))
        sy = float(rng.uniform(size_min, size_max))
        height = float(rng.uniform(height_min, height_max))
        mask = (np.abs(xx - x) <= sx * 0.5) & (np.abs(yy - y) <= sy * 0.5)
        heights[mask] = np.maximum(heights[mask], height)
    return _add_heightfield_surface(terrain, cfg, center, heights)


def _add_open_stairs(terrain: _core.Terrain3D, cfg: TerrainGeneratorCfg, sub_cfg: SubTerrainCfg, center: Vector3, difficulty: float) -> float:
    step_width = float(sub_cfg.kwargs.get("step_width", 0.4))
    step_height = float(sub_cfg.kwargs.get("step_height", 0.08)) * (0.5 + difficulty)
    platform_width = float(sub_cfg.kwargs.get("platform_width", 1.0))
    steps = max(1, int((cfg.size[0] - platform_width) / (2.0 * step_width)))
    max_height = 0.0
    _add_flat(terrain, cfg, center)
    for i in range(-steps, steps + 1):
        height = abs(i) * step_height
        x = center[0] + i * step_width
        _add_box_top(terrain, x, center[1], height, step_width, cfg.size[1], brand_ramp(abs(i) / max(steps, 1)))
        max_height = max(max_height, height)
    return max_height


def _add_random_stairs(terrain: _core.Terrain3D, cfg: TerrainGeneratorCfg, sub_cfg: SubTerrainCfg, center: Vector3, difficulty: float, rng: np.random.Generator) -> float:
    step_width = float(sub_cfg.kwargs.get("step_width", 0.4))
    height_min = float(sub_cfg.kwargs.get("height_min", -0.08)) * (0.5 + difficulty)
    height_max = float(sub_cfg.kwargs.get("height_max", 0.16)) * (0.5 + difficulty)
    nx = max(1, int(round(cfg.size[0] / step_width)))
    max_height = 0.0
    for ix in range(nx):
        x = center[0] - cfg.size[0] * 0.5 + (ix + 0.5) * (cfg.size[0] / nx)
        height = float(rng.uniform(height_min, height_max))
        _add_box_top(terrain, x, center[1], height, cfg.size[0] / nx, cfg.size[1], brand_ramp((height - height_min) / max(height_max - height_min, 1.0e-6)))
        max_height = max(max_height, height)
    return max_height


def _add_stepping_stones(terrain: _core.Terrain3D, cfg: TerrainGeneratorCfg, sub_cfg: SubTerrainCfg, center: Vector3, difficulty: float, rng: np.random.Generator) -> float:
    stone = float(sub_cfg.kwargs.get("stone_size", 0.55))
    distance = float(sub_cfg.kwargs.get("stone_distance", 0.25))
    pitch = stone + distance
    height_min = float(sub_cfg.kwargs.get("height_min", -0.04)) * (0.5 + difficulty)
    height_max = float(sub_cfg.kwargs.get("height_max", 0.08)) * (0.5 + difficulty)
    platform_width = float(sub_cfg.kwargs.get("platform_width", 1.0))
    max_height = 0.0
    nx = max(1, int(cfg.size[0] / pitch))
    ny = max(1, int(cfg.size[1] / pitch))
    _add_base_box(terrain, cfg, center, _heightfield_hsv_color(0.30))
    for ix in range(nx):
        for iy in range(ny):
            x = center[0] - cfg.size[0] * 0.5 + (ix + 0.5) * pitch
            y = center[1] - cfg.size[1] * 0.5 + (iy + 0.5) * pitch
            if abs(x - center[0]) <= platform_width * 0.5 and abs(y - center[1]) <= platform_width * 0.5:
                height = 0.0
                color = _get_platform_color()
            else:
                height = float(rng.uniform(height_min, height_max))
                color = brand_ramp((height - height_min) / max(height_max - height_min, 1.0e-6))
            _add_box_top(terrain, x, y, height, stone, stone, color)
            max_height = max(max_height, height)
    return max_height


def _add_stepping_stones_heightfield(
    terrain: _core.Terrain3D,
    cfg: TerrainGeneratorCfg,
    sub_cfg: SubTerrainCfg,
    center: Vector3,
    difficulty: float,
    rng: np.random.Generator,
) -> float:
    stone = float(sub_cfg.kwargs.get("stone_size", 0.55))
    distance = float(sub_cfg.kwargs.get("stone_distance", 0.25))
    pitch = stone + distance
    height_min = float(sub_cfg.kwargs.get("height_min", -0.04)) * (0.5 + difficulty)
    height_max = float(sub_cfg.kwargs.get("height_max", 0.08)) * (0.5 + difficulty)
    platform_width = float(sub_cfg.kwargs.get("platform_width", 1.0))
    rows, cols = _heightfield_shape(cfg)
    heights = np.full((rows, cols), min(height_min, 0.0), dtype=float)
    xs = np.linspace(-cfg.size[0] * 0.5, cfg.size[0] * 0.5, cols)
    ys = np.linspace(-cfg.size[1] * 0.5, cfg.size[1] * 0.5, rows)
    xx, yy = np.meshgrid(xs, ys)
    nx = max(1, int(cfg.size[0] / pitch))
    ny = max(1, int(cfg.size[1] / pitch))
    for ix in range(nx):
        for iy in range(ny):
            x = -cfg.size[0] * 0.5 + (ix + 0.5) * pitch
            y = -cfg.size[1] * 0.5 + (iy + 0.5) * pitch
            height = 0.0 if abs(x) <= platform_width * 0.5 and abs(y) <= platform_width * 0.5 else float(rng.uniform(height_min, height_max))
            mask = (np.abs(xx - x) <= stone * 0.5) & (np.abs(yy - y) <= stone * 0.5)
            heights[mask] = height
    return _add_heightfield_surface(terrain, cfg, center, heights)


def _add_narrow_beams(terrain: _core.Terrain3D, cfg: TerrainGeneratorCfg, sub_cfg: SubTerrainCfg, center: Vector3, difficulty: float) -> float:
    beam_width = float(sub_cfg.kwargs.get("beam_width", 0.18))
    gap = float(sub_cfg.kwargs.get("gap", 0.45)) * (1.2 - min(difficulty, 1.0) * 0.4)
    height = float(sub_cfg.kwargs.get("height", 0.08)) * (0.5 + difficulty)
    pitch = beam_width + gap
    count = max(1, int(cfg.size[1] / pitch))
    _add_base_box(terrain, cfg, center, _heightfield_hsv_color(0.30))
    for i in range(count):
        y = center[1] - cfg.size[1] * 0.5 + (i + 0.5) * pitch
        _add_box_top(terrain, center[0], y, height, cfg.size[0], beam_width, brand_ramp(i / max(count - 1, 1)))
    return height


def _add_narrow_beams_heightfield(
    terrain: _core.Terrain3D,
    cfg: TerrainGeneratorCfg,
    sub_cfg: SubTerrainCfg,
    center: Vector3,
    difficulty: float,
) -> float:
    beam_width = float(sub_cfg.kwargs.get("beam_width", 0.18))
    gap = float(sub_cfg.kwargs.get("gap", 0.45)) * (1.2 - min(difficulty, 1.0) * 0.4)
    height = float(sub_cfg.kwargs.get("height", 0.08)) * (0.5 + difficulty)
    pitch = beam_width + gap
    rows, cols = _heightfield_shape(cfg)
    heights = np.zeros((rows, cols), dtype=float)
    ys = np.linspace(-cfg.size[1] * 0.5, cfg.size[1] * 0.5, rows)
    count = max(1, int(cfg.size[1] / pitch))
    for i in range(count):
        y = -cfg.size[1] * 0.5 + (i + 0.5) * pitch
        mask = np.abs(ys - y) <= beam_width * 0.5
        heights[mask, :] = height
    return _add_heightfield_surface(terrain, cfg, center, heights)


def _add_tilted_grid(terrain: _core.Terrain3D, cfg: TerrainGeneratorCfg, sub_cfg: SubTerrainCfg, center: Vector3, difficulty: float, rng: np.random.Generator) -> float:
    tile = float(sub_cfg.kwargs.get("tile_size", 0.65))
    gap = float(sub_cfg.kwargs.get("gap", 0.08))
    pitch = tile + gap
    tilt_min = float(sub_cfg.kwargs.get("tilt_min", -12.0)) * (0.5 + difficulty)
    tilt_max = float(sub_cfg.kwargs.get("tilt_max", 12.0)) * (0.5 + difficulty)
    height_min = float(sub_cfg.kwargs.get("height_min", -0.04))
    height_max = float(sub_cfg.kwargs.get("height_max", 0.08)) * (0.5 + difficulty)
    platform_width = float(sub_cfg.kwargs.get("platform_width", 1.0))
    half = tile * 0.5
    vertices = [(-half, -half, 0.0), (half, -half, 0.0), (half, half, 0.0), (-half, half, 0.0)]
    indices = [0, 1, 2, 0, 2, 3]
    max_height = 0.0
    nx = max(1, int(cfg.size[0] / pitch))
    ny = max(1, int(cfg.size[1] / pitch))
    _add_base_box(terrain, cfg, center, _heightfield_hsv_color(0.30))
    for ix in range(nx):
        for iy in range(ny):
            x = center[0] - cfg.size[0] * 0.5 + (ix + 0.5) * pitch
            y = center[1] - cfg.size[1] * 0.5 + (iy + 0.5) * pitch
            height = 0.0 if abs(x - center[0]) <= platform_width * 0.5 and abs(y - center[1]) <= platform_width * 0.5 else float(rng.uniform(height_min, height_max))
            rotation = (float(rng.uniform(tilt_min, tilt_max)), float(rng.uniform(tilt_min, tilt_max)), 0.0)
            terrain.add_mesh_patch((x, y, height), vertices, indices, rotation, brand_ramp((height - height_min) / max(height_max - height_min, 1.0e-6)))
            max_height = max(max_height, height)
    return max_height


def _heightfield(cfg: TerrainGeneratorCfg, sub_cfg: SubTerrainCfg, difficulty: float, rng: np.random.Generator) -> np.ndarray:
    rows, cols = _heightfield_shape(cfg)
    x = np.linspace(-cfg.size[0] * 0.5, cfg.size[0] * 0.5, cols)
    y = np.linspace(-cfg.size[1] * 0.5, cfg.size[1] * 0.5, rows)
    xx, yy = np.meshgrid(x, y)
    border_mask = _heightfield_border_mask(cfg, sub_cfg, rows, cols)

    if sub_cfg.kind == "hf_pyramid_slope":
        slope = _difficulty_value(sub_cfg.kwargs, "slope", "slope_range", difficulty, 0.35)
        platform_width = float(sub_cfg.kwargs.get("platform_width", 1.0))
        border_width = max(0.0, float(sub_cfg.kwargs.get("border_width", 0.0)))
        half_extent = max(1.0e-6, min(cfg.size[0], cfg.size[1]) * 0.5 - border_width)
        edge_distance = np.maximum(half_extent - np.maximum(np.abs(xx), np.abs(yy)), 0.0)
        plateau_distance = max(half_extent - platform_width * 0.5, 0.0)
        heights = np.minimum(edge_distance, plateau_distance) * slope
        heights[border_mask] = 0.0
        return -heights if bool(sub_cfg.kwargs.get("inverted", False)) else heights

    if sub_cfg.kind == "random_rough":
        difficulty_scale = 0.5 + difficulty if bool(sub_cfg.kwargs.get("scale_by_difficulty", True)) else 1.0
        noise_min = float(sub_cfg.kwargs.get("noise_min", -0.05)) * difficulty_scale
        noise_max = float(sub_cfg.kwargs.get("noise_max", 0.05)) * difficulty_scale
        noise_step = max(float(sub_cfg.kwargs.get("noise_step", 0.01)), 1.0e-6)
        downsampled_scale = sub_cfg.kwargs.get("downsampled_scale")
        scale = cfg.horizontal_scale if downsampled_scale is None else max(float(downsampled_scale), cfg.horizontal_scale)
        coarse_rows = max(2, int(round(cfg.size[1] / scale)) + 1)
        coarse_cols = max(2, int(round(cfg.size[0] / scale)) + 1)
        coarse = rng.uniform(noise_min, noise_max, size=(coarse_rows, coarse_cols))
        heights = _resize(coarse, rows, cols)
        heights = np.round(heights / noise_step) * noise_step
        heights[border_mask] = 0.0
        return heights

    if sub_cfg.kind == "wave":
        amplitude = _difficulty_value(sub_cfg.kwargs, "amplitude", "amplitude_range", difficulty, 0.08)
        num_waves = float(sub_cfg.kwargs.get("num_waves", 2.0))
        if "amplitude_range" in sub_cfg.kwargs:
            heights = 0.5 * amplitude * (
                np.sin(2.0 * num_waves * np.pi * xx / cfg.size[0])
                + np.cos(2.0 * num_waves * np.pi * yy / cfg.size[1])
            )
        else:
            heights = amplitude * (
                np.sin(num_waves * np.pi * xx / cfg.size[0])
                + np.cos(num_waves * np.pi * yy / cfg.size[1])
            )
        heights[border_mask] = 0.0
        return heights

    if sub_cfg.kind == "perlin_noise":
        amplitude = float(sub_cfg.kwargs.get("amplitude", 0.08)) * (0.5 + difficulty)
        frequency = float(sub_cfg.kwargs.get("frequency", 2.0))
        octaves = max(1, int(sub_cfg.kwargs.get("octaves", 3)))
        return amplitude * _value_noise(rows, cols, frequency, octaves, rng)

    return np.zeros((rows, cols), dtype=float)


def _difficulty_value(
    kwargs: dict[str, Any],
    value_key: str,
    range_key: str,
    difficulty: float,
    default: float,
) -> float:
    value_range = kwargs.get(range_key)
    if value_range is not None:
        lower, upper = value_range
        return float(lower) + float(difficulty) * (float(upper) - float(lower))
    return float(kwargs.get(value_key, default)) * (0.5 + difficulty)


def _heightfield_border_mask(
    cfg: TerrainGeneratorCfg,
    sub_cfg: SubTerrainCfg,
    rows: int,
    cols: int,
) -> np.ndarray:
    border_width = max(0.0, float(sub_cfg.kwargs.get("border_width", 0.0)))
    if border_width <= 0.0:
        return np.zeros((rows, cols), dtype=bool)
    border_rows = min(rows // 2, int(border_width / max(cfg.horizontal_scale, 1.0e-6)))
    border_cols = min(cols // 2, int(border_width / max(cfg.horizontal_scale, 1.0e-6)))
    mask = np.zeros((rows, cols), dtype=bool)
    if border_rows > 0:
        mask[:border_rows, :] = True
        mask[-border_rows:, :] = True
    if border_cols > 0:
        mask[:, :border_cols] = True
        mask[:, -border_cols:] = True
    return mask


def _heightfield_shape(cfg: TerrainGeneratorCfg) -> tuple[int, int]:
    return (
        max(2, int(round(cfg.size[1] / cfg.horizontal_scale)) + 1),
        max(2, int(round(cfg.size[0] / cfg.horizontal_scale)) + 1),
    )


def _normalize_elevation(values: np.ndarray) -> np.ndarray:
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    return (values - min_value) / max(max_value - min_value, 1.0e-6)


def _resize(values: np.ndarray, rows: int, cols: int) -> np.ndarray:
    if ndimage is not None:
        zoom = (rows / values.shape[0], cols / values.shape[1])
        return ndimage.zoom(values, zoom, order=1)[:rows, :cols]
    return _resize_bilinear(values, rows, cols)


def _value_noise(rows: int, cols: int, frequency: float, octaves: int, rng: np.random.Generator) -> np.ndarray:
    result = np.zeros((rows, cols), dtype=float)
    amplitude = 1.0
    amplitude_sum = 0.0
    for octave in range(octaves):
        coarse_rows = max(2, int(round(frequency * (2**octave))) + 1)
        coarse_cols = max(2, int(round(frequency * (2**octave))) + 1)
        result += amplitude * _resize(rng.uniform(-1.0, 1.0, size=(coarse_rows, coarse_cols)), rows, cols)
        amplitude_sum += amplitude
        amplitude *= 0.5
    return result / max(amplitude_sum, 1.0e-6)


def _resize_bilinear(values: np.ndarray, rows: int, cols: int) -> np.ndarray:
    src_y = np.linspace(0.0, values.shape[0] - 1, rows)
    src_x = np.linspace(0.0, values.shape[1] - 1, cols)
    x0 = np.floor(src_x).astype(int)
    y0 = np.floor(src_y).astype(int)
    x1 = np.minimum(x0 + 1, values.shape[1] - 1)
    y1 = np.minimum(y0 + 1, values.shape[0] - 1)
    wx = (src_x - x0)[None, :]
    wy = (src_y - y0)[:, None]
    top = values[y0[:, None], x0[None, :]] * (1.0 - wx) + values[y0[:, None], x1[None, :]] * wx
    bottom = values[y1[:, None], x0[None, :]] * (1.0 - wx) + values[y1[:, None], x1[None, :]] * wx
    return top * (1.0 - wy) + bottom * wy


def _bilinear_resample_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    nx, ny = z.shape
    ix = np.clip(np.searchsorted(x, x_new) - 1, 0, nx - 2)
    iy = np.clip(np.searchsorted(y, y_new) - 1, 0, ny - 2)
    wx = ((x_new - x[ix]) / (x[ix + 1] - x[ix])).reshape(-1, 1)
    wy = ((y_new - y[iy]) / (y[iy + 1] - y[iy])).reshape(1, -1)
    z00 = z[ix[:, None], iy[None, :]]
    z01 = z[ix[:, None], (iy + 1)[None, :]]
    z10 = z[(ix + 1)[:, None], iy[None, :]]
    z11 = z[(ix + 1)[:, None], (iy + 1)[None, :]]
    return z00 * (1.0 - wx) * (1.0 - wy) + z01 * (1.0 - wx) * wy + z10 * wx * (1.0 - wy) + z11 * wx * wy


ROUGH_TERRAINS_CFG: dict[str, SubTerrainCfg] = {
    "flat": flat(proportion=0.05),
    "stairs": pyramid_stairs(proportion=0.12),
    "stairs_inv": pyramid_stairs_inv(proportion=0.08),
    "slope": hf_pyramid_slope(proportion=0.12, slope=0.85),
    "slope_inv": hf_pyramid_slope_inv(proportion=0.08, slope=0.85),
    "rough": random_rough(proportion=0.15),
    "waves": wave_terrain(proportion=0.08),
    "obstacles": discrete_obstacles(proportion=0.10),
    "grid": box_random_grid(proportion=0.10),
    "stones": stepping_stones(proportion=0.08),
    "tilted_grid": tilted_grid(proportion=0.04),
}


__all__ = [
    "SubTerrainCfg",
    "TerrainGeneratorCfg",
    "ROUGH_TERRAINS_CFG",
    "brand_ramp",
    "darken_rgba",
    "go1_rough_terrain_cfg",
    "go1_training_terrains",
    "showcase_terrains",
    "create_terrain_node",
    "flat",
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "random_rough",
    "wave",
    "wave_terrain",
    "discrete_obstacles",
    "perlin_noise",
    "random_grid",
    "box_random_grid",
    "random_spread_boxes",
    "open_stairs",
    "random_stairs",
    "stepping_stones",
    "narrow_beams",
    "nested_rings",
    "tilted_grid",
    "radial_mound",
    "radial_pit",
]
