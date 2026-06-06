"""Terrain3D generation helpers with MJLab-style visuals."""

from __future__ import annotations

from dataclasses import dataclass, field
import colorsys
from typing import Literal

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


@dataclass(slots=True)
class SubTerrainCfg:
    kind: TerrainKind
    proportion: float = 1.0
    kwargs: dict[str, float | int | bool] = field(default_factory=dict)


@dataclass(slots=True)
class TerrainGeneratorCfg:
    size: Vector2 = (8.0, 8.0)
    num_rows: int = 1
    num_cols: int = 1
    border_width: float = 0.0
    seed: int | None = None
    curriculum: bool = False
    difficulty_range: Vector2 = (0.0, 1.0)
    sub_terrains: dict[str, SubTerrainCfg] = field(default_factory=dict)
    horizontal_scale: float = 0.1
    base_thickness: float = 0.1
    surface_color: Color4 = (1.0, 1.0, 1.0, 1.0)
    color_mode: _core.TerrainColorMode = _core.TerrainColorMode.MjLab
    height_low_color: Color4 = (0.10, 0.34, 0.30, 1.0)
    height_high_color: Color4 = (0.62, 0.44, 0.25, 1.0)
    height_range_min: float = 0.0
    height_range_max: float = 0.0


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
    step_width: float = 0.35,
    platform_width: float = 1.0,
    inverted: bool = False,
) -> SubTerrainCfg:
    return SubTerrainCfg(
        "pyramid_stairs",
        proportion,
        {
            "step_height": step_height,
            "step_width": step_width,
            "platform_width": platform_width,
            "inverted": inverted,
        },
    )


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
    platform_width: float = 1.0,
    inverted: bool = False,
) -> SubTerrainCfg:
    return SubTerrainCfg(
        "hf_pyramid_slope",
        proportion,
        {"slope": slope, "platform_width": platform_width, "inverted": inverted},
    )


def hf_pyramid_slope_inv(**kwargs) -> SubTerrainCfg:
    kwargs["inverted"] = True
    return hf_pyramid_slope(**kwargs)


def random_rough(
    *,
    proportion: float = 1.0,
    noise_range: Vector2 = (-0.05, 0.05),
    noise_step: float = 0.01,
    downsampled_scale: float = 0.3,
) -> SubTerrainCfg:
    return SubTerrainCfg(
        "random_rough",
        proportion,
        {
            "noise_min": noise_range[0],
            "noise_max": noise_range[1],
            "noise_step": noise_step,
            "downsampled_scale": downsampled_scale,
        },
    )


def wave_terrain(
    *,
    proportion: float = 1.0,
    amplitude: float = 0.08,
    num_waves: float = 2.0,
) -> SubTerrainCfg:
    return SubTerrainCfg("wave", proportion, {"amplitude": amplitude, "num_waves": num_waves})


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
    """Go1-sized rough terrain presets.

    These keep MJLab-style visual variety while capping obstacles to heights a
    small quadruped can plausibly step over during early locomotion training.
    """

    return {
        "flat": flat(proportion=0.16),
        "rough_easy": random_rough(proportion=0.12, noise_range=(-0.015, 0.020), downsampled_scale=0.20),
        "waves": wave_terrain(proportion=0.10, amplitude=0.030, num_waves=2.0),
        "grid": box_random_grid(proportion=0.10, grid_width=0.45, height_range=(-0.020, 0.065), platform_width=1.10),
        "stones": stepping_stones(
            proportion=0.08,
            stone_size=0.55,
            stone_distance=0.18,
            height_range=(-0.020, 0.055),
            platform_width=1.10,
        ),
        "slope": hf_pyramid_slope(proportion=0.10, slope=0.090, platform_width=1.20),
        "slope_inv": hf_pyramid_slope_inv(proportion=0.08, slope=0.080, platform_width=1.20),
        "stairs": pyramid_stairs(proportion=0.10, step_height=0.026, step_width=0.42, platform_width=1.20),
        "stairs_inv": pyramid_stairs_inv(proportion=0.06, step_height=0.022, step_width=0.42, platform_width=1.20),
        "obstacles": discrete_obstacles(
            proportion=0.08,
            obstacle_count=14,
            obstacle_size=(0.14, 0.30),
            height_range=(0.025, 0.085),
            platform_width=1.10,
        ),
        "mound": radial_mound(proportion=0.06, height=0.14, radius=1.10, flat_radius=0.22),
        "pit": radial_pit(proportion=0.04, depth=0.12, radius=1.10, flat_radius=0.22),
    }


def mjlab_showcase_terrains() -> dict[str, SubTerrainCfg]:
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
    terrains = cfg.sub_terrains or {"flat": flat()}
    choices = _normalized_choices(terrains)
    curriculum_choices = list(terrains.values()) or [flat()]
    rng = np.random.default_rng(cfg.seed)
    origins: list[Vector3] = []

    total_x = cfg.num_rows * cfg.size[0] + 2.0 * cfg.border_width
    total_y = cfg.num_cols * cfg.size[1] + 2.0 * cfg.border_width
    origin_x = -total_x * 0.5 + cfg.border_width + cfg.size[0] * 0.5
    origin_y = -total_y * 0.5 + cfg.border_width + cfg.size[1] * 0.5

    if cfg.border_width > 0.0:
        terrain.add_box((0.0, 0.0, -0.55), (total_x, total_y, 0.9), color=darken_rgba(_get_platform_color(), 0.55))

    for row in range(max(cfg.num_rows, 0)):
        for col in range(max(cfg.num_cols, 0)):
            patch_center = (origin_x + row * cfg.size[0], origin_y + col * cfg.size[1], 0.0)
            if cfg.curriculum:
                denom = max(cfg.num_rows - 1, 1)
                difficulty = cfg.difficulty_range[0] + (cfg.difficulty_range[1] - cfg.difficulty_range[0]) * (row / denom)
                sub_cfg = curriculum_choices[(row * max(cfg.num_cols, 1) + col) % len(curriculum_choices)]
            else:
                difficulty = float(rng.uniform(cfg.difficulty_range[0], cfg.difficulty_range[1]))
                sub_cfg = choices[int(rng.integers(0, len(choices)))]
            origins.append((patch_center[0], patch_center[1], _add_patch(terrain, cfg, sub_cfg, patch_center, difficulty, rng)))

    terrain.spawn_origins = origins
    return terrain


def _normalized_choices(sub_terrains: dict[str, SubTerrainCfg]) -> list[SubTerrainCfg]:
    weighted: list[SubTerrainCfg] = []
    for cfg in sub_terrains.values():
        weighted.extend([cfg] * max(1, int(round(max(cfg.proportion, 0.0) * 100.0))))
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
    step_height = float(sub_cfg.kwargs.get("step_height", 0.08)) * (0.5 + difficulty)
    platform_width = float(sub_cfg.kwargs.get("platform_width", 1.0))
    inverted = bool(sub_cfg.kwargs.get("inverted", False))
    if inverted:
        return _add_inverted_pyramid_stairs(terrain, cfg, step_width, step_height, platform_width, center)

    half_x = cfg.size[0] * 0.5
    half_y = cfg.size[1] * 0.5
    layer = 0
    extrema = 0.0
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
    center: Vector3,
) -> float:
    half_x = cfg.size[0] * 0.5
    half_y = cfg.size[1] * 0.5
    target_half = platform_width * 0.5
    layer = 0
    min_height = 0.0

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

    if sub_cfg.kind == "hf_pyramid_slope":
        slope = float(sub_cfg.kwargs.get("slope", 0.35)) * (0.5 + difficulty)
        platform_width = float(sub_cfg.kwargs.get("platform_width", 1.0))
        half_extent = min(cfg.size[0], cfg.size[1]) * 0.5
        edge_distance = np.maximum(half_extent - np.maximum(np.abs(xx), np.abs(yy)), 0.0)
        plateau_distance = max(half_extent - platform_width * 0.5, 0.0)
        heights = np.minimum(edge_distance, plateau_distance) * slope
        return -heights if bool(sub_cfg.kwargs.get("inverted", False)) else heights

    if sub_cfg.kind == "random_rough":
        noise_min = float(sub_cfg.kwargs.get("noise_min", -0.05)) * (0.5 + difficulty)
        noise_max = float(sub_cfg.kwargs.get("noise_max", 0.05)) * (0.5 + difficulty)
        noise_step = max(float(sub_cfg.kwargs.get("noise_step", 0.01)), 1.0e-6)
        scale = max(float(sub_cfg.kwargs.get("downsampled_scale", 0.3)), cfg.horizontal_scale)
        coarse_rows = max(2, int(round(cfg.size[1] / scale)) + 1)
        coarse_cols = max(2, int(round(cfg.size[0] / scale)) + 1)
        coarse = rng.uniform(noise_min, noise_max, size=(coarse_rows, coarse_cols))
        heights = _resize(coarse, rows, cols)
        return np.round(heights / noise_step) * noise_step

    if sub_cfg.kind == "wave":
        amplitude = float(sub_cfg.kwargs.get("amplitude", 0.08)) * (0.5 + difficulty)
        num_waves = float(sub_cfg.kwargs.get("num_waves", 2.0))
        return amplitude * (np.sin(num_waves * np.pi * xx / cfg.size[0]) + np.cos(num_waves * np.pi * yy / cfg.size[1]))

    if sub_cfg.kind == "perlin_noise":
        amplitude = float(sub_cfg.kwargs.get("amplitude", 0.08)) * (0.5 + difficulty)
        frequency = float(sub_cfg.kwargs.get("frequency", 2.0))
        octaves = max(1, int(sub_cfg.kwargs.get("octaves", 3)))
        return amplitude * _value_noise(rows, cols, frequency, octaves, rng)

    return np.zeros((rows, cols), dtype=float)


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
    "go1_training_terrains",
    "mjlab_showcase_terrains",
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
