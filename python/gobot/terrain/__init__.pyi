from typing import Any, Literal

from .._core import TerrainColorMode

Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]
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

class SubTerrainCfg:
    kind: TerrainKind
    proportion: float
    kwargs: dict[str, Any]

class TerrainGeneratorCfg:
    size: Vector2
    num_rows: int
    num_cols: int
    border_width: float
    seed: int | None
    curriculum: bool
    difficulty_range: Vector2
    origin_row: int | None
    origin_sub_terrain: str | None
    sub_terrains: dict[str, SubTerrainCfg]
    horizontal_scale: float
    base_thickness: float
    surface_color: tuple[float, float, float, float]
    color_mode: TerrainColorMode
    height_low_color: tuple[float, float, float, float]
    height_high_color: tuple[float, float, float, float]
    height_range_min: float
    height_range_max: float
    merged_heightfield: bool

def flat(proportion: float = 1.0) -> SubTerrainCfg: ...
def pyramid_stairs(
    *,
    proportion: float = 1.0,
    step_height: float = 0.08,
    step_height_range: Vector2 | None = None,
    step_width: float = 0.35,
    platform_width: float = 1.0,
    border_width: float = 0.0,
    inverted: bool = False,
) -> SubTerrainCfg: ...
def random_grid(
    *,
    proportion: float = 1.0,
    grid_width: float = 0.5,
    height_range: Vector2 = (-0.05, 0.18),
    platform_width: float = 1.0,
) -> SubTerrainCfg: ...
def hf_pyramid_slope(
    *,
    proportion: float = 1.0,
    slope: float = 0.65,
    slope_range: Vector2 | None = None,
    platform_width: float = 1.0,
    border_width: float = 0.0,
    inverted: bool = False,
) -> SubTerrainCfg: ...
def random_rough(
    *,
    proportion: float = 1.0,
    noise_range: Vector2 = (-0.05, 0.05),
    noise_step: float = 0.01,
    downsampled_scale: float | None = 0.3,
    border_width: float = 0.0,
    scale_by_difficulty: bool = True,
) -> SubTerrainCfg: ...
def wave(
    *,
    proportion: float = 1.0,
    amplitude: float = 0.08,
    amplitude_range: Vector2 | None = None,
    num_waves: float = 2.0,
    border_width: float = 0.0,
) -> SubTerrainCfg: ...
def perlin_noise(
    *,
    proportion: float = 1.0,
    amplitude: float = 0.08,
    frequency: float = 2.0,
    octaves: int = 3,
) -> SubTerrainCfg: ...
def pyramid_stairs_inv(**kwargs) -> SubTerrainCfg: ...
def hf_pyramid_slope_inv(**kwargs) -> SubTerrainCfg: ...
def wave_terrain(
    *,
    proportion: float = 1.0,
    amplitude: float = 0.08,
    amplitude_range: Vector2 | None = None,
    num_waves: float = 2.0,
    border_width: float = 0.0,
) -> SubTerrainCfg: ...
def discrete_obstacles(**kwargs) -> SubTerrainCfg: ...
def box_random_grid(**kwargs) -> SubTerrainCfg: ...
def random_spread_boxes(**kwargs) -> SubTerrainCfg: ...
def open_stairs(**kwargs) -> SubTerrainCfg: ...
def random_stairs(**kwargs) -> SubTerrainCfg: ...
def stepping_stones(**kwargs) -> SubTerrainCfg: ...
def narrow_beams(**kwargs) -> SubTerrainCfg: ...
def nested_rings(**kwargs) -> SubTerrainCfg: ...
def tilted_grid(**kwargs) -> SubTerrainCfg: ...
def radial_mound(**kwargs) -> SubTerrainCfg: ...
def radial_pit(**kwargs) -> SubTerrainCfg: ...
def brand_ramp(value: float) -> tuple[float, float, float, float]: ...
def darken_rgba(color: tuple[float, float, float, float], amount: float = 0.72) -> tuple[float, float, float, float]: ...
def showcase_terrains() -> dict[str, SubTerrainCfg]: ...
def go1_training_terrains() -> dict[str, SubTerrainCfg]: ...
def go1_rough_terrain_cfg(*, seed: int = 42, curriculum: bool = False) -> TerrainGeneratorCfg: ...
def go1_mjlab_rough_terrain_cfg(*, seed: int = 42, curriculum: bool = True) -> TerrainGeneratorCfg: ...
def create_terrain_node(cfg: TerrainGeneratorCfg, name: str = "terrain"): ...

__all__: list[str]
