from __future__ import annotations

from ._core import Robot3D


def create_cartpole_scene(
    name: str = "cartpole",
    cart_size: tuple[float, float, float] = (0.35, 0.22, 0.18),
    pole_size: tuple[float, float, float] = (0.05, 0.05, 1.0),
) -> Robot3D: ...


def save_cartpole_scene(path: str = "res://cartpole.jscn", name: str = "cartpole") -> Robot3D: ...


__all__: list[str]
