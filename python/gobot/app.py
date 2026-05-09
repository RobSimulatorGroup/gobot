"""Application runtime helpers."""

from ._core import (
    AppContext,
    app,
    clear_editor_physics_callback,
    clear_editor_tick_callback,
    set_editor_physics_callback,
    set_editor_tick_callback,
)

context = app.context

__all__ = [
    "AppContext",
    "clear_editor_physics_callback",
    "clear_editor_tick_callback",
    "context",
    "set_editor_physics_callback",
    "set_editor_tick_callback",
]
