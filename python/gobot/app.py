"""Application runtime helpers."""

from ._core import (
    AppContext,
    app,
)

context = app.context
create_context = app.create_context

__all__ = [
    "AppContext",
    "create_context",
    "context",
]
