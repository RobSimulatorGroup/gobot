"""Application runtime helpers."""

from ._core import AppContext, app

context = app.context

__all__ = ["AppContext", "context"]
