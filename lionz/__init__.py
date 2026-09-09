"""Public LION package API with lazy inference dependency loading."""

from __future__ import annotations

from typing import Any


__all__ = ["lion"]


def __getattr__(name: str) -> Any:
    if name == "lion":
        from .lionz import lion

        return lion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
