from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, TypeVar

    T = TypeVar("T", bound=Any)
    R = TypeVar("R")


def optional_converter(converter: Callable[[T], R]) -> Callable[[T], T | R]:
    """Create a converter which can pass through None."""

    def wrapped(value: T) -> T | R:  # pragma: no cover
        if value in {None, attrs.NOTHING}:
            return value  # pyright: ignore[reportReturnType]
        return converter(value)

    return wrapped


def field_formatter(fmt: str = "{:.5g}") -> Callable[[Any], str]:
    """Formatter for attrs field."""

    @optional_converter
    def wrapped(value: Any) -> str:
        return fmt.format(value)

    return wrapped


def field_array_formatter(threshold: int = 3, **kws: Any) -> Callable[[Any], str]:
    """Formatter for numpy array field."""

    @optional_converter
    def wrapped(value: Any) -> str:
        with np.printoptions(threshold=threshold, **kws):
            return str(value)

    return wrapped
