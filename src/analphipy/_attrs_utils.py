from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np

if TYPE_CHECKING:
    from typing import Any, Callable, TypeVar

    T = TypeVar("T", bound=Any)
    R = TypeVar("R")


def optional_converter(converter: Callable[[T], R]) -> Callable[[T], T | R]:
    """Create a converter which can pass through None."""

    def wrapped(value: T) -> T | R:  # pragma: no cover
        if value in {None, attrs.NOTHING}:
            return value
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


def private_field(init: bool = False, repr: bool = False, **kws: Any) -> Any:  # noqa: A002
    """Create a private attrs field."""
    return attrs.field(init=init, repr=repr, **kws)  # pytype: disable=not-supported-yet
