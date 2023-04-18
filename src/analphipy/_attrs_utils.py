from __future__ import annotations

import attrs
import numpy as np


def attrs_clear_cache(self, attribute, value):
    """Clear out _cache if setting value"""
    self._cache = {}
    return value


def optional_converter(converter):
    """Create a converter which can pass through None"""

    def wrapped(value):
        if value is None or attrs.NOTHING:
            return value
        else:
            return converter(value)

    return wrapped


def field_formatter(fmt: str = "{:.5g}"):
    """Formatter for attrs field."""

    @optional_converter
    def wrapped(value):
        return fmt.format(value)

    return wrapped


def field_array_formatter(threshold=3, **kws):
    """Formatter for numpy array field"""

    @optional_converter
    def wrapped(value):
        with np.printoptions(threshold=threshold, **kws):
            return str(value)

    return wrapped


def private_field(init=False, repr=False, **kws):
    """Create a private attrs field."""
    return attrs.field(init=init, repr=repr, **kws)
