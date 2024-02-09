import sys

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

if sys.version_info < (3, 10):
    from typing_extensions import Concatenate, ParamSpec, TypeAlias, TypeGuard
else:
    from typing import Concatenate, ParamSpec, TypeAlias, TypeGuard


__all__ = [
    "Concatenate",
    "ParamSpec",
    "Self",
    "TypeAlias",
    "TypeGuard",
]
