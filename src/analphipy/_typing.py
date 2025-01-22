from __future__ import annotations

from collections.abc import Callable, Sequence  # noqa: F401
from typing import (
    TYPE_CHECKING,
    Any,
    TypedDict,
    TypeVar,
)

import numpy as np  # noqa: F401
from numpy.typing import NDArray

from ._typing_compat import ParamSpec, TypeAlias

if TYPE_CHECKING:
    from .base_potential import PhiAbstract

P = ParamSpec("P")
"""Parameter specification"""

R = TypeVar("R")
"""Return Type"""

Array: TypeAlias = NDArray["np.floating[Any]"]
ArrayLike: TypeAlias = "Sequence[float] | NDArray[np.floating[Any]]"
Float_or_ArrayLike: TypeAlias = "float | ArrayLike"
Float_or_Array: TypeAlias = "float | Array"
Phi_Signature: TypeAlias = "Callable[..., Array]"

# To bind input and output to same type
T_Float_or_Array = TypeVar("T_Float_or_Array", float, Array)


T_PhiAbstract = TypeVar("T_PhiAbstract", bound="PhiAbstract")


# Quadrature

QuadSegments_Integrals: TypeAlias = "float | list[float]"
QuadSegments_Errors: TypeAlias = "float | list[float]"
QuadSegments_Outputs: TypeAlias = "dict[str, Any] | list[dict[str, Any]]"

QuadSegments: TypeAlias = """(
    QuadSegments_Integrals
    | tuple[QuadSegments_Integrals, QuadSegments_Errors]
    | tuple[QuadSegments_Integrals, QuadSegments_Outputs]
    | tuple[QuadSegments_Integrals, QuadSegments_Errors, QuadSegments_Outputs]
)
"""


# Minimizing
class OptimizeResultInterface(TypedDict):
    """Interface to scipy.optimize.OptimizeResult."""

    x: Array
    success: bool
    status: int
    message: str
    fun: Array | float
    jac: Array
    hess: Array
    hess_inv: Any
    nfev: int
    njev: int
    nhev: int
    nit: int
    maxcv: float
