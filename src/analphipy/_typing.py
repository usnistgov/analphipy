from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

import numpy as np
from numpy.typing import NDArray  # , ArrayLike
from typing_extensions import ParamSpec, TypeAlias

if TYPE_CHECKING:
    from .base_potential import PhiAbstract

P = ParamSpec("P")
"""Parameter specification"""

R = TypeVar("R")
"""Return Type"""

Array: TypeAlias = NDArray[np.float_]
ArrayLike: TypeAlias = Union[Sequence[float], NDArray[np.float_]]
Float_or_ArrayLike: TypeAlias = Union[float, ArrayLike]
Float_or_Array: TypeAlias = Union[float, Array]
Phi_Signature: TypeAlias = Callable[..., Array]

# To bind input and output to same type
T_Float_or_Array = TypeVar("T_Float_or_Array", float, Array)


T_PhiAbstract = TypeVar("T_PhiAbstract", bound="PhiAbstract")


# Quadrature

QuadSegments_Integrals: TypeAlias = Union[float, List[float]]
QuadSegments_Errors: TypeAlias = Union[float, List[float]]
QuadSegments_Outputs: TypeAlias = Union[Dict[str, Any], List[Dict[str, Any]]]

QuadSegments: TypeAlias = Union[
    QuadSegments_Integrals,
    Tuple[QuadSegments_Integrals, QuadSegments_Errors],
    Tuple[QuadSegments_Integrals, QuadSegments_Outputs],
    Tuple[QuadSegments_Integrals, QuadSegments_Errors, QuadSegments_Outputs],
]


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
