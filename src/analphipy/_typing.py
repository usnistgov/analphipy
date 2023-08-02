from __future__ import annotations

from typing import Callable, Sequence, Union

import numpy as np
from numpy.typing import NDArray  # , ArrayLike

# if TYPE_CHECKING:
#     from .potentials_generic import PhiBase


# T_PhiBase = TypeVar("T_PhiBase", bound="PhiBase")


ArrayLike = Union[Sequence[float], NDArray[np.float_]]
Float_or_ArrayLike = Union[float, ArrayLike]
Float_or_Array = Union[float, NDArray[np.float_]]
Phi_Signature = Callable[..., Union[float, NDArray[np.float_]]]
