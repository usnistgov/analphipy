from __future__ import annotations

from typing import Callable, Sequence, Union

import numpy as np
from numpy.typing import NDArray  # , ArrayLike

ArrayLike = Union[Sequence[float], NDArray[np.float_]]
Float_or_ArrayLike = Union[float, ArrayLike]
Float_or_Array = Union[float, NDArray[np.float_]]
Phi_Signature = Callable[..., Union[float, np.ndarray]]

TWO_PI = 2.0 * np.pi
