# type: ignore
from typing import Callable

import numpy as np

# from ._potentials import Phi_Baseclass
from ._typing import ArrayLike


class PotentialAnalysis:
    def __init__(
        self,
        phi_func: Callable,
        beta: float,
        segments: ArrayLike,
        length_scale: float = 1.0,
        energy_scale: float = 1.0,
    ):
        """
        Parameters
        """
        self.phi_func = phi_func
        self.beta = beta
        self.segments = segments

        self.length_scale = length_scale
        self.energy_scale = energy_scale
        self.energy_scale_inv = 1.0 / self.energy_scale

    def new_like(self, **kws):
        kws = dict(
            {
                "phi": self.phi,
                "beta": self.beta,
                "length_scale": self.length_scale,
                "energy_scale": self.energy_scale,
            },
            **kws
        )
        return type(self)(**kws)

    def __call__(self, x: ArrayLike) -> np.ndarray:
        return self.phi_scaled(x)

    def phi_scaled(self, x: ArrayLike) -> np.ndarray:
        r = np.array(x) * self.length_scale
        return self.phi_func(r) * self.energy_scale_inv

    def boltz(self, x: ArrayLike) -> np.ndarray:
        return np.exp(-self.beta * self(x))

    def mayer(self, x: ArrayLike) -> np.ndarray:
        return self.boltz(x) - 1.0

    # def phi_scaled(self, x: Float_or_ArrayLike, sig: float, eps: float) -> np.ndarray:
    #     """
    #     Scaled potential function

    #     Parameters
    #     ----------
    #     x : array-like
    #         normalized position.  `r/sig`
    #     sig : float
    #         Positive length scale parameter
    #     eps : float
    #         Positive energy scale parameter.

    #     Returns
    #     -------
    #     phi_scaled : array-like
    #         phi_scaled(x) = phi(x * sig) / eps
    #     """

    #     assert eps > 0., sig > 0.
    #     r = np.asarray(x) * sig

    #     return self.phi(r) / eps

    # def phi_scaled_partial(self, sig: float, eps: float) -> Callable[[Float_or_ArrayLike], Union[float, np.ndarray]]:
    #     return partial(self.phi_scaled, sig=sig, eps=eps)
