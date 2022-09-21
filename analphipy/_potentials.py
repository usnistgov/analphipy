# type: ignore

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Union, cast

import numpy as np
from custom_inherit import DocInheritMeta

from analphipy.norofrenkel import NoroFrenkelPair

from ._docstrings import docfiller_shared
from ._typing import Float_or_ArrayLike
from .measures import diverg_js_cont
from .utils import minimize_phi


class Phi_Abstractclass(
    metaclass=DocInheritMeta(style="numpy_with_merge", abstract_base_class=False)
):
    def __call__(self, r: Float_or_ArrayLike) -> np.ndarray:
        """
        return pair potential at requested separation ``r``.

        Parameters
        ----------
        r : array-like
            pair separation

        Returns
        -------
        phi : ndarray
            Evaluated pair potential.
        """
        return self.phi(r)

    def asdict(self) -> dict:
        """Convert class to dictionary"""
        return dataclasses.asdict(self)

    def copy(self):
        """Copy class"""
        return type(self)(**self.asdict())

    @property
    def segments(self) -> list:
        """Segment phi.  Used in analsis"""
        raise NotImplementedError("to be implemented in subclass")

    def phi(self, r: Float_or_ArrayLike) -> np.ndarray:
        """
        Pair potential

        Parameters
        ----------
        r : array-like
            pair separation

        Returns
        -------
        phi : ndarray
            Evaluated pair potential.
        """
        raise NotImplementedError("to be implemented in subclass if appropriate")

    def dphidr(self, r: Float_or_ArrayLike) -> np.ndarray:
        r"""
        Derivative of pair potential.

        This returns the value of ``dphi(r)/dr``:

        .. math::

            \frac{d \phi(r)}{dr} = \frac{d \phi(r)}{d r}A


        Parameters
        ----------
        r : array-like
            pair separation

        Returns
        -------
        dphidr : ndarray
            Pair potential values.

        """
        raise NotImplementedError("to be implemented in subclass if appropriate")

    @property
    def r_min(self) -> float:
        """Position of minimum in potential."""
        raise NotImplementedError("to be implemented in subclass")

    @property
    def phi_min(self):
        """Value of ``phi`` at minimum"""
        return self.phi(self.r_min) * 1.0

    def pipe(self, func, *args, **kwargs):
        """
        Apply function.

        returns ``func(self, *args, **kwargs)``
        """
        return func(self, *args, **kwargs)

    def minimize(self, r0: float, bounds: Optional[tuple[float, float]] = None, **kws):
        """Determine position `r` where ``phi`` is minimized.

        Parameters
        ----------
        r0 : float
            Guess for position of minimum.
        bounds : tuple floats, optional
            If passed, should be of form ``bounds=(lower_bound, upper_bound)``.
        **kws :
            Extra arguments to :func:`analphipy.minimize_phi`

        Returns
        -------
        rmin : float
            ``phi(rmin)`` is found location of minimum.
        phimin : float
            Value of ``phi(rmin)``, i.e., the value of ``phi`` at the minimum
        output :
            Output class from :func:`scipy.optimize.minimize`.


        See Also
        --------
        analphipy.minimize_phi
        """
        return minimize_phi(self.phi, r0=r0, bounds=bounds, **kws)

    def boltz(self, r: Float_or_ArrayLike, beta: float) -> np.ndarray:
        r"""Boltzmann factor of potential :math:`\exp[-\beta \phi(r)]`"""
        return np.exp(-beta * self.phi(r))

    def mayer(self, r: Float_or_ArrayLike, beta: float) -> np.ndarray:
        r"""Mayer f-function of potential :math:`\exp[-\beta \phi(r)] - 1`"""
        return np.exp(-beta * self.phi(r)) - 1.0

    def boltz_partial(
        self,
        beta: float,
        # sig: Optional[float] = None,
        # eps: Optional[float] = None,
    ) -> Callable[[Float_or_ArrayLike], Union[float, np.ndarray]]:
        """
        Returns callable function  ``boltz(x) = exp(-beta * phi(x))``.
        """

        # if sig is not None:
        #     assert eps is not None

        return partial(self.boltz, beta=beta)

    def mayer_partial(
        self, beta: float
    ) -> Callable[[Float_or_ArrayLike], Union[float, np.ndarray]]:
        """Returns callable function mayer(x) = exp(-beta * phi(x)) - 1"""
        return partial(self.mayer, beta=beta)

    def to_nf_fixed(self, quad_kws: Optional[dict] = None) -> NoroFrenkelPair:
        """Create :class:`~analphipy.NoroFrenkelPair` instance from fixed minima parameters.

        Parameters
        ----------
        quad_kws : mapping, optional
            Extra arguments to :func:`scipy.integrate.quad`

        See Also
        --------
        NoroFrenkelPair
        """
        r_min, phi_min = self.r_min, self.phi_min

        return NoroFrenkelPair(
            phi=self.phi,
            segments=self.segments,
            r_min=r_min,
            phi_min=phi_min,
            quad_kws=quad_kws,
        )

    def to_nf_numeric(
        self,
        r_min: Optional[float] = None,
        bounds: Optional[tuple[float, float]] = None,
        quad_kws: Optional[dict] = None,
        **kws,
    ) -> NoroFrenkelPair:
        """Create :class:~analphipy.NoroFrenkelPair` by numerically finding minima parameters.

        See Also
        --------
        minimize
        """
        return NoroFrenkelPair.from_phi(
            phi=self.phi,
            segments=self.segments,
            bounds=bounds,
            r_min=r_min,
            quad_kws=quad_kws,
            **kws,
        )

    def to_nf(
        self,
        r_min: Optional[float] = None,
        bounds: Optional[tuple[float, float]] = None,
        quad_kws: Optional[dict] = None,
        **kws,
    ) -> NoroFrenkelPair:
        try:
            return self.to_nf_fixed(quad_kws=quad_kws)
        except NotImplementedError:
            return self.to_nf_numeric(
                r_min=r_min, bounds=bounds, quad_kws=quad_kws, **kws
            )

    @docfiller_shared
    def boltz_diverg_js(
        self,
        other: Phi_Abstractclass,
        beta: float,
        beta_other: Optional[float] = None,
        volume: Union[str, Callable] = "3d",
        err: bool = False,
        full_output: bool = False,
        **kws,
    ):
        """
        Jenken-Shannon divergence of the boltzman factors of two potentials.

        This calculates the


        Parameters
        ----------
        other : Phi class
            Class wrapping other potential to compare `self` to.
        {beta}
        beta_other : float, optional
            beta value to evaluate other boltzman factor at.
        {volume_int_func}


        See Also
        --------
        `See here for more info <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Symmetrised_divergence>`


        """

        if beta_other is None:
            beta_other = beta

        return diverg_js_cont(
            p=self.boltz_partial(beta=beta),
            q=other.boltz_partial(beta=beta_other),
            segments=self.segments,
            segments_q=other.segments,
            volume=volume,
            err=err,
            full_output=full_output,
            **kws,
        )

    def mayer_diverg_js(
        self,
        other: Phi_Abstractclass,
        beta: float,
        beta_other: Optional[float] = None,
        volume: Union[str, Callable] = "3d",
        err: bool = False,
        full_output: bool = False,
        **kws,
    ):

        if beta_other is None:
            beta_other = beta

        return diverg_js_cont(
            p=self.mayer_partial(beta=beta),
            q=other.mayer_partial(beta=beta_other),
            segments=self.segments,
            segments_q=other.segments,
            volume=volume,
            err=err,
            full_output=full_output,
            **kws,
        )


@dataclass
class Phi_cut_base(Phi_Abstractclass):
    """
    create cut potential from base potential


    phi_cut(r) = phi(r) + _vcorrect(r) if r <= rcut
               = 0 otherwise


    dphi_cut(r) = dphi(r) + _dvcorrect(r) if r <= rcut
                = 0 otherwise

    So, for example, for just cut, `_vcorrect(r) = -v(rcut)`, `dvrcut=0`,
    and for lfs, `_vcorrect(r) = -v(rcut) - dv(rcut)/dr (r - rcut)`
    `_dvcorrect(r) = ...`
    """

    phi_base: Phi_Abstractclass
    rcut: float

    @property
    def segments(self) -> list:
        return [x for x in self.phi_base.segments if x < self.rcut] + [self.rcut]

    @classmethod
    def from_base(cls, base, rcut, *args, **kws) -> Phi_cut_base:
        """
        Create cut potential from base phi class
        """
        phi_base = base(*args, **kws)
        return cls(phi_base=phi_base, rcut=rcut)

    def phi(self, r: Float_or_ArrayLike) -> np.ndarray:
        r = np.asarray(r)
        v = np.empty_like(r)

        left = r <= self.rcut
        right = ~left

        v[right] = 0.0

        if np.any(left):
            v[left] = self.phi_base.phi(r[left]) + self._vcorrect(r[left])
        return v

    def dphidr(self, r: Float_or_ArrayLike) -> np.ndarray:
        r = np.asarray(r)
        dvdr = np.empty_like(r)

        left = r <= self.rcut
        right = ~left

        dvdr[right] = 0.0
        dvdr[left] = self.phi_base.dphidr(r[left]) + self._dvdrcorrect(r[left])

        return dvdr

    # def phidphi(self, r: Float_or_ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    #     r = np.array(r)

    #     v = np.empty_like(r)
    #     dv = np.empty_like(r)

    #     left = r <= self.rcut
    #     right = ~left

    #     v[right] = 0.0
    #     dv[right] = 0.0

    #     if np.any(left):
    #         v[left], dv[left] = self.phi_base.phidphi(r[left])
    #         v[left] += self._vcorrect(r[left])
    #         dv[left] += self._dvcorrect(r[left])

    #     return v, dv

    def _vcorrect(self, r):
        raise NotImplementedError

    def _dvdrcorrect(self, r):
        raise NotImplementedError


@dataclass
class Phi_cut(Phi_cut_base):
    r"""
    Pair potential cut at position ``r_cut``.

    .. math::

        \phi_{\rm cut}(r) =
            \begin{cases}
                \phi(r) - \phi(r_{\rm cut})  & r < r_{\rm cut} \\
                0 & r_{\rm cut} \leq r
            \end{cases}

    Parameters
    ----------
    phi_base : Phi_Baseclass instance
        Potential class to perform cut on.
    rcut : float
        Where to 'cut' the potential.
    """

    def __post_init__(self):
        self.vcut = self.phi_base(self.rcut)

    def _vcorrect(self, r: np.ndarray) -> np.ndarray:
        return -cast(np.ndarray, self.vcut)

    def _dvdrcorrect(self, r: np.ndarray) -> np.ndarray:
        return np.array(0.0)


@dataclass
class Phi_lfs(Phi_cut_base):
    r"""
    Pair potential cut and linear force shifted at ``r_cut``.

    .. math::

        \phi_{rm lfs}(r) =
            \begin{cases}
                \phi(r) - \left( \frac{d \phi}{d r} \right)_{\rm cut} (r - r_{\rm cut}) - \phi(r_{\rm cut}) & r < r_{\rm cut} \\
                0 & r_{\rm cut} < r
            \end{cases}

    """

    def __post_init__(self):

        self.vcut = self.phi_base.phi(self.rcut)
        self.dvdrcut = self.phi_base.dphidr(self.rcut)

        # vcut, dvcut = self.phi_base.phidphi(self.rcut)
        # self.vcut = vcut
        # self.dvdrcut = -dvcut * self.rcut

    def _vcorrect(self, r: np.ndarray) -> np.ndarray:
        out = -(self.vcut + self.dvdrcut * (r - self.rcut))
        return cast(np.ndarray, out)

    # def _dvcorrect(self, r: np.ndarray) -> np.ndarray:
    #     out = self.dvdrcut / r
    #     return cast(np.ndarray, out)
    def _dvdrcorrect(self, r: np.ndarray) -> np.ndarray:
        out = -self.dvdrcut
        return cast(np.ndarray, out)


class Phi_Baseclass(Phi_Abstractclass):
    """
    Baseclass for creating Phi object
    """

    def cut(self, rcut: float) -> Phi_cut:
        return Phi_cut(phi_base=self, rcut=rcut)

    def lfs(self, rcut: float) -> Phi_lfs:
        return Phi_lfs(phi_base=self, rcut=rcut)


# This is kinda a dumb way to do things
# Instead, should just have the Phi Wrapper class, which you then call with all the all the special sauce.
# phi = Phi_hs -> instance of Phi_base just provides phi, phidphi, r_min, phi_min, segments
# phi.nf -> provides noro frenkel like analysis
# phi.
# phi.cut -> another phi_base classmetho
