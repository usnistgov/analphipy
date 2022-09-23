# type: ignore

from __future__ import annotations

from typing import Mapping, Sequence, cast

import numpy as np
from custom_inherit import DocInheritMeta
from typing_extensions import Literal

from analphipy.norofrenkel import NoroFrenkelPair

# from ._docstrings import docfiller_shared
from ._typing import Float_or_ArrayLike
from .cached_decorators import cached_clear, gcached

# from .measures import diverg_js_cont
from .utils import minimize_phi


class PhiBase(
    metaclass=DocInheritMeta(style="numpy_with_merge", abstract_base_class=False)
):
    """
    Base class to interact with pair potentials
    """

    def __init__(
        self,
        r_min: float | None = None,
        phi_min: float | None = None,
        segments: Sequence[float] | None = None,
        quad_kws: Mapping | None = None,
    ):

        self.r_min = r_min
        self.phi_min = phi_min
        self.segments = segments
        self.quad_kws = quad_kws

    def asdict(self) -> dict:
        """Convert class to dictionary"""
        private = {"phi": self._phi_func, "dphidr": self._dphidr_func}
        public = {
            k: getattr(self, k) for k in ["r_min", "phi_min", "segments", "quad_kws"]
        }

        return {**private, **public}

    def copy(self):
        """Copy class"""
        return self.new_like()

    def new_like(self, **kws):
        kws = {**self.asdict(), **kws}
        return type(self)(**kws)

    def assign(self, **kws):
        return self.new_like(**kws)

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

        raise NotImplementedError("Must implement in subclass")

    def dphidr(self, r: Float_or_ArrayLike) -> np.ndarray:
        r"""
                Derivative of pair potential.

                This returns the value of ``dphi(r)/dr``:

                .. math::

                    \frac{d \phi(r)}{dr} = \frac{d \phi(r)}{d r}A


                Parameters
        n       ----------
                r : array-like
                    pair separation

                Returns
                -------
                dphidr : ndarray
                    Pair potential values.

        """
        raise NotImplementedError("Must implement in subclass")

    @property
    def segments(self) -> list:
        """Segment phi.  Used in analsis"""
        return getattr(self, "_segments", None)

    @segments.setter
    def segments(self, val):
        self._segments = val

    @property
    def quad_kws(self) -> dict:
        val = getattr(self, "_quad_kws", None)
        if val is None:
            val = {}
        return val

    @quad_kws.setter
    def quad_kws(self, val: Mapping | None):
        if val is not None:
            val = dict(val)
        self._quad_kws = val

    @property
    def r_min(self) -> float | None:
        """Position of minimum in potential."""
        return getattr(self, "_r_min", None)

    @r_min.setter
    @cached_clear()
    def r_min(self, val):
        self._r_min = val

    @property
    def phi_min(self):
        """Value of ``phi`` at minimum"""
        val = getattr(self, "_phi_min", None)

        if val is None and self.r_min is not None:
            return self.phi(self.r_min)
        else:
            return val

    @phi_min.setter
    @cached_clear()
    def phi_min(self, val):
        self._phi_min = val

    def pipe(self, func, *args, **kwargs):
        """
        Apply function.

        returns ``func(self, *args, **kwargs)``
        """
        return func(self, *args, **kwargs)

    def minimize(
        self,
        r0: float | Literal["mean"],
        bounds: tuple[float, float] | Literal["segments"] | None = None,
        **kws,
    ):
        """Determine position `r` where ``phi`` is minimized.

        Parameters
        ----------
        r0 : float
            Guess for position of minimum.
            If value is `'mean'`, then use ``r0 = mean(bounds)``.

        bounds : tuple floats, {'segments'}, optional
            Bounds for minimization search.
            If tuple, should be of form ``bounds=(lower_bound, upper_bound)``.
            If `'segments`, then ``bounds=(segments[0], segments[-1])``.
            If None, no bounds used.
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

        if bounds == "segments":
            bounds = (self.segments[0], self.segments[-1])

        if r0 == "mean":
            r0 = np.mean(bounds)

        return minimize_phi(self.phi, r0=r0, bounds=bounds, **kws)

    def assign_min_numeric(
        self, r0: float, bounds: tuple[float, float] | None = None, **kws
    ):
        """
        Create new object with minima set by numerical minimization

        call :meth:`minimize`
        """

        r_min, phi_min, output = self.minimize(r0=r0, bounds=bounds, **kws)
        return self.new_like(r_min=r_min, phi_min=phi_min)

    @gcached()
    def nf(self):
        if self.r_min is None:
            raise ValueError("must set `self.r_min` to use NoroFrenkel")

        return NoroFrenkelPair(
            phi=self.phi,
            segments=self.segments,
            r_min=self.r_min,
            phi_min=self.phi_min,
            quad_kws=self.quad_kws,
        )

    @gcached()
    def measures(self):
        from .measures import Measures

        return Measures(phi=self.phi, segments=self.segments, quad_kws=self.quad_kws)

    def __repr__(self):
        # from textwrap import dedent

        v = [super().__repr__(self)] + [
            f"{k}={getattr(self, k)}"
            for k in ["r_min", "phi_min", "segments", "quad_kws"]
        ]

        s = ",\n   ".join(v)

        return f"<{s}>"


class PhiBaseGenericCut(PhiBase):
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

    def __init__(
        self,
        phi_base: PhiBase,
        rcut: float,
        r_min: float | None = None,
        phi_min: float | None = None,
        quad_kws: float | None = None,
    ):

        self.phi_base = phi_base
        self.rcut = rcut

        segments = [x for x in self.phi_base.segments if x < self.rcut] + [self.rcut]
        super().__init__(
            r_min=r_min, phi_min=phi_min, quad_kws=quad_kws, segments=segments
        )

    def asdict(self):
        return {"phi_base": self.phi_base, "rcut": self.rcut}

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

    def _vcorrect(self, r):
        raise NotImplementedError

    def _dvdrcorrect(self, r):
        raise NotImplementedError

    @classmethod
    def from_phi(cls, phi_base, rcut):

        return cls(
            phi_base=phi_base,
            rcut=rcut,
        )


class PhiCut(PhiBaseGenericCut):
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
    phi_base : PhiBaseCuttable instance
        Potential class to perform cut on.
    rcut : float
        Where to 'cut' the potential.
    """

    @gcached()
    def _vcut(self):
        return self.phi_base.phi(self.rcut)

    def _vcorrect(self, r: np.ndarray) -> np.ndarray:
        return -cast(np.ndarray, self._vcut)

    def _dvdrcorrect(self, r: np.ndarray) -> np.ndarray:
        return np.array(0.0)


class PhiLFS(PhiBaseGenericCut):
    r"""
    Pair potential cut and linear force shifted at ``r_cut``.

    .. math::

        \phi_{rm lfs}(r) =
            \begin{cases}
                \phi(r)
                - \left( \frac{d \phi}{d r} \right)_{\rm cut} (r - r_{\rm cut})
                - \phi(r_{\rm cut}) & r < r_{\rm cut} \\
                0 & r_{\rm cut} < r
            \end{cases}
    """

    @gcached()
    def _vcut(self):
        return self.phi_base.phi(self.rcut)

    @gcached()
    def _dvdrcut(self):
        return self.phi_base.dphidr(self.rcut)

    def _vcorrect(self, r: np.ndarray) -> np.ndarray:
        out = -(self._vcut + self._dvdrcut * (r - self.rcut))
        return cast(np.ndarray, out)

    def _dvdrcorrect(self, r: np.ndarray) -> np.ndarray:
        out = -self._dvdrcut
        return cast(np.ndarray, out)


class PhiBaseCuttable(PhiBase):
    """
    Baseclass for creating Phi object which can be cut.

    This adds methods :meth:`cut` and :meth:`lfs`
    """

    def cut(self, rcut: float) -> PhiCut:
        """Creates a :class:`PhiCut` object from `self`"""
        return PhiCut(
            phi_base=self,
            rcut=rcut,
        )

    def lfs(self, rcut: float) -> PhiLFS:
        """Create as :class:`PhiLFS` object from `self`"""
        return PhiLFS(phi_base=self, rcut=rcut)
