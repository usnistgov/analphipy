# type: ignore
# flake8: noqa

from __future__ import annotations

# import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import singledispatch

# from functools import partial
from typing import Optional, Sequence, Tuple, cast

import numpy as np

# from abc import abstractmethod
from custom_inherit import DocInheritMeta
from typing_extensions import Literal

from analphipy.norofrenkel import NoroFrenkelPair
from analphipy.utils import minimize_phi

from ._typing import Float_or_ArrayLike
from .cached_decorators import cached_clear, gcached
from .utils import (
    phi_to_phi_cut,
    phi_to_phi_lfs,
    segments_to_segments_cut,
    wca_decomp_att,
    wca_decomp_rep,
)

# * Utility function


class PhiBase:
    r"""
    Base class for working with pair potentials.

    This is the most general case, where you supply a callable function
    for ``phi`` and (optionally), ``dphi``.  In addition, extra parameters
    to handle the calculation of metrics is also included

    Parameters
    ----------
    phi : callable
        Function to calculate the pair potential.  This should have the form
        ``v = phi(r)``, where ``r`` can be a scalar or ndarray, and ``v`` is an ndarray.
    dphidr : callable, optional
        Function to calculate derivative of ``phi`` with respect to ``r``
        :math:`\frac{d \phi}{d r}`
    r_min : float, optional
        Position of minimum in ``phi``.
    phi_min : float, optional
        Value of ``phi`` at minimum.  If specified, but pass in value for ``r_min``, then
        ``phi_min = phi(r_min)``. This is available, to exactly set the value of
        the minimum.
    segments : sequence of floats, optional
        Bounds specifiying unique regions of the ``phi``.  Used in calculating various
        metric
    quad_kws : mapping, optional
        Extra arguments to :func:`analphipy.utils.quad_segments`

    meta : mapping, optional
        Optional mapping of any meta data to be passed along.
    """

    def __init__(
        self,
        phi: Callable,
        dphidr: Callable | None = None,
        r_min: float | None = None,
        phi_min: float | None = None,
        segments: Sequence[float] | None = None,
        quad_kws: Mapping | None = None,
        meta: Mapping | None = None,
    ):

        self._phi_func = phi
        self._dphidr_func = dphidr

        self._r_min = r_min
        self._phi_min = phi_min
        self._segments = segments
        self.meta = meta

        if quad_kws is None:
            quad_kws = {}
        else:
            quad_kws = dict(quad_kws)

        self.quad_kws = quad_kws

    @property
    def r_min(self):
        return self._r_min

    @r_min.setter
    @cached_clear()
    def r_min(self, val):
        self._r_min = val

    @property
    def phi_min(self):
        if self._phi_min is not None:
            return self._phi_min
        elif self._r_min is not None:
            return self.phi(self._r_min)
        else:
            return self._phi_min

    @phi_min.setter
    @cached_clear()
    def phi_min(self, val):
        self._phi_min = val

    @property
    def quad_kws(self):
        return self._quad_kws

    @quad_kws.setter
    @cached_clear()
    def quad_kws(self, val):
        if val is None:
            val = {}
        else:
            val = dict(val)

        self._quad_kws = val

    @property
    def segments(self):
        return self._segments

    @segments.setter
    @cached_clear()
    def segments(self, val):
        self._segments = val

    def phi(self, r: Float_or_ArrayLike) -> np.ndarray:
        r = np.asarray(r)
        return self._phi_func(r)

    def dphidr(self, r: Float_or_ArrayLike) -> np.ndarray:
        if self._dphidr_func is None:
            raise NotImplementedError("no method dphidr provided")
        else:
            r = np.asarray(r)
            return self._dphidr_func(r)

    def new_like(self, inherit_mins=True, **kws):
        """
        Create new object like ``self``.

        Any parameters specified will override parameters of ``self``.

        Parameters
        ----------
        inherit_mins : bool, default=True
            If True, inherit ``r_min`` and ``phi_min`` parameters unless explicitly specified.

        **kws :
            Parameters to explicitly set.
        """

        defaults = {
            "phi": self._phi_func,
            "dphidr": self._dphidr_func,
            "segments": self.segments,
            "quad_kws": self.quad_kws,
            "meta": self.meta,
        }

        if inherit_mins:
            defaults["r_min"], defaults["phi_min"] = self.r_min, self.phi_min

        kws = {**defaults, **kws}

        return type(self)(**kws)

    def cut(self, rcut):
        """
        Create cut versions of phi/dphidr
        """

        phi_cut, dphidr_cut, meta = phi_to_phi_cut(
            rcut=rcut, phi=self._phi_func, dphidr=self._dphidr_func, meta=self.meta
        )

        return self.new_like(
            phi=phi_cut,
            dphidr=dphidr_cut,
            segments=segments_to_segments_cut(self.segments, rcut),
            meta=meta,
            inherit_mins=False,
        )

    def lfs(self, rcut):
        """
        Create lfs version of phi/dphidr
        """

        if self._dphidr_func is None:
            raise ValueError("Must have ``dphidr`` function to apply lfs")

        phi_lfs, dphidr_lfs, meta = phi_to_phi_lfs(
            rcut=rcut, phi=self._phi_func, dphidr=self._dphidr_func, meta=self.meta
        )

        return self.new_like(
            phi=phi_lfs,
            dphidr=dphidr_lfs,
            segments=segments_to_segments_cut(self.segments, rcut),
            meta=meta,
            inherit_mins=False,
        )

    def minimize(
        self,
        r0: float | None = None,
        bounds: tuple[float, float] | Literal["None"] | None = None,
        **kws,
    ):
        """Determine position `r` where ``phi`` is minimized.

        Parameters
        ----------
        r0 : float
            Guess for position of minimum.
            Defaults to ``mean(bounds)``.
        bounds : tuple floats, {'None'}, optional
            If passed, should be of form ``bounds=(lower_bound, upper_bound)``.
            Defaults to ``(self.segments[0], self.segments[-1])``.
            If pass string value of `'None'`, then use ``bounds=[0, inf]``.
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

        if bounds == "None":
            bounds = [0.0, np.inf]
        elif bounds is None:
            bounds = (self.segments[0], self.segments[-1])

        if r0 is None:
            r0 = np.mean(bounds)

        return minimize_phi(phi=self.phi, r0=r0, bounds=bounds, **kws)

    def set_min(
        self, r0: float | None = None, bounds: tuple[float, float] | None = None, **kws
    ):
        """
        Create new object with ``r_min`` and ``phi_min`` set by numerical minimization.

        See Also
        --------
        :meth:`minimize`
        """
        r_min, phi_min, output = self.minimize(r0=r0, bounds=bounds, **kws)
        return self.new_like(r_min=r_min, phi_min=phi_min)

    @gcached()
    def nf(self):
        """Noro frenkel object"""

        if self.r_min is None:
            raise ValueError("Must have r_min/phi_min specified")

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

        return Measures(self)

    @gcached()
    def wca_rep(self):
        """
        WCA repulsive
        """

        if self.r_min is None:
            raise ValueError("must specify r_min to access WCA decomp")

        phi, dphidr, meta = wca_decomp_rep(
            phi=self._phi_func,
            dphidr=self._dphidr_func,
            meta=self.meta,
            r_min=self.r_min,
            phi_min=self.phi_min,
        )

        return self.new_like(
            phi=phi,
            dphidr=dphidr,
            meta=meta,
            inherit_mins=False,
        )

    @gcached()
    def wca_att(self):
        """
        WCA repulsive
        """

        if self.r_min is None:
            raise ValueError("must specify r_min to access WCA decomp")

        phi, dphidr, meta = wca_decomp_att(
            phi=self._phi_func,
            dphidr=self._dphidr_func,
            meta=self.meta,
            r_min=self.r_min,
            phi_min=self.phi_min,
        )

        return self.new_like(
            phi=phi,
            dphidr=dphidr,
            meta=meta,
            inherit_mins=False,
        )

    def __repr__(self):

        if self.meta is None:
            return "<PhiBase>"
        else:
            kws = dict(self.meta)
            style = "_".join(kws.pop("style"))

            params = ", ".join([f"{k}={v}" for k, v in kws.items()])

            return f"<{style}: {params}>"


def factory_phi_lj(sig: float, eps: float, quad_kws: Mapping | None = None):
    r"""Lennard-Jones potential.

    .. math::

        \phi(r) = 4 \epsilon \left[ \left(\frac{{\sigma}}{{r}}\right)^{{12}} - \left(\frac{{\sigma}}{{r}}\right)^6\right]

    Parameters
    ----------
    sig : float
        Length parameter :math:`\sigma`.
    eps : float
        Energy parameter :math:`\epsilon`.
    """

    sigsq = sig * sig
    four_eps = 4.0 * eps

    def phi(r: Float_or_ArrayLike) -> np.ndarray:
        r = np.array(r)

        x2 = sigsq / (r * r)
        x6 = x2**3
        return cast(np.ndarray, four_eps * x6 * (x6 - 1.0))

    def dphidr(r: Float_or_ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """calculate phi and dphi (=-1/r dphi/dr) at particular r"""
        r = np.array(r)
        rinvsq = 1.0 / (r * r)

        x2 = sigsq * rinvsq
        x6 = x2 * x2 * x2

        dphidr = -12.0 * four_eps * x6 * (x6 - 0.5) / r
        return dphidr

    r_min = sig * 2.0 ** (1.0 / 6.0)

    phi_min = -eps

    segments = [0.0, np.inf]

    meta = {
        "style": ("lj",),
        "sig": sig,
        "eps": eps,
    }

    return PhiBase(
        phi=phi,
        dphidr=dphidr,
        r_min=r_min,
        phi_min=phi_min,
        segments=segments,
        quad_kws=quad_kws,
        meta=meta,
    )
