"""
Classes/routines for pair potentials (:mod:`analphipy.potential`)
=================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np
from attrs import field

from ._attrs_utils import field_array_formatter, field_formatter
from ._docstrings import docfiller
from .base_potential import PhiAbstract, PhiBase

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal, TypeVar

    from ._typing import Array, ArrayLike, Float_or_ArrayLike, Phi_Signature


docfiller_phibase = docfiller.factory_inherit_from_parent(PhiBase)


@attrs.define(frozen=True)
@docfiller.decorate
class Generic(PhiBase):
    """
    Class to define potential using callables.

    Parameters
    ----------
    phi_func : Callable
        Function ``phi(r)``
    dphidr : Callable, optional
        Optional function ``dphidr(r)``.
    {r_min_exact}
    {phi_min_exact}
    {segments}
    """

    #: Function :math:`\phi(r)`
    phi_func: Phi_Signature

    #: Function :math:`d\phi(r)/dr`
    dphidr_func: Phi_Signature | None = None

    @docfiller_phibase()
    def phi(self, r: Float_or_ArrayLike) -> Array:
        r = np.asarray(r)
        return self.phi_func(r)

    @docfiller_phibase()
    def dphidr(self, r: Float_or_ArrayLike) -> Array:
        if self.dphidr_func is None:
            msg = "Must specify dphidr_func"
            raise ValueError(msg)
        r = np.asarray(r)
        return self.dphidr_func(r)  # pylint: disable=not-callable,useless-suppression


@attrs.define(frozen=True)
@docfiller.decorate
class Analytic(PhiBase):
    """
    Base class for defining analytic potentials.

    Parameters
    ----------
    {r_min_exact}
    {phi_min_exact}
    {segments}

    Notes
    -----
    Specific subclasses should set values for ``r_min``,
    ``phi_min``, and ``segments``, as well as
    forms for ``phi`` and ``dphidr``.

    """

    # fmt: off
    #: Position of minimum in :math:`\phi(r)`
    r_min: float = field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default=np.nan, init=False, repr=field_formatter()
    )
    #: Value of ``phi`` at minimum.
    phi_min: float = field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default=np.nan, init=False, repr=False
    )
    #: Integration limits.
    segments: Sequence[float] = field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default=(), init=False
    )
    # fmt: on


docfiller_analytic = docfiller.factory_inherit_from_parent(Analytic)


@attrs.define(frozen=True)
@docfiller_analytic(Analytic)
class LennardJones(Analytic):
    r"""
    Lennard-Jones potential.

    .. math::

        \phi(r) = 4 \epsilon \left[ \left(\frac{{\sigma}}{{r}}\right)^{{12}} - \left(\frac{{\sigma}}{{r}}\right)^6\right]

    Parameters
    ----------
    sig : float
        Length parameter :math:`\sigma`.
    eps : float
        Energy parameter :math:`\epsilon`.

    """

    #: Length parameter :math:`\sigma`
    sig: float = 1.0
    #: Energy parameter :math:`\epsilon`
    eps: float = 1.0

    def __attrs_post_init__(self) -> None:
        self._immutable_setattrs(
            r_min=self.sig * 2.0 ** (1.0 / 6.0),
            phi_min=-self.eps,
            segments=(0.0, np.inf),
        )

    @property
    def _sigsq(self) -> float:
        return self.sig**2

    @property
    def _four_eps(self) -> float:
        return 4.0 * self.eps

    @docfiller_analytic()
    def phi(self, r: Float_or_ArrayLike) -> Array:
        r = np.array(r)
        x2: Array = self._sigsq / (r * r)
        x6: Array = x2 * x2 * x2
        return self._four_eps * x6 * (x6 - 1.0)

    @docfiller_analytic()
    def dphidr(self, r: Float_or_ArrayLike) -> Array:
        """Calculate phi and dphi (=-1/r dphi/dr) at particular r."""
        r = np.array(r)
        rinvsq = 1.0 / (r * r)

        x2: Array = self._sigsq * rinvsq
        x6: Array = x2 * x2 * x2

        return -12.0 * self._four_eps * x6 * (x6 - 0.5) / r


@attrs.define(frozen=True)
@docfiller_analytic(Analytic)
class LennardJonesNM(Analytic):
    r"""
    Generalized Lennard-Jones potential

    .. math::

        \phi(r) = \epsilon \frac{{n}}{{n-m}} \left( \frac{{n}}{{m}} \right) ^{{m / (n-m)}}
        \left[ \left(\frac{{\sigma}}{{r}}\right)^n - \left(\frac{{\sigma}}{{r}}\right)^m\right]


    Parameters
    ----------
    n, m : int
        ``n`` and ``m`` parameters to potential :math:`n, m`.
    sig : float
        Length parameter :math:`\sigma`.
    eps : float
        Energy parameter :math:`\epsilon`.


    Notes
    -----
    with parameters ``n=12`` and ``m=6``, this is equivalent to :class:`LennardJones`.

    """

    n: int = 12  #: ``n`` parameter
    m: int = 6  #: ``m`` parameter
    sig: float = 1.0  #: Length parameter
    eps: float = 1.0  #: Energy parameter

    def __attrs_post_init__(self) -> None:
        n, m = self.n, self.m

        self._immutable_setattrs(
            r_min=self.sig * (n / m) ** (1.0 / (n - m)),
            phi_min=-self.eps,
            segments=(0.0, np.inf),
        )

    @property
    def _prefac(self) -> float:
        n, m, eps = self.n, self.m, self.eps
        out = eps * (n / (n - m)) * (n / m) ** (m / (n - m))
        if isinstance(out, float):
            return out

        msg = f"Bad parameters lead to unknown prefac {out}"
        raise ValueError(msg)

    @docfiller_analytic()
    def phi(self, r: Float_or_ArrayLike) -> Array:
        r = np.array(r)

        x = self.sig / r
        return self._prefac * (x**self.n - x**self.m)

    @docfiller_analytic()
    def dphidr(self, r: Float_or_ArrayLike) -> Array:
        r = np.array(r)
        x = self.sig / r

        xn = x**self.n
        xm = x**self.m

        return -self._prefac * (self.n * xn - self.m * xm) / (r)


@attrs.define(frozen=True)
@docfiller_analytic(Analytic)
class Yukawa(Analytic):
    r"""
    Hard core Yukawa potential


    .. math::

        \phi(r) =
        \begin{{cases}}
            \infty &  r \leq \sigma \\
            -\epsilon \frac{{\sigma}}{{r}} \exp\left[-z (r/\sigma - 1) \right] & r > \sigma
        \end{{cases}}


    Parameters
    ----------
    sig : float
        Length parameters :math:`\sigma`.
    eps : float
        Energy parameter :math:`\epsilon`.
    z : float
        Interaction range parameter :math:`z`

    """

    z: float = 1.0  #: Interaction parameter
    sig: float = 1.0  #: Length parameter
    eps: float = 1.0  #: Energy parameter

    def __attrs_post_init__(self) -> None:
        self._immutable_setattrs(
            r_min=self.sig,
            phi_min=self.eps,
            segments=(0.0, self.sig, np.inf),
        )

    @docfiller_analytic()
    def phi(self, r: Float_or_ArrayLike) -> Array:
        sig, eps = self.sig, self.eps

        r = np.array(r)
        phi = np.empty_like(r)
        m = r >= sig

        phi[~m] = np.inf
        if np.any(m):
            x = r[m] / sig
            phi[m] = -eps * np.exp(-self.z * (x - 1.0)) / x
        return phi


@attrs.define(frozen=True)
@docfiller_analytic(Analytic)
class HardSphere(Analytic):
    r"""
    Hard-sphere pair potential

    .. math::

        \phi(r) =
        \begin{{cases}}
        \infty & r \leq \sigma \\
        0 & r > \sigma
        \end{{cases}}

    Parameters
    ----------
    sig: float
        Length scale parameter :math:`\sigma`

    """

    sig: float = 1.0  #: Length parameter

    def __attrs_post_init__(self) -> None:
        self._immutable_setattrs(segments=(0.0, self.sig))

    @docfiller_analytic()
    def phi(self, r: Float_or_ArrayLike) -> Array:
        r = np.array(r)

        phi = np.empty_like(r)

        m0 = r < self.sig
        phi[m0] = np.inf
        phi[~m0] = 0.0
        return phi


@attrs.define(frozen=True)
@docfiller_analytic(Analytic)
class SquareWell(Analytic):
    r"""
    Square-well pair potential


    .. math::

        \phi(r) =
        \begin{{cases}}
        \infty & r \leq \sigma \\
        \epsilon & \sigma < r \leq \lambda \sigma \\
        0 & r > \lambda \sigma
        \end{{cases}}

    Parameters
    ----------
    sig : float
        Length scale parameter :math:`\sigma`.
    eps : float
        Energy parameter :math:`\epsilon`.  Note that here, ``eps`` is the value inside the well.  So, to specify an attractive square well potential, pass a negative value for ``eps``.
    lam : float
        Width of well parameter :math:`lambda`.

    """

    sig: float = 1.0  #: Length parameter.
    eps: float = 1.0  #: Energy parameter.
    lam: float = 1.5  #: Well width parameter.

    def __attrs_post_init__(self) -> None:
        self._immutable_setattrs(
            r_min=self.sig,
            phi_min=self.eps,
            segments=(0.0, self.sig, self.sig * self.lam),
        )

    @docfiller_analytic()
    def phi(self, r: Float_or_ArrayLike) -> Array:
        sig, eps, lam = self.sig, self.eps, self.lam

        r = np.array(r)

        phi = np.empty_like(r)

        m0 = r < sig
        m2 = r >= lam * sig
        m1 = (~m0) & (~m2)

        phi[m0] = np.inf
        phi[m1] = eps
        phi[m2] = 0.0

        return phi


def _validate_bounds(self: Any, attribute: Any, bounds: Sequence[float]) -> None:  # noqa: ARG001
    if len(bounds) != 2:  # noqa: PLR2004
        msg = "length of bounds must be 2"
        raise ValueError(msg)


if TYPE_CHECKING:
    T_CubicTable = TypeVar("T_CubicTable", bound="CubicTable")


@attrs.define(frozen=True)
@docfiller.decorate
class CubicTable(PhiBase):
    """
    Cubic interpolation table potential.

    Parameters
    ----------
    {r_min_exact}
    {phi_min_exact}
    bounds : sequence of float
        the minimum and maximum values of squared pair separation `r**2` at which `phi_table` is evaluated.
    phi_table : array-like
        Values of potential evaluated on even grid of ``r**2`` values.

    segments : sequence of float, optional
        Integration segments.  Defaults to ``sqrt(bounds)``.

    phi_left, phi_right, dphi_left, dphi_right : float, optional
        Values to set for ``phi``/``-1/r dphidr`` if  (left) ``r < bounds[0]`` or (right) ``r > bounds[1]``.

    """

    #: Minimum and maximum values of squared pair separation :math:`r^2`
    bounds: Sequence[float] = field(
        validator=_validate_bounds
    )  # validator=_validate_bounds, converter=tuple)
    #: Values of potential evaluated on even grid of :math:`r^2` values.
    phi_table: ArrayLike = field(converter=np.asarray, repr=field_array_formatter())
    #: value of `phi` at left bound (`r < bounds[0]`)
    phi_left: float = field(converter=float, default=np.inf)
    #: value of `phi` at right bound (`r > bounds[1]`)
    phi_right: float = field(converter=float, default=0.0)

    #: value of `dphi` at left bound
    dphi_left: float = field(converter=float, default=np.inf)
    #: value of `dphi` at right bound
    dphi_right: float = field(converter=float, default=0.0)

    _ds: float = field(init=False, repr=False)
    _dsinv: float = field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.phi_table, np.ndarray)  # noqa: S101

        ds = (self.bounds[1] - self.bounds[0]) / self.size

        self._immutable_setattrs(
            _ds=ds,
            _dsinv=1.0 / ds,
        )

        if self.segments is None:  # pyright: ignore[reportUnnecessaryComparison]
            self._immutable_setattrs(segments=tuple(np.sqrt(x) for x in self.bounds))

    @classmethod
    def from_phi(
        cls: type[T_CubicTable],
        phi: Phi_Signature,
        rmin: float,
        rmax: float,
        ds: float,
        **kws: Any,
    ) -> T_CubicTable:
        """
        Create object from callable pair potential function.

        This will evaluate ``phi`` at even spacing in ``s = r**2`` space.

        Parameters
        ----------
        phi : callable
            pair potential function
        rmin : float
            minimum pair separation `r` to evaluate at.
        rmax : float
            Maximum pair separation `r` to evaluate at.
        ds : float
            spaceing in ``s = r ** 2``.
        **kws :
            Extra arguments to constructor.

        Returns
        -------
        table : CubicTable

        """
        bounds: tuple[float, float] = (rmin * rmin, rmax * rmax)

        delta = bounds[1] - bounds[0]

        size = int(delta / ds)
        ds = delta / size

        # fmt: off
        r: Array = np.sqrt(np.arange(size + 1) * ds + bounds[0])  # pyright: ignore[reportUnknownMemberType]
        # fmt: on

        phi_table = phi(r)

        return cls(bounds=bounds, phi_table=phi_table, **kws)

    def __len__(self) -> int:
        return len(self.phi_table)

    @property
    def size(self) -> int:
        """Size of the array (less 1)."""
        return len(self) - 1

    @property
    def smin(self) -> float:
        """Minimum value of `s = r**2`."""
        return self.bounds[0]

    @property
    def smax(self) -> float:
        """Maximum value of `s = r**2`."""
        return self.bounds[1]

    def phidphi(self, r: Float_or_ArrayLike) -> tuple[Array, Array]:
        """Values of `phi` and `dphi` at `r`."""
        r = np.asarray(r)

        v = np.empty_like(r)
        dv = np.empty_like(r)

        s = r * r
        left = s <= self.smin
        right = s >= self.smax
        mid = (~left) & (~right)

        v[left] = self.phi_left
        dv[right] = self.dphi_left

        v[right] = self.phi_right
        dv[right] = self.dphi_right

        if np.any(mid):
            sds = (s[mid] - self.smin) * self._dsinv
            k = sds.astype(int)
            k[k < 0] = 0

            xi = sds - k

            # fmt: off
            t: Array = np.take(self.phi_table, [k, k + 1, k + 2], mode="clip")  # pyright: ignore[reportUnknownMemberType]
            # fmt: on
            dt = np.diff(t, axis=0)
            ddt = np.diff(dt, axis=0)

            v[mid] = t[0, :] + xi * (dt[0, :] + 0.5 * (xi - 1.0) * ddt[0, :])
            dv[mid] = -2.0 * self._dsinv * (dt[0, :] + (xi - 0.5) * ddt[0, :])
        return v, dv

    @docfiller_phibase()
    def phi(self, r: Float_or_ArrayLike) -> Array:
        return self.phidphi(r)[0]

    @docfiller_phibase()
    def dphidr(self, r: Float_or_ArrayLike) -> Array:
        r = np.asarray(r)
        return -r * self.phidphi(r)[1]

    @property
    def rsq_table(self) -> Array:
        """Value of ``r**2`` where potential is defined."""
        # fmt: off
        return (self.smin + np.arange(self.size + 1) * self._ds)  # pyright: ignore[reportUnknownMemberType]
        # fmt: on

    @property
    def r_table(self) -> Array:
        """Values of ``r`` where potential is defined."""
        return np.sqrt(self.rsq_table)


if TYPE_CHECKING:
    _PHI_NAMES = Literal["lj", "nm", "sw", "hs", "yk", "LJ", "NM", "SW", "HS", "YK"]


def factory(
    potential_name: _PHI_NAMES,
    rcut: float | None = None,
    lfs: bool = False,
    cut: bool = False,
    **kws: Any,
) -> PhiAbstract:
    """
    Factory function to construct Phi object by name.

    Parameters
    ----------
    potential_name : {lj, nm, sw, hs, yk}
        Name of potential.
    rcut : float, optional
        if passed, Construct either and 'lfs' or 'cut' version of the potential.
    lfs : bool, default=False
        If True, construct a  linear force shifted potential :class:`analphipy.base_potential.PhiLFS`.
    cut : bool, default=False
        If True, construct a cut potential :class:`analphipy.base_potential.PhiCut`.

    Returns
    -------
    phi : :class:`analphipy.base_potential.PhiBase` subclass
        output potential energy class.

    See Also
    --------
    LennardJones
    LennardJonesNM
    SquareWell
    HardSphere
    Yukawa

    """
    name = potential_name.lower()

    phi: PhiAbstract

    if name == "lj":
        phi = LennardJones(**kws)

    elif name == "sw":
        phi = SquareWell(**kws)

    elif name == "nm":
        phi = LennardJonesNM(**kws)

    elif name == "yk":
        phi = Yukawa(**kws)

    elif name == "hs":
        phi = HardSphere(**kws)

    else:
        msg = f"{name} must be in {_PHI_NAMES}"
        raise ValueError(msg)

    if lfs or cut:
        if rcut is None:
            msg = "rcut cannot be None"
            raise TypeError(msg)
        if cut:
            return phi.cut(rcut=rcut)

        if lfs:
            return phi.lfs(rcut=rcut)

    return phi
