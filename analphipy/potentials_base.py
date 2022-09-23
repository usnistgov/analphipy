# type: ignore
from __future__ import annotations

from typing import Literal, Sequence, cast

import attrs
import numpy as np
from attrs import field
from custom_inherit import DocInheritMeta

from ._attrs_utils import _formatter, _private_field, optional_converter
from ._docstrings import docfiller_shared
from ._typing import Float_or_ArrayLike, Phi_Signature
from .measures import Measures
from .norofrenkel import NoroFrenkelPair
from .utils import minimize_phi


# * attrs utilities
@optional_converter
def segments_converter(segments):
    return [float(x) for x in segments]


def validate_segments(self, attribute, segments):
    assert len(segments) >= 2, "Segments must be at least of length 2"


def _kw_only_field(kw_only=True, default=None, **kws):
    return attrs.field(kw_only=kw_only, default=default, **kws)


@attrs.frozen
@docfiller_shared
class PhiBase(metaclass=DocInheritMeta(style="numpy_with_merge")):  # type: ignore
    """
    Base Class for working with potentials.

    Parameters
    ----------
    r_min : float, optional
        Position of minimum.
    phi_min : float, optional
        Value of ``phi`` at ``r_min``.
    {segments}
    """

    r_min: float | None = _kw_only_field(converter=optional_converter(float))
    phi_min: float | None = _kw_only_field(converter=optional_converter(float))
    segments: Sequence[float] | None = _kw_only_field(
        converter=segments_converter,
    )

    def asdict(self):
        """Convert object to dictionary"""
        return attrs.asdict(self, filter=self._get_smart_filter())

    def new_like(self, **kws):
        """Create a new object with optional parameters"""
        return attrs.evolve(self, **kws)

    def assign(self, **kws):
        """Alias to :meth:`new_like`"""
        return self.new_like(**kws)

    def _immutable_setattrs(self, **kws):
        """
        Set attributes of frozen attrs class.

        This should only be used to set 'derived' attributes on creation.

        Parameters
        ----------
        **kws :
            `key`, `value` pairs to be set.  `key` should be the name of the attribute to be set,
            and `value` is the value to set `key` to.

        Examples
        --------
        >>> @attrs.frozen
        ... class Derived(PhiBase):
        ...     _derived = attrs.field(init=False)
        ...
        ...     def __attrs_post_init__(self):
        ...         self._immutable_setattrs(_derived=self.r_min + 10)

        >>> x = Derived(r_min=5)
        >>> x._derived
        15.0
        """

        for key, value in kws.items():
            object.__setattr__(self, key, value)

    def _get_smart_filter(
        self, include=None, exclude=None, exclude_private=True, exclude_no_init=True
    ):
        """
        create a filter to include exclude names

        Parameters
        ----------
        include : sequence of str
            Names to include
        exclude : sequence of str
            names to exclude
        exclude_private : bool, default=True
            If True, exclude any names starting with '_'
        exclude_no_init : bool, default=True
            If True, exclude any fields defined with ``field(init=False)``.

        Notes
        -----
        Precidence is in order
        `include, exclude, exclude_private exclude_no_init`.
        That is, if a name is in include and exclude and is private/no_init,
        it will be included
        """
        fields = attrs.fields(type(self))

        if include is None:
            include = []
        elif isinstance(include, str):
            include = [include]

        if exclude is None:
            exclude = []
        elif isinstance(exclude, str):
            exclude = [exclude]

        includes = []
        for f in fields:

            if f.name in include:
                includes.append(f)

            elif f.name in exclude:
                pass

            elif exclude_private and f.name.startswith("_"):
                pass

            elif exclude_no_init and not f.init:
                pass

            else:
                includes.append(f)
        return attrs.filters.include(*includes)

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
            if self.segments is not None:
                bounds = (self.segments[0], self.segments[-1])
            else:
                bounds = None

        if r0 == "mean":
            if bounds is not None:
                r0 = np.mean(bounds)
            else:
                r0 = None

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

    def to_nf(self, **quad_kws):
        if self.r_min is None:
            raise ValueError("must set `self.r_min` to use NoroFrenkel")

        return NoroFrenkelPair(
            phi=self.phi,
            segments=self.segments,
            r_min=self.r_min,
            phi_min=self.phi_min,
            quad_kws=quad_kws,
        )

    def to_measures(self, **quad_kws):
        return Measures(phi=self.phi, segments=self.segments, quad_kws=quad_kws)


@attrs.frozen
class PhiBaseCut(PhiBase):
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

    phi_base: PhiBase
    rcut: float = field(converter=float)
    segments: float = _private_field()

    def __attrs_post_init__(self):
        self._immutable_setattrs(
            segments=[x for x in self.phi_base.segments if x < self.rcut] + [self.rcut]
        )

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


@attrs.frozen
class PhiCut(PhiBaseCut):
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

    _vcut: float = _private_field()

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self._immutable_setattrs(_vcut=self.phi_base.phi(self.rcut))

    def _vcorrect(self, r: np.ndarray) -> np.ndarray:
        return -cast(np.ndarray, self._vcut)

    def _dvdrcorrect(self, r: np.ndarray) -> np.ndarray:
        return np.array(0.0)


@attrs.frozen
class PhiLFS(PhiBaseCut):
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

    _vcut: float = _private_field()
    _dvdrcut: float = _private_field()

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self._immutable_setattrs(
            _vcut=self.phi_base.phi(self.rcut), _dvdrcut=self.phi_base.dphidr(self.rcut)
        )

    def _vcorrect(self, r: np.ndarray) -> np.ndarray:
        out = -(self._vcut + self._dvdrcut * (r - self.rcut))
        return cast(np.ndarray, out)

    def _dvdrcorrect(self, r: np.ndarray) -> np.ndarray:
        out = -self._dvdrcut
        return cast(np.ndarray, out)


@attrs.frozen
class PhiGeneric(PhiBase):
    """
    Function form of PhiBase
    """

    phi_func: Phi_Signature
    dphidr_func: Phi_Signature | None = None

    def phi(self, r: Float_or_ArrayLike) -> np.ndarray:
        r = np.asarray(r)
        return self.phi_func(r)

    def dphidr(self, r: Float_or_ArrayLike) -> np.ndarray:
        if self.dphidr_func is None:
            raise ValueError("Must specify dphidr_func")
        r = np.asarray(r)
        return self.dphidr_func(r)

    def cut(self, rcut):
        return PhiCut(phi_base=self, rcut=rcut)

    def lfs(self, rcut):
        return PhiLFS(phi_base=self, rcut=rcut)


@attrs.frozen
class PhiDefined(PhiBase):
    """
    Base class for defining a pair potential


    Notes
    -----
    Specific subclasses should set values for ``r_min``,
    ``phi_min``, and ``segments``, as well as
    forms for ``phi`` and ``dphidr``.

    """

    r_min: float = field(init=False, repr=_formatter())
    phi_min: float = field(init=False, repr=False)
    segments: float = field(init=False)


@attrs.frozen
class PhiDefinedCuttable(PhiDefined):
    def cut(self, rcut):
        return PhiCut(phi_base=self, rcut=rcut)

    def lfs(self, rcut):
        return PhiLFS(phi_base=self, rcut=rcut)


@attrs.frozen
class Phi_lj(PhiDefinedCuttable):
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

    sig: float = 1.0
    eps: float = 1.0

    def __attrs_post_init__(self):
        self._immutable_setattrs(
            r_min=self.sig * 2.0 ** (2.0 / 6.0),
            phi_min=-self.eps,
            segments=[0.0, np.inf],
        )

    @property
    def _sigsq(self):
        return self.sig**2

    @property
    def _four_eps(self):
        return 4.0 * self.eps

    def phi(self, r: Float_or_ArrayLike) -> np.ndarray:
        r = np.array(r)
        x2 = self._sigsq / (r * r)
        x6 = x2**3
        return cast(np.ndarray, self._four_eps * x6 * (x6 - 1.0))

    def dphidr(self, r: Float_or_ArrayLike) -> np.ndarray:
        """calculate phi and dphi (=-1/r dphi/dr) at particular r"""
        r = np.array(r)
        rinvsq = 1.0 / (r * r)

        x2 = self._sigsq * rinvsq
        x6 = x2 * x2 * x2
        return cast(np.ndarray, -12.0 * self._four_eps * x6 * (x6 - 0.5) / r)


@attrs.frozen
class Phi_nm(PhiDefinedCuttable):
    r"""
    Generalized Lennard-Jones potential

    .. math::

        \phi(r) = \epsilon \frac{n}{n-m} \left( \frac{n}{m} \right) ^{m / (n-m)}
        \left[ \left(\frac{\sigma}{r}\right)^n - \left(\frac{\sigma}{r}\right)^m\right]


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
    with parameters ``n=12`` and ``m=6``, this is equivalent to :class:`Phi_lj`.
    """
    n: int = 12
    m: int = 6
    sig: float = 1.0
    eps: float = 1.0

    def __attrs_post_init__(self):
        n, m = self.n, self.m

        self._immutable_setattrs(
            r_min=self.sig * (n / m) ** (1.0 / (n - m)),
            phi_min=-self.eps,
            segments=[0.0, np.inf],
        )

    @property
    def _prefac(self):
        n, m, eps = self.n, self.m, self.eps
        return eps * (n / (n - m)) * (n / m) ** (m / (n - m))

    def phi(self, r: Float_or_ArrayLike) -> np.ndarray:
        r = np.array(r)

        x = self.sig / r
        out = self._prefac * (x**self.n - x**self.m)
        return cast(np.ndarray, out)

    def dphidr(self, r: Float_or_ArrayLike) -> np.ndarray:

        r = np.array(r)
        x = self.sig / r

        xn = x**self.n
        xm = x**self.m

        return cast(np.ndarray, -self._prefac * (self.n * xn - self.m * xm) / (r))


@attrs.frozen
class Phi_yk(PhiDefinedCuttable):
    r"""
    Hard core Yukawa potential


    .. math::

        \phi(r) =
        \begin{cases}
            \infty &  r \leq \sigma \\
            -\epsilon \frac{\sigma}{r} \exp\left[-z (r/\sigma - 1) \right] & r > \sigma
        \end{cases}


    Parameters
    ----------
    sig : float
        Length parameters :math:`\sigma`.
    eps : float
        Energy parameter :math:`\epsilon`.
    z : float
        Interaction range parameter :math:`z`

    """

    z: float = 1.0
    sig: float = 1.0
    eps: float = 1.0

    def __attrs_post_init__(self):
        self._immutable_setattrs(
            r_min=self.sig,
            phi_min=self.eps,
            segments=[0.0, self.sig, np.inf],
        )

    def phi(self, r: Float_or_ArrayLike) -> np.ndarray:
        sig, eps = self.sig, self.eps

        r = np.array(r)
        phi = np.empty_like(r)
        m = r >= sig

        phi[~m] = np.inf
        if np.any(m):
            x = r[m] / sig
            phi[m] = -eps * np.exp(-self.z * (x - 1.0)) / x
        return phi


@attrs.frozen
class Phi_hs(PhiDefined):
    r"""
    Hard-sphere pair potential

    .. math::

        \phi(r) =
        \begin{cases}
        \infty & r \leq \sigma \\
        0 & r > \sigma

    Parameters
    ----------
    sig: float
        Length scale parameter :math:`\sigma`
    """

    sig: float = 1.0

    def __attrs_post_init__(self):
        self._immutable_setattrs(segments=[0.0, self.sig])

    def phi(self, r: Float_or_ArrayLike) -> np.ndarray:
        r = np.array(r)

        phi = np.empty_like(r)

        m0 = r < self.sig
        phi[m0] = np.inf
        phi[~m0] = 0.0
        return phi


@attrs.frozen
class Phi_sw(PhiDefined):
    r"""
    Square-well pair potential


    .. math::

        \phi(r) =
        \begin{cases}
        \infty & r \leq \sigma \\
        \epsilon & \sigma < r \leq \lambda \sigma \\
        0 & r > \lambda \sigma

    Parameters
    ----------
    sig : float
        Length scale parameter :math:`\sigma`.
    eps : float
        Energy parameter :math:`\epsilon`.  Note that here, ``eps`` is the value inside the well.  So, to specify an attractive square well potential, pass a negative value for ``eps``.
    lam : float
        Width of well parameter :math:`lambda`.
    """

    sig: float = 1.0
    eps: float = 1.0
    lam: float = 1.5

    def __attrs_post_init__(self):
        self._immutable_setattrs(
            r_min=self.sig,
            phi_min=self.eps,
            segments=[0.0, self.sig, self.sig * self.lam],
        )

    def phi(self, r: Float_or_ArrayLike) -> np.ndarray:

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


@attrs.frozen
class CubicTable(PhiDefinedCuttable):
    """
    Cubic interpolation table potential

    Parameters
    ----------
    bounds : sequence of floats
        the minimum and maximum values of squared pair separation `r**2` at which `phi_array` is evaluated.
    phi_array : array-like
        Values of potential evaluated on even grid of ``r**2`` values.
    """

    bounds: Sequence[float]
    phi_array: Float_or_ArrayLike = field(factory=np.array)

    _ds: float = _private_field()
    _dsinv: float = _private_field()

    def __attrs_post_init__(self):
        ds = (self.bounds[1] - self.bounds[0]) / self.size

        self._immutable_setattrs(
            _ds=ds,
            _dsinv=1.0 / ds,
            segments=[np.sqrt(x) for x in self.bounds],
        )

    @classmethod
    def from_phi(cls, phi: Phi_Signature, rmin: float, rmax: float, ds: float):
        """
        Create object from callable pair potential funciton.

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

        Returns
        -------
        table : CubicTable

        """
        bounds = (rmin * rmin, rmax * rmax)

        delta = bounds[1] - bounds[0]

        size = int(delta / ds)
        ds = delta / size

        phi_array = []

        r = np.sqrt(np.arange(size + 1) * ds + bounds[0])

        phi_array = phi(r)

        return cls(bounds=bounds, phi_array=phi_array)

    def __len__(self):
        return len(self.phi_array)

    @property
    def size(self) -> int:
        return len(self) - 1

    @property
    def smin(self) -> float:
        return self.bounds[0]

    @property
    def smax(self) -> float:
        return self.bounds[1]

    def phidphi(self, r: Float_or_ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(r)

        v = np.empty_like(r)
        dv = np.empty_like(r)

        s = r * r
        left = s <= self.smax
        right = ~left

        v[right] = 0.0
        dv[right] = 0.0

        if np.any(left):

            sds = (s[left] - self.smin) * self._dsinv
            k = sds.astype(int)
            k[k < 0] = 0

            xi = sds - k

            t = np.take(self.phi_array, [k, k + 1, k + 2], mode="clip")
            dt = np.diff(t, axis=0)
            ddt = np.diff(dt, axis=0)

            v[left] = t[0, :] + xi * (dt[0, :] + 0.5 * (xi - 1.0) * ddt[0, :])
            dv[left] = -2.0 * self._dsinv * (dt[0, :] + (xi - 0.5) * ddt[0, :])
        return v, dv

    def phi(self, r: Float_or_ArrayLike) -> np.ndarray:
        return self.phidphi(r)[0]

    def dphidr(self, r: Float_or_ArrayLike) -> np.ndarray:
        r = np.asarray(r)
        return cast(np.ndarray, -r * self.phidphi(r)[1])


_PHI_NAMES = Literal["lj", "nm", "sw", "hs", "yk", "LJ", "NM", "SW", "HS", "YK"]


def factory_phi(
    potential_name: _PHI_NAMES,
    rcut: float | None = None,
    lfs: bool = False,
    cut: bool = False,
    **kws,
) -> "PhiBase":
    """Factory function to construct Phi object by name

    Parameters
    ----------
    potential_name : {lj, nm, sw, hs, yk}
        Name of potential.
    rcut : float, optional
        if passed, Construct either and 'lfs' or 'cut' version of the potential.
    lfs : bool, default=False
        If True, construct a  linear force shifted potential :class:`analphipy.PhiLFS`.
    cut : bool, default=False
        If True, construct a cut potential :class:`analphipy.PhiCut`.

    Returns
    -------
    phi :
        output potential energy class.
    """

    name = potential_name.lower()

    phi: "PhiBase"

    if name == "lj":
        phi = Phi_lj(**kws)

    elif name == "sw":
        phi = Phi_sw(**kws)

    elif name == "nm":
        phi = Phi_nm(**kws)

    elif name == "yk":
        phi = Phi_yk(**kws)

    elif name == "hs":
        phi = Phi_hs(**kws)

    else:
        raise ValueError(f"{name} must be in {_PHI_NAMES}")

    if lfs or cut:
        assert rcut is not None
        if cut:
            phi = phi.cut(rcut=rcut)

        elif lfs:
            phi = phi.lfs(rcut=rcut)

    return phi
