# type: ignore
"""
Base classes (:mod:`analphipy.potentials_base`)
===============================================
"""
from __future__ import annotations

from typing import Literal, Sequence, cast

import attrs
import numpy as np
from attrs import field
from custom_inherit import DocInheritMeta

from ._attrs_utils import field_formatter, optional_converter, private_field
from ._docstrings import docfiller_shared
from ._typing import Float_or_ArrayLike
from .measures import Measures
from .norofrenkel import NoroFrenkelPair
from .utils import minimize_phi


# * attrs utilities
@optional_converter
def segments_converter(segments):
    return tuple(float(x) for x in segments)


def validate_segments(self, attribute, segments):
    assert len(segments) >= 2, "Segments must be at least of length 2"


def _kw_only_field(kw_only=True, default=None, **kws):
    return attrs.field(kw_only=kw_only, default=default, **kws)


@attrs.frozen
@docfiller_shared
class _PhiBase(metaclass=DocInheritMeta(style="numpy_with_merge")):  # type: ignore
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

    #: Position of minimum in :math:`\phi(r)`
    r_min: float | None = _kw_only_field(
        converter=optional_converter(float), repr=field_formatter()
    )

    #: Value of ``phi`` at minimum.
    phi_min: float | None = _kw_only_field(
        converter=optional_converter(float), repr=field_formatter()
    )

    #: Integration limits
    segments: Sequence[float] | None = _kw_only_field(
        converter=segments_converter,
    )

    def asdict(self) -> dict:
        """Convert object to dictionary"""
        return attrs.asdict(self, filter=self._get_smart_filter())

    def new_like(self, **kws):
        """Create a new object with optional parameters

        Parameters
        ----------
        **kws
            `attribute`, `value` pairs.
        """
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
        ...

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

        This returns the value of :math:`d \phi(r) / dr`



        Parameters
        ----------
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
        ~analphipy.utils.minimize_phi
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

    def to_nf(self, **kws):
        """
        Create a :class:`analphipy.NoroFrenkelPair` object.

        Parameters
        ---------
        **kws :
            Extra arguments to :class:`analphipy.NoroFrenkelPair` constructor.
            parameters `phi`, `semgnets`, `r_min', `phi_min` default to values from `self`.

        Returns
        -------
        nf : :class:`analphipy.NoroFrenkelPair`

        """
        if self.r_min is None:
            raise ValueError("must set `self.r_min` to use NoroFrenkel")

        for k in ["phi", "segments", "r_min", "phi_min"]:
            if k not in kws:
                kws[k] = getattr(self, k)

        return NoroFrenkelPair(**kws)

    def to_measures(self, **kws):
        """
        Create a :class:`analphipy.measures.Measures` object.

        Parameters
        ----------
        **kws :
            Extra arguments to :class:`analphipy.Measures` constructor.
            parameters `phi`, `semgnets` default to values from `self`.

        Returns
        -------
        nf : :class:`analphipy.measures.Measures`
        """

        for k in ["phi", "segments"]:
            if k not in kws:
                kws[k] = getattr(self, k)

        return Measures(**kws)


@attrs.frozen
class _PhiBaseCut(_PhiBase):
    """
    Base Class for creating cut potential from base potential


    phi_cut(r) = phi(r) + _vcorrect(r) if r <= rcut
               = 0 otherwise


    dphi_cut(r) = dphi(r) + _dvcorrect(r) if r <= rcut
                = 0 otherwise

    So, for example, for just cut, `_vcorrect(r) = -v(rcut)`, `dvrcut=0`,
    and for lfs, `_vcorrect(r) = -v(rcut) - dv(rcut)/dr (r - rcut)`
    `_dvcorrect(r) = ...`


    Parameters
    ----------
    phi_base : PhiBase
        Instance of (sub)class of :class:`analphipy.PhiBase`.
    rcut : float
        Position to cut the potential.


    """

    #: Instance of (sub)class of :class:`analphipy.PhiBase`
    phi_base: PhiBase
    #: Position to cut the potential
    rcut: float = field(converter=float)
    #: Integration limits
    segments: float = private_field()

    def __attrs_post_init__(self):
        self._immutable_setattrs(
            segments=tuple(x for x in self.phi_base.segments if x < self.rcut)
            + (self.rcut,)
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
class PhiCut(_PhiBaseCut):
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

    _vcut: float = private_field()

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self._immutable_setattrs(_vcut=self.phi_base.phi(self.rcut))

    def _vcorrect(self, r: np.ndarray) -> np.ndarray:
        return -cast(np.ndarray, self._vcut)

    def _dvdrcorrect(self, r: np.ndarray) -> np.ndarray:
        return np.array(0.0)


@attrs.frozen
class PhiLFS(_PhiBaseCut):
    r"""
    Pair potential cut and linear force shifted at ``r_cut``.

    .. math::

        \phi_{\rm lfs}(r) =
            \begin{cases}
                \phi(r)
                - \left( \frac{d \phi}{d r} \right)_{\rm cut} (r - r_{\rm cut})
                - \phi(r_{\rm cut}) & r < r_{\rm cut} \\
                0 & r_{\rm cut} < r
            \end{cases}
    """

    _vcut: float = private_field()
    _dvdrcut: float = private_field()

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
class PhiBase(_PhiBase):
    def cut(self, rcut):
        """Create a :class:`PhiCut` potential."""
        return PhiCut(phi_base=self, rcut=rcut)

    def lfs(self, rcut):
        """Create a :class:`PhiLFS` potential."""
        return PhiLFS(phi_base=self, rcut=rcut)
