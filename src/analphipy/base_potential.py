"""
Base classes (:mod:`analphipy.base_potential`)
==============================================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import attrs
import numpy as np
from attrs import field

from ._attrs_utils import field_formatter, private_field
from ._docstrings import docfiller

if TYPE_CHECKING:
    from typing import Any, Callable, Literal, Sequence

    from typing_extensions import Self

    from ._typing import Array, Float_or_ArrayLike


from .measures import Measures
from .norofrenkel import NoroFrenkelPair
from .utils import minimize_phi


# * attrs utilities
def segments_converter(segments: Sequence[Any]) -> tuple[float, ...]:
    """Make sure segments are in correct format."""
    return tuple(float(x) for x in segments)


# def validate_segments(self: Any, attribute: Any, segments: Sequence[Any]) -> None:
#     assert len(segments) >= 2, "Segments must be at least of length 2"


# def _kw_only_field(kw_only: bool = True, default: Any = None, **kws: Any) -> Any:
#     return attrs.field(kw_only=kw_only, default=default, **kws)


@attrs.define(frozen=True, kw_only=True)
@docfiller.decorate
class PhiAbstract:
    """
    Abstract class from which base classes inherit.

    Parameters
    ----------
    r_min : float, optional
        Position of minimum.
    phi_min : float, optional
        Value of ``phi`` at ``r_min``.
    {segments}
    """

    #: Position of minimum in :math:`\phi(r)`
    r_min: float | None = attrs.field(
        default=None,
        converter=attrs.converters.optional(float),
        repr=field_formatter(),
    )

    #: Value of ``phi`` at minimum.
    phi_min: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float), repr=field_formatter()
    )

    #: Integration limits
    segments: Sequence[float] = attrs.field(
        default=None,
        converter=attrs.converters.optional(segments_converter),
    )

    def asdict(self) -> dict[str, Any]:
        """Convert object to dictionary."""
        return attrs.asdict(self, filter=self._get_smart_filter())

    def new_like(self, **kws: Any) -> Self:
        """
        Create a new object with optional parameters.

        Parameters
        ----------
        **kws
            `attribute`, `value` pairs.
        """
        return attrs.evolve(self, **kws)

    def assign(self, **kws: Any) -> Self:
        """Alias to :meth:`new_like`."""
        return self.new_like(**kws)

    def _immutable_setattrs(self, **kws: Any) -> None:
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
        ... class Derived(PhiAbstract):
        ...     _derived = attrs.field(init=False)
        ...
        ...     def __attrs_post_init__(self):
        ...         self._immutable_setattrs(_derived=self.r_min + 10)

        >>> x = Derived(r_min=5)
        >>> x._derived
        15.0
        """
        for key, value in kws.items():
            object.__setattr__(self, key, value)  # noqa: PLC2801

    def _get_smart_filter(
        self,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        exclude_private: bool = True,
        exclude_no_init: bool = True,
    ) -> Callable[..., Any]:  # pragma: no cover
        """
        Create a filter to include exclude names.

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
        Precedence is in order
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

        includes: list[Any] = []
        for f in fields:
            if f.name in include:
                includes.append(f)

            elif (
                f.name in exclude
                or (exclude_private and f.name.startswith("_"))
                or (exclude_no_init and not f.init)
            ):
                pass

            else:
                includes.append(f)
        return attrs.filters.include(*includes)

    def phi(self, r: Float_or_ArrayLike) -> Array:
        """
        Pair potential.

        Parameters
        ----------
        r : array-like
            pair separation

        Returns
        -------
        phi : ndarray
            Evaluated pair potential.
        """
        msg = "Must implement in subclass"
        raise NotImplementedError(msg)

    def dphidr(self, r: Float_or_ArrayLike) -> Array:
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
        msg = "Must implement in subclass"
        raise NotImplementedError(msg)

    def minimize(
        self,
        r0: float | Literal["mean"],
        bounds: tuple[float, float] | Literal["segments"] | None = None,
        **kws: Any,
    ) -> tuple[float, float, Any]:
        """
        Determine position `r` where ``phi`` is minimized.

        Parameters
        ----------
        r0 : float
            Guess for position of minimum.
            If value is `'mean'`, then use ``r0 = mean(bounds)``.

        bounds : tuple of float, {'segments'}, optional
            Bounds for minimization search.
            If tuple, should be of form ``bounds=(lower_bound, upper_bound)``.
            If `'segments`, then ``bounds=(segments[0], segments[-1])``.
            If None, no bounds used.
        **kws :
            Extra arguments to :func:`analphipy.utils.minimize_phi`

        Returns
        -------
        rmin : float
            ``phi(rmin)`` is found location of minimum.
        phimin : float
            Value of ``phi(rmin)``, i.e., the value of ``phi`` at the minimum
        output : object
            Output class from :func:`scipy.optimize.minimize`.

        See Also
        --------
        ~analphipy.utils.minimize_phi
        """
        if bounds == "segments":
            if self.segments is not None:  # pyright: ignore[reportUnnecessaryComparison]
                bounds = (self.segments[0], self.segments[-1])
            else:
                bounds = None  # pragma: no cover

        if r0 == "mean":
            if bounds is not None:
                if not isinstance(bounds, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]
                    msg = "bounds must be a tuple"
                    raise TypeError(msg)
                r0 = cast(float, np.mean(bounds))
            else:
                msg = 'must specify bounds with r0="mean"'
                raise ValueError(msg)

        return minimize_phi(self.phi, r0=r0, bounds=bounds, **kws)

    def assign_min_numeric(
        self,
        r0: float | Literal["mean"],
        bounds: tuple[float, float] | Literal["segments"] | None = None,
        **kws: Any,
    ) -> Self:
        """
        Create new object with minima set by numerical minimization.

        call :meth:`minimize`
        """
        r_min, phi_min, _ = self.minimize(r0=r0, bounds=bounds, **kws)
        return self.new_like(r_min=r_min, phi_min=phi_min)

    def to_nf(self, **kws: Any) -> NoroFrenkelPair:
        """
        Create a :class:`analphipy.norofrenkel.NoroFrenkelPair` object.

        Parameters
        ----------
        **kws :
            Extra arguments to :class:`analphipy.norofrenkel.NoroFrenkelPair` constructor.
            parameters `phi`, `semgnets`, `r_min', `phi_min` default to values from `self`.

        Returns
        -------
        nf : :class:`analphipy.norofrenkel.NoroFrenkelPair`
        """
        if self.r_min is None:
            msg = "must set `self.r_min` to use NoroFrenkel"
            raise ValueError(msg)

        for k in ["phi", "segments", "r_min", "phi_min"]:
            if k not in kws:
                kws[k] = getattr(self, k)

        return NoroFrenkelPair(**kws)

    def to_measures(self, **kws: Any) -> Measures:
        """
        Create a :class:`analphipy.measures.Measures` object.

        Parameters
        ----------
        **kws :
            Extra arguments to :class:`analphipy.measures.Measures` constructor.
            parameters `phi`, `semgnets` default to values from `self`.

        Returns
        -------
        nf : :class:`analphipy.measures.Measures`
        """
        for k in ["phi", "segments"]:
            if k not in kws:
                kws[k] = getattr(self, k)

        return Measures(**kws)


docfiller_phiabstract = docfiller.factory_inherit_from_parent(PhiAbstract)


@attrs.define(frozen=True)
@docfiller.inherit(PhiAbstract)
class PhiCutBase(PhiAbstract):
    r"""
    Base Class for creating cut potential from base potential


    .. math::

        \phi_{{\rm cut}}(r) =
            \begin{{cases}}
                \phi(r) + {{\text _vcorrect}}(r) & r < r_{{\rm cut}} \\
                0 & r_{{\rm cut}} \leq r
            \end{{cases}}

        \frac{{d \phi_{{\rm cut}}(r)}}{{d r}} =
            \begin{{cases}}
                \frac{{d \phi(r)}}{{d r}} + {{\text _dvdrcorrect}}(r) & r < r_{{\rm cut}} \\
                0 & r_{{\rm cut}} \leq r
            \end{{cases}}

    So, for example, for just cut, `_vcorrect(r) = -v(rcut)`, `dvrcut=0`,
    and for lfs, `_vcorrect(r) = -v(rcut) - dv(rcut)/dr (r - rcut)`
    `_dvcorrect(r) = ...`


    Parameters
    ----------
    phi_base : :class:`analphipy.base_potential.PhiAbstract`
    rcut : float
        Position to cut the potential.


    """

    #: Instance of (sub)class of :class:`analphipy.base_potential.PhiAbstract`
    phi_base: PhiBase
    #: Position to cut the potential
    rcut: float = field(converter=float)
    #: Integration limits
    segments: Sequence[float] = private_field()  # pyright: ignore[reportIncompatibleVariableOverride]

    def __attrs_post_init__(self) -> None:
        if self.phi_base.segments is None:  # pyright: ignore[reportUnnecessaryComparison]
            msg = "must specify segments"
            raise ValueError(msg)  # pragma: no cover
        self._immutable_setattrs(
            segments=(
                *tuple(x for x in self.phi_base.segments if x < self.rcut),
                self.rcut,
            )
        )

    @docfiller_phiabstract()
    def phi(self, r: Float_or_ArrayLike) -> Array:  # noqa: D102
        r = np.asarray(r)
        v = np.empty_like(r)

        left = r <= self.rcut
        right = ~left

        v[right] = 0.0

        if np.any(left):
            v[left] = self.phi_base.phi(r[left]) + self._vcorrect(r[left])
        return v

    @docfiller_phiabstract()
    def dphidr(self, r: Float_or_ArrayLike) -> Array:  # noqa: D102
        r = np.asarray(r)
        dvdr = np.empty_like(r)

        left = r <= self.rcut
        right = ~left

        dvdr[right] = 0.0
        dvdr[left] = self.phi_base.dphidr(r[left]) + self._dvdrcorrect(r[left])

        return dvdr

    def _vcorrect(self, r: Array) -> Array:
        raise NotImplementedError

    def _dvdrcorrect(self, r: Array) -> Array:
        raise NotImplementedError


@attrs.frozen
@docfiller.inherit(PhiCutBase)
class PhiCut(PhiCutBase):
    r"""
    Pair potential cut at position ``r_cut``.

    .. math::

        \phi_{{\rm cut}}(r) =
            \begin{{cases}}
                \phi(r) - \phi(r_{{\rm cut}})  & r < r_{{\rm cut}} \\
                0 & r_{{\rm cut}} \leq r
            \end{{cases}}

    Parameters
    ----------
    phi_base : :class:`analphipy.base_potential.PhiBase` instance
        Potential class to perform cut on.
    rcut : float
        Where to 'cut' the potential.
    """

    _vcut: float = private_field()

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        self._immutable_setattrs(_vcut=self.phi_base.phi(self.rcut))

    def _vcorrect(self, r: Array) -> Array:
        return -cast("Array", self._vcut)

    def _dvdrcorrect(self, r: Array) -> Array:
        return np.array(0.0)


@attrs.frozen
@docfiller.inherit(PhiCutBase)
class PhiLFS(PhiCutBase):
    r"""
    Pair potential cut and linear force shifted at ``r_cut``.

    .. math::

        \phi_{{\rm lfs}}(r) =
            \begin{{cases}}
                \phi(r)
                - \left( \frac{{d \phi}}{{d r}} \right)_{{\rm cut}} (r - r_{{\rm cut}})
                - \phi(r_{{\rm cut}}) & r < r_{{\rm cut}} \\
                0 & r_{{\rm cut}} < r
            \end{{cases}}
    """

    _vcut: float = private_field()
    _dvdrcut: float = private_field()

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        self._immutable_setattrs(
            _vcut=self.phi_base.phi(self.rcut), _dvdrcut=self.phi_base.dphidr(self.rcut)
        )

    def _vcorrect(self, r: Array) -> Array:
        return -(self._vcut + self._dvdrcut * (r - self.rcut))

    def _dvdrcorrect(self, r: Array) -> Array:
        out = -self._dvdrcut
        return cast("Array", out)


@attrs.frozen
@docfiller.inherit(PhiAbstract)
class PhiBase(PhiAbstract):
    """Base class for potentials."""

    def cut(self, rcut: float) -> PhiCut:
        """Create a :class:`PhiCut` potential."""
        return PhiCut(phi_base=self, rcut=rcut)

    def lfs(self, rcut: float) -> PhiLFS:
        """Create a :class:`PhiLFS` potential."""
        return PhiLFS(phi_base=self, rcut=rcut)
