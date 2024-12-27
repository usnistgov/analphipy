"""
Utilities module (:mod:`analphipy.utils`)
=========================================
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, cast

import numpy as np

from ._docstrings import docfiller

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Any, Callable, Protocol, TypeVar

    from ._typing import ArrayLike, OptimizeResultInterface, P, QuadSegments, R
    from ._typing_compat import Concatenate, TypeGuard


TWO_PI: float = 2.0 * np.pi


def combine_segmets(a: ArrayLike, b: ArrayLike) -> list[float]:
    """
    Combine two lists of segments.

    Parameters
    ----------
    a, b : array-like
        list of segments.

    Returns
    -------
    combined : list of float
        sorted unique values for ``a`` and ``b``.

    """
    aa, bb = set(a), set(b)
    return sorted(aa.union(bb))  # type: ignore[arg-type, unused-ignore]


def is_float(val: Any) -> TypeGuard[float]:
    """
    Type guard for float.

    Use to type narrow output for quad_segments
    """
    return isinstance(val, float)


@docfiller.decorate
def quad_segments(
    func: Callable[..., Any],
    segments: ArrayLike,
    args: tuple[Any, ...] = (),
    full_output: bool = False,
    sum_integrals: bool = True,
    sum_errors: bool = False,
    err: bool = True,
    **kws: Any,
) -> QuadSegments:
    """
    Perform quadrature with discontinuities.

    Parameters
    ----------
    func : callable
        function to be integrated
    {segments}
    args : tuple, optional
        Extra positional arguments to `func`.
    full_output : bool, default=False
        If True, return extra information.
    sum_integrals : bool, default=True
        If True, sum the segments in the output.
    sum_errors : bool, default=True
        If True and returning `error` sum errors.
    err : bool, default=True
        If True, return error.
    **kws :
        Extra arguments to :func:`scipy.integrate.quad`

    Returns
    -------
    integral : float or list of float
        If `sum_integrals`,  this is the sum of integrals over each segment.  Otherwise return list
        of values corresponding to integral in each segment.
    errors : float or list of float, optional
        If `err` or `full_output` are True, then return error.  If `sum_errors`, then sum of errors
        Across segments.
    outputs : object
        Output from :func:`scipy.integrate.quad`. If multiple segments, return a list of output.

    See Also
    --------
    scipy.integrate.quad

    """
    from scipy.integrate import (  # pyright: ignore[reportMissingTypeStubs]
        quad,  # pyright: ignore[reportUnknownVariableType]
    )

    out: list[tuple[float, float, dict[str, Any]]] = [
        quad(func, a=a, b=b, args=args, full_output=True, **kws)
        for a, b in zip(segments[:-1], segments[1:])
    ]

    integrals: float | list[float]
    errors: float | list[float]
    outputs: dict[str, Any] | list[dict[str, Any]]

    if len(segments) == 2:  # noqa: PLR2004
        integrals, errors, outputs = out[0]
    else:
        integrals_list: list[float] = []
        errors_list: list[float] = []
        outputs_list: list[dict[str, Any]] = []

        for y, e, o in out:
            integrals_list.append(y)
            errors_list.append(e)
            outputs_list.append(o)

        integrals = (
            cast("float", np.sum(integrals_list))  # pyright: ignore[reportUnknownMemberType]
            if sum_integrals
            else integrals_list
        )

        errors = (
            np.sum(errors_list)  # pyright: ignore[reportUnknownMemberType]
            if sum_errors
            else errors_list
        )

        outputs = outputs_list

    # Gather final results
    if err and full_output:
        return integrals, errors, outputs
    if err and not full_output:
        return integrals, errors
    if not err and full_output:
        return integrals, outputs
    return integrals


def minimize_phi(
    phi: Callable[..., Any], r0: float, bounds: ArrayLike | None = None, **kws: Any
) -> tuple[float, float, OptimizeResultInterface | None]:
    """
    Find value of ``r`` which minimized ``phi``.

    Parameters
    ----------
    phi : callable
        Function to be minimized.
    r0 : float
        Guess for position of minimum.
    bounds : tuple of float
        If passed, should be of form ``bounds=(lower_bound, upper_bound)``.
    **kws :
        Extra arguments to :func:`scipy.optimize.minimize`

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
    scipy.optimize.minimize

    """
    from scipy.optimize import (  # pyright: ignore[reportMissingTypeStubs]
        minimize,  # pyright: ignore[reportUnknownVariableType]
    )

    if bounds is None:
        bounds = (0.0, np.inf)

    if bounds[0] == bounds[1]:
        # force minima to be exactly here
        xmin = bounds[0]
        ymin = phi(xmin)
        return xmin, ymin, None

    outputs = cast("OptimizeResultInterface", minimize(phi, r0, bounds=[bounds], **kws))

    if not outputs["success"]:
        msg = "could not find min of phi"
        raise ValueError(msg)

    xmin = outputs["x"][0]

    tmp = outputs["fun"]
    ymin = tmp[0] if isinstance(tmp, np.ndarray) else tmp

    return cast("tuple[float, float, OptimizeResultInterface]", (xmin, ymin, outputs))


# * Phi utilities
if TYPE_CHECKING:

    class HasQuadKws(Protocol):
        """Class protocol for quad_kws"""

        quad_kws: Mapping[str, Any]

    S = TypeVar("S", bound=HasQuadKws)


def partial_phi(phi: Callable[..., R], **params: Any) -> Callable[..., R]:
    """Partial potential with params set."""
    from functools import partial

    return partial(phi, **params)


def segments_to_segments_cut(segments: Iterable[float], rcut: float) -> list[float]:
    """Update segments for 'cut' potential."""
    return [x for x in segments if x < rcut] + [rcut]


def add_quad_kws(
    func: Callable[Concatenate[S, P], R],
) -> Callable[Concatenate[S, P], R]:
    """Add quad keyword arguments."""

    @wraps(func)
    def wrapped(self: S, /, *args: P.args, **kws: P.kwargs) -> R:
        kws = dict(self.quad_kws, **kws)  # type: ignore[assignment]
        return func(self, *args, **kws)

    return wrapped
