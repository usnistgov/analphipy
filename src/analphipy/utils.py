"""
Utilities module (:mod:`analphipy.utils`)
=========================================
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

from ._docstrings import docfiller_shared

if TYPE_CHECKING:
    from ._typing import ArrayLike

# from typing_extensions import Protocol

# ArrayLike = Union[Sequence[float], NDArray[np.float_]]
# Float_or_ArrayLike = Union[float, ArrayLike]
# Float_or_Array = Union[float, NDArray[np.float_]]

# Phi_Signature = Callable[..., Union[float, np.ndarray]]

TWO_PI = 2.0 * np.pi

# class Phi_Signature(Protocol):
#     def __call__(self, Float_or_ArrayLike) -> Union[float, np.ndarray]: ...

# Vector = list[float]

# def scale(scalar: float, vector: Vector) -> Vector:
#     return [scalar * num for num in vector]


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
    return sorted(aa.union(bb))


@docfiller_shared
def quad_segments(
    func: Callable,
    segments: ArrayLike,
    args: tuple = (),
    full_output: bool = False,
    sum_integrals: bool = True,
    sum_errors: bool = False,
    err: bool = True,
    **kws,
):
    """
    Perform quadrature with discontinuities

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

    out = [
        quad(func, a=a, b=b, args=args, full_output=full_output, **kws)
        for a, b in zip(segments[:-1], segments[1:])
    ]

    if len(segments) == 2:
        if full_output:
            integrals, errors, outputs = out[0]
        else:
            integrals, errors = out[0]
            outputs = None

    else:
        integrals = []
        errors = []

        if full_output:
            outputs = []
            for y, e, o in out:
                integrals.append(y)
                errors.append(e)
                outputs.append(o)
        else:
            outputs = None
            for y, e in out:
                integrals.append(y)
                errors.append(e)

        if sum_integrals:
            integrals = np.sum(integrals)

        if sum_errors:
            errors = np.sum(errors)

    # Gather final results
    result = [integrals]
    if err:
        result.append(errors)
    if outputs is not None:
        result.append(outputs)

    if len(result) == 1:
        return result[0]
    else:
        return result


def minimize_phi(
    phi: Callable, r0: float, bounds: ArrayLike | None = None, **kws
) -> tuple[float, float, Any]:
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

    if bounds is None:
        bounds = (0.0, np.inf)

    if bounds[0] == bounds[1]:
        # force minima to be exactly here
        xmin = bounds[0]
        ymin = phi(xmin)
        outputs = None
    else:
        outputs = minimize(phi, r0, bounds=[bounds], **kws)

        if not outputs.success:
            raise ValueError("could not find min of phi")

        xmin = outputs.x[0]

        try:
            ymin = outputs.fun[0]
        except Exception:
            ymin = outputs.fun

    return xmin, ymin, outputs


# * Phi utilities


def partial_phi(phi, **params):
    from functools import partial

    return partial(phi, **params)


def segments_to_segments_cut(segments, rcut):
    """Update segments for 'cut' potential"""
    return [x for x in segments if x < rcut] + [rcut]


# def phi_to_phi_cut(
#     rcut: float,
#     phi: Callable,
#     dphidr: Callable | None = None,
#     meta: Mapping | None = None,
# ):
#     r"""
#     Create callable ``phi`` and ``dphidr`` cut potentials.

#     Parameters
#     ----------
#     rcut : float
#         Position of cut.
#     phi : callable
#         input callable function.
#     dphidr : callable, optional
#     input callable for :math:`d\phi/dr`

#     Returns
#     -------
#     phi_cut : callable
#     dphidr_cut : callable or None
#         Note that ``dphidr_cut`` is always returned, even if it is None.
#     """

#     vcut = phi(rcut)

#     def phi_cut(r):
#         r = np.asarray(r)
#         v = np.empty_like(r)

#         left = r <= rcut
#         right = ~left

#         v[right] = 0.0

#         if np.any(left):
#             v[left] = phi(r[left]) - vcut
#         return v

#     if dphidr:

#         def dphidr_cut(r):
#             r = np.asarray(r)
#             dvdbeta = np.empty_like(r)

#             left = r <= rcut
#             right = ~left

#             dvdbeta[right] = 0

#             if np.any(left):
#                 dvdbeta[left] = dphidr(r[left])

#             return dvdbeta

#     else:
#         dphidr_cut = None

#     if meta is not None:
#         meta = dict(meta, rcut=rcut)
#         meta["style"] = meta.get("style", ()) + ("cut",)

#     return phi_cut, dphidr_cut, meta


# def phi_to_phi_lfs(
#     rcut: float,
#     phi: Callable,
#     dphidr: Callable,
#     meta: Mapping | None = None,
# ):
#     r"""
#     Create callable ``phi`` and ``dphidr`` cut potentials.

#     Parameters
#     ----------
#     rcut : float
#         Position of cut.
#     phi : callable
#         input callable function.
#     dphidr : callable, optional
#     input callable for :math:`d\phi/dr`

#     Returns
#     -------
#     phi_cut : callable
#     dphidr_cut : callable or None
#         Note that ``dphidr_cut`` is always returned, even if it is None.
#     """

#     vcut = phi(rcut)
#     dvdrcut = dphidr(rcut)

#     def phi_cut(r):
#         r = np.asarray(r)
#         v = np.empty_like(r)

#         left = r <= rcut
#         right = ~left

#         v[right] = 0.0

#         if np.any(left):
#             v[left] = phi(r[left]) - vcut - dvdrcut * (r - rcut)
#         return v

#     def dphidr_cut(r):
#         r = np.asarray(r)
#         dvdbeta = np.empty_like(r)

#         left = r <= rcut
#         right = ~left

#         dvdbeta[right] = 0

#         if np.any(left):
#             dvdbeta[left] = dphidr(r[left]) - dvdrcut

#         return dvdbeta

#     if meta is not None:
#         meta = dict(meta, rcut=rcut)
#         meta["style"] = meta.get("style", ()) + ("lfs",)

#     return phi_cut, dphidr_cut, meta


# def wca_decomp_rep(phi, dphidr, r_min, phi_min, meta):
#     def phi_rep(r: Float_or_ArrayLike) -> np.ndarray:
#         """WCA repulsive potential"""
#         r = np.array(r)
#         v = np.empty_like(r)

#         left = r <= r_min
#         right = ~left

#         v[left] = phi(r[left]) - phi_min
#         v[right] = 0.0
#         return v

#     if dphidr is not None:

#         def dphidr_rep(r: Float_or_ArrayLike) -> np.ndarray:
#             r = np.array(r)
#             dvdr = np.empty_like(r)

#             left = r <= r_min
#             right = ~left

#             dvdr[left] = dphidr(r[left])
#             dvdr[right] = 0.0

#             return dvdr

#     else:
#         dphidr_rep = None

#     if meta is not None:
#         meta = dict(meta)
#         meta["style"] = meta.get("style", ()) + ("wca_rep",)

#     return phi_rep, dphidr_rep, meta


# def wca_decomp_att(phi, dphidr, r_min, phi_min, meta):
#     def phi_att(r: Float_or_ArrayLike) -> np.ndarray:
#         """WCA repulsive potential"""
#         r = np.array(r)
#         v = np.empty_like(r)

#         left = r <= r_min
#         right = ~left

#         v[left] = phi_min
#         v[right] = phi(r[right])
#         return v

#     if dphidr is not None:

#         def dphidr_att(r: Float_or_ArrayLike) -> np.ndarray:
#             r = np.array(r)
#             dvdr = np.empty_like(r)

#             left = r <= r_min
#             right = ~left

#             dvdr[left] = 0.0
#             dvdr[right] = dphidr(r[right])

#             return dvdr

#     else:
#         dphidr_att = None

#     if meta is not None:
#         meta = dict(meta)
#         meta["style"] = meta.get("style", ()) + ("wca_att",)

#     return phi_att, dphidr_att, meta


def add_quad_kws(func: Callable) -> Callable:
    @wraps(func)
    def wrapped(self, *args, **kws):
        kws = dict(self.quad_kws, **kws)
        return func(self, *args, **kws)

    return wrapped
