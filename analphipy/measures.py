from typing import Callable, Optional, Union, cast

import numpy as np

from .utils import (
    TWO_PI,
    ArrayLike,
    Float_or_ArrayLike,
    Phi_Signature,
    combine_segmets,
    quad_segments,
)


def secondvirial(
    phi: Phi_Signature,
    beta: float,
    segments: ArrayLike,
    err: bool = False,
    full_output: bool = False,
    **kws
):
    """
    Calculate the second virial coefficient
    """

    def integrand(r):
        return TWO_PI * r ** 2 * (1 - np.exp(-beta * phi(r)))

    return quad_segments(
        integrand,
        segments=segments,
        sum_integrals=True,
        sum_errors=True,
        err=err,
        full_output=full_output,
        **kws
    )


def secondvirial_dbeta(
    phi: Phi_Signature,
    beta: float,
    segments: ArrayLike,
    err: bool = False,
    full_output: bool = False,
    **kws
):
    """
    derivative w.r.t beta of second virial coefficient
    """

    def integrand(r):
        v = phi(r)
        if np.isinf(v):
            return 0.0
        else:
            return TWO_PI * r ** 2 * v * np.exp(-beta * v)

    return quad_segments(
        integrand,
        segments=segments,
        sum_integrals=True,
        sum_errors=True,
        err=err,
        full_output=full_output,
        **kws
    )


def secondvirial_sw(beta: float, sig: float, eps: float, lam: float):
    """
    Second virial coefficient for a square well (SW) fluid. Note that this assumes that
    the SW fluid is defined by the potential phi:

    * infinity : r < sigma
    * eps : sigma <= r < sigma * lambda
    * 0 : r > sigma * lambda

    This differse by a factor (-eps) some other definitions
    """
    return (
        TWO_PI / 3.0 * sig ** 3 * (1.0 + (1 - np.exp(-beta * eps)) * (lam ** 3 - 1.0))
    )


def diverg_kl_integrand(
    p: Float_or_ArrayLike,
    q: Float_or_ArrayLike,
    volume: Optional[Float_or_ArrayLike] = None,
) -> np.ndarray:
    p, q = np.asarray(p), np.asarray(q)

    out = np.empty_like(p)

    zero = p == 0.0
    hero = ~zero

    out[zero] = 0.0
    out[hero] = p[hero] * np.log(p[hero] / q[hero])

    if volume is not None:
        out *= np.asarray(volume)

    return out


def diverg_kl_disc(
    P: Float_or_ArrayLike, Q: Float_or_ArrayLike, axis: Optional[int] = None
) -> Union[float, np.ndarray]:
    """
    calculate discrete Kullback–Leibler divergence

    Parameteres
    -----------
    P, Q : array-like
        Probabilities to consider
    Returns
    -------
    result : float or array
        value of KB divergence

    See Also
    --------
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Symmetrised_divergence
    """

    P, Q = np.asarray(P), np.asarray(Q)
    out = diverg_kl_integrand(P, Q).sum(axis=axis)

    return cast(Union[float, np.ndarray], out)


def _check_volume_func(volume: Optional[Union[str, Callable]] = None) -> Callable:
    if volume is None:
        volume = "1d"

    if isinstance(volume, str):
        if volume == "1d":
            volume = lambda x: 1.0
        elif volume == "2d":
            volume = lambda x: 2.0 * np.pi * x
        elif volume == "3d":
            volume = lambda x: 4 * np.pi * x ** 2
        else:
            raise ValueError("unknown dimension")
    else:
        assert callable(volume)

    return volume


def diverg_kl_cont(
    p: Callable,
    q: Callable,
    segments: ArrayLike,
    segments_q: Optional[ArrayLike] = None,
    volume: Optional[Union[str, Callable]] = None,
    err: bool = False,
    full_output: bool = False,
    **kws
):
    """
    calculate discrete Kullback–Leibler divergence for contiuous pdf

    Parameteres
    -----------
    p, q: callable
        Probabilities to consider
    volume : callable, optional
        volume element if not linear integration region
        For examle, use volume = lambda x: 4 * np.pi * x ** 2 for spherically symmetric p/q

    segments : list
        segments to integrate over
    segments_q : list, optional
        if supplied, build total segments by combining segments and seqments_q
    Returns
    -------
    result : float or array
        value of KB divergence

    See Also
    --------
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Symmetrised_divergence
    """
    volume = _check_volume_func(volume)

    if segments_q is not None:
        segments = combine_segmets(segments, segments_q)

    def func(x):
        return diverg_kl_integrand(p=p(x), q=q(x), volume=volume(x))

    return quad_segments(
        func,
        segments=segments,
        sum_integrals=True,
        sum_errors=True,
        err=err,
        full_output=full_output,
        **kws
    )


def diverg_js_disc(
    P: Float_or_ArrayLike, Q: Float_or_ArrayLike, axis: Optional[int] = None
) -> Union[float, np.ndarray]:
    """
    Jensen–Shannon divergence

    Parameteres
    -----------
    P, Q : array-like
        Probabilities to consider
    Returns
    -------
    result : float or array
        value of JS divergence

    See Also
    --------
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Symmetrised_divergence
    """

    P, Q = np.asarray(P), np.asarray(Q)

    M = 0.5 * (P + Q)

    out = 0.5 * (diverg_kl_disc(P, M, axis=axis) + diverg_kl_disc(Q, M, axis=axis))
    return out


def diverg_js_integrand(
    p: Float_or_ArrayLike,
    q: Float_or_ArrayLike,
    volume: Optional[Float_or_ArrayLike] = None,
) -> np.ndarray:
    p = np.asarray(p)
    q = np.asarray(q)

    m = 0.5 * (p + q)

    out = 0.5 * (diverg_kl_integrand(p, m) + diverg_kl_integrand(q, m))

    if volume is not None:
        out *= np.asarray(volume)

    return cast(np.ndarray, out)


# def _plogp(p):
#     hero = p != 0.
#     out = np.zeros_like(p)
#     out[hero] = p[hero] * np.log(p[hero])
#     return out

# def diverg_js_integrandb(p, q, volume = None):

#     p, q = np.asarray(p), np.asarray(q)

#     m = 0.5 * (p + q)

#     out = 0.5 * (_plogp(p) + _plogp(q)) - _plogp(m)

#     if volume is not None:
#         out *= np.asarray(volume)

#     return out


def diverg_js_cont(
    p: Callable,
    q: Callable,
    segments: ArrayLike,
    segments_q: Optional[ArrayLike] = None,
    volume: Optional[Union[str, Callable]] = None,
    err: bool = False,
    full_output: bool = False,
    **kws
):
    """
    Jensen–Shannon divergence

    Parameteres
    -----------
    P, Q : array-like
        Probabilities to consider
    Returns
    -------
    result : float or array
        value of JS divergence

    See Also
    --------
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Symmetrised_divergence
    """

    volume = _check_volume_func(volume)

    if segments_q is not None:
        segments = combine_segmets(segments, segments_q)

    def func(x):
        return diverg_js_integrand(p(x), q(x), volume(x))

    return quad_segments(
        func,
        segments=segments,
        sum_integrals=True,
        sum_errors=True,
        err=err,
        full_output=full_output,
        **kws
    )


# class DivergPhi:

#     def __init__(self, func0, func1, segments0, segments1, volume='3d'):
#         """
#         Parameters
#         ----------
#         phi0, phi1 : Phi objects
#         volume : callable or str
#         if str, then

#         * '1d': volume = 1.
#         * '2d': volume = 2 * np.pi * r
#         * '3d': volume = 4 * np.pi * r**2
#         """

#         self.phi0 = phi0
#         self.phi1 = phi1

#         self.segments = combine_segmets(segments0, segments1)


#         if type(volume) == 'str':
#             if volume == '1d':
#                 volume = lambda x: 1.
#             elif volume == '2d':
#                 volume = lambda x: 2. * np.pi * r
#             elif volume == '3d':
#                 volume = lambda x: 4 * np.pi * r ** 2
#             else:
#                 raise ValueError('unknown dimension')

#         assert callable(volume)
#         self.volume = volume


# class NormalizedFunc(object):
#     def __init__(self, func, segments, volume=None, norm_fac=None, **kws):

#         self.func = lambda x: func(x) * volume(x)
#         self.segments = segments
#         self.kws = kws

#         if volume is None:
#             volume = lambda x: 1.0
#         self.volume = volume

#         if norm_fac is None:
#             self.set_norm_fac()
#         self.norm_fac = norm_fac

#     def __call__(self, x):
#         return self.norm_fac * self.func(x)

#     def integrand(self, x):
#         return self.volume(x) * self(x)

#     def set_norm_fac(self, **kws):
#         self.norm_fac = 1.0

#         kws = dict(self.kws, **kws)

#         value = quad_segments(
#             self.integrand,
#             segments=self.segments,
#             sum_integrals=True,
#             sum_errors=True,
#             err=False,
#             full_output=False,
#             **kws
#         )
#         self.norm_fac = 1.0 / value

#     def __add__(self, other):
#         # combine stuff

#         return type(self)(
#             func=lambda x: self(x) + other(x),
#             segments=combine_segmets(self.segments, other.segments),
#             volume=self.volume,
#             norm_fac=1.0,
#             kws=dict(other.kws, **self.kws),
#         )

#     def __mul__(self, fac):
#         return type(self)(
#             func=self.func,
#             segments=self.segments,
#             volume=self.volume,
#             norm_fac=self.norm_fac * fac,
#             kws=dict(self.kws),
#         )
