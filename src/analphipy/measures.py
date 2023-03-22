"""
Routines to calculate measures of pair potentials (:mod:`analphipy.measures`)
=============================================================================
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence, Union, cast

import numpy as np
from custom_inherit import doc_inherit

from analphipy.cached_decorators import gcached

from ._docstrings import docfiller_shared
from ._typing import ArrayLike, Float_or_ArrayLike, Phi_Signature
from .utils import TWO_PI, add_quad_kws, combine_segmets, quad_segments

__all__ = [
    "Measures",
    "secondvirial",
    "secondvirial_dbeta",
    "secondvirial_sw",
    "diverg_kl_cont",
    "diverg_js_cont",
]


def other():
    """A function"""
    pass


@docfiller_shared
def secondvirial(
    phi: Phi_Signature,
    beta: float,
    segments: ArrayLike,
    err: bool = False,
    full_output: bool = False,
    **kws,
):
    r"""
    Calculate the second virial coefficient.

    .. math::

        B_2(\beta) = -\int 2\pi r^2 dr \left(\exp(-\beta \phi(r)) - 1\right)

    Parameters
    ----------
    {phi}
    {beta}
    {segments}
    {err}
    {full_output}
    **kws
        Extra arguments to :func:`analphipy.utils.quad_segments`

    Returns
    -------
    B2 : float
        Value of second virial coefficient.
    {error_summed}
    {full_output_summed}

    See Also
    --------
    ~analphipy.utils.quad_segments
    """

    def integrand(r):
        return TWO_PI * r**2 * (1 - np.exp(-beta * phi(r)))

    return quad_segments(
        integrand,
        segments=segments,
        sum_integrals=True,
        sum_errors=True,
        err=err,
        full_output=full_output,
        **kws,
    )


@docfiller_shared
def secondvirial_dbeta(
    phi: Phi_Signature,
    beta: float,
    segments: ArrayLike,
    err: bool = False,
    full_output: bool = False,
    **kws,
):
    r"""
    ``beta`` derivative of second virial coefficient.

    .. math::

        \frac{{d B_2}}{{d \beta}} = \int 2\pi r^2 dr \phi(r) \exp(-\beta \phi(r))

    Parameters
    ----------
    {phi}
    {beta}
    {segments}
    {err}
    {full_output}

    Returns
    -------
    dB2dbeta : float
        Value of derivative.
    {error_summed}
    {full_output_summed}


    """

    def integrand(r):
        v = phi(r)
        if np.isinf(v):
            return 0.0
        else:
            return TWO_PI * r**2 * v * np.exp(-beta * v)

    return quad_segments(
        integrand,
        segments=segments,
        sum_integrals=True,
        sum_errors=True,
        err=err,
        full_output=full_output,
        **kws,
    )


@docfiller_shared
def secondvirial_sw(beta: float, sig: float, eps: float, lam: float):
    r"""
    Second virial coefficient for a square well (SW) fluid. Note that this assumes that
    the SW fluid is defined by the potential:

    .. math::

        \phi(r) =
        \begin{{cases}}
            \infty & r \leq \sigma \\
            \epsilon & \sigma < r \leq \lambda \sigma  \\
            0 & \lambda \sigma < r
        \end{{cases}}


    Parameters
    ----------
    {beta}
    sig : float
        Length parameter :math:`\sigma`.
    eps : float
        Energy parameter :math:`\epsilon`.
    lam : float
        Well width parameter :math:`\lambda`.

    Returns
    -------
    B2 : float
        Value of second virial coefficient.

    """
    return (
        TWO_PI / 3.0 * sig**3 * (1.0 + (1 - np.exp(-beta * eps)) * (lam**3 - 1.0))
    )


def diverg_kl_integrand(
    p: Float_or_ArrayLike,
    q: Float_or_ArrayLike,
    volume: Float_or_ArrayLike | None = None,
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


@docfiller_shared
def diverg_kl_disc(
    P: Float_or_ArrayLike, Q: Float_or_ArrayLike, axis: int | None = None
) -> float | np.ndarray:
    """
    Calculate discrete Kullback–Leibler divergence

    Parameteres
    -----------
    P, Q : array-like
        Probabilities to consider
    Returns
    -------
    result : float or ndarray
        value of KB divergence

    References
    ----------
    {kl_link}
    """

    P, Q = np.asarray(P), np.asarray(Q)
    out = diverg_kl_integrand(P, Q).sum(axis=axis)

    return cast(Union[float, np.ndarray], out)


def _check_volume_func(volume: str | Callable | None = None) -> Callable:
    if volume is None:
        volume = "1d"

    if isinstance(volume, str):
        if volume == "1d":
            volume = lambda x: 1.0
        elif volume == "2d":
            volume = lambda x: 2.0 * np.pi * x
        elif volume == "3d":
            volume = lambda x: 4 * np.pi * x**2
        else:
            raise ValueError("unknown dimension")
    else:
        assert callable(volume)

    return volume


@docfiller_shared
def diverg_kl_cont(
    p: Callable,
    q: Callable,
    segments: ArrayLike,
    segments_q: ArrayLike | None = None,
    volume: str | Callable | None = None,
    err: bool = False,
    full_output: bool = False,
    **kws,
):
    """
    Calculate continuous Kullback–Leibler divergence for continuous pdf

    Parameters
    ----------
    p, q: callable
        Probabilities to consider
    {volume_int_func}
    {segments}
    segments_q : list, optional
        if supplied, build total segments by combining segments and segments_q
    {err}
    {full_output}

    Returns
    -------
    result : float or ndarray
        value of KB divergence
    {error_summed}
    {full_output_summed}

    References
    ----------
    {kl_link}
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
        **kws,
    )


@doc_inherit(diverg_kl_disc, style="numpy_with_merge")
def diverg_js_disc(
    P: Float_or_ArrayLike, Q: Float_or_ArrayLike, axis: int | None = None
) -> float | np.ndarray:
    """
    Discrete Jensen–Shannon divergence

    Returns
    -------
    result : float or ndarray
        value of JS divergence

    References
    ----------
    {js_link}
    """

    P, Q = np.asarray(P), np.asarray(Q)

    M = 0.5 * (P + Q)

    out = 0.5 * (diverg_kl_disc(P, M, axis=axis) + diverg_kl_disc(Q, M, axis=axis))
    return cast(Union[float, np.ndarray], out)


def diverg_js_integrand(
    p: Float_or_ArrayLike,
    q: Float_or_ArrayLike,
    volume: Float_or_ArrayLike | None = None,
) -> np.ndarray:
    p = np.asarray(p)
    q = np.asarray(q)

    m = 0.5 * (p + q)

    out = 0.5 * (diverg_kl_integrand(p, m) + diverg_kl_integrand(q, m))

    if volume is not None:
        out *= np.asarray(volume)

    return cast(np.ndarray, out)


@doc_inherit(diverg_kl_cont, style="numpy_with_merge")
@docfiller_shared
def diverg_js_cont(
    p: Callable,
    q: Callable,
    segments: ArrayLike,
    segments_q: ArrayLike | None = None,
    volume: str | Callable | None = None,
    err: bool = False,
    full_output: bool = False,
    **kws,
):
    """
    Continuous Jensen–Shannon divergence


    Returns
    -------
    result : float or ndarray
        value of JS divergence

    References
    ----------
    {js_link}
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
        **kws,
    )


@docfiller_shared
class Measures:
    """
    Convenience class for calculating measures

    Parameters
    ----------
    {phi}
    {segments}
    {quad_kws}

    """

    def __init__(
        self,
        phi: Phi_Signature,
        segments: Sequence[float],
        quad_kws: Mapping | None = None,
    ) -> None:
        self.phi = phi
        self.segments = segments
        if quad_kws is None:
            quad_kws = {}
        self.quad_kws = quad_kws

    @gcached(prop=False)
    @add_quad_kws
    @docfiller_shared
    def secondvirial(self, beta, err=False, full_output=False, **kws):
        """
        Calculate second virial coefficient.

        Parameters
        ----------
        {beta}
        {err}
        {full_output}

        **kws
            Extra arguments to :func:`analphipy.utils.quad_segments`

        Returns
        -------
        B2 : float
            Value of second virial coefficient.
        {error_summed}
        {full_output_summed}

        See Also
        --------
        ~analphipy.measures.secondvirial
        """
        return secondvirial(
            phi=self.phi,
            beta=beta,
            segments=self.segments,
            err=err,
            full_output=full_output,
            **kws,
        )

    @gcached(prop=False)
    @add_quad_kws
    @docfiller_shared
    def secondvirial_dbeta(self, beta, err=False, full_output=False, **kws):
        """
        Calculate ``beta`` derivative of second virial coefficient.

        Parameters
        ----------
        {beta}
        {err}
        {full_output}

        Returns
        -------
        dB2dbeta : float
            Value of derivative.
        {error_summed}
        {full_output_summed}


        See Also
        --------
        ~analphipy.measures.secondvirial_dbeta
        """
        return secondvirial_dbeta(
            phi=self.phi,
            beta=beta,
            segments=self.segments,
            err=err,
            full_output=full_output,
            **kws,
        )

    @docfiller_shared
    @add_quad_kws
    def boltz_diverg_js(
        self,
        other,
        beta: float,
        beta_other: float | None = None,
        volume: str | Callable = "3d",
        err: bool = False,
        full_output: bool = False,
        **kws,
    ):
        r"""
        Jensen-Shannon divergence of the Boltzmann factors of two potentials.

        The Boltzmann factors are defined as:

        .. math::

            B(r; \beta, \phi) = \exp(-\beta \phi(r))


        Parameters
        ----------
        other : :class:`analphipy.base_potential.PhiBase`
            Class wrapping other potential to compare `self` to.
        {beta}
        beta_other : float, optional
            beta value to evaluate other Boltzmann factor at.
        {volume_int_func}

        See Also
        --------
        ~analphipy.measures.diverg_js_cont

        References
        --------
        `See here for more info <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Symmetrised_divergence>`
        """

        if beta_other is None:
            beta_other = beta

        p = lambda x: np.exp(-beta * self.phi(x))
        q = lambda x: np.exp(-beta_other * other.phi(x))

        return diverg_js_cont(
            p=p,
            q=q,
            segments=self.segments,
            segments_q=other.segments,
            volume=volume,
            err=err,
            full_output=full_output,
            **kws,
        )

    def mayer_diverg_js(
        self,
        other,
        beta: float,
        beta_other: float | None = None,
        volume: str | Callable = "3d",
        err: bool = False,
        full_output: bool = False,
        **kws,
    ):
        r"""
        Jensen-Shannon divergence of the Mayer f-functions of two potentials.

        The Mayer f-function is defined as:

        .. math::

            f(r; \beta, \phi) = \exp(-beta \phi(r)) - 1


        Parameters
        ----------
        other : :class:`analphipy.base_potential.PhiBase`
            Class wrapping other potential to compare `self` to.
        {beta}
        beta_other : float, optional
            beta value to evaluate other Boltzmann factor at.
        {volume_int_func}


        See Also
        --------
        ~analphipy.measures.diverg_js_cont


        References
        ----------
        `See here for more info <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Symmetrised_divergence>`
        """

        if beta_other is None:
            beta_other = beta

        p = lambda x: np.exp(-beta * self.phi(x)) - 1.0
        q = lambda x: np.exp(-beta_other * other.phi(x)) - 1.0

        return diverg_js_cont(
            p=p,
            q=q,
            segments=self.segments,
            segments_q=other.segments,
            volume=volume,
            err=err,
            full_output=full_output,
            **kws,
        )
