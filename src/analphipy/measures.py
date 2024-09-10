"""
Routines to calculate measures of pair potentials (:mod:`analphipy.measures`)
=============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from module_utilities import cached

from ._docstrings import docfiller

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any, Callable

    from ._typing import (
        Array,
        ArrayLike,
        Float_or_Array,
        Float_or_ArrayLike,
        Phi_Signature,
        QuadSegments,
    )
    from .base_potential import PhiAbstract


from .utils import TWO_PI, add_quad_kws, combine_segmets, quad_segments

__all__ = [
    "Measures",
    "diverg_js_cont",
    "diverg_kl_cont",
    "secondvirial",
    "secondvirial_dbeta",
    "secondvirial_sw",
]


@docfiller.decorate
def secondvirial(
    phi: Phi_Signature,
    beta: float,
    segments: ArrayLike,
    err: bool = False,
    full_output: bool = False,
    **kws: Any,
) -> QuadSegments:
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

    def integrand(r: Float_or_Array) -> Array:
        out: Array = TWO_PI * r**2 * (1 - np.exp(-beta * phi(r)))
        return out

    return quad_segments(
        integrand,
        segments=segments,
        sum_integrals=True,
        sum_errors=True,
        err=err,
        full_output=full_output,
        **kws,
    )


@docfiller.decorate
def secondvirial_dbeta(
    phi: Phi_Signature,
    beta: float,
    segments: ArrayLike,
    err: bool = False,
    full_output: bool = False,
    **kws: Any,
) -> QuadSegments:
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

    def integrand(r: Float_or_Array) -> Array:
        v = phi(r)
        out = np.array(0.0) if np.isinf(v) else TWO_PI * r**2 * v * np.exp(-beta * v)
        return cast("Array", out)

    return quad_segments(
        integrand,
        segments=segments,
        sum_integrals=True,
        sum_errors=True,
        err=err,
        full_output=full_output,
        **kws,
    )


@docfiller.decorate
def secondvirial_sw(beta: float, sig: float, eps: float, lam: float) -> float:
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
    out: float = (
        TWO_PI / 3.0 * sig**3 * (1.0 + (1 - np.exp(-beta * eps)) * (lam**3 - 1.0))
    )
    return out


def diverg_kl_integrand(
    p: Float_or_ArrayLike,
    q: Float_or_ArrayLike,
    volume: Float_or_ArrayLike | None = None,
) -> Array:
    p, q = np.asarray(p), np.asarray(q)

    out = np.empty_like(p)

    zero = p == 0.0
    hero = ~zero

    out[zero] = 0.0
    out[hero] = p[hero] * np.log(p[hero] / q[hero])

    if volume is not None:
        out *= np.asarray(volume)

    return out


@docfiller(summary="Calculate discrete Kullback-Leibler divergence")
def diverg_kl_disc(
    p: Float_or_ArrayLike, q: Float_or_ArrayLike, axis: int | None = None
) -> Float_or_Array:
    """
    {summary}.

    Parameters
    ----------
    p, q : array-like
        Probabilities to consider

    Returns
    -------
    results : float or ndarray
        Value of divergence.

    References
    ----------
    {kl_link}

    """
    p, q = np.asarray(p), np.asarray(q)
    diverg = diverg_kl_integrand(p, q)
    # fmt: off
    out: Float_or_Array = diverg.sum(axis=axis)  # pyright: ignore[reportUnknownMemberType]
    # fmt: on
    return out


def _check_volume_func(
    volume: str | Callable[[Float_or_Array], Float_or_Array] | None = None,
) -> Callable[[Float_or_Array], Float_or_Array]:
    if volume is None:
        volume = "1d"
    if isinstance(volume, str):
        if volume == "1d":
            volume = lambda x: 1.0  # noqa: ARG005
        elif volume == "2d":
            volume = lambda x: 2.0 * np.pi * x
        elif volume == "3d":
            volume = lambda x: 4 * np.pi * x**2
        else:
            msg = "unknown dimension"
            raise ValueError(msg)
    elif not callable(volume):
        msg = "volume should be str or callable"
        raise TypeError(msg)

    return volume


@docfiller(
    summary="Calculate continuous Kullback-Leibler divergence for continuous pdf"
)
def diverg_kl_cont(
    p: Callable[[Float_or_Array], Float_or_Array],
    q: Callable[[Float_or_Array], Float_or_Array],
    segments: ArrayLike,
    segments_q: ArrayLike | None = None,
    volume: str | Callable[[Float_or_Array], Float_or_Array] | None = None,
    err: bool = False,
    full_output: bool = False,
    **kws: Any,
) -> QuadSegments:
    """
    {summary}.

    Parameters
    ----------
    p, q : callable
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
        value of divergence
    {error_summed}
    {full_output_summed}

    References
    ----------
    {kl_link}

    """
    volume_callable = _check_volume_func(volume)

    if segments_q is not None:
        segments = combine_segmets(segments, segments_q)

    def func(x: Float_or_Array) -> Array:
        return diverg_kl_integrand(p=p(x), q=q(x), volume=volume_callable(x))

    return quad_segments(
        func,
        segments=segments,
        sum_integrals=True,
        sum_errors=True,
        err=err,
        full_output=full_output,
        **kws,
    )


# @doc_inherit(diverg_kl_disc, style="numpy_with_merge")
@docfiller(diverg_kl_disc, summary="Discrete Jensen-Shannon divergence")
def diverg_js_disc(
    p: Float_or_ArrayLike, q: Float_or_ArrayLike, axis: int | None = None
) -> Float_or_Array:
    p, q = np.asarray(p), np.asarray(q)

    m = 0.5 * (p + q)

    return 0.5 * (diverg_kl_disc(p, m, axis=axis) + diverg_kl_disc(q, m, axis=axis))


def diverg_js_integrand(
    p: Float_or_ArrayLike,
    q: Float_or_ArrayLike,
    volume: Float_or_ArrayLike | None = None,
) -> Array:
    p = np.asarray(p)
    q = np.asarray(q)

    m = 0.5 * (p + q)

    out = 0.5 * (diverg_kl_integrand(p, m) + diverg_kl_integrand(q, m))

    if volume is not None:
        out *= np.asarray(volume)

    return out


@docfiller(diverg_kl_cont, summary="Continuous Jensen-Shannon divergence")
def diverg_js_cont(
    p: Callable[[Float_or_Array], Float_or_Array],
    q: Callable[[Float_or_Array], Float_or_Array],
    segments: ArrayLike,
    segments_q: ArrayLike | None = None,
    volume: str | Callable[[Float_or_Array], Float_or_Array] | None = None,
    err: bool = False,
    full_output: bool = False,
    **kws: Any,
) -> QuadSegments:
    """Calculate Jensen-Shannon divergence for continuous functions."""
    volume_callable = _check_volume_func(volume)

    if segments_q is not None:
        segments = combine_segmets(segments, segments_q)

    def func(x: Float_or_Array) -> Array:
        return diverg_js_integrand(p(x), q(x), volume_callable(x))

    return quad_segments(
        func,
        segments=segments,
        sum_integrals=True,
        sum_errors=True,
        err=err,
        full_output=full_output,
        **kws,
    )


@docfiller.decorate
class Measures:
    """
    Convenience class for calculating measures.

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
        quad_kws: Mapping[str, Any] | None = None,
    ) -> None:
        self.phi = phi
        self.segments = segments
        if quad_kws is None:
            quad_kws = {}
        self.quad_kws = quad_kws
        self._cache: dict[str, Any] = {}

    @cached.meth
    @add_quad_kws
    @docfiller.decorate
    def secondvirial(
        self, /, beta: float, err: bool = False, full_output: bool = False, **kws: Any
    ) -> QuadSegments:
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

    @cached.meth
    @add_quad_kws
    @docfiller.decorate
    def secondvirial_dbeta(
        self, /, beta: float, err: bool = False, full_output: bool = False, **kws: Any
    ) -> QuadSegments:
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

    @docfiller.decorate
    @add_quad_kws
    def boltz_diverg_js(
        self,
        /,
        other: PhiAbstract,
        beta: float,
        beta_other: float | None = None,
        volume: str | Callable[[Float_or_Array], Float_or_Array] = "3d",
        err: bool = False,
        full_output: bool = False,
        **kws: Any,
    ) -> QuadSegments:
        r"""
        Jensen-Shannon divergence of the Boltzmann factors of two potentials.

        The Boltzmann factors are defined as:

        .. math::

            B(r; \beta, \phi) = \exp(-\beta \phi(r))


        Parameters
        ----------
        other : :class:`analphipy.base_potential.PhiAbstract`
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

        def p_func(x: Float_or_Array) -> Array:
            return cast("Array", np.exp(-beta * self.phi(x)))

        def q_func(x: Float_or_Array) -> Array:
            return cast("Array", np.exp(-beta_other * other.phi(x)))

        return diverg_js_cont(
            p=p_func,
            q=q_func,
            segments=self.segments,
            segments_q=other.segments,
            volume=volume,
            err=err,
            full_output=full_output,
            **kws,
        )

    def mayer_diverg_js(
        self,
        other: PhiAbstract,
        beta: float,
        beta_other: float | None = None,
        volume: str | Callable[[Float_or_Array], Float_or_Array] = "3d",
        err: bool = False,
        full_output: bool = False,
        **kws: Any,
    ) -> QuadSegments:
        r"""
        Jensen-Shannon divergence of the Mayer f-functions of two potentials.

        The Mayer f-function is defined as:

        .. math::

            f(r; \beta, \phi) = \exp(-beta \phi(r)) - 1


        Parameters
        ----------
        other : :class:`analphipy.base_potential.PhiAbstract`
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

        def p_func(x: Float_or_Array) -> Array:
            out: Array = np.exp(-beta * self.phi(x)) - 1.0
            return out

        def q_func(x: Float_or_Array) -> Array:
            out: Array = np.exp(-beta_other * other.phi(x)) - 1.0
            return out

        return diverg_js_cont(
            p=p_func,
            q=q_func,
            segments=self.segments,
            segments_q=other.segments,
            volume=volume,
            err=err,
            full_output=full_output,
            **kws,
        )
