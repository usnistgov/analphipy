"""
Noro-Frenkel pair potential analysis (:mod:`analphipy.norofrenkel`)
===================================================================

A collection of routines to analyze pair potentials using Noro-Frenkel analysis.

References
----------
.. [1] {ref_Noro_Frenkel}

.. [2] {ref_Barker_Henderson}

.. [3] {ref_WCA}


"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from module_utilities import cached

from ._docstrings import docfiller
from .measures import secondvirial, secondvirial_dbeta, secondvirial_sw
from .utils import TWO_PI, add_quad_kws, is_float, minimize_phi, quad_segments

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from analphipy.base_potential import PhiAbstract

    from ._typing import (
        Array,
        ArrayLike,
        Float_or_Array,
        Float_or_ArrayLike,
        Phi_Signature,
        QuadSegments,
    )
    from ._typing_compat import Self

# Workaround to document module level docstring
__doc__ = __doc__.format(**docfiller.data)  # pyright: ignore[reportOptionalMemberAccess]  # ty: ignore[possibly-unbound-attribute]

__all__ = [
    "NoroFrenkelPair",
    "lam_nf",
    "lam_nf_dbeta",
    "sig_nf",
    "sig_nf_dbeta",
]

d = docfiller.update(
    summary="Noro-Frenkel/Barker-Henderson effective hard sphere diameter.",
    extended_summary=r"""
    This is calculated using the formula [1]_ [2]_

    .. math::

        \sigma_{{\rm BH}}(\beta) = \int_0^{{\infty}} dr \left( 1 - \exp[-\beta \phi_{{\rm rep}}(r)]\right)

    where :math:`\phi_{{\rm rep}}(r)` is the repulsive part of the potential [3]_.
    """,
).dedent()


@d.decorate
def sig_nf(
    phi_rep: Phi_Signature,
    beta: float,
    segments: ArrayLike,
    err: bool = False,
    full_output: bool = False,
    **kws: Any,
) -> QuadSegments:
    r"""
    {summary}

    {extended_summary}

    Parameters
    ----------
    {phi_rep}
    {beta}
    {segments}
    phi_rep : callable
        Repulsive part of pair potential.
    beta : float
        Inverse temperature.
    segments : array-like
        Integration segments.
    err : bool, default=False
        If True, return error value.
    full_output : bool, default=True
        If True, return full_output.

    Returns
    -------
    sig_nf : float or list of float
        Value of integral.
    errors : float or list of float, optional
        If `err` or `full_output` are True, then return sum of errors.
    outputs : object
        Output from :func:`scipy.integrate.quad`. If multiple segments, return a list of output.



    See Also
    --------
    ~analphipy.utils.quad_segments

    """

    def integrand(r: Float_or_Array) -> Array:
        out: Array = 1.0 - np.exp(-beta * phi_rep(r))
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


@docfiller.inherit(sig_nf)
def sig_nf_dbeta(
    phi_rep: Phi_Signature,
    beta: float,
    segments: ArrayLike,
    err: bool = False,
    full_output: bool = False,
    **kws: Any,
) -> QuadSegments:
    r"""
    Derivative with respect to inverse temperature ``beta`` of ``sig_nf``.

    See refs [1]_ [2]_ [3]_

    .. math::

        \frac{{d \sigma_{{\rm BH}}}}{{d\beta}} = \int_0^{{\infty}} dr \phi_{{\rm rep}}(r) \exp[-\beta \phi_{{\rm rep}}(r)]
    """

    def integrand(r: Float_or_Array) -> Array:
        v = phi_rep(r)
        out = np.array(0.0) if np.isinf(v) else v * np.exp(-beta * v)

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
def lam_nf(beta: float, sig: float, eps: float, B2: float) -> float:
    r"""
    Noro-Frenkel effective lambda parameter

    This is the value of :math:`\lambda` in a square well potential which matches second virial
    coefficients.  The square well fluid is defined as [1]_

    .. math::

        \phi_{{\rm sw}}(r) =
        \begin{{cases}}
            \infty & r \leq \sigma \\
            \epsilon &  \sigma < r \leq \lambda \sigma \\
            0 &  r > \lambda \sigma
        \end{{cases}}


    Parameters
    ----------
    {beta}
    sig : float
        Particle diameter.
    eps : float
        Energy parameter in square well potential. The convention is that ``eps`` is the same as the value of ``phi`` at the minimum.
    B2 : float
        Second virial coefficient to match.

    Returns
    -------
    lam_nf : float
        Value of `lambda` in an effective square well fluid which matches ``B2``.


    """
    B2star = B2 / (TWO_PI / 3.0 * sig**3)

    # B2s_SW = 1 + (1-exp(-beta epsilon)) * (lambda**3 - 1)
    #        = B2s
    lam: float = ((B2star - 1.0) / (1.0 - np.exp(-beta * eps)) + 1.0) ** (1.0 / 3.0)
    return lam


@docfiller.decorate
def lam_nf_dbeta(
    beta: float,
    sig: float,
    eps: float,
    lam: float,
    B2: float,
    B2_dbeta: float,
    sig_dbeta: float,
) -> float:
    """
    Calculate derivative of ``lam_nf``  with respect to ``beta``.

    Parameters
    ----------
    {beta}
    sig : float
        Particle diameter.
    eps : float
        Energy parameter in square well potential. The convention is that ``eps`` is the same as the value of ``phi`` at the minimum.
    B2 : float
        Second virial coefficient to match.
    lam : float
        Value from :func:`analphipy.norofrenkel.lam_nf`.
    B2_dbeta : float
        d(B2)/d(beta) at ``beta``
    sig_dbeta : float
        derivative of Noro-Frenkel sigma w.r.t inverse temperature at `beta`

    Returns
    -------
    lam_nf_dbeta : float
        Value of ``d(lam_nf)/d(beta)``

    See Also
    --------
    lam_nf

    """
    B2_hs = TWO_PI / 3.0 * sig**3

    dB2stardbeta = 1.0 / B2_hs * (B2_dbeta - B2 * 3.0 * sig_dbeta / sig)

    e = np.exp(beta * eps)

    out: float = 1.0 / (3 * lam**2 * (e - 1)) * (dB2stardbeta * e - eps * (lam**3 - 1))

    return out


@docfiller.decorate
class NoroFrenkelPair:
    """
    Class to calculate Noro-Frenkel parameters.

    See [1]_ [2]_ [3]_

    Parameters
    ----------
    {phi}
    {segments}
    {r_min_exact}
    {phi_min_exact}
    {quad_kws}

    """

    def __init__(
        self,
        phi: Phi_Signature,
        segments: ArrayLike,
        r_min: float,
        phi_min: Float_or_Array | None,
        quad_kws: Mapping[str, Any] | None = None,
    ) -> None:
        self.phi = phi
        self.r_min = r_min

        if phi_min is None:
            phi_min = phi(r_min)
        self.phi_min = phi_min
        self.segments = segments

        if quad_kws is None:
            quad_kws = {}
        self.quad_kws = quad_kws

        self._cache: dict[str, Any] = {}

    def __repr__(self) -> str:
        params = ",\n    ".join(
            [
                f"{v}={getattr(self, v)}"
                for v in ("phi", "segments", "r_min", "phi_min", "quad_kws")
            ]
        )

        return f"{type(self).__name__}({params})"

    @docfiller.decorate
    def phi_rep(self, r: Float_or_ArrayLike) -> Array:
        """
        Repulsive part of potential.

        This is the Weeks-Chandler-Anderson decomposition.

        Parameters
        ----------
        {r}

        Returns
        -------
        output : float or ndarray
            Value of ``phi_ref`` at separation(s) ``r``.

        """
        r = np.array(r)
        phi = np.empty_like(r)
        m = r <= self.r_min

        phi[m] = self.phi(r[m]) - self.phi_min
        phi[~m] = 0.0

        return phi

    @classmethod
    @docfiller.decorate
    def from_phi(
        cls,
        phi: Phi_Signature,
        segments: ArrayLike,
        r_min: float | None = None,
        bounds: Sequence[float] | None = None,
        quad_kws: Mapping[str, Any] | None = None,
        **kws: Any,
    ) -> Self:
        """
        Create object from pair potential function.

        Parameters
        ----------
        {phi}
        {segments}
        r_min : float, optional
            Optional guess for numerically finding minimum in `phi`.
        bounds : array-like, optional
            Optional bounds for numerically locating ``r_min``.
        quad_kws : mapping, optional
            Optional arguments to :func:`analphipy.utils.quad_segments`.
        **kws :
            Extra arguments to :func:`analphipy.utils.minimize_phi`.

        Returns
        -------
        output : object
            instance of calling class

        """
        if bounds is None:
            bounds = (segments[0], segments[-1])

        if bounds[-1] == np.inf and r_min is None:
            msg = "if specify infinite bounds, must supply guess"
            raise ValueError(msg)

        if r_min is None:
            r_min = cast("float", np.mean(bounds))

        r_min, phi_min, _ = minimize_phi(phi, r0=r_min, bounds=bounds, **kws)
        return cls(
            phi=phi, r_min=r_min, phi_min=phi_min, segments=segments, quad_kws=quad_kws
        )

    @classmethod
    def from_phi_class(
        cls,
        phi: PhiAbstract,
        r_min: float | None = None,
        bounds: Sequence[float] | None = None,
        quad_kws: dict[str, Any] | None = None,
        **kws: Any,
    ) -> Self:
        """
        Create object, trying to use pre computed values for ``r_min``, ``phi_min``.

        Parameters
        ----------
        phi : :class:`analphipy.base_potential.PhiAbstract`
        r_min : float, optional
            Optional guess for numerically finding minimum in `phi`.
        bounds : array-like, optional
            Optional bounds for numerically locating ``r_min``.
        quad_kws : mapping, optional
            Optional arguments to :func:`analphipy.utils.quad_segments`.
        **kws :
            Extra arguments to :func:`analphipy.utils.minimize_phi`.

        Returns
        -------
        output : object
            instance of calling class

        """
        if phi.segments is None:  # pyright: ignore[reportUnnecessaryComparison]
            msg = "need phi.segments to be set"  # type: ignore[unreachable]
            raise ValueError(msg)

        if phi.r_min is not None and phi.phi_min is not None:
            return cls(
                phi=phi.phi,
                segments=phi.segments,
                r_min=phi.r_min,
                phi_min=phi.phi_min,
                quad_kws=quad_kws,
            )

        return cls.from_phi(
            phi=phi.phi,
            segments=phi.segments,
            r_min=r_min,
            bounds=bounds,
            quad_kws=quad_kws,
            **kws,
        )

    @cached.meth
    @add_quad_kws
    def secondvirial(self, /, beta: float, **kws: Any) -> QuadSegments:
        """
        Second virial coefficient.

        See Also
        --------
        ~analphipy.measures.secondvirial

        """
        return secondvirial(phi=self.phi, beta=beta, segments=self.segments, **kws)

    @cached.prop
    def _segments_rep(self) -> list[float]:
        return [float(x) for x in self.segments if x < self.r_min] + [self.r_min]

    @cached.meth
    @add_quad_kws
    def sig(self, /, beta: float, **kws: Any) -> QuadSegments:
        """
        Effective hard sphere diameter.

        See Also
        --------
        ~analphipy.norofrenkel.sig_nf

        """
        return sig_nf(
            self.phi_rep,
            beta=beta,
            segments=self._segments_rep,
            **kws,
        )

    def eps(self, beta: float | None = None, **kws: Any) -> float:  # noqa: ARG002
        """
        Effective square well epsilon.

        This is equal to the minimum in ``phi``.
        """
        return cast("float", self.phi_min)

    @cached.meth
    @add_quad_kws
    def lam(self, /, beta: float, **kws: Any) -> float:
        """
        Effective square well lambda.

        See Also
        --------
        ~analphipy.norofrenkel.lam_nf

        """
        sig = self.sig(beta, **kws)
        B2 = self.secondvirial(beta, **kws)

        if is_float(sig) and is_float(B2):
            return lam_nf(
                beta=beta,
                sig=sig,
                eps=self.eps(beta, **kws),
                B2=B2,
            )

        msg = f"Bad kws={kws}"
        raise ValueError(msg)

    @cached.meth
    @add_quad_kws
    def sw_dict(self, /, beta: float, **kws: Any) -> dict[str, float]:
        """Dictionary view of Noro-Frenkel parameters."""
        sig = self.sig(beta, **kws)
        eps = self.eps(beta, **kws)
        B2 = self.secondvirial(beta, **kws)

        if is_float(sig) and is_float(B2):
            lam = lam_nf(beta=beta, sig=sig, eps=eps, B2=B2)
            return {"sig": sig, "eps": eps, "lam": lam}

        msg = f"Bad kws={kws}"
        raise ValueError(msg)

    @cached.meth
    @add_quad_kws
    def secondvirial_dbeta(self, /, beta: float, **kws: Any) -> QuadSegments:
        """
        Derivative of ``secondvirial`` with respect to ``beta``.

        See Also
        --------
        ~analphipy.measures.secondvirial_dbeta

        """
        return secondvirial_dbeta(
            phi=self.phi, beta=beta, segments=self.segments, **kws
        )

    @cached.meth
    @add_quad_kws
    def sig_dbeta(self, /, beta: float, **kws: Any) -> QuadSegments:
        """
        Derivative of effective hard-sphere diameter with respect to ``beta``.

        See Also
        --------
        ~analphipy.norofrenkel.sig_nf_dbeta

        """
        return sig_nf_dbeta(self.phi_rep, beta=beta, segments=self.segments, **kws)

    @cached.meth
    def lam_dbeta(self, /, beta: float, **kws: Any) -> float:
        """
        Derivative of effective lambda parameter with respect to ``beta``.

        See Also
        --------
        ~analphipy.norofrenkel.lam_nf_dbeta

        """
        sig = self.sig(beta, **kws)
        eps = self.eps(beta, **kws)
        lam = self.lam(beta, **kws)
        B2 = self.secondvirial(beta, **kws)
        B2_dbeta = self.secondvirial_dbeta(beta, **kws)
        sig_dbeta = self.sig_dbeta(beta, **kws)

        if (
            is_float(sig)
            and is_float(lam)
            and is_float(B2)
            and is_float(B2_dbeta)
            and is_float(sig_dbeta)
        ):
            return lam_nf_dbeta(
                beta=beta,
                sig=sig,
                eps=eps,
                lam=lam,
                B2=B2,
                B2_dbeta=B2_dbeta,
                sig_dbeta=sig_dbeta,
            )

        msg = f"Bad kws={kws}"
        raise ValueError(msg)  # pragma: no cover

    @cached.meth
    def secondvirial_sw(self, /, beta: float, **kws: Any) -> float:
        """
        Second virial coefficient of effective square well fluid.

        For testing.  This should be the same of value from :meth:`secondvirial`
        """
        sig = self.sig(beta, **kws)
        eps = self.eps(beta, **kws)
        lam = self.lam(beta, **kws)

        if is_float(sig) and is_float(lam):
            return secondvirial_sw(
                beta=beta,
                sig=sig,
                eps=eps,
                lam=lam,
            )

        msg = f"Bad kws={kws}"
        raise ValueError(msg)  # pragma: no cover

    def B2(self, beta: float, **kws: Any) -> QuadSegments:
        """Alias to :meth:`secondvirial`."""
        return self.secondvirial(beta, **kws)

    def B2_dbeta(self, beta: float, **kws: Any) -> QuadSegments:
        """Alias to :meth:`secondvirial_dbeta`."""
        return self.secondvirial_dbeta(beta, **kws)

    def B2_sw(self, beta: float, **kws: Any) -> QuadSegments:
        """Alias to :meth:`secondvirial_sw`."""
        return self.secondvirial_sw(beta, **kws)

    def table(
        self,
        betas: ArrayLike,
        props: Sequence[str] | None = None,
        key_format: str = "{prop}",
        **kws: Any,
    ) -> dict[str, Any]:
        """
        Create a dictionary of outputs for multiple values of inverse temperature ``beta``.

        Parameters
        ----------
        betas : array-like
            Array of values of inverse temperature ``beta``.
        props : sequence of string
            Name of methods to access.
        key_format : string, default="{prop}"
            Each key in the output dictionary will have the value ``key_format.format(prop=prop)``

        Returns
        -------
        output : dict
            dictionary of arrays.

        """
        if props is None:
            props = ["B2", "sig", "eps", "lam"]

        table = {"beta": betas}

        for prop in props:
            f = getattr(self, prop)
            key = key_format.format(prop=prop)
            table[key] = [f(beta=beta, **kws) for beta in betas]
        return table
