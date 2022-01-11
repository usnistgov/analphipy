from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Mapping, Optional, Sequence, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .cached_decorators import gcached
from .measures import secondvirial, secondvirial_dbeta, secondvirial_sw
from .utils import TWO_PI, Seq_float, minimize_phi, quad_segments


def add_quad_kws(func: Callable) -> Callable:
    @wraps(func)
    def wrapped(self, *args, **kws):
        kws = dict(self.quad_kws, **kws)
        return func(self, *args, **kws)

    return wrapped


def sig_nf(
    phi_rep: Callable,
    beta: float,
    segments: Seq_float,
    err: bool = False,
    full_output: bool = False,
    **kws
):
    """Noro-Frenkel/Barker-Henderson effective hard sphere diameter"""

    def integrand(r):
        return 1.0 - np.exp(-beta * phi_rep(r))

    return quad_segments(
        integrand,
        segments=segments,
        sum_integrals=True,
        sum_errors=True,
        err=err,
        full_output=full_output,
        **kws
    )


def sig_nf_dbeta(
    phi_rep: Callable,
    beta: float,
    segments: Seq_float,
    err: bool = False,
    full_output: bool = False,
    **kws
):
    """derivative w.r.t. beta of sig_nf"""

    def integrand(r):
        v = phi_rep(r)
        if np.isinf(v):
            return 0.0
        else:
            return v * np.exp(-beta * v)

    return quad_segments(
        integrand,
        segments=segments,
        sum_integrals=True,
        sum_errors=True,
        err=err,
        full_output=full_output,
        **kws
    )


def lam_nf(beta: float, sig: float, eps: float, B2: float):
    B2star = B2 / (TWO_PI / 3.0 * sig ** 3)

    # B2s_SW = 1 + (1-exp(-beta epsilon)) * (lambda**3 - 1)
    #        = B2s
    lam = ((B2star - 1.0) / (1.0 - np.exp(-beta * eps)) + 1.0) ** (1.0 / 3.0)
    return lam


def lam_nf_dbeta(
    beta: float,
    sig: float,
    eps: float,
    lam: float,
    B2: float,
    B2_dbeta: float,
    sig_dbeta: float,
):
    """
    calculate d(lam_nf)/d(beta) from known parameters
    Parameters
    ----------
    beta : float
        inverse temperature
    sig, eps, lam : float
        Noro-frenkel sigma/eps/lambda parameters
    B2 : float
        Actual second virial coef
    B2_dbeta : float
        d(B2)/d(beta) at `beta`
    sig_dbeta : float
        derivative of noro-frenkel sigma w.r.t inverse temperature at `beta`
    """

    B2_hs = TWO_PI / 3.0 * sig ** 3

    dB2stardbeta = 1.0 / B2_hs * (B2_dbeta - B2 * 3.0 * sig_dbeta / sig)

    # e = np.exp(-beta * eps)
    # f = 1. - e

    e = np.exp(beta * eps)

    out = 1.0 / (3 * lam ** 2 * (e - 1)) * (dB2stardbeta * e - eps * (lam ** 3 - 1))

    return out


class NoroFrenkelPair:
    def __init__(
        self,
        phi: Callable,
        segments: Seq_float,
        x_min: float,
        phi_min: Union[float, NDArray[np.float_]],
        quad_kws: Optional[Mapping[str, Any]] = None,
    ):

        self.phi = phi
        self.x_min = x_min

        if phi_min is None:
            phi_min = phi(x_min)
        self.phi_min = phi_min
        self.segments = segments

        if quad_kws is None:
            quad_kws = {}
        self.quad_kws = quad_kws

    def phi_rep(self, r: ArrayLike) -> np.ndarray:
        r = np.array(r)
        phi = np.empty_like(r)
        m = r <= self.x_min

        phi[m] = self.phi(r[m]) - self.phi_min
        phi[~m] = 0.0

        return phi

    @classmethod
    def from_phi(
        cls,
        phi: Callable,
        segments: Seq_float,
        x_min: Optional[float] = None,
        bounds: Optional[Seq_float] = None,
        quad_kws: Optional[Mapping[str, Any]] = None,
        **kws
    ) -> NoroFrenkelPair:

        if bounds is None:
            bounds = (segments[0], segments[-1])

        if bounds[-1] == np.inf and x_min is None:
            raise ValueError("if specify infinite bounds, must supply guess")

        if x_min is None:
            x_min = np.mean(bounds)

        x_min, phi_min, _ = minimize_phi(phi, r0=x_min, bounds=bounds, **kws)
        return cls(
            phi=phi, x_min=x_min, phi_min=phi_min, segments=segments, quad_kws=quad_kws
        )

    @gcached(prop=False)
    @add_quad_kws
    def secondvirial(self, beta: float, **kws):
        return secondvirial(phi=self.phi, beta=beta, segments=self.segments, **kws)

    @gcached()
    def _segments_rep(self) -> list:
        return [x for x in self.segments if x < self.x_min] + [self.x_min]

    @gcached(prop=False)
    @add_quad_kws
    def sig(self, beta: float, **kws):
        return sig_nf(self.phi_rep, beta=beta, segments=self._segments_rep, **kws)

    def eps(self, beta: float, **kws) -> float:
        return cast(float, self.phi_min)

    @gcached(prop=False)
    @add_quad_kws
    def lam(self, beta: float, **kws):
        return lam_nf(
            beta=beta,
            sig=self.sig(beta, **kws),
            eps=self.eps(beta, **kws),
            B2=self.secondvirial(beta, **kws),
        )

    @gcached(prop=False)
    @add_quad_kws
    def sw_dict(self, beta: float, **kws) -> dict[str, float]:
        sig = self.sig(beta, **kws)
        eps = self.eps(beta, **kws)
        lam = lam_nf(beta=beta, sig=sig, eps=eps, B2=self.secondvirial(beta, **kws))
        return {"sig": sig, "eps": eps, "lam": lam}

    @gcached(prop=False)
    @add_quad_kws
    def secondvirial_dbeta(self, beta: float, **kws):
        return secondvirial_dbeta(
            phi=self.phi, beta=beta, segments=self.segments, **kws
        )

    @gcached(prop=False)
    @add_quad_kws
    def sig_dbeta(self, beta: float, **kws):
        return sig_nf_dbeta(self.phi_rep, beta=beta, segments=self.segments, **kws)

    @gcached(prop=False)
    def lam_dbeta(self, beta: float, **kws):
        return lam_nf_dbeta(
            beta=beta,
            sig=self.sig(beta, **kws),
            eps=self.eps(beta, **kws),
            lam=self.lam(beta, **kws),
            B2=self.secondvirial(beta, **kws),
            B2_dbeta=self.secondvirial_dbeta(beta, **kws),
            sig_dbeta=self.sig_dbeta(beta, **kws),
        )

    @gcached(prop=False)
    def secondvirial_sw(self, beta: float, **kws):
        return secondvirial_sw(
            beta=beta,
            sig=self.sig(beta, **kws),
            eps=self.eps(beta, **kws),
            lam=self.lam(beta, **kws),
        )

    def B2(self, beta: float, **kws):
        return self.secondvirial(beta, **kws)

    def B2_dbeta(self, beta: float, **kws):
        return self.secondvirial_dbeta(beta, **kws)

    def B2_sw(self, beta, **kws):
        return self.secondvirial_sw(beta, **kws)

    def table(
        self,
        betas: Seq_float,
        props: Sequence[str] = None,
        key_format: str = "{prop}",
        **kws
    ) -> dict[str, Any]:
        if props is None:
            props = ["B2", "sig", "eps", "lam"]

        table = {"beta": betas}

        for prop in props:
            f = getattr(self, prop)
            key = key_format.format(prop=prop)
            table[key] = [f(beta=beta, **kws) for beta in betas]

        return table
