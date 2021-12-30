import numpy as np

from .utils import TWO_PI, combine_segmets, quad_segments


def secondvirial(phi, beta, segments, err=False, full_output=False, **kws):
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


def secondvirial_dbeta(phi, beta, segments, err=False, full_output=False, **kws):
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


def secondvirial_sw(beta, sig, eps, lam):
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


def kl_diverg_disc(P, Q, axis=None):
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
    result = np.sum(P * np.log(P / Q), axis=axis)

    return result


def kl_diverg_cont(p, q, segments, volume=None, err=False, full_output=False, **kws):
    """
    calculate discrete Kullback–Leibler divergence for contiuous pdf

    Parameteres
    -----------
    p, q: callable
        Probabilities to consider
    volume : callable, optional
        volume element if not linear integration region
        For examle, use volume = lambda x: 4 * np.pi * x ** 2 for spherically symmetric p/q
    Returns
    -------
    result : float or array
        value of KB divergence

    See Also
    --------
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Symmetrised_divergence
    """

    if volume is None:
        volume = lambda x: 1.0

    def func(x):
        pval, qval, f = p(x), q(x), volume(x)

        if pval == 0.0:
            return 0.0
        else:
            return f * pval * np.log(pval / qval)

    return quad_segments(
        func,
        segments=segments,
        sum_integrals=True,
        sum_errors=True,
        err=err,
        full_output=full_output,
        **kws
    )


def js_diverg_disc(P, Q, axis=None):
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

    return 0.5 * (kl_diverg_disc(P, M) + kl_diverg_disc(Q, M))


def js_diverg_cont(p, q, segments, volume=None, err=False, full_output=False, **kws):
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

    if volume is None:
        volume = lambda x: 1.0

    def plogp(p):
        if p == 0.0:
            return 0.0
        else:
            return p * np.log(p)

    def func(x):
        pval, qval, f = p(x), q(x), volume(x)

        mval = 0.5 * (pval + qval)
        # 2out/f = p log(p/m) + q log(q/m)
        #         = p log(p) - p log(m) + q log(q) - q log(m)
        #  out/f  = 0.5 * (p log(p) + q log(q)) - m log(m)

        return f * (0.5 * (plogp(pval) + plogp(qval)) - plogp(mval))

    return quad_segments(
        func,
        segments=segments,
        sum_integrals=True,
        sum_errors=True,
        err=err,
        full_output=full_output,
        **kws
    )


class NormalizedFunc(object):
    def __init__(self, func, segments, volume=None, norm_fac=None, **kws):

        self.func = lambda x: func(x) * volume(x)
        self.segments = segments
        self.kws = kws

        if volume is None:
            volume = lambda x: 1.0
        self.volume = volume

        if norm_fac is None:
            self.set_norm_fac()
        self.norm_fac = norm_fac

    def __call__(self, x):
        return self.norm_fac * self.func(x)

    def integrand(self, x):
        return self.volume(x) * self(x)

    def set_norm_fac(self, **kws):
        self.norm_fac = 1.0

        kws = dict(self.kws, **kws)

        value = quad_segments(
            self.integrand,
            segments=self.segments,
            sum_integrals=True,
            sum_errors=True,
            err=False,
            full_output=False,
            **kws
        )
        self.norm_fac = 1.0 / value

    def __add__(self, other):
        # combine stuff

        return type(self)(
            func=lambda x: self(x) + other(x),
            segments=combine_segmets(self.segments, other.segments),
            volume=self.volume,
            norm_fac=1.0,
            kws=dict(other.kws, **self.kws),
        )

    def __mul__(self, fac):
        return type(self)(
            func=self.func,
            segments=self.segments,
            volume=self.volume,
            norm_fac=self.norm_fac * fac,
            kws=dict(self.kws),
        )
