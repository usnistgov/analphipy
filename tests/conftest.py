# mypy: disable-error-code="no-untyped-def, no-untyped-call"
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Any
from typing_extensions import Self

import numpy as np
import pytest


def phidphi_lj(r, sig=1.0, eps=1.0):
    """
    Lennard jones potential and radial force.

    Returns
    -------
    phi : array
        potential at `r`
    dphi : array
        value of -d(phi)/d(r) * 1/r at `r`
    """

    r = np.array(r)

    rinvsq = 1.0 / (r * r)

    x2 = (sig * sig) * rinvsq
    x6 = x2 * x2 * x2

    phi = 4.0 * eps * x6 * (x6 - 1.0)
    dphi = 48.0 * eps * x6 * (x6 - 0.5) * rinvsq

    return phi, dphi


def phidphi_lj_cut(r, rcut, sig=1.0, eps=1.0):
    r = np.array(r)
    phi = np.empty_like(r)
    dphi = np.empty_like(r)

    m = r <= rcut
    phi[~m] = 0.0
    dphi[~m] = 0.0

    if np.any(m):
        v, dv = phidphi_lj(r[m], sig=sig, eps=eps)
        vcut, dvcut = phidphi_lj(rcut, sig=sig, eps=eps)

        phi[m] = v - vcut
        dphi[m] = dv

    return phi, dphi


def phidphi_lj_lfs(r, rcut, sig=1.0, eps=1):
    r = np.array(r)
    phi = np.empty_like(r)
    dphi = np.empty_like(r)

    m = r <= rcut
    phi[~m] = 0.0
    dphi[~m] = 0.0

    if np.any(m):
        v, dv = phidphi_lj(r[m], sig=sig, eps=eps)
        vcut, dvcut = phidphi_lj(rcut, sig=sig, eps=eps)

        phi[m] = v - vcut + dvcut * rcut * (r[m] - rcut)
        dphi[m] = dv - dvcut * rcut / r[m]
    return phi, dphi


def _prefac_nm(n, m, eps):
    return eps * (n / (n - m)) * (n / m) ** (m / (n - m))


def phidphi_nm(r, n, m, sig=1.0, eps=1.0):
    r = np.array(r)
    prefac = _prefac_nm(n=n, m=m, eps=eps)

    x = sig / r

    xn = x**n
    xm = x**m
    phi = prefac * (xn - xm)

    # dphi = -1/r dphi/dr = x dphi/dx * 1/r**2
    # where x = sig / r
    dphi = prefac * (n * xn - m * xm) / r**2

    return phi, dphi


def get_r(rmin, rmax, n=100):
    rmin = np.random.rand() * 0.1 + rmin
    rmax = np.random.rand() * 0.1 + rmax

    return np.linspace(rmin, rmax, n)


def kws_to_ld(**kws: Iterable[Any]) -> list[dict[str, Any]]:
    return [dict(zip(kws, t)) for t in zip(*kws.values())]


@dataclass
class Base_params:
    def __post_init__(self):
        self.r = None

    def phidphi(self, r):
        raise NotImplementedError

    def get_phidphi(self):
        if hasattr(self, "phidphi"):
            return self.phidphi(self.r)
        else:
            raise NotImplementedError

    def phi(self, r):
        return self.phidphi(r)[0]

    def get_phi(self):
        return self.phi(self.r)

    @classmethod
    def get_params(cls, N):
        raise NotImplementedError

    @classmethod
    def get_objects(cls, N=10) -> Iterable[Self]:
        for params in cls.get_params(N=N):
            yield cls(**params)

    def asdict(self):
        return asdict(self)


@dataclass
class LJ_params(Base_params):
    sig: float
    eps: float

    def __post_init__(self):
        self.r = get_r(rmin=0.1 * self.sig, rmax=5.0 * self.sig, n=100)

    def phidphi(self, r):
        return phidphi_lj(r, sig=self.sig, eps=self.eps)

    @classmethod
    def get_params(cls, N=10):
        return kws_to_ld(
            sig=np.random.rand(N) * 4,
            eps=np.random.rand(N),
        )


@dataclass
class LJ_cut_params(LJ_params):
    rcut: float

    def __post_init__(self):
        self.r = get_r(rmin=0.1 * self.sig, rmax=self.rcut + self.sig, n=100)

    def phidphi(self, r):
        return phidphi_lj_cut(r, sig=self.sig, eps=self.eps, rcut=self.rcut)

    @classmethod
    def get_params(cls, N=10):
        return kws_to_ld(
            sig=np.random.rand(N),
            eps=np.random.rand(N),
            rcut=np.random.rand(N) + 3.0,
        )


@dataclass
class LJ_lfs_params(LJ_cut_params):
    def phidphi(self, r):
        return phidphi_lj_lfs(r, sig=self.sig, eps=self.eps, rcut=self.rcut)


@pytest.fixture(params=LJ_params.get_objects(), scope="module")
def lj_params(request):
    return request.param


@pytest.fixture(params=LJ_cut_params.get_objects(), scope="module")
def lj_cut_params(request):
    return request.param


@pytest.fixture(params=LJ_lfs_params.get_objects(), scope="module")
def lj_lfs_params(request):
    return request.param


@dataclass
class NM_params(LJ_params):
    n: int
    m: int

    def phidphi(self, r):
        return phidphi_nm(r, n=self.n, m=self.m, sig=self.sig, eps=self.eps)

    @classmethod
    def get_params(cls, N=None):
        n = [8, 10, 12, 18]
        m = [6, 7, 6, 9]

        if N is None:
            N = len(n)

        return kws_to_ld(sig=np.random.rand(N), eps=np.random.rand(N), n=n, m=m)


@pytest.fixture(params=NM_params.get_objects(), scope="module")
def nm_params(request):
    return request.param


def phi_sw(r, sig=1.0, eps=-1.0, lam=1.5):
    """
    Square well potential with value.

    * inf : r < sig
    * eps : sig <= r < lam * sig
    * 0 :  r >= lam * sig
    """

    r = np.array(r)

    phi = np.empty_like(r)

    m0 = r < sig
    m1 = (r >= sig) & (r < lam * sig)
    m2 = r >= lam * sig

    phi[m0] = np.inf
    phi[m1] = eps
    phi[m2] = 0.0

    return phi


@dataclass
class SW_params(LJ_params):
    lam: float

    def phi(self, r):
        return phi_sw(r, sig=self.sig, eps=self.eps, lam=self.lam)

    @classmethod
    def get_params(cls, N=10):
        return kws_to_ld(
            sig=np.random.rand(N),
            eps=(np.random.rand(N) - 0.5) * 2.0,
            lam=np.random.rand(N) + 1.0,
        )


@pytest.fixture(params=SW_params.get_objects(), scope="module")
def sw_params(request):
    return request.param


def phi_yk(r, z, sig=1.0, eps=1.0):
    r = np.array(r)
    phi = np.empty_like(r)
    m = r >= sig

    phi[~m] = np.inf
    if np.any(m):
        x = r[m] / sig
        phi[m] = -eps * np.exp(-z * (x - 1.0)) / x
    return phi


@dataclass
class YK_params(LJ_params):
    z: float

    def phi(self, r):
        return phi_yk(r, z=self.z, sig=self.sig, eps=self.eps)

    @classmethod
    def get_params(cls, N=10):
        return kws_to_ld(
            sig=np.random.rand(N),
            eps=(np.random.rand(N) - 0.5) * 2.0,
            z=np.random.rand(N) * 4,
        )


@pytest.fixture(params=YK_params.get_objects(), scope="module")
def yk_params(request):
    return request.param


def phi_hs(r, sig=1.0):
    r = np.array(r)

    phi = np.empty_like(r)

    m0 = r < sig
    phi[m0] = np.inf
    phi[~m0] = 0.0

    return phi


@dataclass
class HS_params(Base_params):
    sig: float

    def __post_init__(self):
        self.r = get_r(rmin=0.1 * self.sig, rmax=2 * self.sig, n=100)

    def phi(self, r):
        return phi_hs(self.r, sig=self.sig)

    @classmethod
    def get_params(cls, N=10):
        return kws_to_ld(sig=np.random.rand(N))


@pytest.fixture(params=HS_params.get_objects(), scope="module")
def hs_params(request):
    return request.param
