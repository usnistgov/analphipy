# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pylint: disable=missing-class-docstring,duplicate-code
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable

    from analphipy._typing_compat import Self

rng = np.random.default_rng()


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
        vcut, _dvcut = phidphi_lj(rcut, sig=sig, eps=eps)

        phi[m] = v - vcut
        dphi[m] = dv

    return phi, dphi


def phidphi_lj_lfs(r, rcut, sig=1.0, eps=1.0):
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
    rmin += rng.random() * 0.1
    rmax += rng.random() * 0.1

    return np.linspace(rmin, rmax, n)


def kws_to_ld(**kws: Iterable[Any]) -> list[dict[str, Any]]:
    return [dict(zip(kws, t, strict=True)) for t in zip(*kws.values(), strict=False)]


@dataclass
class BaseParams:
    def __post_init__(self):
        self.r = None  # pyright: ignore[reportUninitializedInstanceVariable]

    def phidphi(self, r) -> None:
        raise NotImplementedError

    def get_phidphi(self):
        if hasattr(self, "phidphi"):
            return self.phidphi(self.r)
        raise NotImplementedError

    def phi(self, r):
        # pyrefly: ignore [unsupported-operation]
        return self.phidphi(r)[0]  # type: ignore[func-returns-value]  # pyright: ignore[reportOptionalSubscript]  # ty:ignore[not-subscriptable]

    def get_phi(self):
        return self.phi(self.r)

    @classmethod
    def get_params(cls, nsamp) -> None:
        raise NotImplementedError

    @classmethod
    def get_objects(cls, nsamp=10) -> Iterable[Self]:
        # pyrefly: ignore [not-iterable]
        for params in cls.get_params(nsamp=nsamp):  # type: ignore[attr-defined]  # pyright: ignore[reportGeneralTypeIssues]  # ty:ignore[not-iterable]
            yield cls(**params)

    def asdict(self):
        return asdict(self)


@dataclass
class LJParams(BaseParams):
    sig: float
    eps: float

    def __post_init__(self):
        self.r = get_r(rmin=0.1 * self.sig, rmax=5.0 * self.sig, n=100)

    # pyrefly: ignore [bad-override]
    def phidphi(self, r):  # pyright: ignore[reportIncompatibleMethodOverride]
        return phidphi_lj(r, sig=self.sig, eps=self.eps)

    @classmethod
    # pyrefly: ignore [bad-override]
    def get_params(cls, nsamp=10):  # pyright: ignore[reportIncompatibleMethodOverride]
        return kws_to_ld(
            sig=rng.random(nsamp) * 4,
            eps=rng.random(nsamp),
        )


@dataclass
class LJCutParams(LJParams):
    rcut: float

    def __post_init__(self):
        self.r = get_r(rmin=0.1 * self.sig, rmax=self.rcut + self.sig, n=100)

    def phidphi(self, r):
        return phidphi_lj_cut(r, sig=self.sig, eps=self.eps, rcut=self.rcut)

    @classmethod
    def get_params(cls, nsamp=10):
        return kws_to_ld(
            sig=rng.random(nsamp),
            eps=rng.random(nsamp),
            rcut=rng.random(nsamp) + 3.0,
        )


@dataclass
class LJLFSParams(LJCutParams):
    def phidphi(self, r):
        return phidphi_lj_lfs(r, sig=self.sig, eps=self.eps, rcut=self.rcut)


@pytest.fixture(params=LJParams.get_objects(), scope="module")
def lj_params(request):
    return request.param


@pytest.fixture(params=LJCutParams.get_objects(), scope="module")
def lj_cut_params(request):
    return request.param


@pytest.fixture(params=LJLFSParams.get_objects(), scope="module")
def lj_lfs_params(request):
    return request.param


@dataclass
class NMParams(LJParams):
    n: int
    m: int

    def phidphi(self, r):
        return phidphi_nm(r, n=self.n, m=self.m, sig=self.sig, eps=self.eps)

    @classmethod
    def get_params(cls, nsamp=None):
        n = [8, 10, 12, 18]
        m = [6, 7, 6, 9]

        if nsamp is None:
            nsamp = len(n)

        return kws_to_ld(sig=rng.random(nsamp), eps=rng.random(nsamp), n=n, m=m)


@pytest.fixture(params=NMParams.get_objects(), scope="module")
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
class SWParams(LJParams):
    lam: float

    def phi(self, r):
        return phi_sw(r, sig=self.sig, eps=self.eps, lam=self.lam)

    @classmethod
    def get_params(cls, nsamp=10):
        return kws_to_ld(
            sig=rng.random(nsamp),
            eps=(rng.random(nsamp) - 0.5) * 2.0,
            lam=rng.random(nsamp) + 1.0,
        )


@pytest.fixture(params=SWParams.get_objects(), scope="module")
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
class YKParams(LJParams):
    z: float

    def phi(self, r):
        return phi_yk(r, z=self.z, sig=self.sig, eps=self.eps)

    @classmethod
    def get_params(cls, nsamp=10):
        return kws_to_ld(
            sig=rng.random(nsamp),
            eps=(rng.random(nsamp) - 0.5) * 2.0,
            z=rng.random(nsamp) * 4,
        )


@pytest.fixture(params=YKParams.get_objects(), scope="module")
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
class HSParams(BaseParams):  # pylint: disable=abstract-method
    sig: float

    def __post_init__(self):
        self.r = get_r(rmin=0.1 * self.sig, rmax=2 * self.sig, n=100)

    def phi(self, r):  # noqa: ARG002
        return phi_hs(self.r, sig=self.sig)

    @classmethod
    # pyrefly: ignore [bad-override]
    def get_params(cls, nsamp=10):  # pyright: ignore[reportIncompatibleMethodOverride]
        return kws_to_ld(sig=rng.random(nsamp))


@pytest.fixture(params=HSParams.get_objects(), scope="module")
def hs_params(request):
    return request.param
