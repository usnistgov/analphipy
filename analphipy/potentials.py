from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional, Sequence, Union, cast

import numpy as np

from analphipy.norofrenkel import NoroFrenkelPair

from .measures import diverg_js_cont
from .utils import Float_or_ArrayLike, minimize_phi

# classes to handle pair potentials


class Phidphi_class:
    def __call__(
        self, r: Float_or_ArrayLike, dphi: bool = False
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        if dphi:
            return self.phidphi(r)
        else:
            return self.phi(r)

    def asdict(self) -> dict:
        return dataclasses.asdict(self)

    def copy(self):
        return type(self)(**self.asdict())

    @property
    def segments(self) -> list:
        raise NotImplementedError("to be implemented in subclass")

    def phi(self, r: Float_or_ArrayLike) -> np.ndarray:
        raise NotImplementedError("to be implemented in subclass if appropriate")

    def phidphi(self, r: Float_or_ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("to be implemented in subclass if appropriate")

    @property
    def x_min(self) -> float:
        raise NotImplementedError("to be implemented in subclass")

    def boltz(self, r: Float_or_ArrayLike, beta: float) -> np.ndarray:
        return np.exp(-beta * self.phi(r))

    def mayer(self, r: Float_or_ArrayLike, beta: float) -> np.ndarray:
        return np.exp(-beta * self.phi(r)) - 1.0

    def boltz_partial(
        self, beta: float
    ) -> Callable[[Float_or_ArrayLike], Union[float, np.ndarray]]:
        """
        returns callable function  boltz(x) = exp(-beta * phi(x))
        """
        return partial(self.boltz, beta=beta)

    def mayer_partial(
        self, beta: float
    ) -> Callable[[Float_or_ArrayLike], Union[float, np.ndarray]]:
        """callable function mayer(x) = exp(-beta * phi(x)) - 1."""
        return partial(self.mayer, beta=beta)

    @property
    def phi_min(self):
        return self.phi(self.x_min) * 1.0

    def minimize(self, r0: float, bounds: Optional[tuple[float, float]] = None, **kws):
        return minimize_phi(self.phi, r0=r0, bounds=bounds, **kws)

    def to_nf_fixed(self, quad_kws: Optional[dict] = None) -> NoroFrenkelPair:
        x_min, phi_min = self.x_min, self.phi_min

        return NoroFrenkelPair(
            phi=self.phi,
            segments=self.segments,
            x_min=x_min,
            phi_min=phi_min,
            quad_kws=quad_kws,
        )

    def to_nf_numeric(
        self,
        x_min: Optional[float] = None,
        bounds: Optional[tuple[float, float]] = None,
        quad_kws: Optional[dict] = None,
        **kws,
    ) -> NoroFrenkelPair:
        return NoroFrenkelPair.from_phi(
            phi=self.phi,
            segments=self.segments,
            bounds=bounds,
            x_min=x_min,
            quad_kws=quad_kws,
            **kws,
        )

    def to_nf(
        self,
        x_min: Optional[float] = None,
        bounds: Optional[tuple[float, float]] = None,
        quad_kws: Optional[dict] = None,
        **kws,
    ) -> NoroFrenkelPair:
        try:
            return self.to_nf_fixed(quad_kws=quad_kws)
        except NotImplementedError:
            return self.to_nf_numeric(
                x_min=x_min, bounds=bounds, quad_kws=quad_kws, **kws
            )

    def boltz_diverg_js(
        self,
        other: Phidphi_class,
        beta: float,
        beta_other: Optional[float] = None,
        volume: Union[str, Callable] = "3d",
        err: bool = False,
        full_output: bool = False,
        **kws,
    ):

        if beta_other is None:
            beta_other = beta

        return diverg_js_cont(
            p=self.boltz_partial(beta=beta),
            q=other.boltz_partial(beta=beta_other),
            segments=self.segments,
            segments_q=other.segments,
            volume=volume,
            err=err,
            full_output=full_output,
            **kws,
        )

    def mayer_diverg_js(
        self,
        other: Phidphi_class,
        beta: float,
        beta_other: Optional[float] = None,
        volume: Union[str, Callable] = "3d",
        err: bool = False,
        full_output: bool = False,
        **kws,
    ):

        if beta_other is None:
            beta_other = beta

        return diverg_js_cont(
            p=self.mayer_partial(beta=beta),
            q=other.mayer_partial(beta=beta_other),
            segments=self.segments,
            segments_q=other.segments,
            volume=volume,
            err=err,
            full_output=full_output,
            **kws,
        )


@dataclass
class Phi_cut_base(Phidphi_class):
    """
    create cut potential from base potential


    phi_cut(r) = phi(r) + vcorrect(r) if r <= rcut
               = 0 otherwise


    dphi_cut(r) = dphi(r) + dvcorrect(r) if r <= rcut
                = 0 otherwise

    So, for example, for just cut, `vcorrect(r) = -v(rcut)`, `dvrcut=0`,
    and for lfs, `vcorrect(r) = -v(rcut) - dv(rcut)/dr (r - rcut)`
    `dvcorrect(r) = ...`
    """

    phi_base: Phidphi_class
    rcut: float

    @property
    def segments(self) -> list:
        return [x for x in self.phi_base.segments if x < self.rcut] + [self.rcut]

    @classmethod
    def from_base(cls, base, rcut, *args, **kws) -> Phi_cut_base:
        """
        Create cut potential from base phi class
        """
        phi_base = base(*args, **kws)
        return cls(phi_base=phi_base, rcut=rcut)

    def phi(self, r: Float_or_ArrayLike) -> np.ndarray:
        r = np.asarray(r)
        v = np.empty_like(r)

        left = r <= self.rcut
        right = ~left

        v[right] = 0.0

        if np.any(left):
            v[left] = self.phi_base(r[left]) + self.vcorrect(r[left])

        return v

    def phidphi(self, r: Float_or_ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        r = np.array(r)

        v = np.empty_like(r)
        dv = np.empty_like(r)

        left = r <= self.rcut
        right = ~left

        v[right] = 0.0
        dv[right] = 0.0

        if np.any(left):
            v[left], dv[left] = self.phi_base.phidphi(r[left])
            v[left] += self.vcorrect(r[left])
            dv[left] += self.dvcorrect(r[left])

        return v, dv

    def vcorrect(self, r):
        raise NotImplementedError

    def dvcorrect(self, r):
        raise NotImplementedError


@dataclass
class Phi_cut(Phi_cut_base):
    """
    potential phi cut at position r
    """

    def __post_init__(self):
        self.vcut = self.phi_base(self.rcut)

    def vcorrect(self, r: np.ndarray) -> np.ndarray:
        return -cast(np.ndarray, self.vcut)

    def dvcorrect(self, r: np.ndarray) -> np.ndarray:
        return np.array(0.0)


@dataclass
class Phi_lfs(Phi_cut_base):
    def __post_init__(self):
        vcut, dvcut = self.phi_base.phidphi(self.rcut)
        self.vcut = vcut
        self.dvdrcut = -dvcut * self.rcut

    def vcorrect(self, r: np.ndarray) -> np.ndarray:
        out = -(self.vcut + self.dvdrcut * (r - self.rcut))
        return cast(np.ndarray, out)

    def dvcorrect(self, r: np.ndarray) -> np.ndarray:
        out = self.dvdrcut / r
        return cast(np.ndarray, out)


class Phi_base(Phidphi_class):
    def cut(self, rcut: float) -> Phi_cut:
        return Phi_cut(phi_base=self, rcut=rcut)

    def lfs(self, rcut: float) -> Phi_lfs:
        return Phi_lfs(phi_base=self, rcut=rcut)

    def phi(self, r: Float_or_ArrayLike) -> np.ndarray:
        raise NotImplementedError("to be implemented in subclass")

    def phidphi(self, r: Float_or_ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("to be implemented in subclass or not available")

    @property
    def segments(self) -> list:
        return [0.0, np.inf]


@dataclass
class Phi_lj(Phi_base):
    sig: float = 1.0
    eps: float = 1.0

    def __post_init__(self):
        sig, eps = self.sig, self.eps
        self.sigsq = sig * sig
        self.four_eps = 4.0 * eps

    def phi(self, r: Float_or_ArrayLike) -> np.ndarray:
        r = np.array(r)

        x2 = self.sigsq / (r * r)
        x6 = x2 ** 3
        return cast(np.ndarray, self.four_eps * x6 * (x6 - 1.0))

    def phidphi(self, r: Float_or_ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """calculate phi and dphi (=-1/r dphi/dr) at particular r"""
        r = np.array(r)
        rinvsq = 1.0 / (r * r)

        x2 = self.sigsq * rinvsq
        x6 = x2 * x2 * x2

        phi = self.four_eps * x6 * (x6 - 1.0)
        dphi = 12.0 * self.four_eps * x6 * (x6 - 0.5) * rinvsq
        return phi, dphi

    @property
    def x_min(self) -> float:
        return cast(float, self.sig * 2.0 ** (1.0 / 6.0))

    @property
    def phi_min(self) -> float:
        return -self.eps


@dataclass
class Phi_nm(Phi_base):
    n: int = 12
    m: int = 6
    sig: float = 1.0
    eps: float = 1.0

    def __post_init__(self):
        n, m, eps = self.n, self.m, self.eps
        self.prefac = eps * (n / (n - m)) * (n / m) ** (m / (n - m))

    @property
    def x_min(self) -> float:
        n, m = self.n, self.m
        return cast(float, self.sig * (n / m) ** (1.0 / (n - m)))

    @property
    def phi_min(self) -> float:
        return -self.eps

    def phi(self, r: Float_or_ArrayLike) -> np.ndarray:
        r = np.array(r)

        x = self.sig / r
        out = self.prefac * (x ** self.n - x ** self.m)
        return cast(np.ndarray, out)

    def phidphi(self, r: Float_or_ArrayLike) -> tuple[np.ndarray, np.ndarray]:

        r = np.array(r)
        x = self.sig / r

        xn = x ** self.n
        xm = x ** self.m

        phi = np.empty_like(r)
        dphi = np.empty_like(r)

        phi = self.prefac * (xn - xm)

        # dphi = -1/r dphi/dr = x dphi/dx * 1/r**2
        # where x = sig / r
        dphi = self.prefac * (self.n * xn - self.m * xm) / (r ** 2)

        return cast(tuple[np.ndarray, np.ndarray], (phi, dphi))


@dataclass
class Phi_yk(Phi_base):

    z: float = 1.0
    sig: float = 1.0
    eps: float = 1.0

    @property
    def segments(self) -> list:
        return [0.0, self.sig, np.inf]

    @property
    def x_min(self) -> float:
        return self.sig

    @property
    def phi_min(self) -> float:
        return -self.eps

    def phi(self, r: Float_or_ArrayLike) -> np.ndarray:
        sig, eps = self.sig, self.eps

        r = np.array(r)
        phi = np.empty_like(r)
        m = r >= sig

        phi[~m] = np.inf
        if np.any(m):
            x = r[m] / sig
            phi[m] = -eps * np.exp(-self.z * (x - 1.0)) / x
        return phi


@dataclass
class Phi_hs(Phi_base):
    sig: float = 1.0

    def phi(self, r: Float_or_ArrayLike) -> np.ndarray:
        r = np.array(r)

        phi = np.empty_like(r)

        m0 = r < self.sig
        phi[m0] = np.inf
        phi[~m0] = 0.0
        return phi


@dataclass
class Phi_sw(Phi_base):
    sig: float = 1.0
    eps: float = 1.0
    lam: float = 1.5

    @property
    def x_min(self) -> float:
        return self.sig

    @property
    def phi_min(self) -> float:
        return self.eps

    @property
    def segments(self) -> list:
        return [0.0, self.sig, self.sig * self.lam]

    def phi(self, r: Float_or_ArrayLike) -> np.ndarray:

        sig, eps, lam = self.sig, self.eps, self.lam

        r = np.array(r)

        phi = np.empty_like(r)

        m0 = r < sig
        m2 = r >= lam * sig
        m1 = (~m0) & (~m2)

        phi[m0] = np.inf
        phi[m1] = eps
        phi[m2] = 0.0

        return phi


class CubicTable(Phi_base):
    def __init__(self, bounds: Sequence[float], phi_array: Float_or_ArrayLike):

        self.phi_array = np.asarray(phi_array)
        self.bounds = bounds

        self.ds = (self.bounds[1] - self.bounds[0]) / self.size
        self.dsinv = 1.0 / self.ds

    @property
    def segments(self):
        return [np.sqrt(x) for x in self.bounds]

    @classmethod
    def from_phi(cls, phi: Callable, rmin: float, rmax: float, ds: float):
        bounds = (rmin * rmin, rmax * rmax)

        delta = bounds[1] - bounds[0]

        size = int(delta / ds)
        ds = delta / size

        phi_array = []

        r = np.sqrt(np.arange(size + 1) * ds + bounds[0])

        phi_array = phi(r)

        return cls(bounds=bounds, phi_array=phi_array)

    def __len__(self):
        return len(self.phi_array)

    @property
    def size(self) -> int:
        return len(self) - 1

    @property
    def smin(self) -> float:
        return self.bounds[0]

    @property
    def smax(self) -> float:
        return self.bounds[1]

    def phidphi(self, r: Float_or_ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(r)

        v = np.empty_like(r)
        dv = np.empty_like(r)

        s = r * r
        left = s <= self.smax
        right = ~left

        v[right] = 0.0
        dv[right] = 0.0

        if np.any(left):

            sds = (s[left] - self.smin) * self.dsinv
            k = sds.astype(int)
            k[k < 0] = 0

            xi = sds - k

            t = np.take(self.phi_array, [k, k + 1, k + 2], mode="clip")
            dt = np.diff(t, axis=0)
            ddt = np.diff(dt, axis=0)

            v[left] = t[0, :] + xi * (dt[0, :] + 0.5 * (xi - 1.0) * ddt[0, :])
            dv[left] = -2.0 * self.dsinv * (dt[0, :] + (xi - 0.5) * ddt[0, :])

        return v, dv

    def phi(self, r: Float_or_ArrayLike) -> np.ndarray:
        return self.phidphi(r)[0]


_PHI_NAMES = Literal["lj", "nm", "sw", "hs", "yk", "LJ", "NM", "SW", "HS", "YK"]


def factory_phi(
    potential_name: _PHI_NAMES,
    rcut: Optional[float] = None,
    lfs: bool = False,
    cut: bool = False,
    **kws,
) -> Phidphi_class:

    name = potential_name.lower()

    phi: Phidphi_class

    if name == "lj":
        phi = Phi_lj(**kws)

    elif name == "sw":
        phi = Phi_sw(**kws)

    elif name == "nm":
        phi = Phi_nm(**kws)

    elif name == "yk":
        phi = Phi_yk(**kws)

    elif name == "hs":
        phi = Phi_hs(**kws)

    else:
        raise ValueError(f"{name} must be in {_PHI_NAMES}")

    if lfs or cut:
        assert rcut is not None
        if cut:
            phi = phi.cut(rcut=rcut)

        elif lfs:
            phi = phi.lfs(rcut=rcut)

    return phi
