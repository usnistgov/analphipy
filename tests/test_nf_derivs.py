# mypy: disable-error-code="no-untyped-def, no-untyped-call"
from typing import TYPE_CHECKING

import numpy as np
import pytest

import analphipy.potential as pots
from analphipy.norofrenkel import NoroFrenkelPair

if TYPE_CHECKING:
    from analphipy.base_potential import PhiAbstract


def _do_test(nf, beta, prop, dprop, dx=1e-8, rtol=1e-3) -> None:
    a = getattr(nf, dprop)(beta)

    f = getattr(nf, prop)

    # central difference:
    b = (f(beta + dx * 0.5) - f(beta - dx * 0.5)) / dx

    np.testing.assert_allclose(a, b, rtol=rtol)


def factory_lj(lfs=False, cut=False, **kws):
    rcut = kws.pop("rcut") if lfs or cut else None

    p: PhiAbstract

    p = pots.LennardJones(**kws)

    if cut and lfs:
        msg = "Can only specify one of lfs and cut"
        raise ValueError(msg)

    if cut:
        assert isinstance(rcut, float)
        return p.cut(rcut=rcut)

    if lfs:
        assert isinstance(rcut, float)
        return p.lfs(rcut=rcut)

    return p


BETAS = [0.1, 0.5, 1.0]


def test_nf_deriv_lj() -> None:
    sig = eps = 1.0

    p = pots.LennardJones(sig=sig, eps=eps)

    nf = NoroFrenkelPair.from_phi(
        p.phi, p.segments, r_min=sig, bounds=[0.5 * sig, 1.5 * sig]
    )

    for prop in ("sig", "B2", "lam"):
        for beta in BETAS:
            _do_test(nf, beta, prop, prop + "_dbeta", dx=1e-8, rtol=1e-2)


@pytest.mark.parametrize("rcut", [2.0, 3.0, 5.0])
def test_nf_deriv_lj_cut(rcut) -> None:
    sig = eps = 1.0

    p = pots.LennardJones(sig=sig, eps=eps).cut(rcut)

    nf = NoroFrenkelPair.from_phi(
        p.phi, p.segments, r_min=sig, bounds=[0.5 * sig, 1.5 * sig]
    )

    for prop in ("sig", "B2", "lam"):
        for beta in BETAS:
            _do_test(nf, beta, prop, prop + "_dbeta", dx=1e-8, rtol=1e-2)


@pytest.mark.parametrize("z", [1.0, 2.0, 4.0])
def test_nf_deriv_yk(z) -> None:
    sig = eps = 1.0

    p = pots.Yukawa(sig=sig, eps=eps, z=z)

    nf = NoroFrenkelPair(p.phi, p.segments, r_min=sig, phi_min=-1.0)

    for prop in ("sig", "B2", "lam"):
        for beta in BETAS:
            _do_test(nf, beta, prop, prop + "_dbeta", dx=1e-8, rtol=1e-2)
