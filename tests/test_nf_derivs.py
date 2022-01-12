import numpy as np
import pytest
from scipy.misc import derivative

import analphipy.potentials as pots
from analphipy.norofrenkel import NoroFrenkelPair


def _do_test(nf, beta, prop, dprop, dx=1e-8, rtol=1e-3):

    a = getattr(nf, dprop)(beta)

    f = getattr(nf, prop)

    b = derivative(f, beta, dx=dx)

    np.testing.assert_allclose(a, b, rtol=rtol)


def factory_lj(lfs=False, cut=False, **kws):

    if lfs or cut:
        rcut = kws.pop("rcut")
    else:
        rcut = None

    p = pots.Phi_lj(**kws)

    if cut:
        p = p.cut(rcut=rcut)
    if lfs:
        p = p.lfs(rcut=rcut)
    return p


BETAS = [0.1, 0.5, 1.0]


def test_nf_deriv_lj():

    sig = eps = 1.0

    p = pots.Phi_lj(sig=sig, eps=eps)

    nf = NoroFrenkelPair.from_phi(
        p, p.segments, x_min=sig, bounds=[0.5 * sig, 1.5 * sig]
    )

    for prop in ["sig", "B2", "lam"]:
        for beta in BETAS:
            _do_test(nf, beta, prop, prop + "_dbeta", dx=1e-8, rtol=1e-2)


@pytest.mark.parametrize("rcut", [2.0, 3.0, 5.0])
def test_nf_deriv_lj_cut(rcut):

    sig = eps = 1.0

    p = pots.Phi_lj(sig=sig, eps=eps).cut(rcut)

    nf = NoroFrenkelPair.from_phi(
        p, p.segments, x_min=sig, bounds=[0.5 * sig, 1.5 * sig]
    )

    for prop in ["sig", "B2", "lam"]:
        for beta in BETAS:
            _do_test(nf, beta, prop, prop + "_dbeta", dx=1e-8, rtol=1e-2)


@pytest.mark.parametrize("z", [1.0, 2.0, 4.0])
def test_nf_deriv_yk(z):

    sig = eps = 1.0

    p = pots.Phi_yk(sig=sig, eps=eps, z=z)

    nf = NoroFrenkelPair(p, p.segments, x_min=sig, phi_min=-1.0)

    for prop in ["sig", "B2", "lam"]:
        for beta in BETAS:
            _do_test(nf, beta, prop, prop + "_dbeta", dx=1e-8, rtol=1e-2)
