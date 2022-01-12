import numpy as np

import analphipy.potentials as pots

# import pytest


def _do_test(params, factory, kws=None, cut=False, lfs=False, phidphi=True):

    if kws is None:
        kws = params.asdict()

    if cut or lfs:
        rcut = kws.pop("rcut")
    else:
        rcut = None

    p = factory(**kws)

    if cut:
        p = p.cut(rcut=rcut)
    if lfs:
        p = p.lfs(rcut=rcut)

    if phidphi:
        phi0, dphi0 = params.get_phidphi()
        phi1, dphi1 = p.phidphi(params.r)

        np.testing.assert_allclose(dphi0, dphi1)

    else:
        phi0 = params.get_phi()
        phi1 = p.phi(params.r)

    np.testing.assert_allclose(phi0, phi1)


def test_lj(lj_params):
    _do_test(lj_params, pots.Phi_lj)


def test_lj_cut(lj_cut_params):
    _do_test(lj_cut_params, pots.Phi_lj, cut=True)


def test_lj_lfs(lj_lfs_params):
    _do_test(lj_lfs_params, pots.Phi_lj, lfs=True)


def test_nm(nm_params):
    _do_test(nm_params, pots.Phi_nm)


def test_nm_cut(lj_cut_params):
    params = lj_cut_params

    phi0, dphi0 = params.get_phidphi()

    p = pots.Phi_nm(n=12, m=6, sig=params.sig, eps=params.eps).cut(rcut=params.rcut)

    phi1, dphi1 = p.phidphi(params.r)

    np.testing.assert_allclose(phi0, phi1)
    np.testing.assert_allclose(dphi0, dphi1)


def test_nm_lfs(lj_lfs_params):
    params = lj_lfs_params

    phi0, dphi0 = params.get_phidphi()

    p = pots.Phi_nm(n=12, m=6, sig=params.sig, eps=params.eps).lfs(rcut=params.rcut)

    phi1, dphi1 = p.phidphi(params.r)

    np.testing.assert_allclose(phi0, phi1)
    np.testing.assert_allclose(dphi0, dphi1)


def test_sw(sw_params):
    _do_test(sw_params, pots.Phi_sw, phidphi=False)


def test_yk(yk_params):
    _do_test(yk_params, pots.Phi_yk, kws=yk_params.asdict(), phidphi=False)


def test_hs(hs_params):
    _do_test(hs_params, pots.Phi_hs, phidphi=False)
