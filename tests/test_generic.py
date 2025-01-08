# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pylint: disable=duplicate-code
import numpy as np

from analphipy import potential


def _get_generic(p):
    return potential.Generic(phi_func=p.phi, dphidr_func=p.dphidr, segments=p.segments)


def _do_test(params, factory, kws=None, cut=False, lfs=False, phidphi=True) -> None:
    if kws is None:
        kws = params.asdict()

    rcut = kws.pop("rcut") if cut or lfs else None

    p = factory(**kws)
    p = _get_generic(p)

    if cut and lfs:
        msg = "Can only specify one of `cut` and `lfs`"
        raise ValueError(msg)

    if cut:
        assert isinstance(rcut, float)
        p = p.cut(rcut=rcut)
    elif lfs:
        assert isinstance(rcut, float)
        p = p.lfs(rcut=rcut)

    if phidphi:
        phi0, dphi0 = params.get_phidphi()

        phi1 = p.phi(params.r)
        dphi1 = -p.dphidr(params.r) / params.r

        np.testing.assert_allclose(dphi0, dphi1)

    else:
        phi0 = params.get_phi()
        phi1 = p.phi(params.r)

    np.testing.assert_allclose(phi0, phi1)


def test_lj(lj_params) -> None:
    _do_test(lj_params, potential.LennardJones)


def test_lj_cut(lj_cut_params) -> None:
    _do_test(lj_cut_params, potential.LennardJones, cut=True)


def test_lj_lfs(lj_lfs_params) -> None:
    _do_test(lj_lfs_params, potential.LennardJones, lfs=True)


def test_nm(nm_params) -> None:
    _do_test(nm_params, potential.LennardJonesNM)


def test_nm_cut(lj_cut_params) -> None:
    params = lj_cut_params

    phi0, dphi0 = params.get_phidphi()

    p = potential.LennardJonesNM(n=12, m=6, sig=params.sig, eps=params.eps).cut(
        rcut=params.rcut
    )

    p = _get_generic(p)

    phi1 = p.phi(params.r)
    dphi1 = -p.dphidr(params.r) / params.r

    np.testing.assert_allclose(phi0, phi1)
    np.testing.assert_allclose(dphi0, dphi1)


def test_nm_lfs(lj_lfs_params) -> None:
    params = lj_lfs_params

    phi0, dphi0 = params.get_phidphi()

    p = potential.LennardJonesNM(n=12, m=6, sig=params.sig, eps=params.eps).lfs(
        rcut=params.rcut
    )

    p = _get_generic(p)

    phi1 = p.phi(params.r)
    dphi1 = -p.dphidr(params.r) / params.r

    np.testing.assert_allclose(phi0, phi1)
    np.testing.assert_allclose(dphi0, dphi1)


def test_sw(sw_params) -> None:
    _do_test(sw_params, potential.SquareWell, phidphi=False)


def test_yk(yk_params) -> None:
    _do_test(yk_params, potential.Yukawa, kws=yk_params.asdict(), phidphi=False)
