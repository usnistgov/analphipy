# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pylint: disable=duplicate-code
import numpy as np
import pytest

from analphipy import potential as pots
from analphipy.base_potential import PhiAbstract


def test_asdict() -> None:
    # pylint: disable=redefined-variable-type
    p_lj = pots.LennardJones(sig=1.5, eps=1.5)

    p = p_lj

    assert p.asdict() == {"sig": 1.5, "eps": 1.5}

    p_new = p.new_like(sig=1.0, eps=1.0)

    assert p_new.asdict() == {"sig": 1.0, "eps": 1.0}

    assert p.assign(sig=2.0).asdict() == {"sig": 2.0, "eps": 1.5}

    pa = PhiAbstract()

    with pytest.raises(NotImplementedError):
        pa.phi(1.0)

    with pytest.raises(NotImplementedError):
        pa.dphidr(1.0)

    from analphipy.base_potential import PhiCutBase

    p = PhiCutBase(p_lj, rcut=2.5)  # type: ignore[assignment]

    with pytest.raises(NotImplementedError):
        p.phi(1.0)

    with pytest.raises(NotImplementedError):
        p.dphidr(1.0)

    p_lj = pots.LennardJones()

    p_lfs = p_lj.lfs(rcut=2.5).assign_min_numeric(1.1, bounds=(0.5, 1.5))
    # pyrefly: ignore [no-matching-overload]
    np.testing.assert_allclose(p_lfs.r_min, 1.123148919)  # pyright: ignore[reportCallIssue]  # ty:ignore[no-matching-overload]

    p_lfs = p_lj.lfs(rcut=2.5).assign_min_numeric(1.1, bounds="segments")
    # pyrefly: ignore [no-matching-overload]
    np.testing.assert_allclose(p_lfs.r_min, 1.123148919)  # pyright: ignore[reportCallIssue]  # ty:ignore[no-matching-overload]

    p_lfs = p_lj.lfs(rcut=2.5).assign_min_numeric(r0="mean", bounds=(0.5, 1.5))
    # pyrefly: ignore [no-matching-overload]
    np.testing.assert_allclose(p_lfs.r_min, 1.123148919, atol=1e-5)  # pyright: ignore[reportCallIssue]  # ty:ignore[no-matching-overload]

    with pytest.raises(ValueError):
        p_lj.lfs(rcut=2.5).assign_min_numeric(r0="mean", bounds=None)

    with pytest.raises(ValueError):
        p_lj.lfs(rcut=2.5).to_nf()


def _do_test(params, factory, kws=None, cut=False, lfs=False, phidphi=True) -> None:
    if kws is None:
        kws = params.asdict()

    rcut = kws.pop("rcut") if cut or lfs else None

    p = factory(**kws)

    if cut:
        p = p.cut(rcut=rcut)
    if lfs:
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
    _do_test(lj_params, pots.LennardJones)


def test_lj_cut(lj_cut_params) -> None:
    _do_test(lj_cut_params, pots.LennardJones, cut=True)


def test_lj_lfs(lj_lfs_params) -> None:
    _do_test(lj_lfs_params, pots.LennardJones, lfs=True)


def test_nm(nm_params) -> None:
    _do_test(nm_params, pots.LennardJonesNM)


def test_nm_cut(lj_cut_params) -> None:
    params = lj_cut_params

    phi0, dphi0 = params.get_phidphi()

    p = pots.LennardJonesNM(n=12, m=6, sig=params.sig, eps=params.eps).cut(
        rcut=params.rcut
    )

    phi1 = p.phi(params.r)
    dphi1 = -p.dphidr(params.r) / params.r

    np.testing.assert_allclose(phi0, phi1)
    np.testing.assert_allclose(dphi0, dphi1)


def test_nm_lfs(lj_lfs_params) -> None:
    params = lj_lfs_params

    phi0, dphi0 = params.get_phidphi()

    p = pots.LennardJonesNM(n=12, m=6, sig=params.sig, eps=params.eps).lfs(
        rcut=params.rcut
    )

    phi1 = p.phi(params.r)
    dphi1 = -p.dphidr(params.r) / params.r

    np.testing.assert_allclose(phi0, phi1)
    np.testing.assert_allclose(dphi0, dphi1)


def test_sw(sw_params) -> None:
    _do_test(sw_params, pots.SquareWell, phidphi=False)


def test_yk(yk_params) -> None:
    _do_test(yk_params, pots.Yukawa, kws=yk_params.asdict(), phidphi=False)


def test_hs(hs_params) -> None:
    _do_test(hs_params, pots.HardSphere, phidphi=False)
