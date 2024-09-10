# mypy: disable-error-code="no-untyped-def, no-untyped-call"
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

import analphipy.potential as pots
from analphipy.norofrenkel import NoroFrenkelPair

if TYPE_CHECKING:
    from analphipy.base_potential import PhiAbstract

data = Path(__file__).parent / "data"

nsamp = 20


def test_simple() -> None:
    p = pots.LennardJones()

    n = p.to_nf()

    np.testing.assert_allclose(n.secondvirial(1.0), n.secondvirial_sw(1.0))  # type: ignore[arg-type]

    np.testing.assert_allclose(n.secondvirial(1.0), n.B2_sw(1.0))  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        n.lam(beta=1.0, err=True)

    with pytest.raises(ValueError):
        n.sw_dict(beta=1.0, err=True)

    with pytest.raises(ValueError):
        n.lam_dbeta(beta=1.0, err=True)

    with pytest.raises(ValueError):
        n.secondvirial_sw(beta=1.0, err=True)

    assert isinstance(n.sw_dict(1.0), dict)


def test_nf_sw() -> None:
    cols = ["sig", "eps", "lam", "B2"]
    table = pd.read_csv(data / "eff_sw_.csv").assign(beta=lambda x: 1.0 / x["temp"])

    if len(table) > nsamp:
        table = table.sample(nsamp)

    for (sig, eps, lam), g in table.groupby(["sig", "eps", "lam"]):
        p = pots.SquareWell(sig=sig, eps=eps, lam=lam)

        a = NoroFrenkelPair(phi=p.phi, segments=p.segments, r_min=sig, phi_min=eps)

        out = a.table(g["beta"], props=None)  # type: ignore[arg-type]

        for col in cols:
            left = col + "_eff"
            np.testing.assert_allclose(g[left], out[col])


def test_nf_lj() -> None:
    cols = ["sig", "eps", "lam", "B2", "sig_dbeta"]

    table = (
        pd.read_csv(data / "eff_lj_.csv")
        .assign(beta=lambda x: 1.0 / x["temp"])
        .drop_duplicates()
        .assign(sig_dbeta_eff=lambda x: -x["dsig_eff"])
    )

    if len(table) > nsamp:
        table = table.sample(nsamp)
    sig = eps = 1.0
    p_base = pots.LennardJones(sig=sig, eps=eps)

    p: PhiAbstract

    for (rcut, tail), g in table.groupby(["rcut", "tail"]):
        if tail == "LFS":
            p = p_base.lfs(rcut=rcut)
        elif tail == "CUT":
            p = p_base.cut(rcut=rcut)
        elif tail == "LRC":
            p = p_base

        a = NoroFrenkelPair.from_phi(
            phi=p.phi, segments=p.segments, r_min=sig, bounds=[0.5, 1.5]
        )

        out = a.table(g["beta"], cols)  # type: ignore[arg-type]

        for col in cols:
            left = col + "_eff"
            np.testing.assert_allclose(g[left], out[col], rtol=1e-3)


def test_nf_nm() -> None:
    cols = ["sig", "eps", "lam", "B2"]
    table = (
        pd.read_csv(data / "eff_lj_nm_.csv")
        .assign(beta=lambda x: 1.0 / x["temp"])
        .drop_duplicates()
    )

    if len(table) > nsamp:
        table = table.sample(nsamp)
    sig = eps = 1.0

    for (n_exp, m_exp), g in table.groupby(["n_exp", "m_exp"]):
        p = pots.LennardJonesNM(n=n_exp, m=m_exp, sig=sig, eps=eps)

        a = NoroFrenkelPair.from_phi(
            phi=p.phi, segments=p.segments, r_min=sig, bounds=[0.5, 1.5]
        )

        out = a.table(g["beta"], cols)  # type: ignore[arg-type]

        for col in cols:
            left = col + "_eff"
            np.testing.assert_allclose(g[left], out[col], rtol=1e-3)


def test_nf_yk() -> None:
    cols = ["sig", "eps", "lam", "B2"]

    table = (
        pd.read_csv(data / "eff_yukawa_.csv")
        .assign(beta=lambda x: 1.0 / x["temp"])
        .drop_duplicates()
    )

    if len(table) > nsamp:
        table = table.sample(nsamp)
    sig = eps = 1.0

    for (z_yukawa), g in table.groupby("z_yukawa"):
        p = pots.Yukawa(z=z_yukawa, sig=sig, eps=eps)  # type: ignore[arg-type]

        a = NoroFrenkelPair(phi=p.phi, segments=p.segments, r_min=sig, phi_min=-1.0)

        out = a.table(g["beta"], cols)  # type: ignore[arg-type]

        for col in cols:
            left = col + "_eff"
            np.testing.assert_allclose(g[left], out[col], rtol=1e-3)
