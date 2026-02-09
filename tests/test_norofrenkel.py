# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pylint: disable=duplicate-code
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analphipy.potential as pots
from analphipy.norofrenkel import NoroFrenkelPair

# pyrefly: ignore [missing-import]
from .utils import iter_phi_lj

data = Path(__file__).parent / "data"

nsamp = 20


def test_simple() -> None:
    p = pots.LennardJones()

    n = p.to_nf()

    # pyrefly: ignore [no-matching-overload]
    np.testing.assert_allclose(n.secondvirial(1.0), n.secondvirial_sw(1.0))  # pyright: ignore[reportCallIssue]

    # pyrefly: ignore [no-matching-overload]
    np.testing.assert_allclose(n.secondvirial(1.0), n.B2_sw(1.0))  # pyright: ignore[reportCallIssue]  # ty:ignore[no-matching-overload]

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
        # pyrefly: ignore [bad-argument-type]
        p = pots.SquareWell(sig=sig, eps=eps, lam=lam)  # ty:ignore[invalid-argument-type]

        # pyrefly: ignore [bad-argument-type]
        a = NoroFrenkelPair(phi=p.phi, segments=p.segments, r_min=sig, phi_min=eps)  # ty:ignore[invalid-argument-type]

        # pyrefly: ignore [bad-argument-type]
        out = a.table(g["beta"], props=None)  # ty:ignore[invalid-argument-type]

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

    for g, a in iter_phi_lj(table, nsamp):
        out = a.table(g["beta"], cols)  # ty:ignore[invalid-argument-type]

        for col in cols:
            left = col + "_eff"
            np.testing.assert_allclose(g[left], out[col], rtol=1e-3)


def test_nf_nm() -> None:
    cols = ["sig", "eps", "lam", "B2"]
    table: pd.DataFrame = (
        pd.read_csv(data / "eff_lj_nm_.csv")
        .assign(beta=lambda x: 1.0 / x["temp"])
        .drop_duplicates()
    )

    if len(table) > nsamp:
        table = table.sample(nsamp)
    sig = eps = 1.0

    for (n_exp, m_exp), g in table.groupby(["n_exp", "m_exp"]):
        # pyrefly: ignore [bad-argument-type]
        p = pots.LennardJonesNM(n=n_exp, m=m_exp, sig=sig, eps=eps)  # ty:ignore[invalid-argument-type]

        a = NoroFrenkelPair.from_phi(
            phi=p.phi, segments=p.segments, r_min=sig, bounds=[0.5, 1.5]
        )

        # pyrefly: ignore [bad-argument-type]
        out = a.table(g["beta"], cols)  # ty:ignore[invalid-argument-type]

        for col in cols:
            left = col + "_eff"
            np.testing.assert_allclose(g[left], out[col], rtol=1e-3)


def test_nf_yk() -> None:
    cols = ["sig", "eps", "lam", "B2"]

    table: pd.DataFrame = (
        pd.read_csv(data / "eff_yukawa_.csv")
        .assign(beta=lambda x: 1.0 / x["temp"])
        .drop_duplicates()
    )

    if len(table) > nsamp:
        table = table.sample(nsamp)
    sig = eps = 1.0

    for (z_yukawa), g in table.groupby("z_yukawa"):
        # pyrefly: ignore [bad-argument-type]
        p = pots.Yukawa(z=z_yukawa, sig=sig, eps=eps)  # ty:ignore[invalid-argument-type]

        a = NoroFrenkelPair(phi=p.phi, segments=p.segments, r_min=sig, phi_min=-1.0)

        # pyrefly: ignore [bad-argument-type]
        out = a.table(g["beta"], cols)  # ty:ignore[invalid-argument-type]

        for col in cols:
            left = col + "_eff"
            np.testing.assert_allclose(g[left], out[col], rtol=1e-3)
