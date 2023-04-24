from pathlib import Path

import numpy as np
import pandas as pd

import analphipy.measures as measures
import analphipy.potential as pots
from analphipy.norofrenkel import NoroFrenkelPair

data = Path(__file__).parent / "data"

nsamp = 20


def test_B2_sw():
    df = (
        pd.read_csv(data / "vir_sw_.csv")
        .assign(beta=lambda x: 1.0 / x["temp"])
        .drop_duplicates()
    )

    if len(df) > nsamp:
        df = df.sample(nsamp)
    for (sig, eps, lam), g in df.groupby(["sig", "eps", "lam"]):
        p = pots.SquareWell(sig=sig, eps=eps, lam=lam)

        a = NoroFrenkelPair(phi=p.phi, segments=p.segments, r_min=sig, phi_min=eps)

        out = a.table(g["beta"], ["B2", "B2_dbeta"])

        np.testing.assert_allclose(g["Bn_2"], out["B2"])
        np.testing.assert_allclose(-g["dBn_2"], out["B2_dbeta"])


def test_B2_lj():
    df = (
        pd.read_csv(data / "vir_lj_.csv")
        .assign(beta=lambda x: 1.0 / x["temp"])
        .drop_duplicates()
        .sample(40)
    )

    if len(df) > nsamp:
        df = df.sample(nsamp)
    sig = eps = 1.0
    p_base = pots.LennardJones(sig=sig, eps=eps)

    for (rcut, tail), g in df.groupby(["rcut", "tail"]):
        if tail == "LFS":
            p = p_base.lfs(rcut=rcut)
        elif tail == "CUT":
            p = p_base.cut(rcut=rcut)
        elif tail == "LRC":
            p = p_base

        a = NoroFrenkelPair.from_phi(
            phi=p.phi, segments=p.segments, r_min=sig, bounds=[0.5, 1.5]
        )

        out = a.table(g["beta"], ["B2", "B2_dbeta"])

        np.testing.assert_allclose(g["Bn_2"], out["B2"], rtol=1e-3)
        np.testing.assert_allclose(-g["dBn_2"], out["B2_dbeta"], rtol=1e-3)


def test_B2_nm():
    df = (
        pd.read_csv(data / "vir_lj_nm_.csv")
        .assign(beta=lambda x: 1.0 / x["temp"])
        .drop_duplicates()
    )

    if len(df) > nsamp:
        df = df.sample(nsamp)
    sig = eps = 1.0

    for (n_exp, m_exp), g in df.groupby(["n_exp", "m_exp"]):
        p = pots.LennardJonesNM(n=n_exp, m=m_exp, sig=sig, eps=eps)

        a = NoroFrenkelPair.from_phi(
            phi=p.phi, segments=p.segments, r_min=sig, bounds=[0.5, 1.5]
        )

        out = a.table(g["beta"], ["B2", "B2_dbeta"])

        np.testing.assert_allclose(g["Bn_2"], out["B2"], rtol=1e-3)
        np.testing.assert_allclose(-g["dBn_2"], out["B2_dbeta"], rtol=1e-3)


def test_B2_yk():
    df = (
        pd.read_csv(data / "vir_yukawa_.csv")
        .assign(beta=lambda x: 1.0 / x["temp"])
        .drop_duplicates()
    )

    if len(df) > nsamp:
        df = df.sample(nsamp)
    sig = eps = 1.0

    for (z_yukawa), g in df.groupby("z_yukawa"):
        p = pots.Yukawa(z=z_yukawa, sig=sig, eps=eps)

        a = NoroFrenkelPair(phi=p.phi, segments=p.segments, r_min=sig, phi_min=-1.0)

        out = a.table(g["beta"], ["B2", "B2_dbeta"])

        np.testing.assert_allclose(g["Bn_2"], out["B2"], rtol=1e-6)
        np.testing.assert_allclose(-g["dBn_2"], out["B2_dbeta"], rtol=1e-6)


def test_B2_hs():
    sig = np.random.rand()

    p = pots.HardSphere(sig=sig)

    B2a = 2 * np.pi / 3.0 * sig**3

    B2b = measures.secondvirial(p.phi, beta=1.0, segments=[0.0, sig])

    np.testing.assert_allclose(B2a, B2b)

    B2_dbeta = measures.secondvirial_dbeta(p.phi, beta=1.0, segments=[0.0, sig])

    np.testing.assert_allclose(0.0, B2_dbeta)
