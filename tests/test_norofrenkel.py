from pathlib import Path

import numpy as np
import pandas as pd

# import analphipy.measures as measures
import analphipy.potential as pots
from analphipy.norofrenkel import NoroFrenkelPair

data = Path(__file__).parent / "data"

nsamp = 20


def test_nf_sw():
    cols = ["sig", "eps", "lam", "B2"]
    df = pd.read_csv(data / "eff_sw_.csv").assign(beta=lambda x: 1.0 / x["temp"])

    if len(df) > nsamp:
        df = df.sample(nsamp)

    for (sig, eps, lam), g in df.groupby(["sig", "eps", "lam"]):
        p = pots.SquareWell(sig=sig, eps=eps, lam=lam)

        a = NoroFrenkelPair(phi=p.phi, segments=p.segments, r_min=sig, phi_min=eps)

        out = a.table(g["beta"], cols)

        for col in cols:
            left = col + "_eff"
            np.testing.assert_allclose(g[left], out[col])


def test_nf_lj():
    cols = ["sig", "eps", "lam", "B2", "sig_dbeta"]

    df = (
        pd.read_csv(data / "eff_lj_.csv")
        .assign(beta=lambda x: 1.0 / x["temp"])
        .drop_duplicates()
        .assign(sig_dbeta_eff=lambda x: -x["dsig_eff"])
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

        out = a.table(g["beta"], cols)

        for col in cols:
            left = col + "_eff"
            np.testing.assert_allclose(g[left], out[col], rtol=1e-3)


def test_nf_nm():
    cols = ["sig", "eps", "lam", "B2"]
    df = (
        pd.read_csv(data / "eff_lj_nm_.csv")
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

        out = a.table(g["beta"], cols)

        for col in cols:
            left = col + "_eff"
            np.testing.assert_allclose(g[left], out[col], rtol=1e-3)


def test_nf_yk():
    cols = ["sig", "eps", "lam", "B2"]

    df = (
        pd.read_csv(data / "eff_yukawa_.csv")
        .assign(beta=lambda x: 1.0 / x["temp"])
        .drop_duplicates()
    )

    if len(df) > nsamp:
        df = df.sample(nsamp)
    sig = eps = 1.0

    for (z_yukawa), g in df.groupby("z_yukawa"):
        p = pots.Yukawa(z=z_yukawa, sig=sig, eps=eps)

        a = NoroFrenkelPair(phi=p.phi, segments=p.segments, r_min=sig, phi_min=-1.0)

        out = a.table(g["beta"], cols)

        for col in cols:
            left = col + "_eff"
            np.testing.assert_allclose(g[left], out[col], rtol=1e-3)
