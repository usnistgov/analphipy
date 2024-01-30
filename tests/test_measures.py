# mypy: disable-error-code="no-untyped-def, no-untyped-call"

from pathlib import Path
from typing import TYPE_CHECKING

# from typing import Sequence, Any
# from typing_extensions import reveal_type
import numpy as np
import pandas as pd

import analphipy.potential as pots
from analphipy import measures
from analphipy.norofrenkel import NoroFrenkelPair

if TYPE_CHECKING:
    from analphipy.base_potential import PhiAbstract

data = Path(__file__).parent / "data"

nsamp = 20

# def test_typing():
#     table = (
#         pd.read_csv(data / "vir_sw_.csv")
#         .assign(beta=lambda x: 1.0 / x["temp"])
#         .drop_duplicates()
#     )

#     def hello(x: Sequence[float]) -> Sequence[float]:
#         return x * 2


#     reveal_type(table)
#     # reveal_type(table["sig"].to_numpy(dtype=np.float_))
#     reveal_type(np.asarray(table['sig'], dtype=np.float_))
#     reveal_type(np.array(table['sig'], dtype=np.float_))
#     reveal_type(np.array([1,2,3], dtype=np.float_))
#     reveal_type(np.array([1,2,3], dtype=float))
#     # reveal_type(hello)
#     # reveal_type(hello(table['sig'].to_numpy(dtype=float)))


def test_B2_sw() -> None:
    table = (
        pd.read_csv(data / "vir_sw_.csv")
        .assign(beta=lambda x: 1.0 / x["temp"])
        .drop_duplicates()
    )

    if len(table) > nsamp:
        table = table.sample(nsamp)
    for (sig, eps, lam), g in table.groupby(["sig", "eps", "lam"]):
        p = pots.SquareWell(sig=sig, eps=eps, lam=lam)

        a = NoroFrenkelPair(phi=p.phi, segments=p.segments, r_min=sig, phi_min=eps)

        out = a.table(np.asarray(g["beta"], dtype=np.float64), ["B2", "B2_dbeta"])

        np.testing.assert_allclose(g["Bn_2"], out["B2"])
        np.testing.assert_allclose(-g["dBn_2"], out["B2_dbeta"])


def test_B2_lj() -> None:
    table = (
        pd.read_csv(data / "vir_lj_.csv")
        .assign(beta=lambda x: 1.0 / x["temp"])
        .drop_duplicates()
        .sample(40)
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

        out = a.table(np.asarray(g["beta"], dtype=np.float64), ["B2", "B2_dbeta"])

        np.testing.assert_allclose(g["Bn_2"], out["B2"], rtol=1e-3)
        np.testing.assert_allclose(-g["dBn_2"], out["B2_dbeta"], rtol=1e-3)


def test_B2_nm() -> None:
    table = (
        pd.read_csv(data / "vir_lj_nm_.csv")
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

        out = a.table(np.asarray(g["beta"], dtype=np.float64), ["B2", "B2_dbeta"])

        np.testing.assert_allclose(g["Bn_2"], out["B2"], rtol=1e-3)
        np.testing.assert_allclose(-g["dBn_2"], out["B2_dbeta"], rtol=1e-3)


def test_B2_yk() -> None:
    table = (
        pd.read_csv(data / "vir_yukawa_.csv")
        .assign(beta=lambda x: 1.0 / x["temp"])
        .drop_duplicates()
    )

    if len(table) > nsamp:
        table = table.sample(nsamp)
    sig = eps = 1.0

    for (z_yukawa), g in table.groupby("z_yukawa"):
        assert isinstance(z_yukawa, float)
        p = pots.Yukawa(z=z_yukawa, sig=sig, eps=eps)

        a = NoroFrenkelPair(phi=p.phi, segments=p.segments, r_min=sig, phi_min=-1.0)

        out = a.table(np.asarray(g["beta"], dtype=np.float64), ["B2", "B2_dbeta"])

        np.testing.assert_allclose(g["Bn_2"], out["B2"], rtol=1e-6)
        np.testing.assert_allclose(-g["dBn_2"], out["B2_dbeta"], rtol=1e-6)


def test_B2_hs() -> None:
    rng = np.random.default_rng()
    sig = rng.random()

    p = pots.HardSphere(sig=sig)

    B2a = 2 * np.pi / 3.0 * sig**3

    B2b = measures.secondvirial(p.phi, beta=1.0, segments=[0.0, sig])

    assert isinstance(B2b, float)

    np.testing.assert_allclose(B2a, B2b)

    B2_dbeta = measures.secondvirial_dbeta(p.phi, beta=1.0, segments=[0.0, sig])

    assert isinstance(B2_dbeta, float)

    np.testing.assert_allclose(0.0, B2_dbeta)
