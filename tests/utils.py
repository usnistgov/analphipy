from __future__ import annotations

from typing import TYPE_CHECKING

from analphipy import potential as pots
from analphipy.norofrenkel import NoroFrenkelPair

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd

    from analphipy.base_potential import PhiAbstract


def iter_phi_lj(
    table: pd.DataFrame,
    nsamp: int,
) -> Iterator[tuple[pd.DataFrame, NoroFrenkelPair]]:
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
        else:
            raise ValueError

        a = NoroFrenkelPair.from_phi(
            phi=p.phi, segments=p.segments, r_min=sig, bounds=[0.5, 1.5]
        )

        yield g, a
