# mypy: disable-error-code="no-untyped-def, no-untyped-call"
"""Simple tests for js divergence..."""
import numpy as np
import pytest

import analphipy
from analphipy.measures import diverg_js_cont


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("beta_other", [0.5, 1.0])
def test_boltz_js_lj(beta, beta_other):
    p0 = analphipy.potential.LennardJones(sig=1.0, eps=1.0)
    p1 = analphipy.potential.LennardJonesNM(n=14, m=7, sig=1.0, eps=1)

    v0 = p0.to_measures().boltz_diverg_js(p1, beta=beta, beta_other=beta_other)

    f0 = lambda x: np.exp(-beta * p0.phi(x))
    f1 = lambda x: np.exp(-beta_other * p1.phi(x))

    v1 = diverg_js_cont(
        f0, f1, segments=p0.segments, segments_q=p1.segments, volume="3d"
    )

    np.testing.assert_allclose(v0, v1)  # type: ignore
