from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import minimize

Seq_float = Union[Sequence[float], NDArray[np.float_]]

TWO_PI = 2.0 * np.pi


def combine_segmets(a: Seq_float, b: Seq_float) -> list[float]:
    aa, bb = set(a), set(b)
    return sorted(aa.union(bb))


def quad_segments(
    func: Callable,
    segments: Seq_float,
    args: tuple = (),
    full_output: bool = False,
    sum_integrals: bool = True,
    sum_errors: bool = False,
    err: bool = True,
    **kws
):
    """
    perform quadrature with discontinuities

    Parameters
    ----------
    func : callable
        function to be integrated
    segments : list
        integration limits.  If len(segments) == 2, then straight quad segments.  Otherwise,
        calculate integral from [segments[0], segments[1]], [segments[1], segments[2]], ....




    """

    out = [
        quad(func, a=a, b=b, args=args, full_output=full_output, **kws)
        for a, b in zip(segments[:-1], segments[1:])
    ]

    if len(segments) == 2:
        if full_output:
            integrals, errors, outputs = out[0]
        else:
            integrals, errors = out[0]
            outputs = None

    else:
        integrals = []
        errors = []

        if full_output:
            outputs = []
            for y, e, o in out:
                integrals.append(y)
                errors.append(e)
                outputs.append(o)
        else:
            outputs = None
            for y, e in out:
                integrals.append(y)
                errors.append(e)

        if sum_integrals:
            integrals = np.sum(integrals)

        if sum_errors:
            errors = np.sum(errors)

    # Gather final results
    result = [integrals]
    if err:
        result.append(errors)
    if outputs is not None:
        result.append(outputs)

    if len(result) == 1:
        return result[0]
    else:
        return result


def minimize_phi(
    phi: Callable, r0: float, bounds: Optional[Seq_float] = None, **kws
) -> tuple[float, float, Any]:

    if bounds is None:
        bounds = (0.0, np.inf)

    if bounds[0] == bounds[1]:
        # force minima to be exactly here
        xmin = bounds[0]
        ymin = phi(xmin)
        outputs = None
    else:
        outputs = minimize(phi, r0, bounds=[bounds], **kws)

        if not outputs.success:
            raise ValueError("could not find min of phi")

        xmin = outputs.x[0]
        ymin = outputs.fun[0]

    return xmin, ymin, outputs
