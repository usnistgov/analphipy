import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

TWO_PI = 2.0 * np.pi


def combine_segmets(a, b):
    a, b = set(a), set(b)
    return sorted(a.union(b))


def quad_segments(
    func,
    segments,
    args=(),
    full_output=0,
    sum_integrals=True,
    sum_errors=False,
    err=True,
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


def minimize_phi(phi, r0, bounds=None, **kws):

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
