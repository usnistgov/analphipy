from __future__ import annotations

import os
from textwrap import dedent
from typing import Any, Callable, TypeVar

FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)

try:
    # Default is DOC_SUB is True
    DOC_SUB = os.getenv("ANALPHIPY_DOC_SUB", "True").lower() not in ("0", "f", "false")
except KeyError:
    DOC_SUB = True


# doc comes from pandas library.  placed directly here because
# 1. This is in a private module of pandas
# 2. pandas not a required dependency of this library.
#
# See https://github.com/pandas-dev/pandas/blob/main/pandas/util/_decorators.py#:~:text=def%20doc


def doc(*docstrings: None | str | Callable, **params) -> Callable[[F], F]:
    """
    A decorator take docstring templates, concatenate them and perform string
    substitution on it.
    This decorator will add a variable "_docstring_components" to the wrapped
    callable to keep track the original docstring template for potential usage.
    If it should be consider as a template, it will be saved as a string.
    Otherwise, it will be saved as callable, and later user __doc__ and dedent
    to get docstring.
    Parameters
    ----------
    *docstrings : None, str, or callable
        The string / docstring / docstring template to be appended in order
        after default docstring under callable.
    **params
        The string which would be used to format docstring template.
    """

    def decorator(decorated: F) -> F:
        # collecting docstring and docstring templates
        docstring_components: list[str | Callable] = []
        if decorated.__doc__:
            docstring_components.append(dedent(decorated.__doc__))

        for docstring in docstrings:
            if docstring is None:
                continue
            if hasattr(docstring, "_docstring_components"):
                # error: Item "str" of "Union[str, Callable[..., Any]]" has no attribute
                # "_docstring_components"
                # error: Item "function" of "Union[str, Callable[..., Any]]" has no
                # attribute "_docstring_components"
                docstring_components.extend(
                    docstring._docstring_components  # type: ignore[union-attr]
                )
            elif isinstance(docstring, str) or docstring.__doc__:
                docstring_components.append(docstring)

        params_applied = [
            component.format(**params)
            if isinstance(component, str) and len(params) > 0
            else component
            for component in docstring_components
        ]

        decorated.__doc__ = "".join(
            [
                component
                if isinstance(component, str)
                else dedent(component.__doc__ or "")
                for component in params_applied
            ]
        )

        # error: "F" has no attribute "_docstring_components"
        decorated._docstring_components = (  # type: ignore[attr-defined]
            docstring_components
        )
        return decorated

    return decorator


def docfiller(*args, **kwargs):
    """To fill common docs.

    Taken from pandas.utils._decorators
    """
    if DOC_SUB:
        return doc(*args, **kwargs)
    else:

        def decorated(func):
            return func

        return decorated


# fmt: off
_shared_docs = {
    "segments":
    """
    segments : sequence of ints
        Integration limits. For ``n = len(segmetns)`` integration will be perfomed over ranges
        ``(segments[0], segments[1]), (segments[1], segments[2]), ..., (segments[n-2], segments[n-]])``
    """,
    "phi_rep":
    """
    phi_rep : callable
        Repulsive part of pair potential.
    """,
    "beta":
    """
    beta : float
        Inverse temperature.
    """,
    "phi":
    """
    phi : callable
        Potential function.
    """,
    "r":
    """
    r : float or array-like
        Pair separation distance(s).
    """,
    "quad_kws":
    """
    quad_kws : mapping, optional
        Extra arguments to :func:`analphipy.utils.quad_segments`
    """,
    "r_min_exact":
    """
    r_min : float
        Location of minimum in potential energy.
    """,
    "phi_min_exact":
    """
    phi_min : float, optional
        Value of potential energy at minimum.
    """,
    "full_output":
    """
    full_output : bool, optional
        If True, return extra information.
    """,
    "err":
    """
    err : bool, optional
        If True, return error.
    """,
    "error_summed":
    """
    error : float, optional
        Total integration error. Returned if ``err`` or ``full_output`` are `True`.
    """,
    "full_output_summed":
    """
    outputs :
        Output(s) from :func:`scipy.integrate.quad`.  Returned if ``full_output`` is True.
    """,
    "volume_int_func":
    """
    volume : str or callable, optional
        Volume element in integration.
        For examle, use volume = lambda x: 4 * np.pi * x ** 2 for spherically symmetric 3d integration.
        Can also pass string value of {'1d', '2d','3d'}.  If passed None, then assume '1d' integration.
    """,
    # References
    "ref_Noro_Frenkel":
    """
    M.G. Noro and D. Frenkel (2000), "Extended corresponding-states behavior for particles with variable range attractions". Journal of Chemical Physics, 113, 2941.
    """,
    "ref_Barker_Henderson":
    """
    J.A. Barker and D. Henderson (1976), "What Is Liquid? Understanding the States of Matter". Reviews of Modern Physics, 48, 587-671
    """,
    "ref_WCA":
    """
    J.D. Weeks, D. Chandler and H.C. Andersen (1971), "Role of Repulsive Forces in Determining the Equilibrium Structure of Simple Liquids", Journal of Chemical Physics 54, 5237-5247
    """,
    "kl_link":
    """
    `See here for more info <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Definition>`_
    """,
    "js_link":
    """
    `See here for more info <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Symmetrised_divergence>`_
    """,


}
# fmt: on


def prepare_shared_docs(shared_docs):
    return {k: dedent(v).strip() for k, v in _shared_docs.items()}


_shared_docs = prepare_shared_docs(_shared_docs)

docfiller_shared = docfiller(**_shared_docs)


def factory_docfiller_shared(**kws):

    if len(kws) == 0:
        return docfiller_shared

    else:
        kws = dict(_shared_docs, **prepare_shared_docs(kws))
        return docfiller(**kws)
