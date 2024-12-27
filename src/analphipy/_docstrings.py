"""Common docstrings."""

from __future__ import annotations

from module_utilities.docfiller import DocFiller


def _docstrings_shared() -> None:
    """
    Parameters
    ----------
    segments : sequence of int
        Integration limits. For ``n = len(segments)`` integration will be performed over ranges
        ``(segments[0], segments[1]), (segments[1], segments[2]), ..., (segments[n-2], segments[n-]])``
    phi_rep : callable
        Repulsive part of pair potential.
    beta : float
        Inverse temperature.
    phi : callable
        Potential function.
    r : float or array-like
        Pair separation distance(s).
    quad_kws : mapping, optional
        Extra arguments to :func:`analphipy.utils.quad_segments`
    r_min_exact | r_min : float
        Location of minimum in potential energy.
    phi_min_exact | phi_min : float, optional
        Value of potential energy at minimum.
    full_output : bool, optional
        If True, return extra information.
    err : bool, optional
        If True, return error.
    error_summed | error : float, optional
        Total integration error. Returned if ``err`` or ``full_output`` are `True`.
    full_output_summed | outputs : object
        Output(s) from :func:`scipy.integrate.quad`.  Returned if ``full_output`` is True.
    volume_int_func | volume : str or callable, optional
        Volume element in integration.
        For example, use ``volume = lambda x: 4 * np.pi * x ** 2`` for spherically symmetric 3d integration.
        Can also pass string value of {'1d', '2d', '3d'}.  If passed None, then assume '1d' integration.

    """


_references = {
    "ref_Noro_Frenkel": "M.G. Noro and D. Frenkel (2000), 'Extended corresponding-states behavior for particles with variable range attractions'. Journal of Chemical Physics, 113, 2941.",
    "ref_Barker_Henderson": "J.A. Barker and D. Henderson (1976), 'What Is Liquid? Understanding the States of Matter'. Reviews of Modern Physics, 48, 587-671",
    "ref_WCA": "J.D. Weeks, D. Chandler and H.C. Andersen (1971), 'Role of Repulsive Forces in Determining the Equilibrium Structure of Simple Liquids', Journal of Chemical Physics 54, 5237-5247",
    "kl_link": "`See here for more info on Kullback & Leibeler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Definition>`_",
    "js_link": "`See here for more info on Jeffreys symmetric divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Symmetrised_divergence>`_",
}


docfiller = DocFiller.from_docstring(
    _docstrings_shared, combine_keys="parameters"
).update(**_references)
