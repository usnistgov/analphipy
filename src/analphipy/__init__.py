"""
Top level API :mod:`analphipy`
==============================

The top level API provides Pair potentials and analysis routines.
"""
# from . import measures, norofrenkel
from .norofrenkel import NoroFrenkelPair
from .potentials import (
    CubicTable,
    Phi_hs,
    Phi_lj,
    Phi_nm,
    Phi_sw,
    Phi_yk,
    PhiAnalytic,
    PhiGeneric,
    factory_phi,
)

# from .potentials_base import PhiAnalytic, PhiBase, PhiCut, PhiGeneric, PhiLFS

# updated versioning scheme
try:
    from importlib.metadata import version as _version
except ImportError:
    # if the fallback library is missing, we are doomed.
    from importlib_metadata import version as _version  # type: ignore[no-redef]

try:
    __version__ = _version("analphipy")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"


__all__ = [
    # "PhiBaseCuttable",
    # "PhiBaseGenericCut",
    "PhiAnalytic",
    "PhiGeneric",
    "Phi_lj",
    "Phi_nm",
    "Phi_sw",
    "Phi_hs",
    "Phi_yk",
    "CubicTable",
    "factory_phi",
    "NoroFrenkelPair",
    # "measures",
    # "norofrenkel",
    "__version__",
]
