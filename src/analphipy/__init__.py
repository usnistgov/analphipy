"""
Top level API :mod:`analphipy`
==============================

The top level API provides Pair potentials and analysis routines.
"""
from . import measures, norofrenkel, potential

# from .potentials_base import PhiAnalytic, PhiBase, PhiCut, PhiGeneric, PhiLFS

# updated versioning scheme
try:
    from importlib.metadata import version as _version

    __version__ = _version("analphipy")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"


__all__ = [
    # "PhiBaseCuttable",
    # "PhiBaseGenericCut",
    "potential",
    "norofrenkel",
    "measures",
    # "PhiAnalytic",
    # "PhiGeneric",
    # "LennardJones",
    # "LennardJonesNM",
    # "SquareWell",
    # "HardSphere",
    # "Yukawa",
    # "CubicTable",
    # "factory_phi",
    # "NoroFrenkelPair",
    # "measures",
    # "norofrenkel",
    "__version__",
]
