"""
Top level API :mod:`analphipy`
==============================

The top level API provides Pair potentials and analysis routines.
"""
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from . import measures, norofrenkel, potential

try:
    __version__ = _version("analphipy")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "999"

__author__ = """William P. Krekelberg"""
__email__ = "wpk@nist.gov"

__all__ = [
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
    "measures",
    "norofrenkel",
    # "PhiBaseCuttable",
    # "PhiBaseGenericCut",
    "potential",
]
