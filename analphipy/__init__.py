# type: ignore
# from ._potentials import (
#     PhiBaseCuttable,
#     PhiBaseGenericCut,
#     PhiCut,
#     PhiLFS,
# )

from . import measures, norofrenkel
from .norofrenkel import NoroFrenkelPair
from .potentials import CubicTable, Phi_hs, Phi_lj, Phi_nm, Phi_sw, Phi_yk, factory_phi
from .potentials_base import PhiAnalytic, PhiBase, PhiCut, PhiGeneric, PhiLFS

try:
    import pkg_resources

    __version__ = pkg_resources.get_distribution("MATS").version
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"


__all__ = [
    # "PhiBaseCuttable",
    # "PhiBaseGenericCut",
    "PhiCut",
    "PhiLFS",
    "PhiBase",
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
    "measures",
    "norofrenkel",
    "__version__",
]
