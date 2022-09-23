from ._potentials import (  # type: ignore
    PhiBaseCuttable,
    PhiBaseGenericCut,
    PhiCut,
    PhiLFS,
)
from .norofrenkel import NoroFrenkelPair
from .potentials import Phi_hs, Phi_lj, Phi_nm, Phi_sw, Phi_yk, factory_phi

try:
    import pkg_resources

    __version__ = pkg_resources.get_distribution("MATS").version
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"


__all__ = [
    "PhiBaseCuttable",
    "PhiBaseGenericCut",
    "PhiCut",
    "PhiLFS",
    "NoroFrenkelPair",
    "Phi_lj",
    "Phi_nm",
    "Phi_sw",
    "Phi_hs",
    "Phi_yk",
    "factory_phi",
    "__version__",
]
