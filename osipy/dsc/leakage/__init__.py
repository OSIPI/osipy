"""DSC leakage correction module (OSIPI: P.LC1.001).

This module implements contrast agent leakage correction for DSC-MRI,
addressing T1 and T2* effects from extravasation in tumors with
blood-brain barrier breakdown. Estimates leakage coefficients K1
(OSIPI: Q.LC1.001) and K2 (OSIPI: Q.LC1.002).

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.29840
"""

from osipy.dsc.leakage.base import BaseLeakageCorrector
from osipy.dsc.leakage.correction import (
    BidirectionalCorrector,
    BSWCorrector,
    LeakageCorrectionParams,
    LeakageCorrectionResult,
    correct_leakage,
)
from osipy.dsc.leakage.registry import (
    LEAKAGE_REGISTRY,
    get_leakage_corrector,
    list_leakage_correctors,
    register_leakage_corrector,
)

__all__ = [
    "LEAKAGE_REGISTRY",
    "BSWCorrector",
    "BaseLeakageCorrector",
    "BidirectionalCorrector",
    "LeakageCorrectionParams",
    "LeakageCorrectionResult",
    "correct_leakage",
    "get_leakage_corrector",
    "list_leakage_correctors",
    "register_leakage_corrector",
]
