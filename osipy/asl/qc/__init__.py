"""ASL Quality Control sub-module.

Provides tools for assessing the quality of ASL perfusion MRI acquisitions,
following the OSIPI ASL Lexicon and ISMRM Perfusion Study Group guidelines.

Included metrics
----------------
- **SNR**: Signal-to-Noise Ratio of the CBF map (``compute_snr``).

References
----------
.. [1] Alsop DC et al. (2015). Recommended implementation of arterial
   spin-labeled perfusion MRI for clinical applications.
   Magn Reson Med 73(1):102-116. doi:10.1002/mrm.25197
"""

from osipy.asl.qc.snr import (
    ASLSNRParams,
    ASLSNRResult,
    compute_snr,
)

__all__ = [
    "ASLSNRParams",
    "ASLSNRResult",
    "compute_snr",
]
