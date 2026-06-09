"""DSC signal-to-concentration conversion.

This module provides functions for converting DSC-MRI signal intensity
to contrast agent concentration via delta-R2* (OSIPI: Q.EL1.009)
relaxation rate changes. Requires echo time TE (OSIPI: Q.MS1.005).

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Dickie BR et al. A community-endorsed open-source lexicon for contrast
       agent-based perfusion MRI: A consensus guidelines report from the ISMRM
       Open Science Initiative for Perfusion Imaging (OSIPI). MRM 2024;91(5):
       1761-1773. doi:10.1002/mrm.29840
"""

from osipy.dsc.concentration.signal_to_conc import (
    delta_r2_to_concentration,
    gamma_variate_fit,
    signal_to_delta_r2,
)

__all__ = [
    "delta_r2_to_concentration",
    "gamma_variate_fit",
    "signal_to_delta_r2",
]
