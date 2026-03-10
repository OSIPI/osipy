"""DSC signal-to-concentration conversion.

This module provides functions for converting DSC-MRI signal intensity
to contrast agent concentration via delta-R2* (OSIPI: Q.EL1.009)
relaxation rate changes. Requires echo time TE (OSIPI: Q.MS1.005).

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.30101
"""

from osipy.dsc.concentration.signal_to_conc import (
    DSCAcquisitionParams,
    delta_r2_to_concentration,
    gamma_variate_fit,
    signal_to_delta_r2,
)

__all__ = [
    "DSCAcquisitionParams",
    "delta_r2_to_concentration",
    "gamma_variate_fit",
    "signal_to_delta_r2",
]
