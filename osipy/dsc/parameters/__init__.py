"""DSC perfusion parameter computation.

This module provides functions for computing cerebral perfusion
parameters from DSC-MRI data, including CBV, CBF (OSIPI: Q.PH1.003),
MTT (OSIPI: Q.PH1.006), TTP (OSIPI: Q.CD1.010), and arterial delay
Ta (OSIPI: Q.PH1.007).

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.29840
"""

from osipy.dsc.parameters.maps import (
    DSCPerfusionMaps,
    compute_cbv,
    compute_mtt,
    compute_perfusion_maps,
)

__all__ = [
    "DSCPerfusionMaps",
    "compute_cbv",
    "compute_mtt",
    "compute_perfusion_maps",
]
