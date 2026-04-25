"""DSC-MRI perfusion analysis module.

This module provides tools for Dynamic Susceptibility Contrast MRI (DSC-MRI)
analysis following OSIPI CAPLEX naming conventions, including:

- Signal-to-concentration conversion via delta-R2* (OSIPI: Q.EL1.009)
- Leakage correction (OSIPI: P.LC1.001) with K1/K2 estimation
- SVD deconvolution (sSVD, cSVD, oSVD) for residue function recovery
- Perfusion parameter maps: CBF (Q.PH1.003), MTT (Q.PH1.006),
  TTP (Q.CD1.010), and arterial delay Ta (Q.PH1.007)

Use ``get_deconvolver()`` and ``get_leakage_corrector()`` for
registry-driven dispatch to deconvolution and leakage methods.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Ostergaard L et al. (1996). High resolution measurement of cerebral blood
   flow using intravascular tracer bolus passages. Magn Reson Med 36(5):715-725.
.. [3] Dickie BR et al. MRM 2024. doi:10.1002/mrm.29840
"""

# Signal to concentration
# Bolus arrival time detection
from osipy.dsc.arrival import (
    get_arrival_detector,
    list_arrival_detectors,
    register_arrival_detector,
)
from osipy.dsc.concentration import (
    DSCAcquisitionParams,
    delta_r2_to_concentration,
    gamma_variate_fit,
    signal_to_delta_r2,
)

# Deconvolution
from osipy.dsc.deconvolution import (
    DeconvolutionResult,
    SVDDeconvolutionParams,
    get_deconvolver,
    list_deconvolvers,
)

# Leakage correction
from osipy.dsc.leakage import (
    LeakageCorrectionParams,
    LeakageCorrectionResult,
    correct_leakage,
    get_leakage_corrector,
    list_leakage_correctors,
)

# Normalization
from osipy.dsc.normalization import (
    NormalizationResult,
    compute_relative_cbv,
    get_normalizer,
    list_normalizers,
    normalize_to_white_matter,
    register_normalizer,
)

# Perfusion parameters
from osipy.dsc.parameters import (
    DSCPerfusionMaps,
    compute_cbv,
    compute_mtt,
    compute_perfusion_maps,
)

__all__ = [
    "DSCAcquisitionParams",
    "DSCPerfusionMaps",
    "DeconvolutionResult",
    "LeakageCorrectionParams",
    "LeakageCorrectionResult",
    "NormalizationResult",
    # Deconvolution
    "SVDDeconvolutionParams",
    "compute_cbv",
    "compute_mtt",
    # Parameters
    "compute_perfusion_maps",
    "compute_relative_cbv",
    # Leakage correction
    "correct_leakage",
    "delta_r2_to_concentration",
    "gamma_variate_fit",
    "get_arrival_detector",
    "get_deconvolver",
    "get_leakage_corrector",
    "get_normalizer",
    "list_arrival_detectors",
    "list_deconvolvers",
    "list_leakage_correctors",
    "list_normalizers",
    # Normalization
    "normalize_to_white_matter",
    # Arrival detection
    "register_arrival_detector",
    "register_normalizer",
    # Signal/Concentration
    "signal_to_delta_r2",
]
