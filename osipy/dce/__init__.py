"""DCE-MRI perfusion analysis module.

This module provides tools for Dynamic Contrast-Enhanced MRI analysis,
including T1 mapping, signal-to-concentration conversion, and
pharmacokinetic model fitting following the OSIPI CAPLEX lexicon.

Available models:
    - Tofts (OSIPI: M.IC1.004)
    - Extended Tofts (OSIPI: M.IC1.005)
    - Patlak (OSIPI: M.IC1.006)
    - Two-Compartment Exchange Model (OSIPI: M.IC1.009)

Key quantities:
    - Ktrans: Volume transfer constant (OSIPI: Q.PH1.008), 1/min
    - ve: Extravascular extracellular volume fraction (OSIPI: Q.PH1.001), unitless fraction [0, 1]
    - vp: Plasma volume fraction (OSIPI: Q.PH1.001), unitless fraction [0, 1]
    - Fp: Plasma flow (OSIPI: Q.PH1.002), mL/min/100mL
    - PS: Permeability-surface area product (OSIPI: Q.PH1.004), mL/min/100mL

GPU Acceleration
----------------
This module supports GPU acceleration via CuPy when available.
Input arrays can be NumPy (CPU) or CuPy (GPU) arrays. When GPU is
available, operations are automatically accelerated.

Functions
---------
compute_t1_map
    Compute T1 map from multi-flip-angle or Look-Locker data.
signal_to_concentration
    Convert signal intensity to indicator concentration (OSIPI: Q.IC1.001).
fit_model
    Fit any registered DCE model by name.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.29840
.. [3] Tofts PS et al. J Magn Reson Imaging 1999;10(3):223-232.
"""

# T1 mapping
# Signal to concentration
from osipy.dce.concentration import (
    DCEAcquisitionParams,
    get_concentration_model,
    list_concentration_models,
    register_concentration_model,
    signal_to_concentration,
)

# High-level fitting
from osipy.dce.fitting import (
    DCEFitResult,
    fit_model,
)

# Pharmacokinetic models
from osipy.dce.models import (
    MODEL_REGISTRY,
    BasePerfusionModel,
    ExtendedToftsModel,
    ExtendedToftsParams,
    ModelParameters,
    PatlakModel,
    PatlakParams,
    ToftsModel,
    ToftsParams,
    TwoCompartmentModel,
    TwoCompartmentParams,
    get_model,
    list_models,
    register_model,
)
from osipy.dce.t1_mapping import (
    compute_t1_look_locker,
    compute_t1_map,
    compute_t1_vfa,
    get_t1_method,
    list_t1_methods,
    register_t1_method,
)

__all__ = [
    # Model registry
    "MODEL_REGISTRY",
    # Models - base
    "BasePerfusionModel",
    "DCEAcquisitionParams",
    # Fitting
    "DCEFitResult",
    "ExtendedToftsModel",
    "ExtendedToftsParams",
    "ModelParameters",
    "PatlakModel",
    "PatlakParams",
    # Models - implementations
    "ToftsModel",
    "ToftsParams",
    "TwoCompartmentModel",
    "TwoCompartmentParams",
    "compute_t1_look_locker",
    # T1 mapping
    "compute_t1_map",
    "compute_t1_vfa",
    "fit_model",
    "get_concentration_model",
    "get_model",
    "get_t1_method",
    "list_concentration_models",
    "list_models",
    "list_t1_methods",
    # Concentration registry
    "register_concentration_model",
    "register_model",
    # T1 registry
    "register_t1_method",
    # Signal to concentration
    "signal_to_concentration",
]
