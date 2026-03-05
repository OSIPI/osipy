"""IVIM (Intravoxel Incoherent Motion) analysis module.

This module provides tools for IVIM-DWI analysis, implementing bi-exponential
signal models to separate diffusion from perfusion contributions.

Functions
---------
fit_ivim
    Fit IVIM bi-exponential model to DWI data.

References
----------
Le Bihan D et al. (1988). Separation of diffusion and perfusion in intravoxel
incoherent motion MR imaging. Radiology 168(2):497-505.

Federau C (2017). Intravoxel incoherent motion MRI as a means to measure
in vivo perfusion: A review of the evidence. NMR Biomed 30(11):e3780.
"""

# Models
# Fitting
from osipy.ivim.fitting import (
    FittingMethod,
    IVIMFitParams,
    IVIMFitResult,
    fit_ivim,
)
from osipy.ivim.fitting.registry import (
    get_ivim_fitter,
    list_ivim_fitters,
    register_ivim_fitter,
)
from osipy.ivim.models import (
    IVIMBiexponentialModel,
    IVIMModel,
    IVIMParams,
    IVIMSimplifiedModel,
    get_ivim_model,
    register_ivim_model,
)

__all__ = [
    "FittingMethod",
    "IVIMBiexponentialModel",
    "IVIMFitParams",
    "IVIMFitResult",
    # Models
    "IVIMModel",
    "IVIMParams",
    "IVIMSimplifiedModel",
    # Fitting
    "fit_ivim",
    "get_ivim_fitter",
    # Registry
    "get_ivim_model",
    "list_ivim_fitters",
    # IVIM fitter registry
    "register_ivim_fitter",
    "register_ivim_model",
]
