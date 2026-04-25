"""Pharmacokinetic models for DCE-MRI.

This module provides implementations of standard tracer kinetic
models for DCE-MRI analysis following the OSIPI CAPLEX lexicon.

Available models:
    - Tofts (OSIPI: M.IC1.004)
    - Extended Tofts (OSIPI: M.IC1.005)
    - Patlak (OSIPI: M.IC1.006)
    - Two-Compartment Exchange Model (OSIPI: M.IC1.009)
    - Two-Compartment Uptake Model (2CUM)

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.29840
.. [3] Sourbron SP, Buckley DL. MRM 2011;66(3):735-745.
"""

from osipy.dce.models.base import BasePerfusionModel, ModelParameters

# Import model classes AFTER registry is set up so @register_model decorators fire
from osipy.dce.models.extended_tofts import ExtendedToftsModel, ExtendedToftsParams
from osipy.dce.models.patlak import PatlakModel, PatlakParams

# Import registry first (empty dict initialized on import)
from osipy.dce.models.registry import (
    MODEL_REGISTRY,
    get_model,
    list_models,
    register_model,
)
from osipy.dce.models.tofts import ToftsModel, ToftsParams
from osipy.dce.models.two_compartment import TwoCompartmentModel, TwoCompartmentParams
from osipy.dce.models.two_compartment_uptake import (
    TwoCompartmentUptakeModel,
    TwoCompartmentUptakeParams,
)

__all__ = [
    "MODEL_REGISTRY",
    "BasePerfusionModel",
    "ExtendedToftsModel",
    "ExtendedToftsParams",
    "ModelParameters",
    "PatlakModel",
    "PatlakParams",
    "ToftsModel",
    "ToftsParams",
    "TwoCompartmentModel",
    "TwoCompartmentParams",
    "TwoCompartmentUptakeModel",
    "TwoCompartmentUptakeParams",
    "get_model",
    "list_models",
    "register_model",
]
