"""Common utilities for osipy perfusion analysis.

This module provides shared functionality used across all perfusion
modality modules including I/O, fitting, AIF handling, signal processing,
and visualization. Parameter naming follows the OSIPI CAPLEX convention.

Submodules
----------
io
    Data loading and export (NIfTI, DICOM, BIDS).
fitting
    Model fitting infrastructure (least squares, Bayesian).
aif
    Arterial input function handling (population AIFs: M.IC2.001, M.IC2.002).
signal
    Signal processing utilities.
convolution
    Convolution and deconvolution operations for pharmacokinetic modeling.
visualization
    Plotting functions for curves and parameter maps.
validation
    Comparison against reference values.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Dickie BR et al. A community-endorsed open-source lexicon for
   contrast agent-based perfusion MRI: A consensus guidelines report from
   the ISMRM Open Science Initiative for Perfusion Imaging (OSIPI).
   MRM 2024;91(5):1761-1773. doi:10.1002/mrm.29840
"""

from osipy.common.convolution import convolve_aif, expconv
from osipy.common.dataset import PerfusionDataset
from osipy.common.exceptions import (
    AIFError,
    DataValidationError,
    FittingError,
    IOError,
    OsipyError,
)
from osipy.common.fitting.registry import get_fitter, list_fitters, register_fitter
from osipy.common.parameter_map import ParameterMap
from osipy.common.types import (
    AcquisitionParams,
    AIFType,
    ASLAcquisitionParams,
    DCEAcquisitionParams,
    DSCAcquisitionParams,
    FittingMethod,
    IVIMAcquisitionParams,
    LabelingType,
    Modality,
)

__all__ = [
    "AIFError",
    "AIFType",
    "ASLAcquisitionParams",
    # Acquisition parameters
    "AcquisitionParams",
    "DCEAcquisitionParams",
    "DSCAcquisitionParams",
    "DataValidationError",
    "FittingError",
    "FittingMethod",
    "IOError",
    "IVIMAcquisitionParams",
    "LabelingType",
    # Enums
    "Modality",
    # Exceptions
    "OsipyError",
    "ParameterMap",
    # Core data containers
    "PerfusionDataset",
    # Convolution functions
    "convolve_aif",
    "expconv",
    "get_fitter",
    "list_fitters",
    # Fitter registry
    "register_fitter",
]
