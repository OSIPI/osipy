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
.. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.30101
"""

from osipy.common.convolution import (
    biexpconv,
    conv,
    convmat,
    deconv,
    expconv,
    fft_convolve,
    invconvmat,
    nexpconv,
    uconv,
)
from osipy.common.convolution.registry import (
    get_convolution,
    list_convolutions,
    register_convolution,
)
from osipy.common.dataset import PerfusionDataset
from osipy.common.exceptions import (
    AIFError,
    DataValidationError,
    FittingError,
    IOError,
    MetadataError,
    OsipyError,
    ValidationError,
)
from osipy.common.fitting.registry import get_fitter, list_fitters, register_fitter
from osipy.common.parameter_map import ParameterMap
from osipy.common.types import (
    AcquisitionParams,
    AIFType,
    ASLAcquisitionParams,
    AnalysisResult,
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
    # Result contract
    "AnalysisResult",
    "DCEAcquisitionParams",
    "DSCAcquisitionParams",
    "DataValidationError",
    "FittingError",
    "FittingMethod",
    "IOError",
    "IVIMAcquisitionParams",
    "LabelingType",
    "MetadataError",
    # Enums
    "Modality",
    # Exceptions
    "OsipyError",
    "ParameterMap",
    # Core data containers
    "PerfusionDataset",
    "ValidationError",
    "biexpconv",
    # Convolution functions
    "conv",
    "convmat",
    "deconv",
    "expconv",
    "fft_convolve",
    "get_convolution",
    "get_fitter",
    "invconvmat",
    "list_convolutions",
    "list_fitters",
    "nexpconv",
    # Convolution registry
    "register_convolution",
    # Fitter registry
    "register_fitter",
    "uconv",
]
