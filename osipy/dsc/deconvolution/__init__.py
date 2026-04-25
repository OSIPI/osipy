"""DSC deconvolution module.

This module implements SVD-based deconvolution methods for estimating
the residue function and perfusion parameters including CBF
(OSIPI: Q.PH1.003), MTT (OSIPI: Q.PH1.006), and arterial delay Ta
(OSIPI: Q.PH1.007) from DSC-MRI data.

Two pathways are available:

1. **Fitter-based** (preferred): Use ``DSCConvolutionModel`` +
   ``BoundDSCModel`` with an SVD fitter (``get_fitter("oSVD")``).
2. **Legacy deconvolver**: Use ``get_deconvolver("oSVD")`` for the
   original ``BaseDeconvolver`` interface.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.29840
"""

from osipy.dsc.deconvolution.base import BaseDeconvolver
from osipy.dsc.deconvolution.registry import (
    DECONVOLUTION_REGISTRY,
    get_deconvolver,
    list_deconvolvers,
    register_deconvolver,
)
from osipy.dsc.deconvolution.signal_model import BoundDSCModel, DSCConvolutionModel
from osipy.dsc.deconvolution.svd import (
    CircularSVDDeconvolver,
    DeconvolutionResult,
    OscillationSVDDeconvolver,
    StandardSVDDeconvolver,
    SVDDeconvolutionParams,
)
from osipy.dsc.deconvolution.svd_fitters import (
    CSVDFitter,
    OSVDFitter,
    SSVDFitter,
    TikhonovFitter,
)

__all__ = [
    "DECONVOLUTION_REGISTRY",
    "BaseDeconvolver",
    "BoundDSCModel",
    "CSVDFitter",
    "CircularSVDDeconvolver",
    "DSCConvolutionModel",
    "DeconvolutionResult",
    "OSVDFitter",
    "OscillationSVDDeconvolver",
    "SSVDFitter",
    "SVDDeconvolutionParams",
    "StandardSVDDeconvolver",
    "TikhonovFitter",
    "get_deconvolver",
    "list_deconvolvers",
    "register_deconvolver",
]
