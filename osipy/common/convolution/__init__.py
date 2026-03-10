"""Convolution and deconvolution operations for pharmacokinetic modeling.

This module provides accurate numerical convolution and deconvolution operations
following the dcmri approach (https://github.com/dcmri/dcmri) for improved
numerical accuracy, particularly for non-uniform time grids.

Key functions:
    - conv(): Piecewise-linear convolution with analytic integration
    - uconv(): Optimized convolution for uniform time grids
    - expconv(): Recursive exponential convolution (Flouri et al. 2016)
    - biexpconv(): Analytical bi-exponential convolution
    - nexpconv(): N-exponential gamma-variate convolution
    - deconv(): Matrix-based deconvolution with TSVD/Tikhonov regularization
    - convmat(): Convolution matrix construction
    - invconvmat(): Regularized pseudo-inverse of convolution matrix
    - fft_convolve(): FFT-based convolution for large uniform datasets

References
----------
.. [1] Flouri D, Lesnic D, Mayrovitz HN (2016). Numerical solution of the
   convolution integral equations in pharmacokinetics. Comput Methods
   Biomech Biomed Engin. 19(13):1359-1367.
.. [2] Sourbron & Buckley (2013). Tracer kinetic modelling in MRI.
   NMR Biomed. 26(8):1034-1048.
.. [3] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
"""

from osipy.common.convolution.conv import conv, uconv
from osipy.common.convolution.deconv import (
    deconv,
    deconvolve_svd,
    deconvolve_svd_batch,
)
from osipy.common.convolution.expconv import biexpconv, expconv, nexpconv
from osipy.common.convolution.fft import convolve_aif, fft_convolve
from osipy.common.convolution.matrix import convmat, invconvmat
from osipy.common.convolution.registry import (
    get_convolution,
    list_convolutions,
    register_convolution,
)

__all__ = [
    "biexpconv",
    "conv",
    "convmat",
    "convolve_aif",
    "deconv",
    "deconvolve_svd",
    "deconvolve_svd_batch",
    "expconv",
    "fft_convolve",
    "get_convolution",
    "invconvmat",
    "list_convolutions",
    "nexpconv",
    # Registry
    "register_convolution",
    "uconv",
]
