"""Convolution operations for pharmacokinetic modeling.

Key functions:
    - expconv(): Recursive exponential convolution (Flouri et al. 2016)
    - convolve_aif(): FFT-based AIF/impulse-response convolution

References
----------
.. [1] Flouri D, Lesnic D, Sourbron SP (2016). Fitting the two-compartment
   model in DCE-MRI by linear inversion. Magn Reson Med. 76(3):998-1006.
   doi:10.1002/mrm.25991
.. [2] Sourbron & Buckley (2013). Classic models for dynamic
   contrast-enhanced MRI. NMR Biomed. 26(8):1004-1027.
   doi:10.1002/nbm.2940
.. [3] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/

The expconv function in this package is a derivative work adapted from dcmri
(Apache-2.0); see the project NOTICE file for attribution.
"""

from osipy.common.convolution.expconv import expconv
from osipy.common.convolution.fft import convolve_aif

__all__ = ["convolve_aif", "expconv"]
