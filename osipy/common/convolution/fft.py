"""FFT-based convolution for uniform time grids.

This module provides FFT-based convolution operations that are
efficient for large datasets with uniform time sampling. GPU-compatible
using the array module pattern.

Includes batch operations optimized for processing many voxels
simultaneously (e.g., convolving a single AIF with many IRFs).

References
----------
    - scipy.signal.fftconvolve documentation
    - Sourbron & Buckley (2013). Classic models for dynamic contrast-enhanced MRI.
      NMR in Biomedicine, 26(8), 1004-1027.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from osipy.common.backend import get_array_module
from osipy.common.exceptions import DataValidationError

if TYPE_CHECKING:
    from numpy.typing import NDArray


def convolve_aif(
    aif: NDArray[Any],
    impulse_response: NDArray[Any],
    dt: float = 1.0,
) -> NDArray[Any]:
    """Convolve AIF with impulse response function(s) using FFT.

    Computes the tissue concentration C(t) from the convolution of the
    arterial input function (AIF) with the impulse response function (IRF):

        C(t) = AIF(t) * IRF(t) * dt

    Handles both single-voxel and batch cases via broadcasting:
    a 1-D AIF is automatically broadcast against a 2-D IRF matrix.

    Parameters
    ----------
    aif : NDArray
        Arterial input function, shape ``(n_timepoints,)`` or
        ``(n_timepoints, 1)`` or ``(n_timepoints, n_voxels)``.
    impulse_response : NDArray
        Impulse response function, shape ``(n_timepoints,)`` or
        ``(n_timepoints, n_voxels)``.
    dt : float, optional
        Time step in seconds. Default is 1.0.

    Returns
    -------
    NDArray
        Convolution result.  Shape matches the broadcast of *aif* and
        *impulse_response* along the time axis.

    Notes
    -----
    Uses FFT-based convolution which is O(n log n) and highly
    parallelizable on GPU. The result is truncated to the same length
    as the input (causal convolution).
    """
    xp = get_array_module(aif, impulse_response)

    n_time = aif.shape[0]

    if impulse_response.shape[0] != n_time:
        msg = (
            f"Time dimension mismatch: AIF has {n_time}, "
            f"IRF has {impulse_response.shape[0]}"
        )
        raise DataValidationError(msg)

    n_fft = 2 * n_time - 1

    # Expand 1-D AIF so broadcasting works against a 2-D IRF
    if aif.ndim == 1 and impulse_response.ndim > 1:
        aif = aif[:, xp.newaxis]

    aif_fft = xp.fft.rfft(aif, n=n_fft, axis=0)
    irf_fft = xp.fft.rfft(impulse_response, n=n_fft, axis=0)

    result_fft = aif_fft * irf_fft
    result = xp.fft.irfft(result_fft, n=n_fft, axis=0)

    return result[:n_time] * dt
