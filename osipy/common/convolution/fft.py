"""FFT-based convolution for uniform time grids.

This module provides FFT-based convolution operations that are
efficient for large datasets with uniform time sampling. GPU-compatible
using the array module pattern.

Includes batch operations optimized for processing many voxels
simultaneously (e.g., convolving a single AIF with many IRFs).

References
----------
    - scipy.signal.fftconvolve documentation
    - Sourbron & Buckley (2013). Tracer kinetic modelling in MRI.
      NMR in Biomedicine, 26(8), 1034-1048.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

from osipy.common.backend import get_array_module
from osipy.common.convolution.registry import register_convolution
from osipy.common.exceptions import DataValidationError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@register_convolution("fft")
def fft_convolve(
    f: NDArray[np.floating],
    h: NDArray[np.floating],
    dt: float,
    *,
    mode: Literal["full", "same", "valid"] = "same",
) -> NDArray[np.floating]:
    """Convolve two signals using FFT.

    Efficient convolution for large datasets with uniform time sampling.
    Uses FFT for O(n log n) complexity instead of O(n^2) direct convolution.

    Parameters
    ----------
    f : ndarray
        First input signal. Shape: (n,).
    h : ndarray
        Second input signal (impulse response). Shape: (m,).
    dt : float
        Time step between samples in seconds.
    mode : {"full", "same", "valid"}, default "same"
        Output mode:
        - "full": Full convolution result. Shape: (n + m - 1,).
        - "same": Output same size as first input. Shape: (n,).
        - "valid": Only fully overlapping region. Shape: (max(n, m) - min(n, m) + 1,).

    Returns
    -------
    ndarray
        Convolution result scaled by dt.

    Notes
    -----
    FFT convolution assumes periodic boundary conditions. For pharmacokinetic
    modeling, this may introduce artifacts at boundaries. Consider using
    piecewise-linear convolution (conv()) for more accurate results when:
    - Time sampling is non-uniform
    - Boundary behavior is important
    - Dealing with small datasets where O(n^2) is acceptable

    This function is GPU-compatible and will use CuPy FFT when the input
    arrays are on GPU.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.convolution import fft_convolve
    >>> dt = 0.1  # 100 ms time step
    >>> n = 1000
    >>> t = np.arange(n) * dt
    >>> f = np.exp(-t / 5)  # Input signal
    >>> h = np.exp(-t / 10)  # Impulse response
    >>> result = fft_convolve(f, h, dt, mode="same")

    References
    ----------
    .. [1] scipy.signal.fftconvolve documentation
    """
    xp = get_array_module(f)

    f = xp.asarray(f)
    h = xp.asarray(h)

    n = len(f)
    m = len(h)

    if n == 0 or m == 0:
        return xp.array([], dtype=f.dtype)

    # Compute FFT convolution
    # Zero-pad to avoid circular convolution artifacts
    fft_size = n + m - 1

    # Use next power of 2 for efficiency
    fft_size = int(2 ** math.ceil(math.log2(fft_size)))

    # Compute FFT — both numpy and cupy provide xp.fft.fft
    F = xp.fft.fft(f, n=fft_size)
    H = xp.fft.fft(h, n=fft_size)

    # Multiply in frequency domain
    Y = F * H

    # Inverse FFT
    y = xp.fft.ifft(Y)

    # Take real part (imaginary should be ~0)
    y = xp.real(y)

    # Extract appropriate portion based on mode
    if mode == "full":
        result = y[: n + m - 1]
    elif mode == "same":
        start = (m - 1) // 2
        result = y[start : start + n]
    elif mode == "valid":
        start = m - 1
        end = n
        result = y[start:end] if end > start else xp.array([], dtype=f.dtype)
    else:
        raise DataValidationError(
            f"Unknown mode: {mode}. Use 'full', 'same', or 'valid'."
        )

    # Scale by dt for physical convolution
    return result * dt


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
