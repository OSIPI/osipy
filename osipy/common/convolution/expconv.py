"""Exponential convolution functions for pharmacokinetic modeling.

This module implements efficient recursive formulas for convolving signals
with exponential and multi-exponential functions. These are the core
building blocks for compartmental pharmacokinetic models.

GPU/CPU agnostic using the xp array module pattern.
NO scipy dependency - see XP Compatibility Requirements in plan.md.

References
----------
Flouri D, Lesnic D, Sourbron SP (2016). Fitting the two-compartment
model in DCE-MRI by linear inversion. Magn Reson Med. 76(3):998-1006.
doi:10.1002/mrm.25991

Attribution
-----------
``expconv`` adapts the recursive exponential-convolution algorithm from
dcmri v0.6.20 (https://github.com/dcmri/dcmri, module ``dcmri/utils.py``),
licensed under the Apache License, Version 2.0. Modified: ``(f, T, t)``
signature, batched/GPU array-module path, and non-normalized output. See the
project NOTICE file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from osipy.common.backend import get_array_module
from osipy.common.exceptions import DataValidationError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def expconv(
    f: NDArray[np.floating],
    T: float | NDArray[np.floating],
    t: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Convolve a signal with exponential decay function(s).

    Computes the convolution:
        (f * exp(-t/T))(t) = integral_0^t f(u) * exp(-(t-u)/T) du

    using an efficient recursive formula that avoids explicit numerical
    integration. This is the fundamental operation for compartmental
    pharmacokinetic models.

    Handles both single-voxel (scalar *T*) and batch (array *T*) cases.
    When *T* is an array, every voxel is processed in a single pass
    with the loop running over time points — efficient on CPU and GPU.

    Parameters
    ----------
    f : ndarray
        Input signal (e.g., arterial input function).
        Shape ``(n_times,)`` or ``(n_times, 1)`` or
        ``(n_times, n_voxels)``.
    T : float or ndarray
        Time constant(s) of the exponential decay in seconds.
        Scalar for a single convolution, or array of shape
        ``(n_voxels,)`` for batch convolution.
    t : ndarray
        Time points in seconds. Shape ``(n_times,)`` or
        ``(n_times, 1)``. Must be monotonically increasing.

    Returns
    -------
    ndarray
        Convolution result.  Shape ``(n_times,)`` for scalar *T*,
        ``(n_times, n_voxels)`` for array *T*.

    Notes
    -----
    The recursive formula from Flouri et al. (2016) is used:

        E[i] = E[i-1] * exp(-dt/T) + integral_{t[i-1]}^{t[i]} f(u) * exp(-(t[i]-u)/T) du

    where the integral within each interval is evaluated analytically
    assuming piecewise-linear interpolation of f.

    This is O(n) in computation time, compared to O(n^2) for naive
    numerical integration.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.convolution import expconv
    >>> t = np.linspace(0, 10, 101)
    >>> aif = np.exp(-t / 2)  # Input function
    >>> T = 5.0  # 5 second time constant
    >>> result = expconv(aif, T, t)

    References
    ----------
    .. [1] Flouri D, Lesnic D, Sourbron SP (2016). Fitting the
           two-compartment model in DCE-MRI by linear inversion.
           Magn Reson Med. 76(3):998-1006. doi:10.1002/mrm.25991
    """
    xp = get_array_module(f)

    f = xp.asarray(f)
    t = xp.asarray(t)

    # Flatten t to 1-D (may arrive as (n_time, 1) from batch predict)
    t_flat = t.ravel()
    n = len(t_flat)

    # Determine scalar vs. array T
    T_val = xp.asarray(T)
    is_scalar = T_val.ndim == 0

    if is_scalar:
        # --- Scalar T: single convolution ---
        T_scalar = float(T_val)
        f_flat = f.ravel()

        if len(f_flat) != n:
            raise DataValidationError("f and t must have the same length")
        if n == 0:
            return xp.array([], dtype=f.dtype)
        if T_scalar <= 0:
            return xp.zeros(n, dtype=f.dtype)

        dt_arr = t_flat[1:] - t_flat[:-1]
        x = dt_arr / T_scalar

        E = xp.exp(-x)
        E0 = 1.0 - E
        E1 = x - E0

        df = xp.where(xp.abs(x) > 1e-10, (f_flat[1:] - f_flat[:-1]) / x, 0.0)
        add = f_flat[:-1] * E0 + df * E1

        result = xp.zeros(n, dtype=f.dtype)
        for i in range(n - 1):
            result[i + 1] = E[i] * result[i] + add[i]

        return result * T_scalar

    # --- Array T: batch convolution ---
    T_arr = T_val
    n_voxels = len(T_arr)

    if n == 0:
        return xp.zeros((0, n_voxels), dtype=f.dtype)

    # Normalize f to 2-D so broadcasting works for all input shapes
    if f.ndim == 1:
        f = f[:, xp.newaxis]

    dt_arr = t_flat[1:] - t_flat[:-1]
    x = dt_arr[:, xp.newaxis] / T_arr[xp.newaxis, :]  # (n-1, n_voxels)

    E = xp.exp(-x)
    E0 = 1.0 - E
    E1 = x - E0

    f_diff = f[1:] - f[:-1]
    df = xp.where(xp.abs(x) > 1e-10, f_diff / x, 0.0)
    add = f[:-1] * E0 + df * E1  # (n-1, n_voxels)

    result = xp.zeros((n, n_voxels), dtype=f.dtype)
    for i in range(n - 1):
        result[i + 1] = E[i] * result[i] + add[i]

    return result * T_arr[xp.newaxis, :]
