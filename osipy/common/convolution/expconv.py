"""Exponential convolution functions for pharmacokinetic modeling.

This module implements efficient recursive formulas for convolving signals
with exponential and multi-exponential functions. These are the core
building blocks for compartmental pharmacokinetic models.

GPU/CPU agnostic using the xp array module pattern.
NO scipy dependency - see XP Compatibility Requirements in plan.md.

References
----------
Flouri D, Lesnic D, Mayrovitz HN (2016). Numerical solution of the
convolution integral equations in pharmacokinetics. Comput Methods
Biomech Biomed Engin. 19(13):1359-1367.

dcmri library: https://github.com/dcmri/dcmri
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from osipy.common.backend import get_array_module
from osipy.common.convolution.registry import register_convolution
from osipy.common.exceptions import DataValidationError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def _factorial(n: int) -> float:
    """Compute factorial using pure Python.

    GPU/CPU agnostic helper function for small n values.

    Parameters
    ----------
    n : int
        Non-negative integer.

    Returns
    -------
    float
        n! as a float.
    """
    if n < 0:
        raise DataValidationError("factorial not defined for negative values")
    if n <= 1:
        return 1.0
    result = 1.0
    for i in range(2, n + 1):
        result *= i
    return result


@register_convolution("exponential")
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
    .. [1] Flouri D, Lesnic D, Mayrovitz HN (2016). Comput Methods
           Biomech Biomed Engin. 19(13):1359-1367.
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


def biexpconv(
    f: NDArray[np.floating],
    T1: float,
    T2: float,
    t: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Convolve a signal with a bi-exponential function.

    Computes the convolution of f with the bi-exponential function:
        h(t) = (exp(-t/T1) - exp(-t/T2)) / (T1 - T2)

    This is useful for two-compartment exchange models.

    Parameters
    ----------
    f : ndarray
        Input signal. Shape: (n_times,).
    T1 : float
        First time constant in seconds. Must be positive.
    T2 : float
        Second time constant in seconds. Must be positive and different from T1.
    t : ndarray
        Time points in seconds. Shape: (n_times,).

    Returns
    -------
    ndarray
        Convolution result. Shape: (n_times,).

    Notes
    -----
    The bi-exponential convolution is computed as:
        biexpconv(f, T1, T2, t) = (expconv(f, T1, t) - expconv(f, T2, t)) / (T1 - T2)

    For T1 -> T2, uses the limiting form involving the derivative.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.convolution import biexpconv
    >>> t = np.linspace(0, 10, 101)
    >>> aif = np.exp(-t / 2)
    >>> result = biexpconv(aif, 3.0, 5.0, t)

    References
    ----------
    .. [1] dcmri library: https://github.com/dcmri/dcmri
    """
    xp = get_array_module(f)

    if T1 <= 0 or T2 <= 0:
        return xp.zeros(len(t), dtype=xp.asarray(f).dtype)

    # Handle case where T1 == T2
    if abs(T1 - T2) < 1e-10 * max(T1, T2):
        # Limiting case: derivative of expconv with respect to T
        return _biexpconv_limit(f, T1, t)

    # Standard case
    E1 = expconv(f, T1, t)
    E2 = expconv(f, T2, t)

    return (E1 - E2) / (T1 - T2)


def _biexpconv_limit(
    f: NDArray[np.floating],
    T: float,
    t: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Bi-exponential convolution in the limit T1 -> T2.

    This computes the derivative of expconv with respect to T:
        d/dT [expconv(f, T, t)]
    """
    xp = get_array_module(f)

    f = xp.asarray(f)
    t = xp.asarray(t)

    # Use numerical differentiation with small delta
    delta = T * 1e-6
    E_plus = expconv(f, T + delta, t)
    E_minus = expconv(f, T - delta, t)

    return (E_plus - E_minus) / (2 * delta)


def nexpconv(
    f: NDArray[np.floating],
    T: float,
    n_exp: int,
    t: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Convolve a signal with an n-exponential (gamma variate) function.

    Computes the convolution of f with the n-exponential function:
        h(t) = (t/T)^(n-1) * exp(-t/T) / (T * (n-1)!)

    This represents a chain of n identical compartments, each with
    time constant T. Also known as the gamma variate function.

    Parameters
    ----------
    f : ndarray
        Input signal. Shape: (n_times,).
    T : float
        Time constant of each compartment in seconds. Must be positive.
    n_exp : int
        Number of exponentials (compartments). Must be >= 1.
    t : ndarray
        Time points in seconds. Shape: (n_times,).

    Returns
    -------
    ndarray
        Convolution result. Shape: (n_times,).

    Notes
    -----
    For n=1, this is equivalent to expconv.
    For n>1, the function is computed recursively:
        nexpconv(f, T, n, t) = expconv(nexpconv(f, T, n-1, t), T, t) / T

    For large n, the gamma variate approaches a Gaussian centered at
    t = n*T with standard deviation sqrt(n)*T.

    For n > 20, numerical overflow may occur in the factorial term.
    In this case, a Gaussian approximation is used.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.convolution import nexpconv
    >>> t = np.linspace(0, 20, 201)
    >>> aif = np.exp(-t / 2)
    >>> result = nexpconv(aif, 2.0, 3, t)  # 3-compartment chain

    References
    ----------
    .. [1] dcmri library: https://github.com/dcmri/dcmri
    """
    xp = get_array_module(f)

    if T <= 0 or n_exp < 1:
        return xp.zeros(len(t), dtype=xp.asarray(f).dtype)

    if n_exp == 1:
        return expconv(f, T, t)

    # For large n, use Gaussian approximation to avoid overflow
    if n_exp > 20:
        return _nexpconv_gaussian(f, T, n_exp, t)

    # Recursive computation
    result = expconv(f, T, t)
    for _ in range(2, n_exp + 1):
        result = expconv(result, T, t) / T

    # Normalize by factorial
    norm = _factorial(n_exp - 1)
    return result * T / norm


def _nexpconv_gaussian(
    f: NDArray[np.floating],
    T: float,
    n_exp: int,
    t: NDArray[np.floating],
) -> NDArray[np.floating]:
    """N-exponential convolution using Gaussian approximation.

    For large n, the gamma variate function approaches a Gaussian.
    The gamma variate h(t) = t^(n-1) * exp(-t/T) / (T^n * (n-1)!)
    has mean n*T and std sqrt(n)*T.
    """
    xp = get_array_module(f)

    from osipy.common.convolution.conv import conv

    f = xp.asarray(f)
    t = xp.asarray(t)

    # Gamma variate parameters
    mean = n_exp * T
    std = math.sqrt(n_exp) * T

    # Create Gaussian impulse response centered at the mean
    # The impulse response should be causal (zero for t < 0)
    # and peaked at t = mean
    h = xp.exp(-((t - mean) ** 2) / (2 * std**2))

    # Normalize so integral equals 1 (approximately)
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    h = h / (xp.sum(h) * dt + 1e-20)

    return conv(f, h, t)
