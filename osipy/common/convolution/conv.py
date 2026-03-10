"""Piecewise-linear convolution with analytic integration.

This module implements convolution using piecewise-linear interpolation
with analytical integration between time points. This approach is more
accurate than FFT-based methods for pharmacokinetic modeling, particularly
when time sampling is non-uniform.

References
----------
    - dcmri (https://github.com/dcmri/dcmri) utils.conv()
    - Sourbron & Buckley (2013). Tracer kinetic modelling in MRI.
      NMR Biomed. 26(8):1034-1048.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from osipy.common.backend import get_array_module
from osipy.common.convolution.registry import register_convolution
from osipy.common.exceptions import DataValidationError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@register_convolution("piecewise_linear")
def conv(
    f: NDArray[np.floating],
    h: NDArray[np.floating],
    t: NDArray[np.floating],
    *,
    dt: float | None = None,
) -> NDArray[np.floating]:
    """Convolve two signals using piecewise-linear integration.

    Computes the convolution integral:
        (f * h)(t) = integral_0^t f(u) * h(t - u) du

    using piecewise-linear interpolation and analytical integration
    between time points. This method is accurate for non-uniform time
    grids and preserves endpoint behavior.

    Parameters
    ----------
    f : ndarray
        Input signal (e.g., arterial input function). Shape: (n_times,).
    h : ndarray
        Impulse response function (e.g., residue function). Shape: (n_times,).
    t : ndarray
        Time points in seconds. Shape: (n_times,). Must be monotonically
        increasing and start at 0.
    dt : float, optional
        If provided, assumes uniform time grid with this step size.
        Enables optimized computation path.

    Returns
    -------
    ndarray
        Convolution result at each time point. Shape: (n_times,).

    Notes
    -----
    The convolution is computed using the trapezoidal rule with
    piecewise-linear interpolation. For each time point t[i], the
    integral is computed as:

        conv[i] = sum_{j=0}^{i-1} integral_{t[j]}^{t[j+1]} f(u) * h(t[i]-u) du

    where the integral within each interval is evaluated analytically
    assuming linear interpolation of both f and h.

    For uniform time grids (when dt is provided), a more efficient
    algorithm is used.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.convolution import conv
    >>> t = np.linspace(0, 10, 101)  # 0 to 10 seconds
    >>> aif = np.exp(-t / 2)  # Exponential decay AIF
    >>> irf = np.exp(-t / 5)  # Exponential IRF
    >>> result = conv(aif, irf, t)

    References
    ----------
    .. [1] dcmri library: https://github.com/dcmri/dcmri
    """
    xp = get_array_module(f)

    f = xp.asarray(f)
    h = xp.asarray(h)
    t = xp.asarray(t)

    n = len(t)
    if len(f) != n or len(h) != n:
        raise DataValidationError("f, h, and t must have the same length")

    if n == 0:
        return xp.array([], dtype=f.dtype)

    if n == 1:
        return xp.array([0.0], dtype=f.dtype)

    # Check for uniform time grid
    if dt is not None:
        return uconv(f, h, dt)

    # Check if time grid is approximately uniform
    dt_arr = xp.diff(t)
    dt_mean = xp.mean(dt_arr)
    if xp.allclose(dt_arr, dt_mean, rtol=1e-6):
        return uconv(f, h, float(dt_mean))

    # Non-uniform time grid: use full piecewise-linear integration
    return _conv_nonuniform(f, h, t)


def _conv_nonuniform(
    f: NDArray[np.floating],
    h: NDArray[np.floating],
    t: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Convolution for non-uniform time grids.

    Uses piecewise-linear integration with analytical evaluation
    of the integral within each time interval.
    """
    xp = get_array_module(f)
    n = len(t)
    result = xp.zeros(n, dtype=f.dtype)

    # Convolution using trapezoidal integration
    # For each output point i, sum contributions from intervals [j, j+1]
    for i in range(1, n):
        total = 0.0
        for j in range(i):
            # Time interval
            t0 = t[j]
            t1 = t[j + 1] if j + 1 < n else t[j]
            dt_j = t1 - t0

            if dt_j <= 0:
                continue

            # Linear interpolation coefficients for f on [t0, t1]
            f0 = f[j]
            f1 = f[j + 1] if j + 1 < n else f[j]

            # h is evaluated at (t[i] - u) for u in [t0, t1]
            # So h argument ranges from (t[i] - t1) to (t[i] - t0)
            tau0 = t[i] - t1  # lower argument for h
            tau1 = t[i] - t0  # upper argument for h

            # Interpolate h at tau0 and tau1
            h0 = _interp_value(h, t, tau0)
            h1 = _interp_value(h, t, tau1)

            # Trapezoidal integration: (f0*h1 + f1*h0) * dt_j / 2
            # This is the standard trapezoidal rule for f(u)*h(t-u)
            total += (f0 * h1 + f1 * h0) * dt_j / 2.0

        result[i] = total

    return result


def _interp_value(
    y: NDArray[np.floating],
    t: NDArray[np.floating],
    t_query: float,
) -> float:
    """Linear interpolation of y at t_query."""
    xp = get_array_module(y)

    if t_query <= t[0]:
        return float(y[0])
    if t_query >= t[-1]:
        return float(y[-1])

    # Find interval containing t_query
    idx = int(xp.searchsorted(t, t_query)) - 1
    idx = max(0, min(idx, len(t) - 2))

    t0, t1 = float(t[idx]), float(t[idx + 1])
    y0, y1 = float(y[idx]), float(y[idx + 1])

    if t1 == t0:
        return y0

    # Linear interpolation
    alpha = (t_query - t0) / (t1 - t0)
    return y0 + alpha * (y1 - y0)


def uconv(
    f: NDArray[np.floating],
    h: NDArray[np.floating],
    dt: float,
) -> NDArray[np.floating]:
    """Convolve two signals on a uniform time grid.

    Optimized convolution for uniform time sampling using the
    trapezoidal rule. More efficient than the general non-uniform
    algorithm when time points are equally spaced.

    Parameters
    ----------
    f : ndarray
        Input signal. Shape: (n_times,).
    h : ndarray
        Impulse response function. Shape: (n_times,).
    dt : float
        Time step between samples in seconds.

    Returns
    -------
    ndarray
        Convolution result. Shape: (n_times,).

    Notes
    -----
    Uses the discrete convolution formula with trapezoidal weighting:

        conv[i] = dt * sum_{j=0}^{i} w[j] * f[j] * h[i-j]

    where w[j] = 0.5 for j=0 and j=i, and w[j] = 1 otherwise
    (trapezoidal weights).

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.convolution import uconv
    >>> dt = 0.1  # 100 ms time step
    >>> n = 100
    >>> f = np.exp(-np.arange(n) * dt / 2)
    >>> h = np.exp(-np.arange(n) * dt / 5)
    >>> result = uconv(f, h, dt)
    """
    xp = get_array_module(f)

    f = xp.asarray(f)
    h = xp.asarray(h)

    n = len(f)
    if len(h) != n:
        raise DataValidationError("f and h must have the same length")

    if n == 0:
        return xp.array([], dtype=f.dtype)

    if n == 1:
        return xp.array([0.0], dtype=f.dtype)

    # Discrete convolution with trapezoidal rule
    result = xp.zeros(n, dtype=f.dtype)

    for i in range(1, n):
        # Trapezoidal integration
        total = 0.5 * f[0] * h[i]  # First point weight = 0.5
        for j in range(1, i):
            total += f[j] * h[i - j]
        total += 0.5 * f[i] * h[0]  # Last point weight = 0.5
        result[i] = total * dt

    return result
