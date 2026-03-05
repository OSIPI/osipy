"""Temporal filtering and interpolation for perfusion data.

This module provides functions for temporal filtering, smoothing,
and interpolation of time-series perfusion data.

All operations are GPU/CPU agnostic using the xp array module pattern.
NO scipy dependencies - see XP Compatibility Requirements in plan.md.

"""

from typing import TYPE_CHECKING, Any

from osipy.common.backend.array_module import get_array_module, to_numpy
from osipy.common.exceptions import DataValidationError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def temporal_filter(
    data: "NDArray[np.floating[Any]]",
    filter_type: str = "gaussian",
    sigma: float = 1.0,
    window_size: int = 3,
) -> "NDArray[np.floating[Any]]":
    """Apply temporal filtering to time-series data.

    GPU/CPU agnostic implementation using FFT-based convolution.
    No scipy dependency.

    Parameters
    ----------
    data : NDArray[np.floating]
        Time-series data. Last dimension is time.
    filter_type : str, default="gaussian"
        Filter type:
        - "gaussian": Gaussian smoothing (sigma controls width)
        - "moving_average": Moving average (window_size controls width)
        - "median": Median filter (window_size controls width)
    sigma : float, default=1.0
        Standard deviation for Gaussian filter (in time points).
    window_size : int, default=3
        Window size for moving average or median filter.

    Returns
    -------
    NDArray[np.floating]
        Filtered data with same shape as input.

    Raises
    ------
    ValueError
        If invalid filter_type or parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.signal.filtering import temporal_filter
    >>> data = np.random.rand(64, 64, 20, 30)
    >>> smoothed = temporal_filter(data, filter_type="gaussian", sigma=1.0)
    """
    if filter_type == "gaussian":
        if sigma <= 0:
            msg = "sigma must be positive"
            raise DataValidationError(msg)
        return _gaussian_filter1d_xp(data, sigma=sigma, axis=-1)

    elif filter_type == "moving_average":
        if window_size < 1:
            msg = "window_size must be >= 1"
            raise DataValidationError(msg)
        return _uniform_filter1d_xp(data, size=window_size, axis=-1)

    elif filter_type == "median":
        if window_size < 1:
            msg = "window_size must be >= 1"
            raise DataValidationError(msg)
        # Median filter requires sorting - more complex on GPU
        # For now, use a simple implementation
        return _median_filter1d_xp(data, size=window_size, axis=-1)

    else:
        msg = f"Unknown filter_type: {filter_type}"
        raise DataValidationError(msg)


def _gaussian_filter1d_xp(
    data: "NDArray[np.floating[Any]]",
    sigma: float,
    axis: int = -1,
) -> "NDArray[np.floating[Any]]":
    """Apply 1D Gaussian filter along an axis using FFT convolution.

    GPU/CPU agnostic implementation.

    Parameters
    ----------
    data : NDArray
        Input data.
    sigma : float
        Standard deviation of Gaussian kernel.
    axis : int
        Axis along which to filter.

    Returns
    -------
    NDArray
        Filtered data.
    """
    xp = get_array_module(data)

    # Move target axis to the end for easier processing
    data = xp.moveaxis(data, axis, -1)
    n = data.shape[-1]

    # Create Gaussian kernel
    # Kernel size: 4*sigma is typically sufficient (covers >99.99% of mass)
    kernel_radius = int(4 * sigma + 0.5)
    kernel_size = 2 * kernel_radius + 1

    # Create kernel coordinates centered at 0
    x = xp.arange(-kernel_radius, kernel_radius + 1, dtype=data.dtype)
    kernel = xp.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / xp.sum(kernel)  # Normalize

    # Use FFT-based convolution for efficiency
    # Pad data for valid convolution
    pad_width = [(0, 0)] * (data.ndim - 1) + [(kernel_radius, kernel_radius)]
    data_padded = xp.pad(data, pad_width, mode="edge")

    # Flatten spatial dimensions for batch processing
    spatial_shape = data.shape[:-1]
    n_voxels = int(xp.prod(xp.asarray(spatial_shape)))
    data_flat = data_padded.reshape(n_voxels, -1)

    # FFT convolution
    n_padded = data_flat.shape[-1]
    fft_size = n_padded + kernel_size - 1

    # Zero-pad kernel to match
    kernel_padded = xp.zeros(fft_size, dtype=data.dtype)
    kernel_padded[:kernel_size] = kernel

    # FFT
    data_fft = xp.fft.rfft(data_flat, n=fft_size, axis=-1)
    kernel_fft = xp.fft.rfft(kernel_padded)

    # Multiply and inverse FFT
    result_fft = data_fft * kernel_fft
    result = xp.fft.irfft(result_fft, n=fft_size, axis=-1)

    # Extract valid region (accounting for kernel center)
    start = kernel_radius
    result = result[:, start : start + n]

    # Reshape back
    result = result.reshape((*spatial_shape, n))

    # Move axis back to original position
    result = xp.moveaxis(result, -1, axis)

    return result


def _uniform_filter1d_xp(
    data: "NDArray[np.floating[Any]]",
    size: int,
    axis: int = -1,
) -> "NDArray[np.floating[Any]]":
    """Apply 1D uniform (moving average) filter along an axis.

    GPU/CPU agnostic implementation using cumulative sum for efficiency.

    Parameters
    ----------
    data : NDArray
        Input data.
    size : int
        Window size.
    axis : int
        Axis along which to filter.

    Returns
    -------
    NDArray
        Filtered data.
    """
    xp = get_array_module(data)

    # Move target axis to the end
    data = xp.moveaxis(data, axis, -1)
    n = data.shape[-1]

    # Pad for edge handling
    pad_left = size // 2
    pad_right = size - pad_left - 1
    pad_width = [(0, 0)] * (data.ndim - 1) + [(pad_left, pad_right)]
    data_padded = xp.pad(data, pad_width, mode="edge")

    # Use cumulative sum for efficient moving average
    cumsum = xp.cumsum(data_padded, axis=-1)

    # Moving average = (cumsum[i+size] - cumsum[i]) / size
    result = (cumsum[..., size:] - cumsum[..., :-size]) / size

    # Handle edge case where result might be slightly off size
    if result.shape[-1] > n:
        result = result[..., :n]
    elif result.shape[-1] < n:
        # This shouldn't happen with correct padding, but handle anyway
        pad_needed = n - result.shape[-1]
        result = xp.pad(
            result, [(0, 0)] * (result.ndim - 1) + [(0, pad_needed)], mode="edge"
        )

    # Move axis back
    result = xp.moveaxis(result, -1, axis)

    return result


def _median_filter1d_xp(
    data: "NDArray[np.floating[Any]]",
    size: int,
    axis: int = -1,
) -> "NDArray[np.floating[Any]]":
    """Apply 1D median filter along an axis.

    GPU/CPU agnostic implementation. Note: median filter is inherently
    less parallelizable than linear filters.

    Parameters
    ----------
    data : NDArray
        Input data.
    size : int
        Window size.
    axis : int
        Axis along which to filter.

    Returns
    -------
    NDArray
        Filtered data.
    """
    xp = get_array_module(data)

    # Move target axis to the end
    data = xp.moveaxis(data, axis, -1)
    n = data.shape[-1]

    # Pad for edge handling
    pad_left = size // 2
    pad_right = size - pad_left - 1
    pad_width = [(0, 0)] * (data.ndim - 1) + [(pad_left, pad_right)]
    data_padded = xp.pad(data, pad_width, mode="edge")

    # Create sliding window view
    # This is more complex - we'll use a loop-based approach
    # that still leverages GPU parallelism for the median computation

    spatial_shape = data.shape[:-1]
    n_padded = data_padded.shape[-1]

    # Flatten spatial dimensions
    n_voxels = int(xp.prod(xp.asarray(spatial_shape))) if spatial_shape else 1
    data_flat = data_padded.reshape(n_voxels, n_padded)

    # Sliding window median
    # Build a matrix of windows and compute median along window axis
    # Shape: (n_voxels, n, size)
    windows = xp.zeros((n_voxels, n, size), dtype=data.dtype)
    for i in range(size):
        windows[:, :, i] = data_flat[:, i : i + n]

    # Compute median along last axis
    result_flat = xp.median(windows, axis=-1)

    # Reshape back
    result = result_flat.reshape((*spatial_shape, n))

    # Move axis back
    result = xp.moveaxis(result, -1, axis)

    return result


def temporal_interpolate(
    data: "NDArray[np.floating[Any]]",
    time_old: "NDArray[np.floating[Any]]",
    time_new: "NDArray[np.floating[Any]]",
    method: str = "linear",
) -> "NDArray[np.floating[Any]]":
    """Interpolate time-series data to new time points.

    GPU/CPU agnostic implementation. No scipy dependency.

    Parameters
    ----------
    data : NDArray[np.floating]
        Time-series data. Last dimension is time with length len(time_old).
    time_old : NDArray[np.floating]
        Original time points.
    time_new : NDArray[np.floating]
        Target time points for interpolation.
    method : str, default="linear"
        Interpolation method:
        - "linear": Linear interpolation
        - "cubic": Cubic spline interpolation
        - "nearest": Nearest neighbor

    Returns
    -------
    NDArray[np.floating]
        Interpolated data with last dimension length len(time_new).

    Raises
    ------
    ValueError
        If time arrays have invalid shapes or method is unknown.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.signal.filtering import temporal_interpolate
    >>> data = np.random.rand(64, 64, 20, 30)
    >>> time_old = np.linspace(0, 300, 30)
    >>> time_new = np.linspace(0, 300, 60)  # Upsample to 60 points
    >>> interpolated = temporal_interpolate(data, time_old, time_new)
    >>> print(interpolated.shape)
    (64, 64, 20, 60)
    """
    xp = get_array_module(data)

    # Ensure time arrays are on same device as data
    time_old = xp.asarray(time_old)
    time_new = xp.asarray(time_new)

    if len(time_old) != data.shape[-1]:
        msg = f"time_old length ({len(time_old)}) must match data time dimension ({data.shape[-1]})"
        raise DataValidationError(msg)

    valid_methods = {"linear", "cubic", "nearest"}
    if method not in valid_methods:
        msg = f"Unknown method: {method}. Use one of {valid_methods}"
        raise DataValidationError(msg)

    if method == "linear":
        return _interpolate_linear_xp(data, time_old, time_new)
    elif method == "cubic":
        return _interpolate_cubic_xp(data, time_old, time_new)
    else:
        return _interpolate_nearest_xp(data, time_old, time_new)


def _interpolate_linear_xp(
    data: "NDArray[np.floating[Any]]",
    time_old: "NDArray[np.floating[Any]]",
    time_new: "NDArray[np.floating[Any]]",
) -> "NDArray[np.floating[Any]]":
    """Linear interpolation using xp operations.

    Parameters
    ----------
    data : NDArray
        Input data, last dimension is time.
    time_old : NDArray
        Original time points.
    time_new : NDArray
        Target time points.

    Returns
    -------
    NDArray
        Interpolated data.
    """
    xp = get_array_module(data)

    # Get shapes
    spatial_shape = data.shape[:-1]
    n_time_old = len(time_old)
    n_time_new = len(time_new)

    # Flatten spatial dimensions
    n_voxels = int(xp.prod(xp.asarray(spatial_shape))) if spatial_shape else 1
    data_flat = data.reshape(n_voxels, n_time_old)

    # Find indices for interpolation using searchsorted
    # indices[i] is the index of the first element in time_old >= time_new[i]
    indices = xp.searchsorted(time_old, time_new, side="right")

    # Clip to valid range [1, n_time_old - 1] for interpolation
    indices = xp.clip(indices, 1, n_time_old - 1)

    # Get bracketing indices
    idx_low = indices - 1
    idx_high = indices

    # Get bracketing values
    t_low = time_old[idx_low]
    t_high = time_old[idx_high]

    # Compute interpolation weights
    # Handle case where t_high == t_low (shouldn't happen with valid data)
    dt = t_high - t_low
    dt = xp.where(dt == 0, 1.0, dt)  # Avoid division by zero
    weight = (time_new - t_low) / dt
    weight = xp.clip(weight, 0.0, 1.0)  # Clip for extrapolation

    # Apply interpolation: result = (1 - weight) * y_low + weight * y_high
    # Broadcasting: data_flat is (n_voxels, n_time_old), weight is (n_time_new,)
    y_low = data_flat[:, idx_low]  # (n_voxels, n_time_new)
    y_high = data_flat[:, idx_high]  # (n_voxels, n_time_new)

    result_flat = (1.0 - weight) * y_low + weight * y_high

    # Reshape back
    return result_flat.reshape((*spatial_shape, n_time_new))


def _interpolate_nearest_xp(
    data: "NDArray[np.floating[Any]]",
    time_old: "NDArray[np.floating[Any]]",
    time_new: "NDArray[np.floating[Any]]",
) -> "NDArray[np.floating[Any]]":
    """Nearest neighbor interpolation using xp operations.

    Parameters
    ----------
    data : NDArray
        Input data, last dimension is time.
    time_old : NDArray
        Original time points.
    time_new : NDArray
        Target time points.

    Returns
    -------
    NDArray
        Interpolated data.
    """
    xp = get_array_module(data)

    # Get shapes
    spatial_shape = data.shape[:-1]
    n_time_old = len(time_old)
    n_time_new = len(time_new)

    # Flatten spatial dimensions
    n_voxels = int(xp.prod(xp.asarray(spatial_shape))) if spatial_shape else 1
    data_flat = data.reshape(n_voxels, n_time_old)

    # Find nearest indices
    # For each new time point, find the closest old time point
    # Use searchsorted and compare distances
    indices_right = xp.searchsorted(time_old, time_new, side="right")
    indices_right = xp.clip(indices_right, 0, n_time_old - 1)
    indices_left = xp.clip(indices_right - 1, 0, n_time_old - 1)

    # Compute distances to left and right neighbors
    dist_left = xp.abs(time_new - time_old[indices_left])
    dist_right = xp.abs(time_new - time_old[indices_right])

    # Choose closer neighbor
    nearest_indices = xp.where(dist_left <= dist_right, indices_left, indices_right)

    # Gather values
    result_flat = data_flat[:, nearest_indices]

    # Reshape back
    return result_flat.reshape((*spatial_shape, n_time_new))


def _interpolate_cubic_xp(
    data: "NDArray[np.floating[Any]]",
    time_old: "NDArray[np.floating[Any]]",
    time_new: "NDArray[np.floating[Any]]",
) -> "NDArray[np.floating[Any]]":
    """Cubic spline interpolation using xp operations.

    Uses natural cubic spline interpolation. GPU/CPU agnostic.

    Parameters
    ----------
    data : NDArray
        Input data, last dimension is time.
    time_old : NDArray
        Original time points (must be sorted).
    time_new : NDArray
        Target time points.

    Returns
    -------
    NDArray
        Interpolated data.
    """
    xp = get_array_module(data)

    # Get shapes
    spatial_shape = data.shape[:-1]
    n = len(time_old)
    n_time_new = len(time_new)

    # Flatten spatial dimensions
    n_voxels = int(xp.prod(xp.asarray(spatial_shape))) if spatial_shape else 1
    data_flat = data.reshape(n_voxels, n)  # (n_voxels, n)

    # Compute spline coefficients for all voxels simultaneously
    # h[i] = time_old[i+1] - time_old[i]
    h = xp.diff(time_old)  # (n-1,)

    # Build tridiagonal system for second derivatives (natural spline)
    # For natural spline: M[0] = M[n-1] = 0
    # System: h[i-1]*M[i-1] + 2*(h[i-1]+h[i])*M[i] + h[i]*M[i+1] = 6*(y[i+1]-y[i])/h[i] - 6*(y[i]-y[i-1])/h[i-1]

    if n < 3:
        # Fall back to linear for too few points
        return _interpolate_linear_xp(data, time_old, time_new)

    # Compute divided differences for all voxels
    # dy/h for each interval
    dy = xp.diff(data_flat, axis=1)  # (n_voxels, n-1)
    slopes = dy / h  # (n_voxels, n-1)

    # Right-hand side: 6 * (slopes[i] - slopes[i-1])
    rhs = 6.0 * (slopes[:, 1:] - slopes[:, :-1])  # (n_voxels, n-2)

    # Tridiagonal matrix coefficients (same for all voxels since time_old is shared)
    # Diagonal: 2*(h[i-1] + h[i])
    diag = 2.0 * (h[:-1] + h[1:])  # (n-2,)
    # Sub-diagonal: h[i-1]
    sub_diag = h[:-1]  # (n-2,) but we use (n-3,) entries
    # Super-diagonal: h[i]
    super_diag = h[1:]  # (n-2,) but we use (n-3,) entries

    # Solve tridiagonal system using Thomas algorithm for each voxel
    # This is vectorized across voxels
    M_interior = _solve_tridiagonal_batch(
        sub_diag[:-1], diag, super_diag[:-1], rhs, xp
    )  # (n_voxels, n-2)

    # Build full M array with boundary conditions M[0] = M[-1] = 0
    M = xp.zeros((n_voxels, n), dtype=data.dtype)
    M[:, 1:-1] = M_interior

    # Now interpolate using the spline coefficients
    # For each new point, find the interval and evaluate the cubic

    # Find intervals for new points
    indices = xp.searchsorted(time_old, time_new, side="right")
    indices = xp.clip(indices, 1, n - 1)
    idx_low = indices - 1

    # Evaluate cubic spline
    # S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3
    # where:
    # a_i = y_i
    # c_i = M_i / 2
    # d_i = (M_{i+1} - M_i) / (6*h_i)
    # b_i = (y_{i+1} - y_i)/h_i - h_i*(2*M_i + M_{i+1})/6

    # Get interval-specific values for each new point
    h_i = h[idx_low]  # (n_time_new,)
    y_i = data_flat[:, idx_low]  # (n_voxels, n_time_new)
    y_ip1 = data_flat[:, idx_low + 1]  # (n_voxels, n_time_new)
    M_i = M[:, idx_low]  # (n_voxels, n_time_new)
    M_ip1 = M[:, idx_low + 1]  # (n_voxels, n_time_new)

    # Distance from interval start
    dx = time_new - time_old[idx_low]  # (n_time_new,)

    # Spline coefficients (computed per new point, vectorized over voxels)
    a = y_i
    c = M_i / 2.0
    d = (M_ip1 - M_i) / (6.0 * h_i)
    b = (y_ip1 - y_i) / h_i - h_i * (2.0 * M_i + M_ip1) / 6.0

    # Evaluate: S(x) = a + b*dx + c*dx^2 + d*dx^3
    result_flat = a + b * dx + c * dx**2 + d * dx**3

    # Reshape back
    return result_flat.reshape((*spatial_shape, n_time_new))


def _solve_tridiagonal_batch(
    sub: "NDArray[np.floating[Any]]",
    diag: "NDArray[np.floating[Any]]",
    sup: "NDArray[np.floating[Any]]",
    rhs: "NDArray[np.floating[Any]]",
    xp: Any,
) -> "NDArray[np.floating[Any]]":
    """Solve tridiagonal systems using Thomas algorithm, batched over voxels.

    Solves A @ x = rhs where A is tridiagonal with sub-diagonal 'sub',
    diagonal 'diag', and super-diagonal 'sup'.

    Parameters
    ----------
    sub : NDArray
        Sub-diagonal coefficients, shape (n-1,).
    diag : NDArray
        Diagonal coefficients, shape (n,).
    sup : NDArray
        Super-diagonal coefficients, shape (n-1,).
    rhs : NDArray
        Right-hand side, shape (n_voxels, n).
    xp : module
        Array module (numpy or cupy).

    Returns
    -------
    NDArray
        Solution, shape (n_voxels, n).
    """
    n = len(diag)
    n_voxels = rhs.shape[0]

    # Copy to avoid modifying inputs
    c = xp.zeros(n, dtype=rhs.dtype)  # Modified super-diagonal
    d = rhs.copy()  # Modified RHS

    # Forward sweep
    c[0] = sup[0] / diag[0]
    d[:, 0] = d[:, 0] / diag[0]

    for i in range(1, n):
        denom = diag[i] - sub[i - 1] * c[i - 1]
        if i < n - 1:
            c[i] = sup[i] / denom
        d[:, i] = (d[:, i] - sub[i - 1] * d[:, i - 1]) / denom

    # Back substitution
    x = xp.zeros((n_voxels, n), dtype=rhs.dtype)
    x[:, -1] = d[:, -1]

    for i in range(n - 2, -1, -1):
        x[:, i] = d[:, i] - c[i] * x[:, i + 1]

    return x


def resample_to_uniform(
    data: "NDArray[np.floating[Any]]",
    time: "NDArray[np.floating[Any]]",
    dt: float | None = None,
) -> tuple["NDArray[np.floating[Any]]", "NDArray[np.floating[Any]]"]:
    """Resample non-uniform time series to uniform temporal spacing.

    GPU/CPU agnostic implementation. No scipy dependency.

    Parameters
    ----------
    data : NDArray[np.floating]
        Time-series data. Last dimension is time.
    time : NDArray[np.floating]
        Original (possibly non-uniform) time points.
    dt : float | None
        Target temporal resolution. If None, uses minimum time
        difference in original data.

    Returns
    -------
    tuple[NDArray, NDArray]
        (resampled_data, new_time_points)

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.signal.filtering import resample_to_uniform
    >>> # Non-uniform sampling
    >>> time = np.array([0, 1, 2, 5, 10, 15, 20])
    >>> data = np.random.rand(64, 64, 20, len(time))
    >>> resampled, new_time = resample_to_uniform(data, time, dt=1.0)
    """
    xp = get_array_module(data)
    time = xp.asarray(time)

    if dt is None:
        dt = float(xp.min(xp.diff(time)))

    # Create new uniform time array
    # Use to_numpy for scalar extraction to avoid GPU issues
    t_start = float(to_numpy(time[0]))
    t_end = float(to_numpy(time[-1]))
    n_new = int((t_end - t_start) / dt) + 1
    time_new = xp.linspace(t_start, t_end, n_new)

    data_new = temporal_interpolate(data, time, time_new, method="linear")

    return data_new, time_new
