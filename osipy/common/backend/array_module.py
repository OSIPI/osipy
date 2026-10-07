"""Array module utilities for CPU/GPU agnostic operations.

This module provides the core functions for writing CPU/GPU agnostic code
using the array module pattern. The key function `get_array_module()` returns
either NumPy or CuPy depending on the input array type and configuration.

Supports CPU/GPU agnostic array operations using CuPy, allowing the same
code to execute on CPU (via NumPy) or GPU (via CuPy) without modification.

Example
-------
>>> import numpy as np
>>> from osipy.common.backend import get_array_module
>>>
>>> def compute_mean(data):
...     xp = get_array_module(data)
...     return xp.mean(data, axis=0)
>>>
>>> # Works with NumPy arrays
>>> np_data = np.random.randn(100, 50)
>>> result = compute_mean(np_data)  # Uses NumPy
>>>
>>> # Works with CuPy arrays (if available)
>>> # gpu_data = cupy.asarray(np_data)
>>> # result = compute_mean(gpu_data)  # Uses CuPy

References
----------
.. [1] CuPy array module pattern: https://docs.cupy.dev/en/stable/user_guide/basic.html
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.exceptions import GPUTransferError

# NumPy 2.0 renamed trapz -> trapezoid. Ensure compatibility with NumPy <2.0.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined] # noqa: NPY201

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)

# Cache the CuPy module to avoid repeated imports
_cupy_module: Any = None
_cupy_available: bool | None = None


def _get_cupy() -> Any:
    """Get the CuPy module, caching the result.

    Returns
    -------
    module or None
        The CuPy module if available, None otherwise.
    """
    global _cupy_module, _cupy_available

    if _cupy_available is None:
        try:
            import cupy as cp

            # CuPy < 14 uses trapz; ensure trapezoid alias exists
            if not hasattr(cp, "trapezoid") and hasattr(cp, "trapz"):
                cp.trapezoid = cp.trapz  # type: ignore[attr-defined]

            _cupy_module = cp
            _cupy_available = True
        except ImportError:
            _cupy_module = None
            _cupy_available = False

    return _cupy_module


def get_array_module(*arrays: ArrayLike) -> Any:
    """Get the array module (NumPy or CuPy) for the given arrays.

    This function returns the appropriate array module based on the input
    array types and the current backend configuration. If any input array
    is a CuPy array and GPU is not forced to CPU mode, returns CuPy.
    Otherwise, returns NumPy.

    Parameters
    ----------
    *arrays : ArrayLike
        Input arrays. Can be NumPy arrays, CuPy arrays, or any array-like.

    Returns
    -------
    module
        Either numpy or cupy module, depending on input types and configuration.

    Notes
    -----
    This function checks:
    1. If the backend is forced to CPU mode → returns NumPy
    2. If any input has `__cuda_array_interface__` → returns CuPy
    3. If CuPy's `get_array_module` is available → uses it
    4. Otherwise → returns NumPy

    Example
    -------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3])
    >>> xp = get_array_module(data)
    >>> xp.__name__
    'numpy'
    """
    # Import here to avoid circular imports
    from osipy.common.backend.config import get_backend

    config = get_backend()

    # If forced to CPU, always return NumPy
    if config.force_cpu:
        return np

    # Check if CuPy is available
    cp = _get_cupy()
    if cp is None:
        return np

    # Check if any array is on GPU
    for arr in arrays:
        if arr is None:
            continue
        # Check for CUDA array interface (CuPy arrays have this)
        if hasattr(arr, "__cuda_array_interface__"):
            return cp

    # Use CuPy's get_array_module if available (handles edge cases)
    if hasattr(cp, "get_array_module"):
        try:
            return cp.get_array_module(*[a for a in arrays if a is not None])
        except TypeError:
            pass

    # Default to NumPy
    return np


def to_numpy(array: ArrayLike) -> NDArray[Any]:
    """Convert an array to NumPy, transferring from GPU if necessary.

    Parameters
    ----------
    array : ArrayLike
        Input array. Can be NumPy, CuPy, or any array-like.

    Returns
    -------
    NDArray
        NumPy array with the same data.

    Notes
    -----
    If the input is already a NumPy array, it is returned as-is (no copy).
    If the input is a CuPy array, data is transferred from GPU to CPU.
    For other array-likes, np.asarray is used.

    Example
    -------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3])
    >>> result = to_numpy(data)
    >>> result is data  # No copy for NumPy arrays
    True
    """
    if isinstance(array, np.ndarray):
        return array

    # Check if it's a CuPy array
    if hasattr(array, "__cuda_array_interface__"):
        cp = _get_cupy()
        if cp is not None and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)

    # For other array-likes, convert via NumPy
    return np.asarray(array)


def to_gpu(array: ArrayLike) -> Any:
    """Transfer an array to GPU if available, otherwise return as NumPy.

    Parameters
    ----------
    array : ArrayLike
        Input array. Can be NumPy, CuPy, or any array-like.

    Returns
    -------
    array
        CuPy array if GPU is available and not forced to CPU mode,
        otherwise NumPy array.

    Notes
    -----
    This function respects the global backend configuration:
    - If force_cpu is True, returns NumPy array
    - If CuPy is not available, returns NumPy array
    - If input is already on GPU, returns as-is (no copy)
    - If a GPU is available, ``force_cpu`` is not set, and the transfer
      itself fails (e.g. out of memory), the error is logged and
      re-raised as a ``GPUTransferError`` rather than silently falling
      back to CPU. A silent fallback here would let callers that decide
      GPU-vs-CPU behavior (chunk sizing, threading) keep assuming GPU
      execution when it never happened. Callers that want a graceful
      CPU fallback on transfer failure (e.g. batch processing that
      retries a batch on CPU after a GPU out-of-memory error) must catch
      this exception themselves.

    Raises
    ------
    GPUTransferError
        If a GPU is available, ``force_cpu`` is not set, and the
        transfer to GPU fails for any reason.

    Example
    -------
    >>> import numpy as np
    >>> from osipy.common.backend import to_gpu, is_gpu_available
    >>> data = np.array([1, 2, 3])
    >>> gpu_data = to_gpu(data)
    >>> # Returns CuPy array if GPU available, else NumPy array
    """
    # Import here to avoid circular imports
    from osipy.common.backend.config import get_backend

    config = get_backend()

    # If forced to CPU, return as NumPy
    if config.force_cpu:
        return to_numpy(array)

    # Check if CuPy is available
    cp = _get_cupy()
    if cp is None:
        return to_numpy(array)

    # If already on GPU, return as-is
    if hasattr(array, "__cuda_array_interface__"):
        return array

    # Transfer to GPU
    try:
        return cp.asarray(array)
    except Exception as e:
        msg = (
            f"GPU transfer failed ({e}). GPU was requested and detected, "
            "so refusing to silently fall back to CPU; pass force_cpu=True "
            "to run on CPU instead."
        )
        logger.error(msg)
        raise GPUTransferError(msg) from e
