"""GPU configuration and detection utilities.

This module provides global configuration for GPU acceleration and functions
to detect GPU availability at runtime, including a global configuration
option to force CPU-only execution even when GPU is available.

Example
-------
>>> from osipy.common.backend import is_gpu_available, set_backend, GPUConfig
>>>
>>> # Check if GPU is available
>>> if is_gpu_available():
...     print("GPU acceleration available")
>>>
>>> # Force CPU-only execution
>>> set_backend(GPUConfig(force_cpu=True))
>>>
>>> # Or use environment variable: OSIPY_FORCE_CPU=1

References
----------
.. [1] CuPy Installation Guide: https://docs.cupy.dev/en/stable/install.html
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from osipy.common.exceptions import DataValidationError

logger = logging.getLogger(__name__)

# Global backend configuration
_global_config: GPUConfig | None = None

# Cache for GPU availability check
_gpu_available_cache: bool | None = None


@dataclass
class GPUConfig:
    """Configuration for GPU/CPU backend selection.

    This dataclass holds configuration options for controlling GPU acceleration.
    It can be used with `set_backend()` to configure global behavior.

    Parameters
    ----------
    force_cpu : bool, optional
        If True, force all operations to run on CPU even if GPU is available.
        Default is False, which allows automatic GPU usage when available.
    default_batch_size : int, optional
        Default batch size for GPU batch processing. Larger values use more
        GPU memory but may be faster. Default is 10000.
    memory_limit_fraction : float, optional
        Fraction of GPU memory to use (0.0 to 1.0). Default is 0.9 (90%).
        This helps prevent out-of-memory errors by leaving headroom.
    device_id : int, optional
        CUDA device ID to use. Default is 0 (first GPU).
    n_workers : int, optional
        Number of threads for CPU-parallel chunk processing in ``fit_image()``.
        0 (default) = auto (``os.cpu_count()``); 1 = disable threading.
        Ignored when running on GPU. Can also be set via the
        ``OSIPY_NUM_THREADS`` environment variable.

    Attributes
    ----------
    force_cpu : bool
        Whether to force CPU-only execution.
    default_batch_size : int
        Default batch size for GPU operations.
    memory_limit_fraction : float
        Fraction of GPU memory to use.
    device_id : int
        CUDA device ID.
    n_workers : int
        Number of threads for CPU-parallel chunk processing.

    Example
    -------
    >>> config = GPUConfig(force_cpu=True)
    >>> set_backend(config)
    """

    force_cpu: bool = field(default=False)
    default_batch_size: int = field(default=10000)
    memory_limit_fraction: float = field(default=0.9)
    device_id: int = field(default=0)
    n_workers: int = field(default=0)
    gpu_dtype: str = field(default="float32")

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 < self.memory_limit_fraction <= 1.0:
            msg = "memory_limit_fraction must be between 0 and 1"
            raise DataValidationError(msg)
        if self.default_batch_size < 1:
            msg = "default_batch_size must be positive"
            raise DataValidationError(msg)
        if self.device_id < 0:
            msg = "device_id must be non-negative"
            raise DataValidationError(msg)
        if self.n_workers < 0:
            msg = "n_workers must be non-negative"
            raise DataValidationError(msg)
        if self.gpu_dtype not in ("float32", "float64"):
            msg = "gpu_dtype must be 'float32' or 'float64'"
            raise DataValidationError(msg)


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available.

    This function checks for CUDA-capable GPU and CuPy installation.
    The result is cached after the first call for performance.

    Returns
    -------
    bool
        True if GPU acceleration is available, False otherwise.

    Notes
    -----
    GPU is considered available if:
    1. CuPy is installed and can be imported
    2. At least one CUDA device is detected
    3. The OSIPY_FORCE_CPU environment variable is not set to "1"

    The availability check is cached after the first call. To force
    a recheck, use `_reset_gpu_cache()` (internal use only).

    Example
    -------
    >>> if is_gpu_available():
    ...     print("GPU acceleration enabled")
    ... else:
    ...     print("Running on CPU only")
    """
    global _gpu_available_cache

    # Check environment variable for forced CPU mode
    if os.environ.get("OSIPY_FORCE_CPU", "0") == "1":
        return False

    if _gpu_available_cache is not None:
        return _gpu_available_cache

    try:
        import cupy as cp

        # Check if any CUDA device is available
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count > 0:
            # Try to allocate a small array to verify GPU is working
            try:
                test_array = cp.zeros(10)
                del test_array
                _gpu_available_cache = True
                logger.info(
                    "GPU acceleration available: %d CUDA device(s) detected",
                    device_count,
                )
                return True
            except Exception as e:
                logger.warning("GPU detected but allocation failed: %s", e)
                _gpu_available_cache = False
                return False
        else:
            logger.info("No CUDA devices detected")
            _gpu_available_cache = False
            return False

    except ImportError:
        logger.debug("CuPy not installed - GPU acceleration unavailable")
        _gpu_available_cache = False
        return False
    except Exception as e:
        logger.warning("Error checking GPU availability: %s", e)
        _gpu_available_cache = False
        return False


def get_backend() -> GPUConfig:
    """Get the current backend configuration.

    Returns
    -------
    GPUConfig
        The current global GPU configuration. If not explicitly set,
        returns a default configuration that respects environment variables.

    Example
    -------
    >>> config = get_backend()
    >>> print(f"Force CPU: {config.force_cpu}")
    """
    global _global_config

    if _global_config is None:
        # Check environment variables for defaults
        force_cpu = os.environ.get("OSIPY_FORCE_CPU", "0") == "1"
        n_workers_env = os.environ.get("OSIPY_NUM_THREADS")
        n_workers = int(n_workers_env) if n_workers_env is not None else 0
        _global_config = GPUConfig(force_cpu=force_cpu, n_workers=n_workers)

    return _global_config


def set_backend(config: GPUConfig) -> None:
    """Set the global backend configuration.

    Parameters
    ----------
    config : GPUConfig
        The configuration to use globally.

    Notes
    -----
    This affects all subsequent calls to `get_array_module()`, `to_gpu()`,
    and other backend functions. Changes take effect immediately.

    Example
    -------
    >>> # Force CPU-only execution
    >>> set_backend(GPUConfig(force_cpu=True))
    >>>
    >>> # Re-enable GPU with custom batch size
    >>> set_backend(GPUConfig(force_cpu=False, default_batch_size=50000))
    """
    global _global_config
    _global_config = config
    logger.info(
        "Backend configuration updated: force_cpu=%s, device_id=%d",
        config.force_cpu,
        config.device_id,
    )


def get_gpu_memory_info() -> dict[str, Any]:
    """Get information about GPU memory usage.

    Returns
    -------
    dict
        Dictionary containing:
        - 'available': bool - whether GPU is available
        - 'total_bytes': int - total GPU memory in bytes (0 if unavailable)
        - 'used_bytes': int - used GPU memory in bytes (0 if unavailable)
        - 'free_bytes': int - free GPU memory in bytes (0 if unavailable)
        - 'device_name': str - GPU device name (empty if unavailable)

    Example
    -------
    >>> info = get_gpu_memory_info()
    >>> if info['available']:
    ...     print(f"GPU: {info['device_name']}")
    ...     print(f"Free: {info['free_bytes'] / 1e9:.2f} GB")
    """
    result: dict[str, Any] = {
        "available": False,
        "total_bytes": 0,
        "used_bytes": 0,
        "free_bytes": 0,
        "device_name": "",
    }

    if not is_gpu_available():
        return result

    try:
        import cupy as cp

        config = get_backend()
        with cp.cuda.Device(config.device_id):
            mempool = cp.get_default_memory_pool()
            mem_info = cp.cuda.runtime.memGetInfo()

            result["available"] = True
            result["free_bytes"] = mem_info[0]
            result["total_bytes"] = mem_info[1]
            result["used_bytes"] = mempool.used_bytes()

            # Get device name
            props = cp.cuda.runtime.getDeviceProperties(config.device_id)
            result["device_name"] = (
                props["name"].decode()
                if isinstance(props["name"], bytes)
                else props["name"]
            )

    except Exception as e:
        logger.warning("Error getting GPU memory info: %s", e)

    return result


def get_gpu_batch_size() -> int:
    """Get optimal batch size based on GPU thread capacity.

    Returns the total number of concurrent threads the GPU can handle
    (multiProcessorCount * maxThreadsPerMultiProcessor), which is a
    good heuristic for batch sizing.

    Returns 0 if GPU is not available.
    """
    if not is_gpu_available():
        return 0
    try:
        import cupy as cp

        config = get_backend()
        props = cp.cuda.runtime.getDeviceProperties(config.device_id)
        return props["multiProcessorCount"] * props["maxThreadsPerMultiProcessor"]
    except Exception as e:
        logger.debug("Could not determine GPU batch size: %s", e)
        return 0


def _reset_gpu_cache() -> None:
    """Reset the GPU availability cache (internal use only).

    This forces a recheck of GPU availability on the next call to
    `is_gpu_available()`. Useful for testing.
    """
    global _gpu_available_cache
    _gpu_available_cache = None
