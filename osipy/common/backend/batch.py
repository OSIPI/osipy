"""Batch processing utilities for GPU memory management.

This module provides the BatchProcessor class for processing large datasets
in batches on GPU while managing memory. It supports batch processing of
multiple voxels on GPU with configurable batch sizes and graceful fallback
to CPU execution if GPU memory is exhausted.

Example
-------
>>> from osipy.common.backend import BatchProcessor
>>> import numpy as np
>>>
>>> def process_batch(data):
...     return data ** 2
>>>
>>> processor = BatchProcessor(batch_size=10000)
>>> large_data = np.random.randn(100000, 50)
>>> result = processor.process(large_data, process_batch)

References
----------
.. [1] CuPy Memory Management: https://docs.cupy.dev/en/stable/user_guide/memory.html
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.backend.array_module import to_gpu, to_numpy
from osipy.common.backend.config import (
    get_backend,
    get_gpu_memory_info,
    is_gpu_available,
)
from osipy.common.exceptions import DataValidationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result from batch processing.

    Attributes
    ----------
    data : NDArray
        The processed data.
    used_gpu : bool
        Whether GPU was used for processing.
    fallback_occurred : bool
        Whether CPU fallback occurred due to GPU memory issues.
    batches_processed : int
        Number of batches processed.
    """

    data: NDArray[Any]
    used_gpu: bool = False
    fallback_occurred: bool = False
    batches_processed: int = 0


@dataclass
class BatchProcessor:
    """Process data in batches with automatic GPU memory management.

    This class provides efficient batch processing for large datasets,
    automatically managing GPU memory and falling back to CPU when necessary.

    Parameters
    ----------
    batch_size : int, optional
        Number of elements per batch. Default uses the global configuration.
    use_gpu : bool, optional
        Whether to attempt GPU acceleration. Default is True.
    auto_fallback : bool, optional
        Whether to automatically fall back to CPU on GPU memory errors.
        Default is True.
    memory_safety_margin : float, optional
        Fraction of estimated memory to keep free (0.0 to 0.5).
        Default is 0.1 (10% safety margin).

    Example
    -------
    >>> processor = BatchProcessor(batch_size=5000)
    >>> result = processor.map(data, lambda x: x ** 2)
    """

    batch_size: int | None = field(default=None)
    use_gpu: bool = field(default=True)
    auto_fallback: bool = field(default=True)
    memory_safety_margin: float = field(default=0.1)

    def __post_init__(self) -> None:
        """Initialize with defaults from global config if needed."""
        if self.batch_size is None:
            self.batch_size = get_backend().default_batch_size
        if not 0.0 <= self.memory_safety_margin <= 0.5:
            msg = "memory_safety_margin must be between 0 and 0.5"
            raise DataValidationError(msg)

    def _estimate_batch_memory(self, sample: NDArray[Any], batch_size: int) -> int:
        """Estimate memory required for a batch.

        Parameters
        ----------
        sample : NDArray
            Sample array to estimate from.
        batch_size : int
            Proposed batch size.

        Returns
        -------
        int
            Estimated memory in bytes.
        """
        # Estimate based on array dtype and shape
        element_size = sample.dtype.itemsize
        if sample.ndim == 1:
            return batch_size * element_size
        else:
            # For multi-dimensional, estimate per-element size from sample
            elements_per_item = np.prod(sample.shape[1:])
            return int(batch_size * elements_per_item * element_size)

    def _get_optimal_batch_size(self, data: NDArray[Any]) -> int:
        """Calculate optimal batch size based on available GPU memory.

        Parameters
        ----------
        data : NDArray
            Input data array.

        Returns
        -------
        int
            Optimal batch size that fits in GPU memory.
        """
        assert self.batch_size is not None  # Set in __post_init__

        if not self.use_gpu or not is_gpu_available():
            return self.batch_size

        mem_info = get_gpu_memory_info()
        if not mem_info["available"]:
            return self.batch_size

        # Calculate available memory with safety margin
        available = mem_info["free_bytes"] * (1.0 - self.memory_safety_margin)

        # Estimate memory per element
        per_element = self._estimate_batch_memory(data, 1)

        # Account for temporary arrays (multiply by 3 for safety: input, output, intermediate)
        optimal = int(available / (per_element * 3))

        # Clamp to reasonable bounds
        min_batch = 100
        optimal = max(min_batch, min(optimal, self.batch_size))

        logger.debug(
            "Optimal batch size: %d (available memory: %.2f MB)",
            optimal,
            available / 1e6,
        )

        return optimal

    def map(
        self,
        data: NDArray[Any],
        func: Callable[[NDArray[Any]], NDArray[Any]],
        axis: int = 0,
    ) -> BatchResult:
        """Apply a function to data in batches.

        Parameters
        ----------
        data : NDArray
            Input data array.
        func : Callable
            Function to apply to each batch. Should accept and return arrays.
        axis : int, optional
            Axis along which to batch. Default is 0.

        Returns
        -------
        BatchResult
            Result containing processed data and metadata.

        Notes
        -----
        If GPU memory is exhausted, this will automatically fall back to CPU
        processing (if auto_fallback is True) with a warning.
        """
        n_items = data.shape[axis]

        if n_items == 0:
            return BatchResult(data=data, batches_processed=0)

        # Determine if we can use GPU
        use_gpu = self.use_gpu and is_gpu_available() and not get_backend().force_cpu

        # Get optimal batch size
        batch_size = self._get_optimal_batch_size(data)

        results: list[NDArray[Any]] = []
        fallback_occurred = False
        batches_processed = 0

        # Process in batches
        for start in range(0, n_items, batch_size):
            end = min(start + batch_size, n_items)

            # Slice batch along specified axis
            batch_slice = [slice(None)] * data.ndim
            batch_slice[axis] = slice(start, end)
            batch = data[tuple(batch_slice)]

            try:
                if use_gpu:
                    # Transfer to GPU and process
                    gpu_batch = to_gpu(batch)
                    result = func(gpu_batch)
                    # Transfer result back to CPU
                    results.append(to_numpy(result))
                else:
                    results.append(func(batch))

                batches_processed += 1

            except Exception as e:
                # Check if this is a GPU memory error
                error_str = str(e).lower()
                is_memory_error = any(
                    phrase in error_str
                    for phrase in ["out of memory", "memory", "cuda", "allocation"]
                )

                if is_memory_error and self.auto_fallback:
                    warnings.warn(
                        f"GPU memory exhausted at batch {batches_processed}. "
                        "Falling back to CPU processing.",
                        UserWarning,
                        stacklevel=2,
                    )
                    fallback_occurred = True
                    use_gpu = False

                    # Clear GPU memory
                    self._clear_gpu_memory()

                    # Retry on CPU
                    results.append(func(batch))
                    batches_processed += 1
                else:
                    raise

        # Concatenate results
        output = np.concatenate(results, axis=axis)

        return BatchResult(
            data=output,
            used_gpu=use_gpu and not fallback_occurred,
            fallback_occurred=fallback_occurred,
            batches_processed=batches_processed,
        )

    def _clear_gpu_memory(self) -> None:
        """Clear GPU memory pools to recover from out-of-memory errors."""
        if not is_gpu_available():
            return

        try:
            import cupy as cp

            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()

            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()

            logger.debug("GPU memory cleared")
        except Exception as e:
            logger.warning("Failed to clear GPU memory: %s", e)


def batch_apply(
    data: NDArray[Any],
    func: Callable[[NDArray[Any]], NDArray[Any]],
    batch_size: int | None = None,
    axis: int = 0,
) -> NDArray[Any]:
    """Convenience function to apply a function in batches.

    Parameters
    ----------
    data : NDArray
        Input data array.
    func : Callable
        Function to apply to each batch.
    batch_size : int, optional
        Batch size. Default uses global configuration.
    axis : int, optional
        Axis along which to batch. Default is 0.

    Returns
    -------
    NDArray
        Processed data.

    Example
    -------
    >>> result = batch_apply(large_array, lambda x: x ** 2, batch_size=10000)
    """
    processor = BatchProcessor(batch_size=batch_size)
    result = processor.map(data, func, axis=axis)
    return result.data
