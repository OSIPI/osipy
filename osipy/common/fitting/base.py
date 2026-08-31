"""Base classes for model fitting algorithms.

This module provides the abstract base class for all fitting algorithms.
Fitters operate on ``FittableModel`` instances — models with all
independent variables already bound so only free parameters remain.

The ``BaseFitter.fit_image()`` concrete method handles mask extraction,
chunking, GPU transfer, and ParameterMap assembly. Subclasses only need
to implement ``fit_batch()``.
"""

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.backend.array_module import get_array_module, to_gpu, to_numpy
from osipy.common.backend.config import (
    get_backend,
    get_gpu_batch_size,
    is_gpu_available,
)
from osipy.common.fitting.batch import create_empty_maps, create_parameter_maps

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from osipy.common.models.fittable import FittableModel

logger = logging.getLogger(__name__)


class BaseFitter(ABC):
    """Abstract base class for model fitting algorithms.

    Subclasses implement ``fit_batch()`` for the core optimization.
    ``fit_image()`` is a concrete method that handles image-level
    boilerplate (masking, chunking, GPU, ParameterMap assembly).
    """

    r2_threshold: float = 0.5
    chunk_size: int = 10000
    fitting_method_name: str = "unknown"

    @staticmethod
    def _merge_bounds(
        model_bounds: dict[str, tuple[float, float]],
        overrides: dict[str, tuple[float, float]] | None,
    ) -> dict[str, tuple[float, float]]:
        """Merge user bound overrides with model defaults."""
        if overrides is None:
            return model_bounds
        merged = dict(model_bounds)
        merged.update(overrides)
        return merged

    @abstractmethod
    def fit_batch(
        self,
        model: FittableModel,
        observed_batch: NDArray[np.floating[Any]],
        bounds_override: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Core fitting algorithm for a batch of voxels.

        Parameters
        ----------
        model : FittableModel
            Model with all independent variables bound.
        observed_batch : NDArray
            Observed data, shape ``(n_observations, n_voxels)``.
        bounds_override : dict, optional
            Per-parameter bound overrides.

        Returns
        -------
        params : NDArray
            Fitted parameters, shape ``(n_params, n_voxels)``.
        r2 : NDArray
            R-squared values, shape ``(n_voxels,)``.
        converged : NDArray
            Convergence flags, shape ``(n_voxels,)``.
        """
        ...

    def fit_image(
        self,
        model: FittableModel,
        data: NDArray[np.floating[Any]],
        mask: NDArray[np.bool_] | None = None,
        bounds_override: dict[str, tuple[float, float]] | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> dict[str, Any]:
        """Fit model to entire image volume.

        Concrete method that handles mask extraction, chunking,
        GPU transfer, and ParameterMap assembly. Calls ``fit_batch()``
        per chunk.

        Parameters
        ----------
        model : FittableModel
            Model with all independent variables bound.
        data : NDArray
            Image data, shape ``(x, y, z, n_observations)``.
        mask : NDArray[np.bool_] | None
            Boolean mask of voxels to fit, shape ``(x, y, z)``.
        bounds_override : dict, optional
            Per-parameter bound overrides.
        progress_callback : Callable[[float], None] | None
            Progress callback (0.0 to 1.0).

        Returns
        -------
        dict[str, ParameterMap]
            Mapping of parameter names to ParameterMap objects.
        """
        nx, ny, nz, _nt = data.shape

        if mask is None:
            mask = np.ones((nx, ny, nz), dtype=bool)

        # Get masked voxel indices (numpy for scatter indexing)
        voxel_indices = np.argwhere(to_numpy(mask))
        n_voxels = len(voxel_indices)

        if n_voxels == 0:
            logger.warning("No voxels in mask to fit")
            return create_empty_maps(model, (nx, ny, nz))

        # Determine device
        use_gpu = is_gpu_available() and not get_backend().force_cpu
        logger.info(
            "Fitting %d voxels using %s (%s)",
            n_voxels,
            "GPU" if use_gpu else "CPU",
            self.fitting_method_name,
        )

        # Extract masked voxel data: (nt, n_voxels)
        observed_masked = data[mask].T

        # Transfer to GPU once (not per-chunk). to_gpu() raises instead of
        # silently falling back to NumPy on transfer failure, so use_gpu
        # stays accurate for the chunk sizing and threading decisions below.
        if use_gpu:
            observed_masked = to_gpu(observed_masked)

        xp = get_array_module(observed_masked)

        # Resolve working dtype for GPU float32 precision
        config = get_backend()
        if use_gpu and config.gpu_dtype == "float32":
            working_dtype = xp.float32
        else:
            working_dtype = xp.float64
        observed_masked = observed_masked.astype(working_dtype)

        # Ensure model's stored arrays live on the same device as observed data
        if hasattr(model, "ensure_device"):
            model.ensure_device(xp)

        n_params = len(model.parameters)

        # Initialize output arrays on the same device
        fitted_params = xp.zeros((n_params, n_voxels), dtype=working_dtype)
        r2_values = xp.zeros(n_voxels, dtype=working_dtype)
        converged = xp.zeros(n_voxels, dtype=bool)

        # Select chunk size: use GPU thread capacity when on GPU
        if use_gpu:
            gpu_threads = get_gpu_batch_size()
            effective_chunk_size = gpu_threads if gpu_threads > 0 else self.chunk_size
        else:
            effective_chunk_size = self.chunk_size

        # Process in chunks
        total_chunks = (n_voxels + effective_chunk_size - 1) // effective_chunk_size

        # Resolve effective worker count
        n_workers = config.n_workers
        if n_workers == 0:
            n_workers = os.cpu_count() or 1

        use_threading = not use_gpu and n_workers > 1 and total_chunks > 1

        if use_threading:
            logger.info("Using %d threads for %d chunks", n_workers, total_chunks)
            progress_lock = threading.Lock()
            completed_count = 0

            def _fit_chunk(
                start: int, end: int
            ) -> tuple[int, NDArray[Any], NDArray[Any], NDArray[Any]]:
                batch_observed = observed_masked[:, start:end]
                bp, br, bc = self.fit_batch(model, batch_observed, bounds_override)
                return start, bp, br, bc

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = []
                for start in range(0, n_voxels, effective_chunk_size):
                    end = min(start + effective_chunk_size, n_voxels)
                    futures.append(pool.submit(_fit_chunk, start, end))

                for future in as_completed(futures):
                    start, batch_params, batch_r2, batch_converged = future.result()
                    end = min(start + effective_chunk_size, n_voxels)
                    fitted_params[:, start:end] = batch_params
                    r2_values[start:end] = batch_r2
                    converged[start:end] = batch_converged

                    if progress_callback is not None:
                        with progress_lock:
                            completed_count += 1
                            progress_callback(completed_count / total_chunks)
        else:
            for chunk_idx, start in enumerate(range(0, n_voxels, effective_chunk_size)):
                end = min(start + effective_chunk_size, n_voxels)
                batch_observed = observed_masked[:, start:end]

                batch_params, batch_r2, batch_converged = self.fit_batch(
                    model, batch_observed, bounds_override
                )

                fitted_params[:, start:end] = batch_params
                r2_values[start:end] = batch_r2
                converged[start:end] = batch_converged

                if progress_callback is not None:
                    progress_callback((chunk_idx + 1) / total_chunks)

        # Transfer results back to CPU for parameter map creation
        fitted_params_np = to_numpy(fitted_params)
        r2_values_np = to_numpy(r2_values)
        converged_np = to_numpy(converged)

        return create_parameter_maps(
            model,
            fitted_params_np,
            r2_values_np,
            converged_np,
            voxel_indices,
            (nx, ny, nz),
            r2_threshold=self.r2_threshold,
            fitting_method=self.fitting_method_name,
        )
