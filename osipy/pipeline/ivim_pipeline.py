"""IVIM analysis pipeline.

This module provides an end-to-end IVIM-DWI analysis pipeline,
integrating preprocessing and bi-exponential model fitting.

The pipeline produces parameter maps: D, D*, and f.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Le Bihan D et al. (1988). Radiology 168(2):497-505.
.. [3] Dickie BR et al. MRM 2024. doi:10.1002/mrm.29840
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.dataset import PerfusionDataset
from osipy.ivim import (
    FittingMethod,
    IVIMFitParams,
    IVIMFitResult,
    fit_ivim,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray


@dataclass
class IVIMPipelineConfig:
    """Configuration for IVIM pipeline.

    Attributes
    ----------
    fitting_method : FittingMethod
        IVIM fitting method.
    b_threshold : float
        b-value threshold for segmented fitting (s/mm²).
    normalize_signal : bool
        Whether to normalize signal by b=0.
    output_dir : Path | None
        Output directory for results.
    bounds : dict[str, tuple[float, float]] | None
        Custom parameter bounds, e.g. ``{"D": (1e-4, 5e-3)}``.
    initial_guess : dict[str, float] | None
        Custom initial parameter estimates, e.g. ``{"D": 1e-3}``.
    max_iterations : int
        Maximum iterations for optimization.
    tolerance : float
        Convergence tolerance.
    bayesian_params : dict | None
        Bayesian-specific parameters (keys: ``prior_std``,
        ``noise_std``, ``compute_uncertainty``). Only used when
        ``fitting_method`` is ``FittingMethod.BAYESIAN``.
    """

    fitting_method: FittingMethod = FittingMethod.SEGMENTED
    b_threshold: float = 200.0
    normalize_signal: bool = True
    output_dir: Path | None = None
    bounds: dict[str, tuple[float, float]] | None = None
    initial_guess: dict[str, float] | None = None
    max_iterations: int = 500
    tolerance: float = 1e-6
    bayesian_params: Any = None


@dataclass
class IVIMPipelineResult:
    """Result of IVIM pipeline.

    Attributes
    ----------
    fit_result : IVIMFitResult
        IVIM fitting results.
    config : IVIMPipelineConfig
        Pipeline configuration used.
    """

    fit_result: IVIMFitResult
    config: IVIMPipelineConfig


class IVIMPipeline:
    """End-to-end IVIM-DWI analysis pipeline.

    This pipeline performs:
    1. Signal normalization (optional)
    2. IVIM bi-exponential model fitting
    3. Parameter map generation (D, D*, f)

    Examples
    --------
    >>> from osipy.pipeline import IVIMPipeline, IVIMPipelineConfig
    >>> config = IVIMPipelineConfig(fitting_method=FittingMethod.SEGMENTED)
    >>> pipeline = IVIMPipeline(config)
    >>> result = pipeline.run(dwi_data, b_values)
    """

    def __init__(self, config: IVIMPipelineConfig | None = None) -> None:
        """Initialize IVIM pipeline.

        Parameters
        ----------
        config : IVIMPipelineConfig | None
            Pipeline configuration.
        """
        self.config = config or IVIMPipelineConfig()

    def run(
        self,
        dwi_data: PerfusionDataset | NDArray[np.floating[Any]],
        b_values: NDArray[np.floating[Any]],
        mask: NDArray[np.bool_] | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> IVIMPipelineResult:
        """Run IVIM analysis pipeline.

        Parameters
        ----------
        dwi_data : PerfusionDataset or NDArray
            DWI signal data, shape (..., n_b_values).
        b_values : NDArray
            b-values in s/mm².
        mask : NDArray, optional
            Brain/tissue mask.
        progress_callback : Callable, optional
            Callback for progress updates.

        Returns
        -------
        IVIMPipelineResult
            Pipeline results.
        """
        # Extract data array
        signal = dwi_data.data if isinstance(dwi_data, PerfusionDataset) else dwi_data

        # Step 1: Preprocessing
        if progress_callback:
            progress_callback("Preprocessing", 0.0)

        if self.config.normalize_signal:
            # Normalize by b=0 signal
            b0_idx = np.argmin(b_values)
            s0 = signal[..., b0_idx : b0_idx + 1]
            s0 = np.maximum(s0, 1e-10)
            signal_normalized = signal / s0
        else:
            signal_normalized = signal

        if progress_callback:
            progress_callback("Preprocessing", 1.0)

        # Step 2: IVIM fitting
        if progress_callback:
            progress_callback("IVIM Fitting", 0.0)

        fit_params = IVIMFitParams(
            method=self.config.fitting_method,
            b_threshold=self.config.b_threshold,
            bounds=self.config.bounds,
            max_iterations=self.config.max_iterations,
            tolerance=self.config.tolerance,
            bayesian_params=self.config.bayesian_params,
        )

        fit_result = fit_ivim(
            signal=signal_normalized,
            b_values=b_values,
            mask=mask,
            params=fit_params,
            progress_callback=lambda p: progress_callback("IVIM Fitting", p)
            if progress_callback
            else None,
        )

        if progress_callback:
            progress_callback("IVIM Fitting", 1.0)

        return IVIMPipelineResult(
            fit_result=fit_result,
            config=self.config,
        )
