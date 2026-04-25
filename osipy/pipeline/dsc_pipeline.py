"""DSC-MRI analysis pipeline.

This module provides an end-to-end DSC-MRI analysis pipeline,
integrating signal conversion, leakage correction, deconvolution,
and perfusion parameter estimation.

The pipeline produces OSIPI CAPLEX-compliant perfusion maps
(CBF, CBV, MTT) via SVD-based deconvolution.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.29840
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.dataset import PerfusionDataset
from osipy.dsc import (
    DSCPerfusionMaps,
    compute_perfusion_maps,
    correct_leakage,
    signal_to_delta_r2,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray


@dataclass
class DSCPipelineConfig:
    """Configuration for DSC-MRI pipeline.

    Attributes
    ----------
    te : float
        Echo time in milliseconds.
    deconvolution_method : str
        Deconvolution method: 'oSVD', 'cSVD', or 'sSVD'.
    apply_leakage_correction : bool
        Whether to apply leakage correction.
    svd_threshold : float
        SVD truncation threshold.
    output_dir : Path | None
        Output directory for results.
    """

    te: float = 30.0
    deconvolution_method: str = "oSVD"
    apply_leakage_correction: bool = True
    svd_threshold: float = 0.2
    output_dir: Path | None = None


@dataclass
class DSCPipelineResult:
    """Result of DSC pipeline.

    Attributes
    ----------
    perfusion_maps : DSCPerfusionMaps
        Computed perfusion parameter maps.
    delta_r2 : NDArray
        ΔR2* data (original or leakage-corrected).
    aif : NDArray
        AIF used for analysis.
    config : DSCPipelineConfig
        Pipeline configuration used.
    """

    perfusion_maps: DSCPerfusionMaps
    delta_r2: NDArray[np.floating[Any]]
    aif: NDArray[np.floating[Any]]
    config: DSCPipelineConfig


class DSCPipeline:
    """End-to-end DSC-MRI analysis pipeline.

    This pipeline performs:
    1. Signal to ΔR2* conversion
    2. AIF extraction
    3. Leakage correction (optional)
    4. SVD deconvolution
    5. Perfusion parameter estimation (CBF, CBV, MTT)

    Examples
    --------
    >>> from osipy.pipeline import DSCPipeline, DSCPipelineConfig
    >>> config = DSCPipelineConfig(te=25.0, apply_leakage_correction=True)
    >>> pipeline = DSCPipeline(config)
    >>> result = pipeline.run(dsc_signal, time, aif_signal)
    """

    def __init__(self, config: DSCPipelineConfig | None = None) -> None:
        """Initialize DSC pipeline.

        Parameters
        ----------
        config : DSCPipelineConfig | None
            Pipeline configuration.
        """
        self.config = config or DSCPipelineConfig()

    def run(
        self,
        dsc_signal: PerfusionDataset | NDArray[np.floating[Any]],
        time: NDArray[np.floating[Any]],
        aif_signal: NDArray[np.floating[Any]] | None = None,
        aif_voxels: NDArray[np.bool_] | None = None,
        mask: NDArray[np.bool_] | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> DSCPipelineResult:
        """Run DSC-MRI analysis pipeline.

        Parameters
        ----------
        dsc_signal : PerfusionDataset or NDArray
            DSC-MRI signal data, shape (..., n_timepoints).
        time : NDArray
            Time points in seconds.
        aif_signal : NDArray, optional
            Pre-extracted AIF signal. If None, extracts from data.
        aif_voxels : NDArray, optional
            Mask of AIF voxels for extraction.
        mask : NDArray, optional
            Brain mask.
        progress_callback : Callable, optional
            Callback for progress updates.

        Returns
        -------
        DSCPipelineResult
            Pipeline results.
        """
        # Extract data array
        if isinstance(dsc_signal, PerfusionDataset):
            signal = dsc_signal.data
        else:
            signal = dsc_signal

        # Step 1: Signal to ΔR2*
        if progress_callback:
            progress_callback("Signal Conversion", 0.0)

        delta_r2 = signal_to_delta_r2(signal, self.config.te)

        if progress_callback:
            progress_callback("Signal Conversion", 1.0)

        # Step 2: Extract or process AIF
        if progress_callback:
            progress_callback("AIF Processing", 0.0)

        if aif_signal is not None:
            aif = signal_to_delta_r2(aif_signal, self.config.te)
        elif aif_voxels is not None:
            # Extract AIF from specified voxels
            aif = np.mean(delta_r2[aif_voxels], axis=0)
        else:
            # Simple AIF detection: use high enhancement voxels
            peak_enhancement = np.max(delta_r2, axis=-1)
            threshold = np.percentile(
                peak_enhancement[mask] if mask is not None else peak_enhancement, 95
            )
            aif_mask = peak_enhancement > threshold
            if mask is not None:
                aif_mask &= mask
            aif = np.mean(delta_r2[aif_mask], axis=0)

        if progress_callback:
            progress_callback("AIF Processing", 1.0)

        # Step 3: Leakage correction
        if progress_callback:
            progress_callback("Leakage Correction", 0.0)

        if self.config.apply_leakage_correction:
            leakage_result = correct_leakage(
                delta_r2=delta_r2,
                aif=aif,
                time=time,
                mask=mask,
            )
            delta_r2_corrected = leakage_result.corrected_delta_r2
        else:
            delta_r2_corrected = delta_r2

        if progress_callback:
            progress_callback("Leakage Correction", 1.0)

        # Step 4: Compute perfusion maps
        if progress_callback:
            progress_callback("Perfusion Computation", 0.0)

        perfusion_maps = compute_perfusion_maps(
            delta_r2=delta_r2_corrected,
            aif=aif,
            time=time,
            mask=mask,
            deconvolve=True,
            deconvolution_method=self.config.deconvolution_method,
            svd_threshold=self.config.svd_threshold,
        )

        if progress_callback:
            progress_callback("Perfusion Computation", 1.0)

        return DSCPipelineResult(
            perfusion_maps=perfusion_maps,
            delta_r2=delta_r2_corrected,
            aif=aif,
            config=self.config,
        )
