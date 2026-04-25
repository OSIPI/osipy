"""DCE-MRI analysis pipeline.

This module provides an end-to-end DCE-MRI analysis pipeline,
integrating T1 mapping, signal-to-concentration conversion,
AIF handling, and pharmacokinetic model fitting.

The pipeline produces OSIPI CAPLEX-compliant parameter maps
(Ktrans, ve, vp, kep) and supports population AIFs from the
CAPLEX model registry (M.IC2.001 Parker, M.IC2.002 Georgiou).

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.29840
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.aif import (
    ArterialInputFunction,
    detect_aif,
    get_population_aif,
)
from osipy.common.backend.array_module import get_array_module
from osipy.common.dataset import PerfusionDataset
from osipy.common.exceptions import DataValidationError
from osipy.common.types import Modality
from osipy.dce import (
    DCEAcquisitionParams,
    DCEFitResult,
    compute_t1_map,
    signal_to_concentration,
)
from osipy.dce.fitting import fit_model

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray

    from osipy.common.parameter_map import ParameterMap


@dataclass
class DCEPipelineConfig:
    """Configuration for DCE-MRI pipeline.

    Attributes
    ----------
    t1_mapping_method : str
        T1 mapping method: 'vfa' or 'look_locker'.
    model : str
        Pharmacokinetic model: 'tofts', 'extended_tofts', 'patlak', '2cxm'.
    aif_source : str
        AIF source: 'population' (Parker), 'detect', or 'manual'.
    population_aif : str
        Population AIF type if aif_source='population'.
    acquisition_params : DCEAcquisitionParams | None
        Acquisition parameters for signal conversion.
    output_dir : Path | None
        Output directory for results.
    save_intermediate : bool
        Whether to save intermediate results.
    fitter : str | None
        Fitter registry name (e.g., 'lm', 'bayesian').
    concentration_method : str
        Signal-to-concentration conversion method.
    bounds_override : dict[str, tuple[float, float]] | None
        Per-parameter bound overrides for fitting.
    aif_detection_method : str
        AIF detection method when aif_source='detect'.
    initial_guess_override : dict[str, float] | None
        Per-parameter initial guess overrides for fitting.
    max_iterations : int | None
        Maximum number of fitting iterations.
    tolerance : float | None
        Convergence tolerance for fitting.
    r2_threshold : float | None
        R-squared threshold for valid fitting results.
    fit_delay : bool
        If True, jointly fit an arterial delay parameter with the DCE model
        (adds one parameter per voxel). Defaults to False.
    """

    t1_mapping_method: str = "vfa"
    model: str = "extended_tofts"
    aif_source: str = "population"
    population_aif: str = "parker"
    acquisition_params: DCEAcquisitionParams | None = None
    output_dir: Path | None = None
    save_intermediate: bool = False
    fitter: str | None = None
    concentration_method: str = "spgr"
    bounds_override: dict[str, tuple[float, float]] | None = None
    aif_detection_method: str = "multi_criteria"
    initial_guess_override: dict[str, float] | None = None
    max_iterations: int | None = None
    tolerance: float | None = None
    r2_threshold: float | None = None
    fit_delay: bool = False


@dataclass
class DCEPipelineResult:
    """Result of DCE pipeline.

    Attributes
    ----------
    fit_result : DCEFitResult
        Model fitting results.
    t1_map : ParameterMap | None
        T1 map (if computed).
    aif : ArterialInputFunction
        AIF used for analysis.
    concentration : NDArray | None
        Concentration data.
    config : DCEPipelineConfig
        Pipeline configuration used.
    """

    fit_result: DCEFitResult
    t1_map: ParameterMap | None
    aif: ArterialInputFunction
    concentration: NDArray[np.floating[Any]] | None
    config: DCEPipelineConfig


class DCEPipeline:
    """End-to-end DCE-MRI analysis pipeline.

    This pipeline performs:
    1. T1 mapping (if VFA/LL data provided)
    2. Signal-to-concentration conversion
    3. AIF extraction or population AIF generation
    4. Pharmacokinetic model fitting
    5. Parameter map generation

    Examples
    --------
    >>> from osipy.pipeline import DCEPipeline, DCEPipelineConfig
    >>> config = DCEPipelineConfig(model='extended_tofts')
    >>> pipeline = DCEPipeline(config)
    >>> result = pipeline.run(dce_data, time, t1_data=vfa_data)
    """

    def __init__(self, config: DCEPipelineConfig | None = None) -> None:
        """Initialize DCE pipeline.

        Parameters
        ----------
        config : DCEPipelineConfig | None
            Pipeline configuration.
        """
        self.config = config or DCEPipelineConfig()

    def run(
        self,
        dce_data: PerfusionDataset | NDArray[np.floating[Any]],
        time: NDArray[np.floating[Any]],
        t1_data: PerfusionDataset | NDArray[np.floating[Any]] | None = None,
        t1_map: ParameterMap | None = None,
        aif: ArterialInputFunction | NDArray[np.floating[Any]] | None = None,
        mask: NDArray[np.bool_] | None = None,
        flip_angles: NDArray[np.floating[Any]] | None = None,
        tr: float | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> DCEPipelineResult:
        """Run DCE-MRI analysis pipeline.

        Parameters
        ----------
        dce_data : PerfusionDataset or NDArray
            DCE-MRI signal data, shape (..., n_timepoints).
        time : NDArray
            Time points in seconds.
        t1_data : PerfusionDataset or NDArray, optional
            T1 mapping data (VFA or Look-Locker).
        t1_map : ParameterMap, optional
            Pre-computed T1 map. If provided, skips T1 mapping.
        aif : ArterialInputFunction or NDArray, optional
            Custom AIF. If None, uses config.aif_source.
        mask : NDArray, optional
            Brain/tissue mask.
        flip_angles : NDArray, optional
            Flip angles for VFA T1 mapping (degrees).
        tr : float, optional
            TR for T1 mapping (ms).
        progress_callback : Callable, optional
            Callback for progress updates (step_name, progress).

        Returns
        -------
        DCEPipelineResult
            Pipeline results.
        """
        # Extract data array
        signal = dce_data.data if isinstance(dce_data, PerfusionDataset) else dce_data

        # Step 1: T1 mapping (if needed)
        if progress_callback:
            progress_callback("T1 Mapping", 0.0)

        if t1_map is None and t1_data is not None:
            t1_map = self._compute_t1_map(t1_data, flip_angles, tr)

        if progress_callback:
            progress_callback("T1 Mapping", 1.0)

        # Step 2: Signal to concentration
        if progress_callback:
            progress_callback("Signal Conversion", 0.0)

        acq_params = self.config.acquisition_params or DCEAcquisitionParams()
        if t1_map is not None or acq_params.t1_assumed is not None:
            concentration = signal_to_concentration(
                signal=signal,
                t1_map=t1_map,
                acquisition_params=acq_params,
                method=self.config.concentration_method,
            )
        else:
            # Assume input is already concentration or use direct signal
            concentration = signal

        if progress_callback:
            progress_callback("Signal Conversion", 1.0)

        # Step 3: Get AIF
        if progress_callback:
            progress_callback("AIF Processing", 0.0)

        if aif is None:
            aif = self._get_aif(concentration, time, mask)

        if progress_callback:
            progress_callback("AIF Processing", 1.0)

        # Step 4: Model fitting (registry-driven)
        if progress_callback:
            progress_callback("Model Fitting", 0.0)

        fit_mask = self._build_fit_mask(concentration, t1_map, mask)

        fit_result = fit_model(
            model_name=self.config.model,
            concentration=concentration,
            aif=aif,
            time=time,
            mask=fit_mask,
            fitter=self.config.fitter,
            bounds_override=self.config.bounds_override,
            fit_delay=self.config.fit_delay,
            progress_callback=lambda p: (
                progress_callback("Model Fitting", p) if progress_callback else None
            ),
        )

        if progress_callback:
            progress_callback("Model Fitting", 1.0)

        return DCEPipelineResult(
            fit_result=fit_result,
            t1_map=t1_map,
            aif=aif
            if isinstance(aif, ArterialInputFunction)
            else ArterialInputFunction(time=time, concentration=aif),
            concentration=concentration,
            config=self.config,
        )

    def _build_fit_mask(
        self,
        concentration: NDArray[np.floating[Any]],
        t1_map: ParameterMap | None,
        user_mask: NDArray[np.bool_] | None,
    ) -> NDArray[np.bool_] | None:
        """Pass through the caller's mask unchanged.

        The pipeline does not add R² or T1-quality filters here — callers
        filter on r² or parameter values themselves after inspecting the
        returned maps.
        """
        if user_mask is None:
            return None
        spatial_shape = concentration.shape[:-1]
        xp = get_array_module(concentration)
        return xp.broadcast_to(xp.asarray(user_mask), spatial_shape).copy()

    def _compute_t1_map(
        self,
        t1_data: PerfusionDataset | NDArray[np.floating[Any]],
        flip_angles: NDArray[np.floating[Any]] | None,
        tr: float | None,
    ) -> ParameterMap:
        """Compute T1 map from input data.

        Returns a ``ParameterMap`` whose ``quality_mask`` carries the T1-fit
        quality info the downstream fit mask filter needs.
        """
        from osipy.common.parameter_map import ParameterMap
        from osipy.dce.t1_mapping import compute_t1_vfa

        signal = t1_data.data if isinstance(t1_data, PerfusionDataset) else t1_data
        affine = t1_data.affine if isinstance(t1_data, PerfusionDataset) else np.eye(4)

        if self.config.t1_mapping_method == "vfa":
            if flip_angles is None or tr is None:
                msg = "flip_angles and tr required for VFA T1 mapping"
                raise DataValidationError(msg)
            t1_result = compute_t1_vfa(
                signal=signal,
                flip_angles=flip_angles,
                tr=tr,
                method="linear",
            )
            return ParameterMap(
                name="T1",
                symbol="T1",
                units="ms",
                values=t1_result.t1_map.values,
                affine=affine,
                quality_mask=t1_result.quality_mask,
            )

        # Look-Locker branch
        if not isinstance(t1_data, PerfusionDataset):
            t1_data = PerfusionDataset(data=signal, modality=Modality.DCE)
        ll_result = compute_t1_map(t1_data, method="look_locker")
        return ParameterMap(
            name="T1",
            symbol="T1",
            units="ms",
            values=ll_result.t1_map.values,
            affine=affine,
            quality_mask=ll_result.quality_mask,
        )

    def _get_aif(
        self,
        concentration: NDArray[np.floating[Any]],
        time: NDArray[np.floating[Any]],
        mask: NDArray[np.bool_] | None,
    ) -> ArterialInputFunction:
        """Get AIF based on configuration."""
        if self.config.aif_source == "population":
            aif_model = get_population_aif(self.config.population_aif)
            return aif_model(time)

        elif self.config.aif_source == "detect":
            dataset = PerfusionDataset(
                data=concentration,
                modality=Modality.DCE,
                time=time,
            )
            result = detect_aif(
                dataset, roi_mask=mask, method=self.config.aif_detection_method
            )
            return result.aif

        else:
            msg = f"Unknown AIF source: {self.config.aif_source}"
            raise DataValidationError(msg)
