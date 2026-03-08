"""High-level fitting functions for DCE-MRI analysis.

This module provides user-friendly interfaces for fitting pharmacokinetic
models to DCE-MRI data, with automatic parameter estimation, uncertainty
quantification, and quality assessment.

Fits OSIPI CAPLEX quantities including:
    - Ktrans (OSIPI: Q.PH1.008): Volume transfer constant, 1/min
    - ve (OSIPI: Q.PH1.001): Extravascular extracellular volume fraction, mL/100mL
    - vp (OSIPI: Q.PH1.001): Plasma volume fraction, mL/100mL
    - Fp (OSIPI: Q.PH1.002): Plasma flow, mL/min/100mL
    - PS (OSIPI: Q.PH1.004): Permeability-surface area product, mL/min/100mL

GPU Acceleration Note:
    This module accepts GPU arrays (CuPy) as input. The default
    LevenbergMarquardtFitter automatically uses GPU when available,
    falling back to CPU. No scipy dependency.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Tofts PS et al. J Magn Reson Imaging 1999;10(3):223-232.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.aif.base import ArterialInputFunction
from osipy.common.backend.array_module import get_array_module, to_gpu, to_numpy
from osipy.common.backend.config import get_backend, is_gpu_available
from osipy.common.exceptions import FittingError
from osipy.common.fitting.least_squares import LevenbergMarquardtFitter
from osipy.common.fitting.registry import get_fitter
from osipy.common.parameter_map import ParameterMap
from osipy.dce.models.binding import BoundDCEModel
from osipy.dce.models.registry import get_model

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from osipy.common.fitting.base import BaseFitter


@dataclass
class DCEFitResult:
    """Result container for DCE model fitting.

    Attributes
    ----------
    parameter_maps : dict[str, ParameterMap]
        Parameter maps with uncertainty and quality information.
    quality_mask : NDArray[np.bool_]
        Boolean mask indicating successfully fitted voxels.
    model_name : str
        Name of the fitted model.
    r_squared_map : NDArray[np.floating] | None
        R-squared goodness-of-fit map.
    residual_map : NDArray[np.floating] | None
        Sum of squared residuals per voxel.
    fitting_stats : dict[str, Any]
        Summary statistics from the fitting process.
    """

    parameter_maps: dict[str, ParameterMap]
    quality_mask: NDArray[np.bool_]
    model_name: str
    r_squared_map: NDArray[np.floating[Any]] | None = None
    residual_map: NDArray[np.floating[Any]] | None = None
    fitting_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSelectionResult:
    """Result container for model selection comparison.

    Holds the per-model fitting results and information-criterion
    score maps so the caller can inspect which model won at each
    voxel and why.

    Attributes
    ----------
    best_model_map : NDArray[np.intp]
        Integer map where each voxel value is the index into
        ``model_names`` of the winning model. Winner is determined
        by lowest AIC/BIC or highest R-squared, depending on
        ``criterion``.
    model_names : list[str]
        Ordered list of successfully compared model names.
    criterion : str
        Criterion used for comparison
        (``'aic'``, ``'bic'``, or ``'r_squared'``).
    aic_maps : dict[str, NDArray[np.floating]]
        Per-model AIC score maps.
    bic_maps : dict[str, NDArray[np.floating]]
        Per-model BIC score maps.
    r_squared_maps : dict[str, NDArray[np.floating]]
        Per-model R-squared maps.
    fit_results : dict[str, DCEFitResult]
        Full fitting results for each model.
    """

    best_model_map: NDArray[np.intp]
    model_names: list[str]
    criterion: str
    aic_maps: dict[str, NDArray[np.floating[Any]]] = field(default_factory=dict)
    bic_maps: dict[str, NDArray[np.floating[Any]]] = field(default_factory=dict)
    r_squared_maps: dict[str, NDArray[np.floating[Any]]] = field(default_factory=dict)
    fit_results: dict[str, DCEFitResult] = field(default_factory=dict)


def fit_model(
    model_name: str,
    concentration: NDArray[np.floating[Any]],
    aif: ArterialInputFunction | NDArray[np.floating[Any]],
    time: NDArray[np.floating[Any]],
    mask: NDArray[np.bool_] | None = None,
    fitter: BaseFitter | str | None = None,
    progress_callback: Callable[[float], None] | None = None,
    bounds_override: dict[str, tuple[float, float]] | None = None,
    fit_delay: bool = False,
) -> DCEFitResult:
    """Fit a named DCE pharmacokinetic model to concentration data.

    This is the unified fitting entry point. It looks up the model from the
    registry and delegates to the shared fitting implementation.

    Parameters
    ----------
    model_name : str
        Model name: 'tofts', 'extended_tofts', 'patlak', '2cxm',
        or any name added via ``register_model()``.
    concentration : NDArray[np.floating]
        Concentration data, shape (x, y, z, t) or (x, y, t) or (n_voxels, t).
    aif : ArterialInputFunction or NDArray[np.floating]
        Arterial input function. Can be an ArterialInputFunction object
        or a 1D array of concentration values.
    time : NDArray[np.floating]
        Time points in seconds.
    mask : NDArray[np.bool_] | None
        Optional mask of voxels to fit. If None, fits all voxels.
    fitter : BaseFitter | str | None
        Fitter instance or registry name (e.g., 'lm', 'bayesian').
        Uses LevenbergMarquardtFitter by default.
    progress_callback : Callable[[float], None] | None
        Optional callback for progress updates.
    bounds_override : dict[str, tuple[float, float]] | None
        Optional per-parameter bound overrides.
    fit_delay : bool
        If True, adds an arterial delay parameter to the model and fits
        it jointly with the other parameters. Default False.

    Returns
    -------
    DCEFitResult
        Fitting result with parameter maps, quality mask, and statistics.

    Raises
    ------
    FittingError
        If fitting fails or data is invalid.
    DataValidationError
        If model name or fitter name is not recognized.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.dce.fitting import fit_model
    >>> from osipy.common.aif import ParkerAIF
    >>>
    >>> t = np.linspace(0, 300, 60)
    >>> aif = ParkerAIF()(t)
    >>> concentration = np.random.rand(64, 64, 20, 60)
    >>>
    >>> result = fit_model("extended_tofts", concentration, aif, t)
    >>> ktrans_map = result.parameter_maps["Ktrans"]
    """
    model = get_model(model_name)
    if fit_delay:
        model = _DelayAwareModel(model)
    return _fit_model_impl(
        model,
        concentration,
        aif,
        time,
        mask,
        fitter,
        progress_callback,
        bounds_override,
    )


def compare_models(
    model_names: list[str],
    concentration: NDArray[np.floating[Any]],
    aif: ArterialInputFunction | NDArray[np.floating[Any]],
    time: NDArray[np.floating[Any]],
    mask: NDArray[np.bool_] | None = None,
    criterion: str = "r_squared",
    fitter: BaseFitter | str | None = None,
    bounds_override: dict[str, tuple[float, float]] | None = None,
) -> ModelSelectionResult:
    """Fit multiple models and select the best one per voxel.

    Fits each model independently using ``fit_model()``, computes
    AIC, BIC, and R-squared at every voxel, and returns a map
    indicating which model won at each spatial location.

    Parameters
    ----------
    model_names : list[str]
        Models to compare, e.g. ``['tofts', 'extended_tofts', 'patlak']``.
        Must contain at least two unique names.
    concentration : NDArray[np.floating]
        Concentration data, shape ``(x, y, z, t)`` or ``(x, y, t)``
        or ``(n_voxels, t)``.
    aif : ArterialInputFunction or NDArray[np.floating]
        Arterial input function. Can be an ArterialInputFunction
        object or a 1D array of concentration values.
    time : NDArray[np.floating]
        Time points in seconds.
    mask : NDArray[np.bool_] | None
        Optional mask of voxels to fit. If None, fits all voxels.
    criterion : str
        Selection criterion: ``'aic'``, ``'bic'``, or ``'r_squared'``.
        Default ``'r_squared'``. Note that ``'r_squared'`` does not
        penalize model complexity and may favor overfitting.
    fitter : BaseFitter | str | None
        Fitter instance or registry name (e.g., ``'lm'``).
        Uses LevenbergMarquardtFitter by default.
    bounds_override : dict[str, tuple[float, float]] | None
        Optional per-parameter bound overrides.

    Returns
    -------
    ModelSelectionResult
        Comparison results including best-model map, AIC/BIC/R-squared
        score maps, and the full ``DCEFitResult`` for each model.

    Raises
    ------
    FittingError
        If fewer than two models are provided, criterion is invalid,
        duplicate model names are given, or fewer than two models
        fit successfully.
    DataValidationError
        If a model name is not recognized in the registry.

    Notes
    -----
    AIC and BIC are computed from the residual sum of squares
    assuming Gaussian errors:

        AIC = n * ln(SS_res / n) + 2 * k
        BIC = n * ln(SS_res / n) + k * ln(n)

    where *n* is the number of time points and *k* is the number
    of model parameters. Lower values indicate a better fit.
    BIC penalizes complexity more heavily than AIC, especially
    for larger *n*.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.dce.fitting import compare_models
    >>> t = np.linspace(0, 300, 60)
    >>> aif = np.exp(-t / 30)
    >>> conc = np.random.rand(10, 10, 5, 60) * 0.01
    >>> result = compare_models(
    ...     ["tofts", "extended_tofts"],
    ...     conc, aif, t,
    ...     criterion="bic",
    ... )
    >>> result.best_model_map.shape
    (10, 10, 5)
    """
    # --- Input validation ---------------------------------------------------
    if len(model_names) < 2:
        msg = "compare_models requires at least two model names."
        raise FittingError(msg)

    if len(model_names) != len(set(model_names)):
        msg = "Duplicate model names are not allowed."
        raise FittingError(msg)

    criterion = criterion.lower()
    if criterion not in ("aic", "bic", "r_squared"):
        msg = f"criterion must be 'aic', 'bic', or 'r_squared', got '{criterion}'"
        raise FittingError(msg)

    n_time = len(time)
    spatial_shape = concentration.shape[:-1]

    # --- Pre-compute SS_tot once (shared across all models) -----------------
    ct = np.asarray(concentration)
    ct_mean = np.mean(ct, axis=-1, keepdims=True)
    ss_tot = np.sum((ct - ct_mean) ** 2, axis=-1)

    # Guard against constant-signal voxels (SS_tot = 0)
    ss_tot_safe = np.where(ss_tot > 1e-30, ss_tot, 1e-30)

    # --- Fit each model and compute AIC/BIC ---------------------------------
    fit_results: dict[str, DCEFitResult] = {}
    aic_maps: dict[str, np.ndarray] = {}
    bic_maps: dict[str, np.ndarray] = {}
    r_squared_maps: dict[str, np.ndarray] = {}
    succeeded: list[str] = []

    for name in model_names:
        logger.info("Fitting model '%s' for model comparison", name)
        try:
            result = fit_model(
                name,
                concentration,
                aif,
                time,
                mask=mask,
                fitter=fitter,
                bounds_override=bounds_override,
            )
        except Exception:
            logger.warning(
                "Model '%s' failed to fit; skipping it in comparison.",
                name,
                exc_info=True,
            )
            continue

        fit_results[name] = result
        succeeded.append(name)

        # Number of free parameters for this model
        n_params = len(result.parameter_maps)

        # Get R-squared map (may be None if computation failed)
        r2 = result.r_squared_map
        if r2 is None:
            r2 = np.zeros(spatial_shape)

        # Clamp R-squared to [0, 1] to prevent negative SS_res
        r2 = np.clip(r2, 0.0, 1.0)
        r_squared_maps[name] = r2

        # SS_res = (1 - R-squared) * SS_tot
        ss_res = (1.0 - r2) * ss_tot_safe

        # Avoid log(0) by clamping
        ss_res_safe = np.where(ss_res > 1e-30, ss_res, 1e-30)

        # AIC = n * ln(SS_res / n) + 2 * k
        aic = n_time * np.log(ss_res_safe / n_time) + 2.0 * n_params
        # BIC = n * ln(SS_res / n) + k * ln(n)
        bic = n_time * np.log(ss_res_safe / n_time) + n_params * np.log(n_time)

        aic_maps[name] = aic
        bic_maps[name] = bic

    # --- Post-fit validation ------------------------------------------------
    if len(succeeded) < 2:
        msg = (
            f"Only {len(succeeded)} model(s) fitted successfully "
            f"({succeeded}); need at least 2 for comparison."
        )
        raise FittingError(msg)

    # --- Build best-model map -----------------------------------------------
    if criterion == "r_squared":
        # Higher R-squared = better fit
        stacked = np.stack([r_squared_maps[n] for n in succeeded], axis=0)
        best_model_map = np.argmax(stacked, axis=0).astype(np.intp)
    else:
        # Lower AIC/BIC = better fit
        score_maps = aic_maps if criterion == "aic" else bic_maps
        stacked = np.stack([score_maps[n] for n in succeeded], axis=0)
        best_model_map = np.argmin(stacked, axis=0).astype(np.intp)

    return ModelSelectionResult(
        best_model_map=best_model_map,
        model_names=list(succeeded),
        criterion=criterion,
        aic_maps=aic_maps,
        bic_maps=bic_maps,
        r_squared_maps=r_squared_maps,
        fit_results=fit_results,
    )


def _fit_model_impl(
    model: Any,
    concentration: NDArray[np.floating[Any]],
    aif: ArterialInputFunction | NDArray[np.floating[Any]],
    time: NDArray[np.floating[Any]],
    mask: NDArray[np.bool_] | None = None,
    fitter: BaseFitter | str | None = None,
    progress_callback: Callable[[float], None] | None = None,
    bounds_override: dict[str, tuple[float, float]] | None = None,
) -> DCEFitResult:
    """Shared fitting implementation for all DCE models.

    Parameters
    ----------
    model : BasePerfusionModel
        Model instance.
    concentration : NDArray[np.floating]
        Concentration data.
    aif : ArterialInputFunction or NDArray[np.floating]
        Arterial input function.
    time : NDArray[np.floating]
        Time points in seconds.
    mask : NDArray[np.bool_] | None
        Optional mask of voxels to fit.
    fitter : BaseFitter | str | None
        Custom fitter instance or registry name.
    progress_callback : Callable[[float], None] | None
        Optional progress callback.
    bounds_override : dict[str, tuple[float, float]] | None
        Optional per-parameter bound overrides.

    Returns
    -------
    DCEFitResult
        Fitting result.
    """
    # Extract AIF concentration array
    aif_conc = aif.concentration if isinstance(aif, ArterialInputFunction) else aif

    # Move all data to GPU once if available — stays there until export
    use_gpu = is_gpu_available() and not get_backend().force_cpu
    if use_gpu:
        concentration = to_gpu(concentration)
        aif_conc = to_gpu(aif_conc)
        time = to_gpu(time)
        if mask is not None:
            mask = to_gpu(mask)

    xp = get_array_module(concentration, aif_conc, time)

    # Validate inputs
    if len(time) != aif_conc.shape[0]:
        msg = (
            f"Time array length ({len(time)}) does not match "
            f"AIF length ({aif_conc.shape[0]})"
        )
        raise FittingError(msg)

    if concentration.shape[-1] != len(time):
        msg = (
            f"Concentration time dimension ({concentration.shape[-1]}) "
            f"does not match time array length ({len(time)})"
        )
        raise FittingError(msg)

    # Resolve fitter
    if isinstance(fitter, str):
        fitter = get_fitter(fitter)
    elif fitter is None:
        fitter = LevenbergMarquardtFitter()

    # Handle different input shapes
    original_shape = concentration.shape[:-1]  # Spatial shape
    ndim = len(original_shape)

    if ndim == 1:
        # (n_voxels, t) - already flat
        ct_4d = concentration.reshape(original_shape[0], 1, 1, len(time))
        spatial_shape = (original_shape[0], 1, 1)
    elif ndim == 2:
        # (x, y, t) -> (x, y, 1, t)
        ct_4d = concentration[:, :, xp.newaxis, :]
        spatial_shape = (original_shape[0], original_shape[1], 1)
    elif ndim == 3:
        # (x, y, z, t)
        ct_4d = concentration
        spatial_shape = original_shape
    else:
        msg = f"Invalid concentration shape: {concentration.shape}"
        raise FittingError(msg)

    # Handle mask
    if mask is None:
        fit_mask = xp.ones(spatial_shape, dtype=bool)
    elif mask.ndim == 2 and ndim >= 2:
        fit_mask = mask[:, :, xp.newaxis]
        fit_mask = xp.broadcast_to(fit_mask, spatial_shape).copy()
    else:
        fit_mask = mask

    # Create bound model (fixes time and AIF so fitter only sees free params)
    bound_model = BoundDCEModel(model, time, aif_conc)

    # Perform fitting
    try:
        param_maps = fitter.fit_image(
            model=bound_model,
            data=ct_4d,
            mask=fit_mask,
            bounds_override=bounds_override,
            progress_callback=progress_callback,
        )
    except Exception as e:
        msg = f"Fitting failed: {e}"
        raise FittingError(msg) from e

    # Build quality mask (voxels with valid fits)
    # Standardized to > 0 for all models
    first_param_name = next(iter(param_maps.keys()))
    first_map = param_maps[first_param_name]
    first_values = xp.asarray(first_map.values)
    quality_mask = xp.isfinite(first_values) & (first_values > 0)

    # Compute R-squared map (vectorized)
    r_squared_map = _compute_r_squared_vectorized(
        ct_4d,
        bound_model,
        param_maps,
        quality_mask,
        xp,
    )

    # Reshape outputs if needed
    if ndim == 1:
        # Flatten back
        for name, pmap in param_maps.items():
            n0 = original_shape[0]
            unc = (
                pmap.uncertainty.ravel()[:n0] if pmap.uncertainty is not None else None
            )
            qm = (
                pmap.quality_mask.ravel()[:n0]
                if pmap.quality_mask is not None
                else None
            )
            param_maps[name] = ParameterMap(
                values=pmap.values.ravel()[:n0],
                name=pmap.name,
                symbol=pmap.symbol,
                units=pmap.units,
                affine=pmap.affine,
                uncertainty=unc,
                quality_mask=qm,
            )
        quality_mask = quality_mask.ravel()[: original_shape[0]]
        if r_squared_map is not None:
            r_squared_map = r_squared_map.ravel()[: original_shape[0]]
    elif ndim == 2:
        # Remove z dimension
        for name, pmap in param_maps.items():
            unc = pmap.uncertainty[:, :, 0] if pmap.uncertainty is not None else None
            qm = pmap.quality_mask[:, :, 0] if pmap.quality_mask is not None else None
            param_maps[name] = ParameterMap(
                values=pmap.values[:, :, 0],
                name=pmap.name,
                symbol=pmap.symbol,
                units=pmap.units,
                affine=pmap.affine,
                uncertainty=unc,
                quality_mask=qm,
            )
        quality_mask = quality_mask[:, :, 0]
        if r_squared_map is not None:
            r_squared_map = r_squared_map[:, :, 0]

    # Ensure outputs are numpy for downstream compatibility (saves, stats)
    quality_mask = to_numpy(quality_mask)
    if r_squared_map is not None:
        r_squared_map = to_numpy(r_squared_map)

    # Compute statistics
    fitting_stats = _compute_fitting_stats(param_maps, quality_mask)

    return DCEFitResult(
        parameter_maps=param_maps,
        quality_mask=quality_mask,
        model_name=model.name,
        r_squared_map=r_squared_map,
        fitting_stats=fitting_stats,
    )


def _compute_r_squared_vectorized(
    ct_4d: NDArray[np.floating[Any]],
    bound_model: BoundDCEModel,
    param_maps: dict[str, ParameterMap],
    quality_mask: NDArray[np.bool_],
    xp: Any,
) -> NDArray[np.floating[Any]]:
    """Compute R-squared goodness-of-fit map using vectorized operations.

    Uses the bound model's ``predict_array_batch`` for GPU-accelerated
    computation instead of per-voxel iteration.

    Parameters
    ----------
    ct_4d : NDArray
        4D concentration data, shape (x, y, z, t).
    bound_model : BoundDCEModel
        Bound model with time and AIF already fixed.
    param_maps : dict
        Fitted parameter maps.
    quality_mask : NDArray
        Quality mask.
    xp : module
        Array module (numpy or cupy).

    Returns
    -------
    NDArray
        R-squared map.
    """
    spatial_shape = ct_4d.shape[:-1]

    # Initialize R-squared map
    r_squared = xp.zeros(spatial_shape, dtype=ct_4d.dtype)

    # Get masked voxels
    n_voxels = int(xp.sum(quality_mask))
    if n_voxels == 0:
        return r_squared

    # Extract masked data: flatten spatial dims, keep time as last axis
    # ct_4d is (x, y, z, t), quality_mask is (x, y, z)
    ct_masked = ct_4d[quality_mask]  # Shape: (n_voxels, n_time)

    # Build parameter array for batch prediction
    # Shape: (n_params, n_voxels)
    param_names = bound_model.parameters
    n_params = len(param_names)
    params_batch = xp.zeros((n_params, n_voxels), dtype=ct_4d.dtype)

    for i, name in enumerate(param_names):
        if name in param_maps:
            params_batch[i, :] = xp.asarray(param_maps[name].values)[quality_mask]

    # predict_array_batch returns (n_time, n_voxels)
    try:
        ct_pred = bound_model.predict_array_batch(
            params_batch,
            xp,
        )  # (n_time, n_voxels)
        ct_pred = ct_pred.T  # (n_voxels, n_time) to match ct_masked

        residuals = ct_masked - ct_pred
        ss_res = xp.sum(residuals**2, axis=1)

        ct_mean = xp.mean(ct_masked, axis=1, keepdims=True)
        ss_tot = xp.sum((ct_masked - ct_mean) ** 2, axis=1)

        r2_values = xp.where(ss_tot > 1e-10, 1.0 - ss_res / ss_tot, 0.0)
        r_squared[quality_mask] = r2_values

    except Exception:
        pass

    return r_squared


def _compute_fitting_stats(
    param_maps: dict[str, ParameterMap],
    quality_mask: NDArray[np.bool_],
) -> dict[str, Any]:
    """Compute summary statistics from fitting results.

    Parameters
    ----------
    param_maps : dict
        Fitted parameter maps.
    quality_mask : NDArray
        Quality mask.

    Returns
    -------
    dict
        Summary statistics.
    """
    # Get array module from quality_mask
    xp = get_array_module(quality_mask)

    stats: dict[str, Any] = {}

    # Overall statistics - convert to Python int/float for JSON serialization
    stats["n_voxels_total"] = int(quality_mask.size)
    stats["n_voxels_fitted"] = int(xp.sum(quality_mask))
    stats["fit_success_rate"] = float(xp.sum(quality_mask) / quality_mask.size)

    # Per-parameter statistics
    for name, pmap in param_maps.items():
        valid_data = pmap.values[quality_mask]
        if len(valid_data) > 0:
            # Use xp operations, convert to float for serialization
            stats[f"{name}_mean"] = float(xp.mean(valid_data))
            stats[f"{name}_std"] = float(xp.std(valid_data))
            stats[f"{name}_median"] = float(xp.median(valid_data))
            stats[f"{name}_min"] = float(xp.min(valid_data))
            stats[f"{name}_max"] = float(xp.max(valid_data))

    return stats


class _DelayAwareModel:
    """Internal wrapper that adds arterial delay fitting to any model.

    Not exported, not registered. Private to fitting.py.
    """

    def __init__(
        self,
        base_model: Any,
        delay_bounds: tuple[float, float] = (0.0, 60.0),
    ) -> None:
        self._base = base_model
        self._delay_bounds = delay_bounds

    @property
    def time_unit(self) -> str:
        return self._base.time_unit

    @property
    def name(self) -> str:
        return f"{self._base.name} (with delay)"

    @property
    def parameters(self) -> list[str]:
        return [*self._base.parameters, "delay"]

    @property
    def parameter_units(self) -> dict[str, str]:
        return {**self._base.parameter_units, "delay": "s"}

    @property
    def reference(self) -> str:
        return self._base.reference

    def predict(
        self,
        t: NDArray[np.floating[Any]],
        aif: NDArray[np.floating[Any]],
        params: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]]:
        from osipy.common.aif.delay import shift_aif

        n_base = len(self._base.parameters)
        base_params = params[:n_base]
        delay = params[n_base]

        shifted = shift_aif(aif, t, delay, xp)
        return self._base.predict(t, shifted, base_params, xp)

    def predict_batch(
        self,
        t: NDArray[np.floating[Any]],
        aif: NDArray[np.floating[Any]],
        params_batch: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]]:
        from osipy.common.aif.delay import shift_aif

        n_base = len(self._base.parameters)
        base_params = params_batch[:n_base, :]
        delays = params_batch[n_base, :]

        # Batch-shift AIF: (n_time, n_voxels) — one shifted AIF per voxel
        shifted_aifs = shift_aif(aif, t, delays, xp)

        # Base model handles 2D AIF via updated convolution functions
        return self._base.predict_batch(t, shifted_aifs, base_params, xp)

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        bounds = self._base.get_bounds().copy()
        bounds["delay"] = self._delay_bounds
        return bounds

    def get_initial_guess(
        self,
        ct: NDArray[np.floating[Any]],
        aif: NDArray[np.floating[Any]],
        t: NDArray[np.floating[Any]],
    ) -> dict[str, float]:
        from dataclasses import asdict

        base_guess = self._base.get_initial_guess(ct, aif, t)
        if hasattr(base_guess, "__dataclass_fields__"):
            d = asdict(base_guess)
        elif isinstance(base_guess, dict):
            d = dict(base_guess)
        else:
            d = {name: getattr(base_guess, name, 0.0) for name in self._base.parameters}
        d["delay"] = 0.0
        return d

    def get_initial_guess_batch(
        self,
        ct_batch: NDArray[np.floating[Any]],
        aif: NDArray[np.floating[Any]],
        t: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]]:
        base_guess = self._base.get_initial_guess_batch(ct_batch, aif, t, xp)
        n_voxels = ct_batch.shape[1]
        delay_row = xp.zeros((1, n_voxels), dtype=ct_batch.dtype)
        return xp.concatenate([base_guess, delay_row], axis=0)

    def params_to_array(self, params: Any) -> NDArray[np.floating[Any]]:
        if isinstance(params, dict):
            return np.array([params.get(p, 0.0) for p in self.parameters])
        base_arr = self._base.params_to_array(params)
        delay = getattr(params, "delay", 0.0)
        return np.append(base_arr, delay)

    def array_to_params(self, values: NDArray[np.floating[Any]]) -> dict[str, float]:
        return {name: float(values[i]) for i, name in enumerate(self.parameters)}

    def bounds_to_arrays(
        self,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        bounds = self.get_bounds()
        lower = np.array([bounds[p][0] for p in self.parameters])
        upper = np.array([bounds[p][1] for p in self.parameters])
        return lower, upper

    def _convert_time(
        self,
        t: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]]:
        return self._base._convert_time(t, xp)
