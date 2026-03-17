"""IVIM parameter estimation algorithms.

This module implements various fitting strategies for IVIM parameters,
including segmented fitting and full bi-exponential fitting.

Output parameter maps:
- D: diffusion coefficient in mm^2/s
- D*: pseudo-diffusion coefficient in mm^2/s
- f: perfusion fraction, dimensionless

GPU/CPU agnostic using the xp array module pattern.
Uses the shared LevenbergMarquardtFitter via BoundIVIMModel binding.

References
----------
.. [1] Federau C et al. (2012). Quantitative measurement of brain perfusion
   with intravoxel incoherent motion MR imaging. Radiology 265(3):874-881.
.. [2] Lemke A et al. (2011). Toward an optimal distribution of b values for
   intravoxel incoherent motion imaging. Magn Reson Imaging 29(6):766-776.
.. [3] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [4] Dickie BR et al. MRM 2024. doi:10.1002/mrm.30101
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.backend.array_module import get_array_module, to_gpu, to_numpy
from osipy.common.backend.config import is_gpu_available
from osipy.common.exceptions import FittingError
from osipy.common.fitting.base import BaseFitter
from osipy.common.fitting.least_squares import LevenbergMarquardtFitter
from osipy.common.parameter_map import ParameterMap
from osipy.ivim.models.biexponential import IVIMBiexponentialModel
from osipy.ivim.models.binding import BoundIVIMModel

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class FittingMethod(Enum):
    """IVIM fitting methods."""

    SEGMENTED = "segmented"  # Two-step fitting
    FULL = "full"  # Full bi-exponential
    BAYESIAN = "bayesian"  # Bayesian estimation


@dataclass
class IVIMFitParams:
    """Parameters for IVIM fitting.

    Attributes
    ----------
    method : FittingMethod
        Fitting method to use.
    b_threshold : float
        b-value threshold for segmented fitting (s/mm²).
        b-values above this are used for D estimation.
    max_iterations : int
        Maximum iterations for optimization.
    tolerance : float
        Convergence tolerance.
    bounds : dict | None
        Custom parameter bounds.
    bayesian_params : dict | None
        Bayesian-specific parameters (prior_std, noise_std,
        compute_uncertainty).  Only used when ``method`` is
        ``FittingMethod.BAYESIAN``.
    """

    method: FittingMethod = FittingMethod.SEGMENTED
    b_threshold: float = 200.0
    max_iterations: int = 500
    tolerance: float = 1e-6
    bounds: dict[str, tuple[float, float]] | None = None
    bayesian_params: Any = None


@dataclass
class IVIMFitResult:
    """Result of IVIM fitting.

    Attributes
    ----------
    d_map : ParameterMap
        Diffusion coefficient (D) map in mm^2/s.
    d_star_map : ParameterMap
        Pseudo-diffusion coefficient (D*) map in mm^2/s.
    f_map : ParameterMap
        Perfusion fraction (f) map, dimensionless.
    s0_map : ParameterMap
        S0 (b=0 signal) map in arbitrary units.
    quality_mask : NDArray[np.bool_]
        Mask of successfully fitted voxels.
    r_squared : NDArray[np.floating] | None
        Goodness of fit (R^2) map.
    fitting_stats : dict[str, Any]
        Fitting statistics.
    """

    d_map: ParameterMap
    d_star_map: ParameterMap
    f_map: ParameterMap
    s0_map: ParameterMap
    quality_mask: "NDArray[np.bool_]"
    r_squared: "NDArray[np.floating[Any]] | None" = None
    fitting_stats: dict[str, Any] = field(default_factory=dict)


def fit_ivim(
    signal: "NDArray[np.floating[Any]]",
    b_values: "NDArray[np.floating[Any]]",
    mask: "NDArray[np.bool_] | None" = None,
    params: IVIMFitParams | None = None,
    method: FittingMethod | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> IVIMFitResult:
    """Fit IVIM model to DWI data.

    Parameters
    ----------
    signal : NDArray[np.floating]
        DWI signal data, shape (..., n_b_values).
        Last dimension corresponds to b-values.
    b_values : NDArray[np.floating]
        b-values in s/mm².
    mask : NDArray[np.bool_] | None
        Brain/tissue mask.
    params : IVIMFitParams | None
        Fitting parameters.
    method : FittingMethod | None
        Fitting method. If provided, overrides params.method.
    progress_callback : Callable[[float], None] | None
        Progress callback function.

    Returns
    -------
    IVIMFitResult
        Fitted parameter maps.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.ivim import fit_ivim
    >>> signal = np.random.rand(64, 64, 10, 8) * 1000
    >>> b_values = np.array([0, 10, 20, 50, 100, 200, 400, 800])
    >>> result = fit_ivim(signal, b_values)
    """
    params = params or IVIMFitParams()

    # Allow method parameter to override params.method
    if method is not None:
        params.method = method

    # Validate inputs
    if signal.shape[-1] != len(b_values):
        msg = (
            f"Signal last dimension ({signal.shape[-1]}) "
            f"!= number of b-values ({len(b_values)})"
        )
        raise FittingError(msg)

    # OSIPI TF2.4 requires minimum 4 b-values for IVIM fitting
    if len(b_values) < 4:
        msg = (
            f"IVIM fitting requires at least 4 b-values (OSIPI TF2.4), "
            f"got {len(b_values)}"
        )
        raise FittingError(msg)

    xp = get_array_module(signal)

    spatial_shape = signal.shape[:-1]

    if mask is None:
        mask = xp.ones(spatial_shape, dtype=bool)

    # Use vectorized fitting for speed
    use_gpu = is_gpu_available()
    logger.debug(
        "Fitting %d voxels using %s (vectorized)",
        int(to_numpy(xp.sum(mask))),
        "GPU" if use_gpu else "CPU",
    )

    # Resolve fitting method via registry
    from osipy.ivim.fitting.registry import get_ivim_fitter

    if isinstance(params.method, FittingMethod):
        method_name = params.method.value
    else:
        method_name = str(params.method)
    fitter_func = get_ivim_fitter(method_name)
    param_maps = fitter_func(signal, b_values, mask, params, use_gpu, progress_callback)

    # Extract named maps (fitter.fit_image already created ParameterMaps)
    d_map = param_maps["D"]
    d_star_map = param_maps.get(
        "D*",
        ParameterMap(
            name="D*",
            symbol="D*",
            units="mm^2/s",
            values=np.zeros_like(d_map.values),
            affine=d_map.affine,
            quality_mask=d_map.quality_mask,
        ),
    )
    f_map = param_maps["f"]
    s0_map = param_maps.get(
        "S0",
        ParameterMap(
            name="S0",
            symbol="S0",
            units="a.u.",
            values=np.zeros_like(d_map.values),
            affine=d_map.affine,
            quality_mask=d_map.quality_mask,
        ),
    )

    quality_mask = d_map.quality_mask if d_map.quality_mask is not None else mask
    r_squared = param_maps.get("r_squared")
    r_squared_values = r_squared.values if r_squared is not None else None

    # Compute statistics
    fitting_stats = _compute_fitting_stats(
        d_map.values, d_star_map.values, f_map.values, quality_mask
    )

    return IVIMFitResult(
        d_map=d_map,
        d_star_map=d_star_map,
        f_map=f_map,
        s0_map=s0_map,
        quality_mask=quality_mask,
        r_squared=r_squared_values,
        fitting_stats=fitting_stats,
    )


def _fit_ivim_vectorized(
    signal: "NDArray[np.floating[Any]]",
    b_values: "NDArray[np.floating[Any]]",
    mask: "NDArray[np.bool_]",
    params: IVIMFitParams,
    use_gpu: bool,
    progress_callback: Callable[[float], None] | None = None,
    fitter: BaseFitter | None = None,
) -> dict[str, ParameterMap]:
    """Vectorized IVIM fitting using a shared fitter.

    Creates a ``BoundIVIMModel`` with analytical Jacobian and delegates
    to the provided fitter (default: ``LevenbergMarquardtFitter``) for
    optimization. Returns the ParameterMaps from ``fitter.fit_image()``
    directly, with IVIM-specific post-processing (D*/D swap, domain
    quality mask) applied.

    Parameters
    ----------
    signal : NDArray
        Signal data, shape (..., n_b_values).
    b_values : NDArray
        b-values in s/mm².
    mask : NDArray
        Boolean mask of voxels to fit.
    params : IVIMFitParams
        Fitting parameters.
    use_gpu : bool
        Whether to use GPU acceleration.
    progress_callback : Callable, optional
        Progress callback.
    fitter : BaseFitter | None, optional
        Fitter instance. Defaults to ``LevenbergMarquardtFitter()``.

    Returns
    -------
    dict[str, ParameterMap]
        Parameter maps including D, D*, f, S0, r_squared.
    """
    # Ensure 3D spatial shape for fit_image
    if signal.ndim == 3:
        # (x, y, n_b) -> (x, y, 1, n_b)
        signal_4d = signal[:, :, np.newaxis, :]
        mask_3d = mask[:, :, np.newaxis]
    elif signal.ndim == 4:
        signal_4d = signal
        mask_3d = mask
    else:
        # (n_voxels, n_b) -> (n_voxels, 1, 1, n_b)
        signal_4d = signal[:, np.newaxis, np.newaxis, :]
        mask_3d = mask[:, np.newaxis, np.newaxis]

    # Move all data to GPU once if available — stays there until export
    from osipy.common.backend.config import get_backend

    if use_gpu and is_gpu_available() and not get_backend().force_cpu:
        signal_4d = to_gpu(signal_4d)
        b_values = to_gpu(b_values)
        mask_3d = to_gpu(mask_3d)

    # Create BoundIVIMModel with analytical Jacobian
    model = IVIMBiexponentialModel()
    bound_model = BoundIVIMModel(model, b_values, b_threshold=params.b_threshold)

    # Use shared fitter — returns dict[str, ParameterMap]
    fitter = fitter or LevenbergMarquardtFitter()
    param_maps = fitter.fit_image(
        model=bound_model,
        data=signal_4d,
        mask=mask_3d,
        bounds_override=params.bounds,
        progress_callback=progress_callback,
    )

    # Apply IVIM-specific post-processing (quality mask + D*/D swap)
    _apply_ivim_quality_and_swap(param_maps)

    return param_maps


def _apply_ivim_quality_and_swap(param_maps: dict[str, ParameterMap]) -> None:
    """Apply IVIM domain constraints and D*/D swap in-place.

    Refines the fitter's quality mask with physiological bounds and
    ensures D* > D by swapping values where needed.

    Parameters
    ----------
    param_maps : dict[str, ParameterMap]
        Parameter maps to modify in-place.
    """
    d_map = param_maps["D"]
    d_vals = d_map.values
    xp = get_array_module(d_vals)

    fitter_qmask = (
        d_map.quality_mask
        if d_map.quality_mask is not None
        else xp.ones_like(d_vals, dtype=bool)
    )

    f_vals = param_maps["f"].values

    if "D*" in param_maps:
        ds_vals = param_maps["D*"].values

        # Ensure D* > D (swap if needed) — must happen BEFORE quality check
        # so the quality mask reflects post-swap values
        swap = d_vals > ds_vals
        d_swapped = xp.where(swap, ds_vals, d_vals)
        ds_swapped = xp.where(swap, d_vals, ds_vals)

        # Update values in-place via array views
        d_map.values[...] = d_swapped
        param_maps["D*"].values[...] = ds_swapped

        # Domain quality mask (computed on post-swap values)
        quality = (
            fitter_qmask
            & (d_swapped > 0)
            & (d_swapped < 5e-3)
            & (ds_swapped > d_swapped)
            & (ds_swapped < 100e-3)
            & (f_vals >= 0)
            & (f_vals <= 0.7)
        )
    else:
        quality = (
            fitter_qmask
            & (d_vals > 0)
            & (d_vals < 5e-3)
            & (f_vals >= 0)
            & (f_vals <= 0.7)
        )

    # Update quality mask on all parameter maps
    for pmap in param_maps.values():
        if pmap.quality_mask is not None:
            pmap.quality_mask[...] = quality


# --- Registered IVIM fitting strategies ---
from osipy.ivim.fitting.registry import register_ivim_fitter


@register_ivim_fitter("segmented")
def _ivim_segmented(
    signal, b_values, mask, params, use_gpu, progress_callback=None
) -> dict[str, ParameterMap]:
    """Segmented IVIM fitting strategy."""
    return _fit_ivim_vectorized(
        signal, b_values, mask, params, use_gpu, progress_callback
    )


@register_ivim_fitter("full")
def _ivim_full(
    signal, b_values, mask, params, use_gpu, progress_callback=None
) -> dict[str, ParameterMap]:
    """Full bi-exponential IVIM fitting (no segmentation)."""
    from copy import copy

    full_params = copy(params)
    full_params.b_threshold = 0
    return _fit_ivim_vectorized(
        signal, b_values, mask, full_params, use_gpu, progress_callback
    )


@register_ivim_fitter("bayesian")
def _ivim_bayesian(
    signal, b_values, mask, params, use_gpu, progress_callback=None
) -> dict[str, ParameterMap]:
    """Two-stage Bayesian MAP IVIM fitting with empirical priors."""
    from osipy.ivim.fitting.bayesian_ivim import TwoStageBayesianIVIMFitter

    # Ensure 4D for fit_image
    if signal.ndim == 3:
        signal_4d = signal[:, :, np.newaxis, :]
        mask_3d = mask[:, :, np.newaxis]
    elif signal.ndim == 4:
        signal_4d = signal
        mask_3d = mask
    else:
        signal_4d = signal[:, np.newaxis, np.newaxis, :]
        mask_3d = mask[:, np.newaxis, np.newaxis]

    # GPU transfer
    from osipy.common.backend.config import get_backend

    if use_gpu and is_gpu_available() and not get_backend().force_cpu:
        signal_4d = to_gpu(signal_4d)
        b_values = to_gpu(b_values)
        mask_3d = to_gpu(mask_3d)

    model = IVIMBiexponentialModel()
    bound_model = BoundIVIMModel(model, b_values, b_threshold=params.b_threshold)

    bayesian_cfg = getattr(params, "bayesian_params", None) or {}
    fitter = TwoStageBayesianIVIMFitter(
        noise_std=bayesian_cfg.get("noise_std", None),
        compute_uncertainty=bayesian_cfg.get("compute_uncertainty", True),
        prior_scale=bayesian_cfg.get("prior_scale", 1.5),
    )

    bounds = {k: tuple(v) for k, v in params.bounds.items()} if params.bounds else None
    param_maps = fitter.fit_image(
        model=bound_model,
        data=signal_4d,
        mask=mask_3d,
        bounds_override=bounds,
        progress_callback=progress_callback,
    )

    _apply_ivim_quality_and_swap(param_maps)
    return param_maps


def _compute_fitting_stats(
    d: "NDArray[np.floating[Any]]",
    d_star: "NDArray[np.floating[Any]]",
    f: "NDArray[np.floating[Any]]",
    quality_mask: "NDArray[np.bool_]",
) -> dict[str, Any]:
    """Compute fitting statistics.

    Parameters
    ----------
    d, d_star, f : NDArray
        Fitted parameter maps.
    quality_mask : NDArray
        Quality mask.

    Returns
    -------
    dict
        Fitting statistics.
    """
    xp = get_array_module(d)

    stats: dict[str, Any] = {}

    stats["n_voxels_total"] = int(quality_mask.size)
    stats["n_voxels_fitted"] = int(to_numpy(xp.sum(quality_mask)))
    stats["fit_success_rate"] = (
        float(to_numpy(xp.sum(quality_mask))) / quality_mask.size
    )

    for name, data in [("D", d), ("D*", d_star), ("f", f)]:
        valid = data[quality_mask]
        if len(valid) > 0:
            stats[f"{name}_mean"] = float(to_numpy(xp.mean(valid)))
            stats[f"{name}_std"] = float(to_numpy(xp.std(valid)))
            # percentile/median: convert to numpy for reduction
            valid_np = to_numpy(valid)
            stats[f"{name}_median"] = float(np.median(valid_np))

    return stats
