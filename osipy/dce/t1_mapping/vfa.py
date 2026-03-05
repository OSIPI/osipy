"""Variable Flip Angle (VFA) T1 mapping (OSIPI: P.NR2.002).

This module implements T1 mapping from spoiled gradient echo (SPGR)
acquisitions at multiple flip angles.

Estimates native longitudinal relaxation rate R1 (OSIPI: Q.EL1.001)
and pre-contrast R1 (OSIPI: Q.EL1.002) via the VFA method
(OSIPI: P.NR1.001 / P.NR2.002).

GPU/CPU agnostic using the xp array module pattern.
Nonlinear fitting uses the shared LevenbergMarquardtFitter.

When input data is on GPU (CuPy array), computations are GPU-accelerated.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Deoni SCL et al. MRM 2003;49(3):515-526.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.backend.array_module import get_array_module, to_numpy
from osipy.common.dataset import PerfusionDataset
from osipy.common.exceptions import DataValidationError
from osipy.common.parameter_map import ParameterMap
from osipy.common.types import DCEAcquisitionParams
from osipy.dce.t1_mapping.registry import register_t1_method

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class T1MappingResult:
    """Result of T1 mapping computation.

    Attributes
    ----------
    t1_map : ParameterMap
        T1 relaxation time map in milliseconds.
    m0_map : ParameterMap | None
        Equilibrium magnetization map (optional).
    quality_mask : NDArray[np.bool_]
        Mask of successfully fitted voxels.
    """

    t1_map: ParameterMap
    m0_map: ParameterMap | None = None
    quality_mask: "NDArray[np.bool_] | None" = None


def _fit_t1_linear_vectorized(
    data: "NDArray[np.floating[Any]]",
    flip_angles_rad: "NDArray[np.floating[Any]]",
    tr: float,
) -> tuple["NDArray[np.floating[Any]]", "NDArray[np.floating[Any]]"]:
    """Vectorized T1 fitting for GPU acceleration.

    Fits T1 for all voxels simultaneously using array operations.

    Parameters
    ----------
    data : NDArray
        4D data array (nx, ny, nz, n_flip_angles).
    flip_angles_rad : NDArray
        Flip angles in radians.
    tr : float
        Repetition time in milliseconds.

    Returns
    -------
    tuple[NDArray, NDArray]
        (T1 map, M0 map) in milliseconds.
    """
    xp = get_array_module(data)

    # Ensure data is on correct device
    data = xp.asarray(data)
    flip_angles_rad = xp.asarray(flip_angles_rad)

    nx, ny, nz, n_fa = data.shape

    # Precompute trigonometric values
    sin_fa = xp.sin(flip_angles_rad)  # Shape: (n_fa,)
    tan_fa = xp.tan(flip_angles_rad)  # Shape: (n_fa,)

    # Reshape data for vectorized operations: (n_voxels, n_fa)
    flat_data = data.reshape(-1, n_fa)

    # Compute x = S/tan(a), y = S/sin(a)
    # Broadcasting: (n_voxels, n_fa) / (n_fa,) -> (n_voxels, n_fa)
    x = flat_data / tan_fa
    y = flat_data / sin_fa

    # Linear regression coefficients (vectorized)
    n = float(n_fa)
    sum_x = xp.sum(x, axis=1)  # (n_voxels,)
    sum_y = xp.sum(y, axis=1)  # (n_voxels,)
    sum_xy = xp.sum(x * y, axis=1)  # (n_voxels,)
    sum_xx = xp.sum(x * x, axis=1)  # (n_voxels,)

    denom = n * sum_xx - sum_x * sum_x

    # Avoid division by zero
    valid = xp.abs(denom) > 1e-10
    denom = xp.where(valid, denom, 1.0)  # Placeholder for invalid

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # Extract T1 and M0
    e1 = slope

    # T1 = -TR / log(E1), valid only if 0 < E1 < 1
    valid_e1 = valid & (e1 > 0) & (e1 < 1)
    e1_safe = xp.where(valid_e1, e1, 0.5)  # Placeholder for invalid

    t1 = xp.where(valid_e1, -tr / xp.log(e1_safe), xp.nan)
    valid_m0 = valid_e1 & (xp.abs(1 - e1) > 1e-10)
    m0 = xp.where(valid_m0, intercept / (1 - e1_safe), xp.nan)

    # Apply physiological bounds
    valid_t1 = (t1 > 1) & (t1 < 10000)
    t1 = xp.where(valid_t1, t1, xp.nan)
    m0 = xp.where(valid_t1, m0, xp.nan)

    # Reshape back to 3D
    t1_map = t1.reshape(nx, ny, nz)
    m0_map = m0.reshape(nx, ny, nz)

    return t1_map, m0_map


def _compute_t1_vfa_impl(
    dataset: PerfusionDataset,
    method: str = "linear",
) -> T1MappingResult:
    """Internal implementation of VFA T1 mapping."""
    # Validate input
    if not isinstance(dataset.acquisition_params, DCEAcquisitionParams):
        msg = "VFA T1 mapping requires DCEAcquisitionParams"
        raise DataValidationError(msg)

    params = dataset.acquisition_params
    if not params.flip_angles:
        msg = "VFA T1 mapping requires flip_angles in acquisition_params"
        raise DataValidationError(msg)

    if params.tr is None:
        msg = "VFA T1 mapping requires TR in acquisition_params"
        raise DataValidationError(msg)

    xp = get_array_module(dataset.data)
    flip_angles = xp.asarray(params.flip_angles)
    tr = params.tr

    # Check data dimensions match flip angles
    n_volumes = dataset.data.shape[3] if dataset.data.ndim == 4 else 1

    if n_volumes != len(flip_angles):
        msg = (
            f"Number of volumes ({n_volumes}) must match "
            f"number of flip angles ({len(flip_angles)})"
        )
        raise DataValidationError(msg)

    # Convert flip angles to radians
    if hasattr(xp, "deg2rad"):
        flip_angles_rad = xp.deg2rad(flip_angles)
    else:
        flip_angles_rad = np.deg2rad(to_numpy(flip_angles))
    flip_angles_rad = xp.asarray(flip_angles_rad)

    # Get spatial dimensions
    if dataset.data.ndim == 4:
        nx, ny, nz, _ = dataset.data.shape
    else:
        nx, ny, nz = dataset.data.shape

    data = dataset.data if dataset.data.ndim == 4 else dataset.data[..., xp.newaxis]

    logger.info(f"Computing VFA T1 map ({method} method) for {nx}x{ny}x{nz} volume")

    if method == "linear":
        t1_values, m0_values = _fit_t1_linear_vectorized(data, flip_angles_rad, tr)

        # Convert to numpy for ParameterMap if on GPU
        t1_values = to_numpy(t1_values)
        m0_values = to_numpy(m0_values)

        # Quality mask: valid T1 values
        quality_mask = np.isfinite(t1_values) & (t1_values > 1) & (t1_values < 10000)
        n_processed = np.sum(quality_mask)

    elif method == "nonlinear":
        # Use shared LevenbergMarquardtFitter via signal model + binding adapter
        from osipy.common.fitting.least_squares import LevenbergMarquardtFitter
        from osipy.dce.t1_mapping.binding import BoundSPGRModel
        from osipy.dce.t1_mapping.models import SPGRSignalModel

        # Run linear fit for initialization (used by BoundSPGRModel internally
        # via get_initial_guess_batch, but we also use it to seed better guesses)
        logger.info("  Running linear fit for initialization...")
        t1_init, _m0_init = _fit_t1_linear_vectorized(data, flip_angles_rad, tr)

        # Create signal model and binding adapter
        signal_model = SPGRSignalModel()
        bound_model = BoundSPGRModel(signal_model, flip_angles_rad, tr)

        # Build mask from linear fit: only fit voxels with valid linear estimates
        t1_init_np = to_numpy(t1_init)
        fit_mask = np.isfinite(t1_init_np) & (t1_init_np > 1) & (t1_init_np < 10000)

        logger.info("  Running nonlinear LM refinement via shared fitter...")
        fitter = LevenbergMarquardtFitter(
            max_iterations=100,
            tolerance=1e-6,
            r2_threshold=0.5,
        )
        param_maps = fitter.fit_image(
            model=bound_model,
            data=data,
            mask=fit_mask,
        )

        # Extract T1 and M0 from returned ParameterMaps
        t1_values = param_maps["T1"].values
        m0_values = param_maps["M0"].values
        quality_mask = param_maps["T1"].quality_mask

        n_processed = np.sum(quality_mask)
    else:
        msg = f"Unknown VFA method: {method!r}. Use 'linear' or 'nonlinear'."
        raise DataValidationError(msg)

    n_voxels = nx * ny * nz
    logger.info(
        f"T1 mapping complete: {n_processed}/{n_voxels} voxels "
        f"({100 * n_processed / n_voxels:.1f}%)",
    )

    t1_param_map = ParameterMap(
        name="T1",
        symbol="T1",
        units="ms",
        values=t1_values,
        affine=dataset.affine,
        quality_mask=quality_mask,
        model_name="VFA",
        fitting_method=method,
        literature_reference="Deoni SCL et al. MRM 2003;49(3):515-526.",
    )

    m0_param_map = ParameterMap(
        name="M0",
        symbol="M0",
        units="a.u.",
        values=m0_values,
        affine=dataset.affine,
        quality_mask=quality_mask,
        model_name="VFA",
        fitting_method=method,
    )

    return T1MappingResult(
        t1_map=t1_param_map,
        m0_map=m0_param_map,
        quality_mask=quality_mask,
    )


@register_t1_method("vfa")
def compute_t1_vfa(
    dataset: PerfusionDataset | None = None,
    method: str = "linear",
    *,
    signal: "NDArray[np.floating[Any]] | None" = None,
    flip_angles: "NDArray[np.floating[Any]] | list[float] | None" = None,
    tr: float | None = None,
    mask: "NDArray[np.bool_] | None" = None,
) -> T1MappingResult:
    """Compute T1 map from Variable Flip Angle data (OSIPI: P.NR2.002).

    Estimates native R1 (OSIPI: Q.EL1.001) and equilibrium signal S0
    (OSIPI: Q.MS1.010) via VFA method (OSIPI: P.NR1.001).

    This function supports two calling conventions:
    1. With a PerfusionDataset: compute_t1_vfa(dataset)
    2. With individual arrays: compute_t1_vfa(signal=..., flip_angles=..., tr=...)

    This function supports GPU acceleration. When the input data is a CuPy
    array, the computation is performed entirely on GPU.

    Parameters
    ----------
    dataset : PerfusionDataset, optional
        Dataset with images acquired at multiple flip angles.
        The acquisition_params must be DCEAcquisitionParams with
        flip_angles (OSIPI: Q.MS1.007) specified.
    method : str, default="linear"
        Fitting method: 'linear' (fast) or 'nonlinear' (more accurate).
    signal : NDArray, optional
        VFA signal data, shape (nx, ny, nz, n_flip_angles).
        Required if dataset is not provided.
    flip_angles : NDArray or list, optional
        Flip angles (OSIPI: Q.MS1.007) in degrees.
        Required if dataset is not provided.
    tr : float, optional
        Repetition time (OSIPI: Q.MS1.006) in milliseconds.
        Required if dataset is not provided.
    mask : NDArray, optional
        Brain/tissue mask. If None, all voxels are processed.

    Returns
    -------
    T1MappingResult
        Result containing T1 map, M0 map, and quality mask.

    Raises
    ------
    DataValidationError
        If input is invalid or incomplete.

    References
    ----------
    .. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    .. [2] Deoni SCL et al. MRM 2003;49(3):515-526.

    Examples
    --------
    >>> from osipy.dce.t1_mapping.vfa import compute_t1_vfa
    >>> result = compute_t1_vfa(dataset)
    >>> print(f"Mean T1: {result.t1_map.statistics()['mean']:.0f} ms")

    >>> # Or with individual arrays:
    >>> result = compute_t1_vfa(signal=vfa_data, flip_angles=[2,5,10,15,20], tr=5.0)
    """
    # If individual arrays provided, create a temporary dataset
    if dataset is None:
        if signal is None or flip_angles is None or tr is None:
            msg = "Either dataset or (signal, flip_angles, tr) must be provided"
            raise DataValidationError(msg)

        flip_angles_array = np.asarray(flip_angles)

        # Create acquisition params
        acq_params = DCEAcquisitionParams(
            tr=tr,
            flip_angles=list(flip_angles_array),
        )

        # Create mask if not provided
        if mask is None:
            spatial_shape = signal.shape[:3]
            mask = np.ones(spatial_shape, dtype=bool)

        from osipy.common.types import Modality

        # Create time_points placeholder (VFA 4th dim is flip angles, not time)
        n_volumes = signal.shape[-1] if signal.ndim == 4 else 1
        time_points = np.arange(n_volumes, dtype=np.float64)

        # Create temporary dataset
        dataset = PerfusionDataset(
            data=signal,
            affine=np.eye(4),
            modality=Modality.DCE,
            time_points=time_points,
            acquisition_params=acq_params,
            quality_mask=mask,
        )

    return _compute_t1_vfa_impl(dataset, method)
