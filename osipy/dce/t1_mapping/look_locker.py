"""Look-Locker T1 mapping (OSIPI: P.NR2.004).

This module implements T1 mapping from Look-Locker inversion recovery
acquisitions.

Estimates native longitudinal relaxation rate R1 (OSIPI: Q.EL1.001)
via multi-delay inversion recovery (OSIPI: P.NR2.004).

GPU/CPU agnostic using the xp array module pattern.
Nonlinear fitting uses the shared LevenbergMarquardtFitter.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Look DC, Locker DR. Rev Sci Instrum 1970;41:250-251.
.. [3] Deichmann R, Haase A. J Magn Reson 1992;96:608-612.
"""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.backend.array_module import get_array_module
from osipy.common.dataset import PerfusionDataset
from osipy.common.exceptions import DataValidationError
from osipy.common.parameter_map import ParameterMap
from osipy.dce.t1_mapping.registry import register_t1_method
from osipy.dce.t1_mapping.vfa import T1MappingResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _t1_star_to_t1(
    t1_star: "NDArray[np.floating[Any]]",
    a: "NDArray[np.floating[Any]]",
    b: "NDArray[np.floating[Any]]",
) -> "NDArray[np.floating[Any]]":
    """Convert apparent T1* to true T1 (vectorized).

    T1 = T1* * (B/A - 1)

    Parameters
    ----------
    t1_star : NDArray
        Apparent T1 values in milliseconds.
    a : NDArray
        Steady-state signal parameter.
    b : NDArray
        Signal amplitude parameter.

    Returns
    -------
    NDArray
        True T1 values in milliseconds. Invalid voxels contain NaN.
    """
    xp = get_array_module(t1_star)

    # Avoid division by zero
    a_safe = xp.where(xp.abs(a) > 1e-10, a, 1e-10)
    ratio = b / a_safe

    # T1 = T1* * (B/A - 1), valid only when B/A > 1
    valid = (xp.abs(a) > 1e-10) & (ratio > 1)
    t1 = xp.where(valid, t1_star * (ratio - 1), xp.nan)
    return t1


@register_t1_method("look_locker")
def compute_t1_look_locker(
    dataset: PerfusionDataset,
    ti_times: "NDArray[np.floating[Any]] | None" = None,
) -> T1MappingResult:
    """Compute T1 map from Look-Locker data (OSIPI: P.NR2.004).

    Estimates native R1 (OSIPI: Q.EL1.001) via multi-delay inversion
    recovery (OSIPI: P.NR2.004).

    GPU/CPU agnostic - operates on same device as dataset data.

    Parameters
    ----------
    dataset : PerfusionDataset
        Dataset with images acquired at multiple inversion times.
    ti_times : NDArray | None
        Inversion times in milliseconds. If None, attempts to extract
        from dataset.time_points.

    Returns
    -------
    T1MappingResult
        Result containing T1 map and quality mask.

    Raises
    ------
    DataValidationError
        If dataset is not suitable for Look-Locker T1 mapping.

    References
    ----------
    .. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    .. [2] Look DC, Locker DR. Rev Sci Instrum 1970;41:250-251.

    Examples
    --------
    >>> from osipy.dce.t1_mapping.look_locker import compute_t1_look_locker
    >>> result = compute_t1_look_locker(dataset, ti_times=ti_array)
    """
    xp = get_array_module(dataset.data)

    # Validate input
    if dataset.data.ndim != 4:
        msg = "Look-Locker T1 mapping requires 4D data"
        raise DataValidationError(msg)

    # Get TI times
    if ti_times is None:
        if dataset.time_points is not None:
            ti_times = xp.asarray(dataset.time_points) * 1000  # Convert s to ms
        else:
            msg = "Look-Locker requires TI times (ti_times or time_points)"
            raise DataValidationError(msg)
    else:
        ti_times = xp.asarray(ti_times)

    n_ti = dataset.data.shape[3]
    if len(ti_times) != n_ti:
        msg = f"TI times ({len(ti_times)}) must match data volumes ({n_ti})"
        raise DataValidationError(msg)

    nx, ny, nz, _ = dataset.data.shape

    logger.info(f"Computing Look-Locker T1 map for {nx}x{ny}x{nz} volume")

    # Use shared LevenbergMarquardtFitter via signal model + binding adapter
    from osipy.common.fitting.least_squares import LevenbergMarquardtFitter
    from osipy.dce.t1_mapping.binding import BoundLookLockerModel
    from osipy.dce.t1_mapping.models import LookLockerSignalModel

    signal_model = LookLockerSignalModel()
    bound_model = BoundLookLockerModel(signal_model, ti_times)

    fitter = LevenbergMarquardtFitter(
        max_iterations=100,
        tolerance=1e-8,
        r2_threshold=0.5,
    )

    param_maps = fitter.fit_image(
        model=bound_model,
        data=dataset.data,
    )

    # Extract fitted parameters
    t1_star_values = param_maps["T1_star"].values
    a_values = param_maps["A"].values
    b_values = param_maps["B"].values
    fit_quality_mask = param_maps["T1_star"].quality_mask

    # Convert T1* to true T1 (vectorized)
    t1_values = _t1_star_to_t1(
        np.asarray(t1_star_values),
        np.asarray(a_values),
        np.asarray(b_values),
    )

    # Build quality mask: valid T1 values within physiological range
    quality_mask = (
        fit_quality_mask
        & np.isfinite(t1_values)
        & (t1_values > 1)
        & (t1_values < 10000)
    )

    # Zero out invalid voxels
    t1_values = np.where(quality_mask, t1_values, np.nan)

    n_voxels = nx * ny * nz
    n_processed = int(np.sum(quality_mask))
    logger.info(
        f"Look-Locker T1 mapping complete: {n_processed}/{n_voxels} voxels "
        f"({100 * n_processed / n_voxels:.1f}%)",
    )

    t1_param_map = ParameterMap(
        name="T1",
        symbol="T1",
        units="ms",
        values=t1_values,
        affine=dataset.affine,
        quality_mask=quality_mask,
        model_name="Look-Locker",
        fitting_method="nonlinear",
        literature_reference="Look DC, Locker DR. Rev Sci Instrum 1970;41:250-251.",
    )

    return T1MappingResult(
        t1_map=t1_param_map,
        m0_map=None,
        quality_mask=quality_mask,
    )
