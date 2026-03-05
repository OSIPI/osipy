"""Signal-to-concentration conversion for DCE-MRI (OSIPI: P.SC1.001).

This module implements conversion of DCE-MRI signal intensity to
indicator concentration (OSIPI: Q.IC1.001) using T1 mapping and
relaxivity (OSIPI: Q.EL1.015).

Estimates baseline signal S0 (OSIPI: P.SC1.001, Q.MS1.010) and converts
signal changes to concentration via R1 (OSIPI: Q.EL1.001).

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Tofts PS. J Magn Reson Imaging 1997;7(1):91-101.
"""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.backend.array_module import get_array_module
from osipy.common.exceptions import DataValidationError
from osipy.common.parameter_map import ParameterMap
from osipy.common.types import DCEAcquisitionParams
from osipy.dce.concentration.registry import (
    get_concentration_model,
    register_concentration_model,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@register_concentration_model("spgr")
def _convert_spgr(
    signal,
    s0,
    t1_pre,
    tr,
    cos_a,
    relaxivity,
    xp,
):
    """SPGR signal-to-concentration conversion."""
    # SPGR signal equation:
    # S = M0 * sin(a) * (1 - E1) / (1 - E1 * cos(a))
    # where E1 = exp(-TR/T1)
    #
    # Solving for T1(t) from signal ratio S(t)/S0:
    # Fully vectorized over the time dimension using broadcasting.

    # Pre-compute E1_pre = exp(-TR/T1_pre)
    e1_pre = xp.exp(-tr / t1_pre)  # (nx, ny, nz)

    # A factor from baseline
    a_factor = (1 - e1_pre * cos_a) / (1 - e1_pre + 1e-10)  # (nx, ny, nz)

    # Signal ratio for all time points at once
    # signal: (nx, ny, nz, nt), s0: (nx, ny, nz)
    ratio = signal / (s0[..., xp.newaxis] + 1e-10)  # (nx, ny, nz, nt)

    # Solve for E1(t) for all time points simultaneously
    # E1_t = (ratio - A) / (ratio * cos_a - A)
    numerator = ratio - a_factor[..., xp.newaxis]
    denominator = ratio * cos_a - a_factor[..., xp.newaxis]
    e1_t = numerator / (denominator + 1e-10)

    # Ensure E1 is in valid range
    e1_t = xp.clip(e1_t, 1e-10, 1 - 1e-10)

    # Convert E1(t) to T1(t)
    t1_t = -tr / xp.log(e1_t)  # (nx, ny, nz, nt)

    # Convert T1 to R1
    r1_pre = 1000 / (t1_pre + 1e-10)  # (nx, ny, nz)
    r1_t = 1000 / (t1_t + 1e-10)  # (nx, ny, nz, nt)

    # Delta R1 and concentration
    delta_r1 = r1_t - r1_pre[..., xp.newaxis]
    concentration = delta_r1 / relaxivity

    return concentration


@register_concentration_model("linear")
def _convert_linear(
    signal,
    s0,
    t1_pre,
    tr,
    cos_a,
    relaxivity,
    xp,
):
    """Linear signal-to-concentration conversion."""
    # Simplified linear approximation for small concentration changes
    # Valid when concentration is low
    # Fully vectorized over the time dimension using broadcasting.

    # Signal ratio for all time points
    delta_s = (signal - s0[..., xp.newaxis]) / (s0[..., xp.newaxis] + 1e-10)

    # Scale factor
    r1_pre = 1000 / (t1_pre + 1e-10)
    scale = r1_pre / relaxivity  # (nx, ny, nz)

    concentration = delta_s * scale[..., xp.newaxis]

    return concentration


def signal_to_concentration(
    signal: "NDArray[np.floating[Any]]",
    t1_map: ParameterMap | None,
    acquisition_params: DCEAcquisitionParams,
    method: str = "spgr",
) -> "NDArray[np.floating[Any]]":
    """Convert DCE signal intensity to indicator concentration (OSIPI: P.SC1.001).

    Uses the SPGR signal equation to convert signal changes to
    changes in R1 (OSIPI: Q.EL1.001), then to concentration
    (OSIPI: Q.IC1.001) using relaxivity (OSIPI: Q.EL1.015).

    Parameters
    ----------
    signal : NDArray[np.floating]
        4D signal intensity data with shape (x, y, z, t).
    t1_map : ParameterMap | None
        Pre-contrast T1 map in milliseconds. If None, will use
        `acquisition_params.t1_assumed` to create a uniform T1 map.
    acquisition_params : DCEAcquisitionParams
        Acquisition parameters including TR (OSIPI: Q.MS1.006),
        flip_angle (OSIPI: Q.MS1.007), and relaxivity (OSIPI: Q.EL1.015).
        If `t1_map` is None, must include `t1_assumed` field.
    method : str, default="spgr"
        Signal model: 'spgr' (spoiled gradient echo) or 'linear'.

    Returns
    -------
    NDArray[np.floating]
        4D concentration data (OSIPI: Q.IC1.001) in mM with same
        shape as input signal.

    Raises
    ------
    DataValidationError
        If input dimensions or parameters are invalid, or if t1_map
        is None and t1_assumed is not specified in acquisition_params.

    References
    ----------
    .. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    .. [2] Tofts PS. J Magn Reson Imaging 1997;7(1):91-101.

    Examples
    --------
    With measured T1 map:

    >>> from osipy.dce.concentration import signal_to_concentration
    >>> concentration = signal_to_concentration(
    ...     signal=dce_data,
    ...     t1_map=t1_map,
    ...     acquisition_params=params,
    ... )

    With assumed T1 value (when T1 mapping data unavailable):

    >>> params = DCEAcquisitionParams(
    ...     tr=6.13,
    ...     flip_angles=[10.0],
    ...     relaxivity=4.5,
    ...     t1_assumed=1400.0,  # Breast tissue at 3T
    ... )
    >>> concentration = signal_to_concentration(
    ...     signal=dce_data,
    ...     t1_map=None,  # Will use t1_assumed
    ...     acquisition_params=params,
    ... )
    """
    from osipy.common.parameter_map import create_uniform_t1_map

    # Handle None t1_map - use assumed T1 if provided
    if t1_map is None:
        if acquisition_params.t1_assumed is None:
            msg = (
                "t1_map is None but acquisition_params.t1_assumed is not set. "
                "Either provide a measured T1 map or set t1_assumed in "
                "DCEAcquisitionParams (e.g., 1400.0 ms for breast tissue at 3T)."
            )
            raise DataValidationError(msg)

        # Create uniform T1 map from assumed value
        logger.info(
            f"Using assumed uniform T1 = {acquisition_params.t1_assumed:.0f} ms "
            "(no measured T1 map provided)",
        )
        t1_map = create_uniform_t1_map(
            t1_ms=acquisition_params.t1_assumed,
            shape=signal.shape[:3],
            affine=np.eye(4),  # Will be overwritten, just needs valid shape
        )
    # Get array module for GPU/CPU agnostic computation
    xp = get_array_module(signal)

    # Validate inputs
    if signal.ndim != 4:
        msg = f"Signal must be 4D, got {signal.ndim}D"
        raise DataValidationError(msg)

    if t1_map.values.shape != signal.shape[:3]:
        msg = (
            f"T1 map shape {t1_map.values.shape} must match "
            f"signal spatial dimensions {signal.shape[:3]}"
        )
        raise DataValidationError(msg)

    if acquisition_params.tr is None:
        msg = "TR is required in acquisition_params"
        raise DataValidationError(msg)

    tr = acquisition_params.tr  # in ms
    relaxivity = acquisition_params.relaxivity  # in mM^-1 s^-1

    # Get flip angle - use first if list, otherwise assume single value
    if acquisition_params.flip_angles:
        flip_angle_deg = acquisition_params.flip_angles[0]
    else:
        msg = "flip_angles required in acquisition_params"
        raise DataValidationError(msg)

    flip_angle = float(xp.deg2rad(flip_angle_deg))

    # Get dimensions
    nx, ny, nz, _nt = signal.shape
    baseline_frames = acquisition_params.baseline_frames

    logger.info(
        f"Converting signal to concentration: "
        f"TR={tr}ms, FA={flip_angle_deg}°, r1={relaxivity}mM⁻¹s⁻¹",
    )

    # Compute baseline signal (pre-contrast)
    s0 = xp.mean(signal[..., :baseline_frames], axis=-1)

    # Get pre-contrast T1 (ensure same array type as signal)
    t1_pre = xp.asarray(t1_map.values)

    # Handle invalid T1 values
    t1_pre = xp.where(t1_pre <= 0, xp.nan, t1_pre)
    t1_pre = xp.where(~xp.isfinite(t1_pre), xp.nan, t1_pre)

    # Pre-compute constants for SPGR equation
    cos_a = float(xp.cos(flip_angle))

    converter = get_concentration_model(method)
    concentration = converter(
        signal,
        s0,
        t1_pre,
        tr,
        cos_a,
        relaxivity,
        xp,
    )

    # Clean up invalid values
    concentration = xp.where(~xp.isfinite(concentration), 0.0, concentration)
    concentration = xp.where(concentration < 0, 0.0, concentration)

    n_valid = int(xp.sum(xp.any(concentration > 0, axis=-1)))
    n_total = nx * ny * nz
    logger.info(
        f"Signal-to-concentration complete: "
        f"{n_valid}/{n_total} voxels with valid concentration",
    )

    return concentration
