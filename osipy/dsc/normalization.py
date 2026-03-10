"""DSC normalization utilities.

This module provides functions for normalizing DSC perfusion parameters
to reference regions, particularly white matter normalization. Produces
relative perfusion measures (e.g. rCBV, rCBF) by dividing by a reference
tissue value.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Boxerman JL et al. (2006). Relative cerebral blood volume maps corrected
   for contrast agent extravasation significantly correlate with glioma tumor
   grade, whereas uncorrected maps do not. AJNR 27(4):859-867.
.. [3] Dickie BR et al. MRM 2024. doi:10.1002/mrm.30101
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from osipy.common.backend.array_module import get_array_module, to_numpy
from osipy.common.exceptions import DataValidationError
from osipy.common.parameter_map import ParameterMap

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

NORMALIZATION_REGISTRY: dict[str, Callable] = {}


def register_normalizer(name: str):
    """Register a normalization method function."""

    def decorator(func):
        if name in NORMALIZATION_REGISTRY:
            logger.warning("Overwriting '%s' with %s", name, func.__name__)
        NORMALIZATION_REGISTRY[name] = func
        return func

    return decorator


def get_normalizer(name: str) -> Callable:
    """Get a normalizer function by name."""
    if name not in NORMALIZATION_REGISTRY:
        valid = ", ".join(sorted(NORMALIZATION_REGISTRY.keys()))
        raise DataValidationError(f"Unknown normalizer: {name}. Valid: {valid}")
    return NORMALIZATION_REGISTRY[name]


def list_normalizers() -> list[str]:
    """List registered normalization methods."""
    return sorted(NORMALIZATION_REGISTRY.keys())


@register_normalizer("mean")
def _normalize_mean(wm_values, xp):
    return float(xp.mean(wm_values))


@register_normalizer("median")
def _normalize_median(wm_values, xp):
    return float(xp.median(wm_values))


@register_normalizer("robust_mean")
def _normalize_robust_mean(wm_values, xp, percentile_range=(25.0, 75.0)):
    low_p, high_p = percentile_range
    low_val = xp.percentile(wm_values, low_p)
    high_val = xp.percentile(wm_values, high_p)
    robust_values = wm_values[(wm_values >= low_val) & (wm_values <= high_val)]
    return (
        float(xp.mean(robust_values))
        if len(robust_values) > 0
        else float(xp.mean(wm_values))
    )


@dataclass
class NormalizationResult:
    """Result of perfusion map normalization.

    Attributes
    ----------
    normalized_map : ParameterMap
        Normalized parameter map (relative values).
    reference_value : float
        Mean value of the reference region.
    reference_std : float
        Standard deviation in reference region.
    reference_mask : NDArray[np.bool_]
        Mask of reference region voxels used.
    """

    normalized_map: ParameterMap
    reference_value: float
    reference_std: float
    reference_mask: NDArray[np.bool_]


def normalize_to_white_matter(
    parameter_map: ParameterMap,
    white_matter_mask: NDArray[np.bool_],
    method: str = "mean",
    percentile_range: tuple[float, float] = (25.0, 75.0),
) -> NormalizationResult:
    """Normalize perfusion parameter map to white matter reference.

    Divides parameter values by the mean (or median) white matter value
    to produce relative perfusion measurements. This is particularly
    important for rCBV in tumor imaging.

    Parameters
    ----------
    parameter_map : ParameterMap
        Parameter map to normalize (e.g., CBV, CBF).
    white_matter_mask : NDArray[np.bool_]
        Binary mask defining white matter reference region.
        Ideally contralateral normal-appearing white matter.
    method : str
        Normalization method: "mean", "median", or "robust_mean".
        "robust_mean" uses values within percentile_range.
    percentile_range : tuple[float, float]
        Percentile range for robust_mean method.

    Returns
    -------
    NormalizationResult
        Normalized map and reference statistics.

    Raises
    ------
    DataValidationError
        If white_matter_mask is empty or shapes don't match.

    Notes
    -----
    For rCBV, typical normal white matter values are used as reference,
    making the normalized rCBV = 1.0 in normal white matter and elevated
    (>1.5-2.0) in high-grade tumors.

    References
    ----------
    Wetzel SG et al. (2002). Relative cerebral blood volume measurements
    in intracranial mass lesions. Radiology 224(2):334-341.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.parameter_map import ParameterMap
    >>> from osipy.dsc.normalization import normalize_to_white_matter
    >>> cbv_values = np.random.rand(64, 64, 20) * 5
    >>> cbv_map = ParameterMap(
    ...     name="CBV", symbol="CBV", units="ml/100g",
    ...     values=cbv_values, affine=np.eye(4)
    ... )
    >>> wm_mask = np.zeros((64, 64, 20), dtype=bool)
    >>> wm_mask[10:20, 10:20, 5:15] = True
    >>> result = normalize_to_white_matter(cbv_map, wm_mask)
    """
    values = parameter_map.values
    xp = get_array_module(values, white_matter_mask)

    # Validate shapes
    if white_matter_mask.shape != values.shape[:3]:
        # Try to broadcast mask to 3D portion
        if white_matter_mask.ndim == 2 and values.ndim == 3:
            # Expand 2D mask to 3D
            white_matter_mask = xp.broadcast_to(
                white_matter_mask[..., np.newaxis],
                values.shape,
            ).copy()
        else:
            msg = (
                f"White matter mask shape {white_matter_mask.shape} "
                f"doesn't match parameter map shape {values.shape}"
            )
            raise DataValidationError(msg)

    # Ensure mask is boolean
    white_matter_mask = white_matter_mask.astype(bool)

    if not xp.any(white_matter_mask):
        msg = "White matter mask is empty"
        raise DataValidationError(msg)

    # Get values in reference region
    wm_values = values[white_matter_mask]

    # Remove any invalid values
    wm_values = wm_values[xp.isfinite(wm_values)]
    wm_values = wm_values[wm_values > 0]

    if len(wm_values) == 0:
        msg = "No valid values in white matter region"
        raise DataValidationError(msg)

    # Compute reference value via registry
    normalizer = get_normalizer(method)
    if method == "robust_mean":
        ref_value = normalizer(wm_values, xp, percentile_range=percentile_range)
    else:
        ref_value = normalizer(wm_values, xp)

    ref_std = float(xp.std(wm_values))

    # Normalize the map
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_values = values / ref_value

    normalized_values = xp.nan_to_num(
        normalized_values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Create normalized parameter map
    normalized_map = ParameterMap(
        name=f"r{parameter_map.name}",
        symbol=f"r{parameter_map.symbol}",
        units="relative",
        values=to_numpy(normalized_values),
        affine=parameter_map.affine,
        quality_mask=parameter_map.quality_mask,
        model_name=parameter_map.model_name,
        fitting_method=parameter_map.fitting_method,
        literature_reference=parameter_map.literature_reference,
    )

    return NormalizationResult(
        normalized_map=normalized_map,
        reference_value=ref_value,
        reference_std=ref_std,
        reference_mask=to_numpy(white_matter_mask),
    )


def compute_relative_cbv(
    cbv_map: ParameterMap,
    white_matter_mask: NDArray[np.bool_],
) -> ParameterMap:
    """Compute relative CBV (rCBV) normalized to white matter.

    Convenience function for the most common normalization use case.

    Parameters
    ----------
    cbv_map : ParameterMap
        Absolute CBV map.
    white_matter_mask : NDArray[np.bool_]
        White matter reference region mask.

    Returns
    -------
    ParameterMap
        Relative CBV map (rCBV).

    Examples
    --------
    >>> rcbv = compute_relative_cbv(cbv_map, wm_mask)
    """
    result = normalize_to_white_matter(cbv_map, white_matter_mask, method="robust_mean")
    return result.normalized_map
