"""Shared batch fitting utilities.

This module provides ParameterMap creation and image-level fitting
helpers used by ``BaseFitter.fit_image()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.parameter_map import ParameterMap

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from osipy.common.models.fittable import FittableModel


def create_parameter_maps(
    model: FittableModel,
    fitted_params: NDArray[Any],
    r2_values: NDArray[Any],
    converged: NDArray[Any],
    voxel_indices: NDArray[Any],
    shape: tuple[int, int, int],
    r2_threshold: float = 0.5,
    fitting_method: str = "levenberg_marquardt",
) -> dict[str, ParameterMap]:
    """Create ParameterMap objects from fitted results.

    Parameters
    ----------
    model : FittableModel
        Model that was fit (provides parameter names, units, reference).
    fitted_params : NDArray
        Fitted parameters, shape ``(n_params, n_voxels)``.
    r2_values : NDArray
        R-squared values, shape ``(n_voxels,)``.
    converged : NDArray
        Convergence flags, shape ``(n_voxels,)``.
    voxel_indices : NDArray
        Original voxel indices, shape ``(n_voxels, 3)``.
    shape : tuple[int, int, int]
        Output volume shape ``(nx, ny, nz)``.
    r2_threshold : float
        Deprecated, ignored. Kept for signature compatibility; quality mask
        no longer applies an R² cutoff.
    fitting_method : str
        Name of fitting method for metadata.

    Returns
    -------
    dict[str, ParameterMap]
        Parameter maps including ``"r_squared"``. The attached
        ``quality_mask`` simply marks voxels that were fit — callers are
        expected to filter on r² or parameter values downstream.
    """
    param_names = model.parameters
    param_units = model.parameter_units

    result: dict[str, ParameterMap] = {}

    # Quality mask marks which voxels were fit; no r²/convergence cutoff.
    quality_mask = np.zeros(shape, dtype=bool)
    quality_mask[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True

    affine = np.eye(4)

    # Get reference safely (may not be available on all models)
    ref = model.reference if hasattr(model, "reference") else ""

    for i, name in enumerate(param_names):
        volume = np.zeros(shape, dtype=np.float64)
        volume[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = (
            fitted_params[i, :]
        )

        result[name] = ParameterMap(
            name=name,
            symbol=name,
            values=volume,
            affine=affine,
            units=param_units.get(name, ""),
            quality_mask=quality_mask.copy(),
            model_name=model.name,
            fitting_method=fitting_method,
            literature_reference=ref,
        )

    # Add R-squared map
    r2_volume = np.zeros(shape, dtype=np.float64)
    r2_volume[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = r2_values

    result["r_squared"] = ParameterMap(
        name="r_squared",
        symbol="R²",
        values=r2_volume,
        affine=affine,
        units="",
        quality_mask=quality_mask.copy(),
        model_name=model.name,
        fitting_method=fitting_method,
        literature_reference=ref,
    )

    return result


def create_empty_maps(
    model: FittableModel,
    shape: tuple[int, int, int],
) -> dict[str, ParameterMap]:
    """Create empty parameter maps when no voxels to fit."""
    param_names = model.parameters
    param_units = model.parameter_units

    result: dict[str, ParameterMap] = {}
    empty_mask = np.zeros(shape, dtype=bool)

    for name in param_names:
        result[name] = ParameterMap(
            name=name,
            values=np.zeros(shape, dtype=np.float64),
            units=param_units.get(name, ""),
            quality_mask=empty_mask.copy(),
        )

    result["r_squared"] = ParameterMap(
        name="r_squared",
        values=np.zeros(shape, dtype=np.float64),
        units="",
        quality_mask=empty_mask.copy(),
    )

    return result
