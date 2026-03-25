"""IVIM signal reconstruction and residual map computation.

This module provides utilities to reconstruct the expected IVIM signal from
fitted parameter maps and to compute per-voxel RMSE residual maps, enabling
quantitative assessment of fitting quality beyond the binary quality mask.

Functions
---------
reconstruct_ivim_signal
    Reconstruct IVIM bi-exponential signal from fitted parameter maps.
compute_rmse_map
    Compute per-voxel root mean square error between observed and fitted signal.

References
----------
.. [1] Le Bihan D et al. (1988). Separation of diffusion and perfusion in
   intravoxel incoherent motion MR imaging. Radiology 168(2):497-505.
.. [2] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from osipy.common.backend.array_module import get_array_module
from osipy.common.parameter_map import ParameterMap

if TYPE_CHECKING:
    from numpy.typing import NDArray


def reconstruct_ivim_signal(
    d_map: ParameterMap,
    d_star_map: ParameterMap,
    f_map: ParameterMap,
    s0_map: ParameterMap,
    b_values: "NDArray",
) -> "NDArray":
    """Reconstruct IVIM bi-exponential signal from fitted parameter maps.

    Computes the predicted signal at each voxel using the IVIM model:

    .. math::

        S(b) = S_0 \\left[ f \\, e^{-b D^*} + (1 - f) \\, e^{-b D} \\right]

    Parameters
    ----------
    d_map : ParameterMap
        Diffusion coefficient (D) map, shape (...,), units mm²/s.
    d_star_map : ParameterMap
        Pseudo-diffusion coefficient (D*) map, shape (...,), units mm²/s.
    f_map : ParameterMap
        Perfusion fraction (f) map, shape (...,), dimensionless.
    s0_map : ParameterMap
        Baseline signal (S0) map, shape (...,), arbitrary units.
    b_values : NDArray
        b-values in s/mm², shape (n_b,).

    Returns
    -------
    NDArray
        Reconstructed signal, shape (..., n_b).

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.ivim import fit_ivim, reconstruct_ivim_signal
    >>> b_values = np.array([0, 50, 100, 200, 400, 800], dtype=float)
    >>> signal = np.ones((4, 4, 6)) * 1000.0
    >>> result = fit_ivim(signal, b_values)
    >>> recon = reconstruct_ivim_signal(
    ...     result.d_map, result.d_star_map,
    ...     result.f_map, result.s0_map, b_values,
    ... )
    >>> recon.shape
    (4, 4, 1, 6)
    """
    xp = get_array_module(d_map.values)

    d = d_map.values        # (...,)
    ds = d_star_map.values  # (...,)
    f = f_map.values        # (...,)
    s0 = s0_map.values      # (...,)

    # b_values: (n_b,) — cast to same array module
    b = xp.asarray(b_values)

    # Broadcast: (..., 1) * (n_b,) → (..., n_b)
    signal = s0[..., xp.newaxis] * (
        f[..., xp.newaxis] * xp.exp(-b * ds[..., xp.newaxis])
        + (1.0 - f[..., xp.newaxis]) * xp.exp(-b * d[..., xp.newaxis])
    )
    return signal


def compute_rmse_map(
    signal: "NDArray",
    d_map: ParameterMap,
    d_star_map: ParameterMap,
    f_map: ParameterMap,
    s0_map: ParameterMap,
    b_values: "NDArray",
    mask: "NDArray | None" = None,
) -> ParameterMap:
    """Compute per-voxel RMSE between observed and fitted IVIM signal.

    Reconstructs the predicted IVIM signal from the fitted parameter maps
    and computes the root mean square error at each voxel:

    .. math::

        \\text{RMSE}(v) = \\sqrt{\\frac{1}{N_b} \\sum_{i=1}^{N_b}
        (S_{\\text{obs},i}(v) - S_{\\text{fit},i}(v))^2}

    Voxels outside ``mask`` are set to ``NaN``.

    Parameters
    ----------
    signal : NDArray
        Observed signal, shape (..., n_b).
    d_map : ParameterMap
        Fitted D map, shape (...,).
    d_star_map : ParameterMap
        Fitted D* map, shape (...,).
    f_map : ParameterMap
        Fitted f map, shape (...,).
    s0_map : ParameterMap
        Fitted S0 map, shape (...,).
    b_values : NDArray
        b-values in s/mm², shape (n_b,).
    mask : NDArray | None
        Boolean spatial mask, shape (...,). If ``None``, all voxels
        are included.

    Returns
    -------
    ParameterMap
        RMSE residual map with:
        - ``name`` = ``"RMSE"``
        - ``symbol`` = ``"RMSE"``
        - ``units`` = ``"a.u."``
        - ``values`` shape = spatial shape of ``d_map.values``
        - unmasked voxels set to ``NaN``

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.ivim import fit_ivim, compute_rmse_map
    >>> b_values = np.array([0, 50, 100, 200, 400, 800], dtype=float)
    >>> signal_1d = 1000.0 * (0.9 * np.exp(-b_values * 1e-3)
    ...                       + 0.1 * np.exp(-b_values * 10e-3))
    >>> signal = np.broadcast_to(signal_1d, (4, 4, 6)).copy()
    >>> result = fit_ivim(signal, b_values)
    >>> rmse = compute_rmse_map(
    ...     signal, result.d_map, result.d_star_map,
    ...     result.f_map, result.s0_map, b_values,
    ... )
    >>> rmse.values.shape
    (4, 4, 1)
    """
    xp = get_array_module(d_map.values)

    signal_arr = xp.asarray(signal)
    spatial_shape = d_map.values.shape

    # Reconstruct fitted signal: shape (*spatial_shape, n_b)
    signal_fit = reconstruct_ivim_signal(
        d_map, d_star_map, f_map, s0_map, b_values
    )

    # Reshape observed signal to match (*spatial_shape, n_b) if needed
    if signal_arr.shape != signal_fit.shape:
        signal_arr = signal_arr.reshape(signal_fit.shape)

    # Per-voxel RMSE over b-value axis (last axis)
    residuals = signal_arr - signal_fit
    rmse_vals = xp.sqrt(xp.mean(residuals ** 2, axis=-1))  # (*spatial_shape,)

    # Apply mask — NaN for unmasked voxels
    if mask is not None:
        mask_arr = xp.asarray(mask)
        # mask may be 2D (x,y) while rmse_vals is (x,y,1) — broadcast
        if mask_arr.shape != rmse_vals.shape:
            mask_arr = mask_arr.reshape(
                mask_arr.shape + (1,) * (rmse_vals.ndim - mask_arr.ndim)
            )
        nan_val = xp.full(rmse_vals.shape, xp.nan if hasattr(xp, "nan") else float("nan"))
        rmse_vals = xp.where(mask_arr, rmse_vals, nan_val)

    return ParameterMap(
        name="RMSE",
        symbol="RMSE",
        units="a.u.",
        values=rmse_vals,
        affine=d_map.affine,
        quality_mask=d_map.quality_mask,
    )
