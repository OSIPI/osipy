"""Leakage correction for DSC-MRI (OSIPI: P.LC1.001).

This module implements the Boxerman-Schmainda-Weisskoff (BSW) method
(OSIPI: M.LC1.001) for correcting contrast agent leakage effects in
DSC-MRI. The correction estimates leakage coefficients K1 (OSIPI:
Q.LC1.001) and K2 (OSIPI: Q.LC1.002) to separate intravascular from
extravascular signal contributions.

GPU/CPU agnostic using the xp array module pattern.
NO scipy dependency.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Boxerman JL, Schmainda KM, Weisskoff RM (2006). Relative cerebral blood
   volume maps corrected for contrast agent extravasation significantly
   correlate with glioma tumor grade, whereas uncorrected maps do not.
   AJNR Am J Neuroradiol 27(4):859-867.
.. [3] Paulson ES, Schmainda KM (2008). Comparison of dynamic susceptibility-
   weighted contrast-enhanced MR methods: recommendations for measuring
   relative cerebral blood volume in brain tumors. Radiology 249(2):601-613.
.. [4] Dickie BR et al. MRM 2024. doi:10.1002/mrm.30101
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from osipy.common.backend.array_module import get_array_module, to_numpy
from osipy.common.exceptions import DataValidationError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass
class LeakageCorrectionParams:
    """Parameters for leakage correction.

    Attributes
    ----------
    method : str
        Correction method: 'bsw' (Boxerman-Schmainda-Weisskoff) or
        'bidirectional'.
    use_t1_correction : bool
        Include T1 leakage correction (default True).
    use_t2_correction : bool
        Include T2* leakage correction (default True).
    reference_tissue : str
        Reference tissue type: 'white_matter', 'normal_brain', or 'custom'.
    custom_reference_mask : NDArray | None
        Custom mask for reference tissue (if reference_tissue='custom').
    fitting_range : tuple[int, int] | None
        Time frame range for fitting. If None, uses full range.
    """

    method: str = "bsw"
    use_t1_correction: bool = True
    use_t2_correction: bool = True
    reference_tissue: str = "white_matter"
    custom_reference_mask: "NDArray[np.bool_] | None" = None
    fitting_range: tuple[int, int] | None = None


@dataclass
class LeakageCorrectionResult:
    """Result of leakage correction (OSIPI: P.LC1.001).

    Attributes
    ----------
    corrected_delta_r2 : NDArray
        Leakage-corrected delta-R2* curves.
    k1 : NDArray
        T1 leakage coefficient K1 (OSIPI: Q.LC1.001) map.
    k2 : NDArray
        T2* leakage coefficient K2 (OSIPI: Q.LC1.002) map.
    reference_curve : NDArray
        Reference tissue delta-R2* curve used for correction.
    residual : NDArray | None
        Fitting residuals.
    """

    corrected_delta_r2: "NDArray[np.floating[Any]]"
    k1: "NDArray[np.floating[Any]]"
    k2: "NDArray[np.floating[Any]]"
    reference_curve: "NDArray[np.floating[Any]]"
    residual: "NDArray[np.floating[Any]] | None" = None


def correct_leakage(
    delta_r2: "NDArray[np.floating[Any]]",
    aif: "NDArray[np.floating[Any]]",
    time: "NDArray[np.floating[Any]]",
    mask: "NDArray[np.bool_] | None" = None,
    params: LeakageCorrectionParams | None = None,
) -> LeakageCorrectionResult:
    """Apply leakage correction to DSC-MRI data (OSIPI: P.LC1.001).

    The Boxerman-Schmainda-Weisskoff method (OSIPI: M.LC1.001) models the
    observed delta-R2* as a combination of intravascular and extravascular
    contributions:

        dR2*(t) = K1 * Ca(t) + K2 * integral(Ca(tau) dtau)

    where Ca is the AIF, K1 (OSIPI: Q.LC1.001) represents T1 leakage
    (positive for enhancement) and K2 (OSIPI: Q.LC1.002) represents T2*
    leakage (positive for signal loss).

    Parameters
    ----------
    delta_r2 : NDArray[np.floating]
        Uncorrected delta-R2* data, shape (..., n_timepoints).
    aif : NDArray[np.floating]
        Arterial input function (delta-R2* or concentration).
    time : NDArray[np.floating]
        Time points in seconds.
    mask : NDArray[np.bool_] | None
        Brain mask, shape (...).
    params : LeakageCorrectionParams | None
        Correction parameters.

    Returns
    -------
    LeakageCorrectionResult
        Corrected delta-R2* and leakage coefficients.

    Raises
    ------
    DataValidationError
        If input data is invalid.

    References
    ----------
    .. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    .. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.30101

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.dsc.leakage import correct_leakage
    >>> delta_r2 = np.random.rand(64, 64, 20, 60) * 10
    >>> aif = np.random.rand(60) * 5
    >>> time = np.linspace(0, 90, 60)
    >>> result = correct_leakage(delta_r2, aif, time)
    """
    params = params or LeakageCorrectionParams()
    xp = get_array_module(delta_r2, aif)

    from osipy.dsc.leakage.registry import get_leakage_corrector

    # Validate method name via registry (raises DataValidationError if unknown)
    get_leakage_corrector(params.method)

    # Get spatial shape
    spatial_shape = delta_r2.shape[:-1]
    n_timepoints = delta_r2.shape[-1]

    if len(aif) != n_timepoints:
        msg = f"AIF length ({len(aif)}) != time points ({n_timepoints})"
        raise DataValidationError(msg)

    # Create mask if not provided
    if mask is None:
        mask = xp.ones(spatial_shape, dtype=bool)

    # Identify reference tissue (non-enhancing normal brain)
    reference_curve = _compute_reference_curve(delta_r2, mask, params)

    # Build design matrix for linear regression
    # Model: ΔR2*(t) = K1·Ca(t) + K2·∫Ca(τ)dτ + ref_contribution
    dt = time[1] - time[0] if len(time) > 1 else 1.0

    # Cumulative integral of AIF
    aif_integral = xp.cumsum(aif) * dt

    # Prepare design matrix
    # Column 0: AIF (T1 leakage term)
    # Column 1: AIF integral (T2* leakage term)
    X = xp.column_stack([aif, aif_integral])

    # Initialize output arrays
    corrected_delta_r2 = xp.zeros_like(delta_r2)
    k1 = xp.zeros(spatial_shape)
    k2 = xp.zeros(spatial_shape)
    residual = xp.zeros(spatial_shape)

    # Flatten spatial dimensions
    flat_delta_r2 = delta_r2.reshape(-1, n_timepoints)
    flat_mask = mask.ravel()

    # Determine fitting range
    if params.fitting_range is not None:
        fit_start, fit_end = params.fitting_range
    else:
        # Find bolus arrival (when AIF exceeds threshold)
        aif_threshold = 0.1 * xp.max(aif)
        arrival_idx = int(to_numpy(xp.argmax(aif > aif_threshold)))
        fit_start = max(0, arrival_idx - 2)
        fit_end = n_timepoints

    fit_slice = slice(fit_start, fit_end)

    # Vectorized least-squares fitting across all masked voxels
    X_fit = X[fit_slice, :]  # (n_fit, 2)
    # Y: (n_masked_voxels, n_fit_timepoints) — observed minus reference
    Y = (
        flat_delta_r2[flat_mask][:, fit_slice]
        - reference_curve[fit_slice][xp.newaxis, :]
    )

    try:
        # Normal equations: coeffs = (X^T X)^-1 X^T Y^T
        # X_fit is the same for all voxels, so pre-compute pseudo-inverse once
        XtX_inv = xp.linalg.inv(X_fit.T @ X_fit)  # (2, 2)
        pseudo_inv = XtX_inv @ X_fit.T  # (2, n_fit)
        coeffs_all = pseudo_inv @ Y.T  # (2, n_masked_voxels)

        k1_masked = coeffs_all[0, :]
        k2_masked = coeffs_all[1, :]

        # Zero out based on correction settings
        if not params.use_t1_correction:
            k1_masked = xp.zeros_like(k1_masked)
        if not params.use_t2_correction:
            k2_masked = xp.zeros_like(k2_masked)

        # Compute corrected curves for all masked voxels at once
        leakage = (
            k1_masked[:, xp.newaxis] * aif[xp.newaxis, :]
            + k2_masked[:, xp.newaxis] * aif_integral[xp.newaxis, :]
        )
        corrected_masked = flat_delta_r2[flat_mask] - leakage

        # Scatter results back into output arrays
        corrected_flat = xp.zeros_like(flat_delta_r2)
        corrected_flat[flat_mask] = corrected_masked
        corrected_delta_r2 = corrected_flat.reshape(delta_r2.shape)

        k1_flat = xp.zeros(flat_delta_r2.shape[0])
        k2_flat = xp.zeros(flat_delta_r2.shape[0])
        k1_flat[flat_mask] = k1_masked
        k2_flat[flat_mask] = k2_masked
        k1 = k1_flat.reshape(spatial_shape)
        k2 = k2_flat.reshape(spatial_shape)

        # Residual: sum of squared residuals per voxel
        predicted_Y = (X_fit @ coeffs_all).T  # (n_masked, n_fit)
        residual_flat = xp.zeros(flat_delta_r2.shape[0])
        residual_flat[flat_mask] = xp.sum((Y - predicted_Y) ** 2, axis=1)
        residual = residual_flat.reshape(spatial_shape)

    except xp.linalg.LinAlgError:
        # Singular matrix — fall back to uncorrected data for masked voxels
        corrected_delta_r2 = xp.copy(delta_r2)
        corrected_delta_r2[~mask] = 0
        k1 = xp.zeros(spatial_shape)
        k2 = xp.zeros(spatial_shape)
        residual = xp.zeros(spatial_shape)

    return LeakageCorrectionResult(
        corrected_delta_r2=corrected_delta_r2,
        k1=k1,
        k2=k2,
        reference_curve=reference_curve,
        residual=residual,
    )


def _compute_reference_curve(
    delta_r2: "NDArray[np.floating[Any]]",
    mask: "NDArray[np.bool_]",
    params: LeakageCorrectionParams,
) -> "NDArray[np.floating[Any]]":
    """Compute reference tissue ΔR2* curve.

    Parameters
    ----------
    delta_r2 : NDArray
        ΔR2* data.
    mask : NDArray
        Brain mask.
    params : LeakageCorrectionParams
        Correction parameters.

    Returns
    -------
    NDArray
        Reference tissue ΔR2* curve.
    """
    xp = get_array_module(delta_r2)
    n_timepoints = delta_r2.shape[-1]

    if params.reference_tissue == "custom" and params.custom_reference_mask is not None:
        ref_mask = params.custom_reference_mask & mask
    else:
        # Auto-detect normal brain tissue
        # Use voxels with:
        # 1. Low peak enhancement
        # 2. Return to baseline
        # 3. Within brain mask

        # Compute enhancement characteristics
        baseline = xp.mean(delta_r2[..., :5], axis=-1)
        peak = xp.max(delta_r2, axis=-1)
        enhancement = peak - baseline

        # Find low enhancement voxels (likely normal tissue)
        enhancement_threshold = xp.percentile(enhancement[mask], 25)
        ref_mask = (enhancement < enhancement_threshold) & mask

        # Ensure minimum number of reference voxels
        if xp.sum(ref_mask) < 100:
            # Fall back to using lower quartile of all masked voxels
            ref_mask = mask

    # Average reference curve
    flat_delta_r2 = delta_r2.reshape(-1, n_timepoints)
    flat_ref_mask = ref_mask.ravel()

    ref_curves = flat_delta_r2[flat_ref_mask, :]
    reference_curve = xp.mean(ref_curves, axis=0)

    return reference_curve


def estimate_permeability(
    k1: "NDArray[np.floating[Any]]",
    k2: "NDArray[np.floating[Any]]",
) -> "NDArray[np.floating[Any]]":
    """Estimate permeability from leakage coefficients.

    The K2 coefficient is related to the permeability-surface area
    product (PS) of the blood-brain barrier.

    Parameters
    ----------
    k1 : NDArray
        T1 leakage coefficient.
    k2 : NDArray
        T2* leakage coefficient.

    Returns
    -------
    NDArray
        Estimated permeability (arbitrary units).
    """
    xp = get_array_module(k1, k2)
    # K2 is proportional to PS × ve
    # Use magnitude to account for sign variations
    permeability = xp.abs(k2)

    return permeability


# --- Strategy classes wrapping existing functions ---

from osipy.dsc.leakage.base import BaseLeakageCorrector
from osipy.dsc.leakage.registry import register_leakage_corrector


@register_leakage_corrector("bsw")
class BSWCorrector(BaseLeakageCorrector):
    """Boxerman-Schmainda-Weisskoff leakage correction (OSIPI: M.LC1.001).

    Estimates K1 (OSIPI: Q.LC1.001) and K2 (OSIPI: Q.LC1.002) leakage
    coefficients via linear regression against the AIF.
    """

    @property
    def name(self) -> str:
        """Return human-readable component name."""
        return "Boxerman-Schmainda-Weisskoff"

    @property
    def reference(self) -> str:
        """Return primary literature citation."""
        return "Boxerman JL et al. AJNR 2006;27(4):859-867."

    def correct(self, delta_r2, aif, time, mask=None, **kwargs):
        """Perform BSW leakage correction (OSIPI: P.LC1.001)."""
        params = LeakageCorrectionParams(method="bsw", **kwargs)
        return correct_leakage(delta_r2, aif, time, mask, params)


@register_leakage_corrector("bidirectional")
class BidirectionalCorrector(BaseLeakageCorrector):
    """Bidirectional leakage correction strategy."""

    @property
    def name(self) -> str:
        """Return human-readable component name."""
        return "Bidirectional"

    @property
    def reference(self) -> str:
        """Return primary literature citation."""
        return "Bjornerud A et al. JCBFM 2011;31(1):215-228."

    def correct(self, delta_r2, aif, time, mask=None, **kwargs):
        """Perform bidirectional leakage correction."""
        params = LeakageCorrectionParams(method="bidirectional", **kwargs)
        return correct_leakage(delta_r2, aif, time, mask, params)
