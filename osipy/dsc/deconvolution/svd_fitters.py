"""SVD-based fitters for DSC deconvolution.

These fitters work with ``BoundDSCModel`` which pre-computes the SVD
of the AIF convolution matrix. Each fitter implements a different
regularization strategy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from osipy.common.backend.array_module import get_array_module, to_numpy
from osipy.common.fitting.base import BaseFitter
from osipy.common.fitting.registry import register_fitter

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from osipy.common.models.fittable import FittableModel


def _extract_perfusion_params(
    residue_batch: NDArray[np.floating[Any]],
    dt: float,
    xp: Any,
) -> tuple[
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
]:
    """Extract CBF, MTT, Ta from residue functions.

    Parameters
    ----------
    residue_batch : NDArray
        Residue functions, shape ``(n_timepoints, n_voxels)``.
    dt : float
        Time step in seconds.
    xp : module
        Array module.

    Returns
    -------
    cbf, mtt, ta : NDArray
        Each shape ``(n_voxels,)``.
    """
    # Non-negative
    residue_batch = xp.maximum(residue_batch, 0.0)

    cbf = xp.max(residue_batch, axis=0)
    mtt = xp.where(
        cbf > 0,
        xp.sum(residue_batch, axis=0) * dt / cbf,
        0.0,
    )
    ta = xp.argmax(residue_batch, axis=0).astype(residue_batch.dtype) * dt
    return cbf, mtt, ta


def _compute_r2(
    observed: NDArray[np.floating[Any]],
    predicted: NDArray[np.floating[Any]],
    xp: Any,
) -> NDArray[np.floating[Any]]:
    """Compute R-squared for batch."""
    ss_res = xp.sum((observed - predicted) ** 2, axis=0)
    ss_tot = xp.sum(
        (observed - xp.mean(observed, axis=0, keepdims=True)) ** 2,
        axis=0,
    )
    return xp.where(ss_tot > 0, 1.0 - ss_res / (ss_tot + 1e-10), 0.0)


def _pad_to_matrix_size(
    arr: NDArray[np.floating[Any]],
    target_len: int,
    xp: Any,
) -> NDArray[np.floating[Any]]:
    """Zero-pad ``arr`` (n_t, n_voxels) along axis 0 to ``target_len`` rows.

    No-op if already at ``target_len`` (the Toeplitz case, where the AIF
    matrix is ``n x n`` rather than the circulant ``2n x 2n``).
    """
    n_t = arr.shape[0]
    if n_t == target_len:
        return arr
    padded = xp.zeros((target_len, arr.shape[1]), dtype=arr.dtype)
    padded[:n_t] = arr
    return padded


def _reconstruct_r2(
    observed: NDArray[np.floating[Any]],
    R_batch: NDArray[np.floating[Any]],
    model: FittableModel,
    xp: Any,
) -> NDArray[np.floating[Any]]:
    """Compute R2 from reconstructed signal.

    ``R_batch`` holds the physical (length n_t) residue; for a circulant
    model (matrix size 2*n_t) it is zero-padded before applying the
    forward operator, and the reconstruction is truncated back to n_t.
    """
    n_t = observed.shape[0]
    R_nn = _pad_to_matrix_size(xp.maximum(R_batch, 0.0), model._U.shape[0], xp)
    recon = (model._U @ (model._S[:, xp.newaxis] * (model._Vh @ R_nn)))[:n_t]
    return _compute_r2(observed, recon, xp)


@register_fitter("sSVD")
class SSVDFitter(BaseFitter):
    """Standard SVD fitter with truncation.

    Uses a Toeplitz (causal) convolution matrix. The ``BoundDSCModel``
    must be created with ``matrix_type='toeplitz'`` for correct results.

    References
    ----------
    .. [1] Ostergaard L et al. MRM 1996;36(5):715-725.
    """

    fitting_method_name = "sSVD"

    def __init__(self, threshold: float = 0.2) -> None:
        self.threshold = threshold

    def fit_batch(
        self,
        model: FittableModel,
        observed_batch: NDArray[np.floating[Any]],
        bounds_override: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Fit a batch of voxels via standard SVD truncation.

        Parameters
        ----------
        model : FittableModel
            Bound DSC model with pre-computed SVD.
        observed_batch : NDArray
            Observed concentration curves, shape ``(n_timepoints, n_voxels)``.
        bounds_override : dict or None
            Not used by SVD fitters; accepted for interface compatibility.

        Returns
        -------
        params : NDArray
            Fitted [CBF, MTT, Ta], shape ``(3, n_voxels)``.
        r2 : NDArray
            R-squared goodness-of-fit, shape ``(n_voxels,)``.
        converged : NDArray
            Boolean convergence flags, shape ``(n_voxels,)``.
        """
        xp = get_array_module(observed_batch)
        _n_t, n_voxels = observed_batch.shape

        S = model._S
        s_max = S[0]
        S_inv = xp.where(self.threshold * s_max < S, 1.0 / S, 0.0)

        UtC = model._U.T @ observed_batch
        R_batch = model._Vh.T @ (S_inv[:, xp.newaxis] * UtC)

        cbf, mtt, ta = _extract_perfusion_params(R_batch, model._dt, xp)
        params = xp.stack([cbf, mtt, ta])

        # R2: reconstruct and compare
        R_nn = xp.maximum(R_batch, 0.0)
        peak = xp.max(R_nn, axis=0, keepdims=True) + 1e-10
        predicted = cbf[xp.newaxis, :] * R_nn / peak
        A_recon = model._U @ (
            model._S[:, xp.newaxis] * (model._Vh @ (predicted * model._dt))
        )
        r2 = _compute_r2(observed_batch, A_recon, xp)

        converged = xp.ones(n_voxels, dtype=bool)
        return params, r2, converged


@register_fitter("cSVD")
class CSVDFitter(BaseFitter):
    """Circular SVD fitter with truncation.

    Uses a block-circulant convolution matrix, making deconvolution
    insensitive to bolus arrival time delays.

    References
    ----------
    .. [1] Wu O et al. MRM 2003;50(1):164-174.
    """

    fitting_method_name = "cSVD"

    def __init__(self, threshold: float = 0.2) -> None:
        self.threshold = threshold

    def fit_batch(
        self,
        model: FittableModel,
        observed_batch: NDArray[np.floating[Any]],
        bounds_override: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Fit a batch of voxels via circular SVD truncation.

        Parameters
        ----------
        model : FittableModel
            Bound DSC model with pre-computed SVD (circulant matrix).
        observed_batch : NDArray
            Observed concentration curves, shape ``(n_timepoints, n_voxels)``.
        bounds_override : dict or None
            Not used by SVD fitters; accepted for interface compatibility.

        Returns
        -------
        params : NDArray
            Fitted [CBF, MTT, Ta], shape ``(3, n_voxels)``.
        r2 : NDArray
            R-squared goodness-of-fit, shape ``(n_voxels,)``.
        converged : NDArray
            Boolean convergence flags, shape ``(n_voxels,)``.
        """
        xp = get_array_module(observed_batch)
        n_t, n_voxels = observed_batch.shape

        S = model._S
        s_max = S[0]
        S_inv = xp.where(self.threshold * s_max < S, 1.0 / S, 0.0)

        # Circulant matrix is 2*n_t x 2*n_t: zero-pad the observed curve
        # before solving, then truncate the residue back to n_t physical
        # points (the wraparound tail is not the residue function).
        padded = _pad_to_matrix_size(observed_batch, model._U.shape[0], xp)
        UtC = model._U.T @ padded
        R_full = model._Vh.T @ (S_inv[:, xp.newaxis] * UtC)
        R_batch = R_full[:n_t]

        cbf, mtt, ta = _extract_perfusion_params(R_batch, model._dt, xp)
        params = xp.stack([cbf, mtt, ta])

        r2 = _reconstruct_r2(observed_batch, R_batch, model, xp)
        converged = xp.ones(n_voxels, dtype=bool)
        return params, r2, converged


@register_fitter("oSVD")
class OSVDFitter(BaseFitter):
    """Oscillation-index SVD fitter.

    Automatically selects the SVD truncation threshold per voxel
    to minimize oscillations in R(t) while preserving CBF estimates.

    References
    ----------
    .. [1] Wu O et al. MRM 2003;50(1):164-174.
    """

    fitting_method_name = "oSVD"

    def __init__(
        self,
        oscillation_index: float = 0.035,
        default_threshold: float = 0.2,
    ) -> None:
        self.oscillation_index = oscillation_index
        self.default_threshold = default_threshold

    def fit_batch(
        self,
        model: FittableModel,
        observed_batch: NDArray[np.floating[Any]],
        bounds_override: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Fit a batch of voxels via oscillation-index SVD.

        Searches, per voxel, for the *least* amount of SVD truncation
        (ascending threshold) whose residue's oscillation index first
        drops below the target -- i.e. the minimal regularization needed
        to suppress ringing, not the maximal one (Wu et al. 2003).

        Parameters
        ----------
        model : FittableModel
            Bound DSC model with pre-computed SVD (circulant matrix).
        observed_batch : NDArray
            Observed concentration curves, shape ``(n_timepoints, n_voxels)``.
        bounds_override : dict or None
            Not used by SVD fitters; accepted for interface compatibility.

        Returns
        -------
        params : NDArray
            Fitted [CBF, MTT, Ta], shape ``(3, n_voxels)``.
        r2 : NDArray
            R-squared goodness-of-fit, shape ``(n_voxels,)``.
        converged : NDArray
            Boolean convergence flags, shape ``(n_voxels,)``.
        """
        from osipy.dsc.deconvolution.svd import (
            _compute_oscillation_index_batch,
        )

        xp = get_array_module(observed_batch)
        n_t, n_voxels = observed_batch.shape

        S = model._S
        s_max = S[0]
        target_oi = self.oscillation_index

        # Circulant matrix is 2*n_t x 2*n_t: zero-pad the observed curve
        # before solving, then truncate each candidate residue back to
        # n_t physical points (the wraparound tail is not the residue
        # function, and must not feed into the oscillation index).
        padded = _pad_to_matrix_size(observed_batch, model._U.shape[0], xp)
        UtC = model._U.T @ padded

        # Search per-voxel threshold, ascending (least truncation first).
        # Take the FIRST threshold whose OI meets the target -- not the one
        # that minimizes OI, which would keep increasing regularization all
        # the way to the top of the range and over-smooth the residue.
        # Span 0-95% is Wu et al. (2003)'s sSVD/cSVD calibration range; the
        # paper does not state oSVD's internal range, but a 0.5 ceiling leaves
        # the target OI unreachable on short acquisitions.
        thresholds = xp.linspace(0.01, 0.95, 40)
        best_threshold = xp.full(
            n_voxels,
            self.default_threshold,
            dtype=observed_batch.dtype,
        )
        found = xp.zeros(n_voxels, dtype=bool)

        for k in range(len(thresholds)):
            thresh_val = thresholds[k]
            s_thresh = thresh_val * s_max
            S_inv = xp.where(s_thresh < S, 1.0 / S, 0.0)
            r_all = (model._Vh.T @ (S_inv[:, xp.newaxis] * UtC)).T[
                :, :n_t
            ]  # (n_voxels, n_t)

            oi = _compute_oscillation_index_batch(r_all, xp)
            meets_target = (oi < target_oi) & (~found)
            best_threshold = xp.where(meets_target, thresh_val, best_threshold)
            found = found | meets_target

        # Apply per-voxel thresholds
        unique_thresholds = xp.unique(best_threshold)
        R_batch = xp.zeros(
            (n_t, n_voxels),
            dtype=observed_batch.dtype,
        )

        for thresh in unique_thresholds:
            thresh_float = float(to_numpy(thresh))
            voxel_sel = best_threshold == thresh
            s_thresh = thresh_float * float(to_numpy(s_max))
            S_inv = xp.where(s_thresh < S, 1.0 / S, 0.0)
            r_sel = model._Vh.T @ (S_inv[:, xp.newaxis] * UtC[:, voxel_sel])
            R_batch[:, voxel_sel] = r_sel[:n_t]

        cbf, mtt, ta = _extract_perfusion_params(
            R_batch,
            model._dt,
            xp,
        )
        params = xp.stack([cbf, mtt, ta])

        r2 = _reconstruct_r2(observed_batch, R_batch, model, xp)
        converged = xp.ones(n_voxels, dtype=bool)
        return params, r2, converged


@register_fitter("tikhonov")
class TikhonovFitter(BaseFitter):
    """Tikhonov regularization fitter for DSC deconvolution.

    Uses Tikhonov filtering: filter_i = S_i / (S_i^2 + lambda^2)
    with optional L-curve criterion for lambda selection.

    References
    ----------
    .. [1] Calamante F, Gadian DG, Connelly A. Quantification of
       bolus-tracking MRI: improved characterization of the tissue
       residue function using Tikhonov regularization. MRM
       2003;50(6):1237-1247.
    """

    fitting_method_name = "tikhonov"

    def __init__(self, lambda_: float = 0.1) -> None:
        self.lambda_ = lambda_

    def fit_batch(
        self,
        model: FittableModel,
        observed_batch: NDArray[np.floating[Any]],
        bounds_override: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Fit a batch of voxels via Tikhonov-regularized SVD.

        Applies Tikhonov filtering S_i / (S_i^2 + lambda^2) to the
        singular values for smooth residue function recovery.

        Parameters
        ----------
        model : FittableModel
            Bound DSC model with pre-computed SVD.
        observed_batch : NDArray
            Observed concentration curves, shape ``(n_timepoints, n_voxels)``.
        bounds_override : dict or None
            Not used by SVD fitters; accepted for interface compatibility.

        Returns
        -------
        params : NDArray
            Fitted [CBF, MTT, Ta], shape ``(3, n_voxels)``.
        r2 : NDArray
            R-squared goodness-of-fit, shape ``(n_voxels,)``.
        converged : NDArray
            Boolean convergence flags, shape ``(n_voxels,)``.
        """
        xp = get_array_module(observed_batch)
        _n_t, n_voxels = observed_batch.shape

        S = model._S
        filter_factors = S / (S**2 + self.lambda_**2)

        UtC = model._U.T @ observed_batch
        R_batch = model._Vh.T @ (filter_factors[:, xp.newaxis] * UtC)

        cbf, mtt, ta = _extract_perfusion_params(
            R_batch,
            model._dt,
            xp,
        )
        params = xp.stack([cbf, mtt, ta])

        r2 = _reconstruct_r2(observed_batch, R_batch, model, xp)
        converged = xp.ones(n_voxels, dtype=bool)
        return params, r2, converged
