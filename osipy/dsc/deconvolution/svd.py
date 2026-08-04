"""SVD-based deconvolution for DSC-MRI.

This module implements Singular Value Decomposition (SVD) based
deconvolution methods for estimating the residue function, cerebral
blood flow (CBF, OSIPI: Q.PH1.003), mean transit time (MTT, OSIPI:
Q.PH1.006), and arterial delay (Ta, OSIPI: Q.PH1.007) from DSC-MRI
concentration data.

GPU/CPU Agnostic:
    Uses xp.linalg.svd where xp is numpy or cupy depending on input.
    When input arrays are CuPy arrays, all computations including SVD
    run on GPU. Results are returned on the same device as input.

NO scipy dependency - uses xp.linalg for SVD (numpy.linalg or cupy.linalg).

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Ostergaard L et al. (1996). High resolution measurement of cerebral blood
   flow using intravascular tracer bolus passages. Part I: Mathematical
   approach and statistical analysis. Magn Reson Med 36(5):715-725.
.. [3] Wu O et al. (2003). Tracer arrival timing-insensitive technique for
   estimating flow in MR perfusion-weighted imaging using singular value
   decomposition with a block-circulant deconvolution matrix.
   Magn Reson Med 50(1):164-174.
.. [4] Dickie BR et al. (2024). A community-endorsed open-source lexicon for
   contrast agent-based perfusion MRI: A consensus guidelines report from the
   ISMRM Open Science Initiative for Perfusion Imaging (OSIPI). Magn Reson Med
   91(5):1761-1773. doi:10.1002/mrm.29840
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from osipy.common.backend.array_module import get_array_module, to_numpy

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass
class SVDDeconvolutionParams:
    """Parameters for SVD deconvolution.

    Attributes
    ----------
    method : str
        Deconvolution method: 'sSVD', 'cSVD', or 'oSVD'.
    threshold : float
        SVD truncation threshold (fraction of max singular value).
        Default 0.2 for standard SVD.
    oscillation_index : float
        Target oscillation index for oSVD. Default 0.035.
    block_circulant : bool
        Use block-circulant matrix (for cSVD/oSVD). Default True for
        cSVD/oSVD methods.
    regularization : str
        Regularization method: 'truncation' or 'tikhonov'.
    """

    method: Literal["sSVD", "cSVD", "oSVD"] = "oSVD"
    threshold: float = 0.2
    oscillation_index: float = 0.035
    block_circulant: bool = True
    regularization: str = "truncation"


@dataclass
class DeconvolutionResult:
    """Result of deconvolution.

    Attributes
    ----------
    residue_function : NDArray
        Estimated residue function R(t), shape (..., n_timepoints).
    cbf : NDArray
        Cerebral blood flow (OSIPI: Q.PH1.003), max of scaled R(t),
        shape (...). Units depend on input scaling.
    mtt : NDArray
        Mean transit time (OSIPI: Q.PH1.006), area/max of R(t),
        shape (...). Units are same as time input.
    delay : NDArray
        Arterial delay Ta (OSIPI: Q.PH1.007) relative to AIF, shape (...).
    threshold_used : float
        Actual SVD threshold used.
    """

    residue_function: "NDArray[np.floating[Any]]"
    cbf: "NDArray[np.floating[Any]]"
    mtt: "NDArray[np.floating[Any]]"
    delay: "NDArray[np.floating[Any]]"
    threshold_used: float


def deconvolve_oSVD(
    concentration: "NDArray[np.floating[Any]]",
    aif: "NDArray[np.floating[Any]]",
    time: "NDArray[np.floating[Any]]",
    mask: "NDArray[np.bool_] | None" = None,
    params: SVDDeconvolutionParams | None = None,
) -> DeconvolutionResult:
    """Oscillation-index SVD (oSVD) deconvolution.

    The oSVD method automatically selects the SVD truncation threshold
    to minimize oscillations in the residue function while preserving
    the CBF estimate.

    GPU/CPU agnostic - uses xp.linalg.svd where xp matches input arrays.

    Parameters
    ----------
    concentration : NDArray[np.floating]
        Tissue concentration curves.
    aif : NDArray[np.floating]
        Arterial input function.
    time : NDArray[np.floating]
        Time points in seconds.
    mask : NDArray[np.bool_] | None
        Brain mask.
    params : SVDDeconvolutionParams | None
        Deconvolution parameters.

    Returns
    -------
    DeconvolutionResult
        Deconvolution results.
    """
    params = params or SVDDeconvolutionParams(method="oSVD")
    xp = get_array_module(concentration, aif, time)

    concentration = xp.asarray(concentration)
    aif = xp.asarray(aif)
    time = xp.asarray(time)

    spatial_shape = concentration.shape[:-1]
    n_timepoints = concentration.shape[-1]
    dt = float(to_numpy(time[1] - time[0])) if len(time) > 1 else 1.0

    mask = xp.ones(spatial_shape, dtype=bool) if mask is None else xp.asarray(mask)

    # Build block-circulant AIF matrix and compute SVD once
    A = _build_circulant_matrix_xp(aif, n_timepoints, xp) * dt
    U, S, Vh = xp.linalg.svd(A, full_matrices=False)

    # Flatten spatial dimensions and extract masked voxels
    flat_conc = concentration.reshape(-1, n_timepoints)
    flat_mask = mask.ravel()
    masked_conc = flat_conc[flat_mask]  # (n_masked, n_timepoints)

    if masked_conc.shape[0] == 0:
        residue = xp.zeros_like(flat_conc).reshape(concentration.shape)
        cbf = xp.zeros(spatial_shape, dtype=concentration.dtype)
        mtt = xp.zeros(spatial_shape, dtype=concentration.dtype)
        delay = xp.zeros(spatial_shape, dtype=concentration.dtype)
        return DeconvolutionResult(
            residue_function=residue,
            cbf=cbf,
            mtt=mtt,
            delay=delay,
            threshold_used=params.threshold,
        )

    # Zero-pad concentration to the circulant length 2n (block-circulant),
    # then precompute U^T @ C_padded for all masked voxels.
    n_pts = masked_conc.shape[1]
    L = U.shape[0]  # 2n
    padded_conc = xp.zeros((masked_conc.shape[0], L), dtype=concentration.dtype)
    padded_conc[:, :n_pts] = masked_conc
    UtC = U.T @ padded_conc.T  # (2n, n_masked)

    # Search per-voxel threshold, ascending (least truncation first). Take
    # the FIRST threshold whose OI meets the target -- not the one that
    # minimizes OI, which would keep increasing regularization all the way
    # to the top of the range and over-smooth the residue (Wu et al. 2003
    # intends the *minimal* regularization needed to suppress ringing).
    thresholds = xp.linspace(0.01, 0.5, 20)
    s_max = S[0]
    target_oi = params.oscillation_index

    best_threshold = xp.full(
        masked_conc.shape[0],
        params.threshold,
        dtype=concentration.dtype,
    )
    found = xp.zeros(masked_conc.shape[0], dtype=bool)

    for k in range(len(thresholds)):
        thresh_val = thresholds[k]
        # Build truncated S_inv for this threshold
        s_thresh = thresh_val * s_max
        S_inv = xp.where(s_thresh < S, 1.0 / S, 0.0)
        # Solve for all voxels: r = Vh^T @ diag(S_inv) @ U^T @ c  (first n points)
        r_all = (Vh.T @ (S_inv[:, None] * UtC)).T[:, :n_pts]  # (n_masked, n_timepoints)

        # Vectorized oscillation index on the physical residue
        oi = _compute_oscillation_index_batch(r_all, xp)

        # Accept the first threshold (per voxel) that meets the target
        meets_target = (oi < target_oi) & (~found)
        best_threshold = xp.where(meets_target, thresh_val, best_threshold)
        found = found | meets_target

    # Apply final per-voxel thresholds -- group by unique threshold for efficiency
    unique_thresholds = xp.unique(best_threshold)
    masked_residue = xp.zeros_like(masked_conc)

    for thresh in unique_thresholds:
        thresh_float = float(to_numpy(thresh))
        voxel_sel = best_threshold == thresh
        s_thresh = thresh_float * float(to_numpy(s_max))
        S_inv = xp.where(s_thresh < S, 1.0 / S, 0.0)
        r_batch = (Vh.T @ (S_inv[:, None] * UtC[:, voxel_sel])).T
        masked_residue[voxel_sel] = r_batch[:, :n_pts]

    # Ensure non-negative
    masked_residue = xp.maximum(masked_residue, 0.0)

    # Compute perfusion metrics vectorized
    masked_cbf = xp.max(masked_residue, axis=1)
    masked_mtt = xp.where(
        masked_cbf > 0,
        xp.sum(masked_residue, axis=1) * dt / masked_cbf,
        0.0,
    )
    masked_delay = xp.argmax(masked_residue, axis=1).astype(concentration.dtype) * dt

    # Scatter results back into full arrays
    residue = xp.zeros_like(flat_conc)
    residue[flat_mask] = masked_residue

    cbf_flat = xp.zeros(flat_conc.shape[0], dtype=concentration.dtype)
    cbf_flat[flat_mask] = masked_cbf

    mtt_flat = xp.zeros(flat_conc.shape[0], dtype=concentration.dtype)
    mtt_flat[flat_mask] = masked_mtt

    delay_flat = xp.zeros(flat_conc.shape[0], dtype=concentration.dtype)
    delay_flat[flat_mask] = masked_delay

    avg_threshold = float(to_numpy(xp.mean(best_threshold)))

    return DeconvolutionResult(
        residue_function=residue.reshape(concentration.shape),
        cbf=cbf_flat.reshape(spatial_shape),
        mtt=mtt_flat.reshape(spatial_shape),
        delay=delay_flat.reshape(spatial_shape),
        threshold_used=avg_threshold,
    )


def deconvolve_cSVD(
    concentration: "NDArray[np.floating[Any]]",
    aif: "NDArray[np.floating[Any]]",
    time: "NDArray[np.floating[Any]]",
    mask: "NDArray[np.bool_] | None" = None,
    params: SVDDeconvolutionParams | None = None,
) -> DeconvolutionResult:
    """Circular SVD (cSVD) deconvolution.

    Uses block-circulant matrix to make deconvolution insensitive
    to bolus arrival time delays.

    GPU/CPU agnostic - uses xp.linalg.svd where xp matches input arrays.

    Parameters
    ----------
    concentration : NDArray[np.floating]
        Tissue concentration curves.
    aif : NDArray[np.floating]
        Arterial input function.
    time : NDArray[np.floating]
        Time points in seconds.
    mask : NDArray[np.bool_] | None
        Brain mask.
    params : SVDDeconvolutionParams | None
        Deconvolution parameters.

    Returns
    -------
    DeconvolutionResult
        Deconvolution results.
    """
    params = params or SVDDeconvolutionParams(method="cSVD")
    xp = get_array_module(concentration, aif, time)

    concentration = xp.asarray(concentration)
    aif = xp.asarray(aif)
    time = xp.asarray(time)

    spatial_shape = concentration.shape[:-1]
    n_timepoints = concentration.shape[-1]
    dt = float(to_numpy(time[1] - time[0])) if len(time) > 1 else 1.0

    mask = xp.ones(spatial_shape, dtype=bool) if mask is None else xp.asarray(mask)

    # Build full 2n x 2n block-circulant AIF matrix and compute SVD once
    A = _build_circulant_matrix_xp(aif, n_timepoints, xp) * dt
    U, S, Vh = xp.linalg.svd(A, full_matrices=False)

    # Vectorized block-circulant solve (pads concentration to 2n, truncates R)
    residue, cbf, mtt, delay = _vectorized_circulant_solve(
        concentration,
        mask,
        U,
        S,
        Vh,
        params.threshold,
        dt,
        xp,
    )

    return DeconvolutionResult(
        residue_function=residue,
        cbf=cbf,
        mtt=mtt,
        delay=delay,
        threshold_used=params.threshold,
    )


def _deconvolve_sSVD(
    concentration: "NDArray[np.floating[Any]]",
    aif: "NDArray[np.floating[Any]]",
    time: "NDArray[np.floating[Any]]",
    mask: "NDArray[np.bool_] | None" = None,
    params: SVDDeconvolutionParams | None = None,
) -> DeconvolutionResult:
    """Standard SVD (sSVD) deconvolution.

    Uses lower-triangular Toeplitz matrix (causal convolution).

    GPU/CPU agnostic - uses xp.linalg.svd where xp matches input arrays.

    Parameters
    ----------
    concentration : NDArray[np.floating]
        Tissue concentration curves.
    aif : NDArray[np.floating]
        Arterial input function.
    time : NDArray[np.floating]
        Time points in seconds.
    mask : NDArray[np.bool_] | None
        Brain mask.
    params : SVDDeconvolutionParams | None
        Deconvolution parameters.

    Returns
    -------
    DeconvolutionResult
        Deconvolution results.
    """
    params = params or SVDDeconvolutionParams(method="sSVD")
    xp = get_array_module(concentration, aif, time)

    concentration = xp.asarray(concentration)
    aif = xp.asarray(aif)
    time = xp.asarray(time)

    spatial_shape = concentration.shape[:-1]
    n_timepoints = concentration.shape[-1]
    dt = float(to_numpy(time[1] - time[0])) if len(time) > 1 else 1.0

    mask = xp.ones(spatial_shape, dtype=bool) if mask is None else xp.asarray(mask)

    # Build lower-triangular Toeplitz AIF matrix and compute SVD once
    A = _build_toeplitz_matrix_xp(aif, n_timepoints, xp) * dt
    U, S, Vh = xp.linalg.svd(A, full_matrices=False)

    # Vectorized solve for all masked voxels
    residue, cbf, mtt, delay = _vectorized_svd_solve(
        concentration,
        mask,
        U,
        S,
        Vh,
        params.threshold,
        dt,
        xp,
    )

    return DeconvolutionResult(
        residue_function=residue,
        cbf=cbf,
        mtt=mtt,
        delay=delay,
        threshold_used=params.threshold,
    )


def _vectorized_svd_solve(
    concentration: "NDArray[np.floating[Any]]",
    mask: "NDArray[np.bool_]",
    U: "NDArray[np.floating[Any]]",
    S: "NDArray[np.floating[Any]]",
    Vh: "NDArray[np.floating[Any]]",
    threshold: float,
    dt: float,
    xp: Any,
) -> tuple[
    "NDArray[np.floating[Any]]",
    "NDArray[np.floating[Any]]",
    "NDArray[np.floating[Any]]",
    "NDArray[np.floating[Any]]",
]:
    """Vectorized SVD-based deconvolution for all masked voxels.

    Since the AIF convolution matrix (and its SVD) is the same for all
    voxels, we compute it once and apply the truncated pseudo-inverse
    to all voxels simultaneously via matrix multiplication.

    Parameters
    ----------
    concentration : NDArray
        Tissue concentration curves, shape (..., n_timepoints).
    mask : NDArray
        Boolean mask, shape (...).
    U, S, Vh : NDArray
        SVD components of the AIF convolution matrix.
    threshold : float
        Truncation threshold as fraction of max singular value.
    dt : float
        Time step in seconds.
    xp : module
        Array module (numpy or cupy).

    Returns
    -------
    residue : NDArray
        Residue functions, shape same as concentration.
    cbf : NDArray
        CBF values, shape same as mask.
    mtt : NDArray
        MTT values, shape same as mask.
    delay : NDArray
        Delay values, shape same as mask.
    """
    spatial_shape = concentration.shape[:-1]
    n_timepoints = concentration.shape[-1]

    flat_conc = concentration.reshape(-1, n_timepoints)
    flat_mask = mask.ravel()
    masked_conc = flat_conc[flat_mask]  # (n_masked, n_timepoints)

    if masked_conc.shape[0] == 0:
        return (
            xp.zeros_like(flat_conc).reshape(concentration.shape),
            xp.zeros(spatial_shape, dtype=concentration.dtype),
            xp.zeros(spatial_shape, dtype=concentration.dtype),
            xp.zeros(spatial_shape, dtype=concentration.dtype),
        )

    # Build truncated pseudo-inverse: Vh^T @ diag(S_inv) @ U^T
    s_max = S[0]
    s_thresh = threshold * s_max
    S_inv = xp.where(s_thresh < S, 1.0 / S, 0.0)

    # Solve all voxels at once: r = (Vh^T @ diag(S_inv) @ U^T) @ c^T
    # UtC shape: (n_timepoints, n_masked)
    UtC = U.T @ masked_conc.T
    masked_residue = (Vh.T @ (S_inv[:, None] * UtC)).T  # (n_masked, n_timepoints)

    # Ensure non-negative
    masked_residue = xp.maximum(masked_residue, 0.0)

    # Compute perfusion metrics vectorized
    masked_cbf = xp.max(masked_residue, axis=1)
    masked_mtt = xp.where(
        masked_cbf > 0,
        xp.sum(masked_residue, axis=1) * dt / masked_cbf,
        0.0,
    )
    masked_delay = xp.argmax(masked_residue, axis=1).astype(concentration.dtype) * dt

    # Scatter results back into full arrays
    residue = xp.zeros_like(flat_conc)
    residue[flat_mask] = masked_residue

    cbf_flat = xp.zeros(flat_conc.shape[0], dtype=concentration.dtype)
    cbf_flat[flat_mask] = masked_cbf

    mtt_flat = xp.zeros(flat_conc.shape[0], dtype=concentration.dtype)
    mtt_flat[flat_mask] = masked_mtt

    delay_flat = xp.zeros(flat_conc.shape[0], dtype=concentration.dtype)
    delay_flat[flat_mask] = masked_delay

    return (
        residue.reshape(concentration.shape),
        cbf_flat.reshape(spatial_shape),
        mtt_flat.reshape(spatial_shape),
        delay_flat.reshape(spatial_shape),
    )


def _vectorized_circulant_solve(
    concentration: "NDArray[np.floating[Any]]",
    mask: "NDArray[np.bool_]",
    U: "NDArray[np.floating[Any]]",
    S: "NDArray[np.floating[Any]]",
    Vh: "NDArray[np.floating[Any]]",
    threshold: float,
    dt: float,
    xp: Any,
) -> tuple[
    "NDArray[np.floating[Any]]",
    "NDArray[np.floating[Any]]",
    "NDArray[np.floating[Any]]",
    "NDArray[np.floating[Any]]",
]:
    """Block-circulant SVD deconvolution for all masked voxels.

    The circulant matrix is ``2n x 2n``, so each concentration curve is
    zero-padded to length ``2n`` before deconvolution and the residue is
    truncated to the first ``n`` points. The zero-padding plus the matrix
    wraparound make this insensitive to bolus arrival delay (Wu et al. 2003).
    """
    spatial_shape = concentration.shape[:-1]
    n = concentration.shape[-1]
    L = U.shape[0]  # 2n

    flat_conc = concentration.reshape(-1, n)
    flat_mask = mask.ravel()
    masked_conc = flat_conc[flat_mask]  # (n_masked, n)

    if masked_conc.shape[0] == 0:
        return (
            xp.zeros_like(flat_conc).reshape(concentration.shape),
            xp.zeros(spatial_shape, dtype=concentration.dtype),
            xp.zeros(spatial_shape, dtype=concentration.dtype),
            xp.zeros(spatial_shape, dtype=concentration.dtype),
        )

    # Zero-pad the concentration curves to the circulant length 2n
    padded = xp.zeros((masked_conc.shape[0], L), dtype=concentration.dtype)
    padded[:, :n] = masked_conc

    s_max = S[0]
    s_thresh = threshold * s_max
    S_inv = xp.where(s_thresh < S, 1.0 / S, 0.0)

    UtC = U.T @ padded.T  # (2n, n_masked)
    r_full = (Vh.T @ (S_inv[:, None] * UtC)).T  # (n_masked, 2n)

    # Physical residue is the first n points; drop the wraparound tail
    masked_residue = xp.maximum(r_full[:, :n], 0.0)

    masked_cbf = xp.max(masked_residue, axis=1)
    masked_mtt = xp.where(
        masked_cbf > 0,
        xp.sum(masked_residue, axis=1) * dt / masked_cbf,
        0.0,
    )
    masked_delay = xp.argmax(masked_residue, axis=1).astype(concentration.dtype) * dt

    residue = xp.zeros_like(flat_conc)
    residue[flat_mask] = masked_residue

    cbf_flat = xp.zeros(flat_conc.shape[0], dtype=concentration.dtype)
    cbf_flat[flat_mask] = masked_cbf

    mtt_flat = xp.zeros(flat_conc.shape[0], dtype=concentration.dtype)
    mtt_flat[flat_mask] = masked_mtt

    delay_flat = xp.zeros(flat_conc.shape[0], dtype=concentration.dtype)
    delay_flat[flat_mask] = masked_delay

    return (
        residue.reshape(concentration.shape),
        cbf_flat.reshape(spatial_shape),
        mtt_flat.reshape(spatial_shape),
        delay_flat.reshape(spatial_shape),
    )


def _compute_oscillation_index_batch(
    r: "NDArray[np.floating[Any]]",
    xp: Any,
) -> "NDArray[np.floating[Any]]":
    """Compute oscillation index for a batch of residue functions.

    Parameters
    ----------
    r : NDArray
        Residue functions, shape (n_voxels, n_timepoints).
    xp : module
        Array module (numpy or cupy).

    Returns
    -------
    NDArray
        Oscillation index per voxel, shape (n_voxels,).
    """
    r_max = xp.max(xp.abs(r), axis=1)  # (n_voxels,)
    n_t = r.shape[1]

    # Sum of absolute differences along time axis
    diff_sum = xp.sum(xp.abs(r[:, 1:] - r[:, :-1]), axis=1)  # (n_voxels,)

    oi = xp.where(
        r_max > 1e-10,
        diff_sum / (r_max * n_t),
        0.0,
    )
    return oi


def _build_circulant_matrix_xp(
    aif: "NDArray[np.floating[Any]]",
    n: int,
    xp: Any,
) -> "NDArray[np.floating[Any]]":
    """Build block-circulant convolution matrix (xp-compatible).

    GPU/CPU agnostic implementation.

    Parameters
    ----------
    aif : NDArray
        Arterial input function.
    n : int
        Matrix size.
    xp : module
        Array module (numpy or cupy).

    Returns
    -------
    NDArray
        Block-circulant matrix, shape ``(2n, 2n)``.
    """
    # Pad AIF to double length for the block-circulant matrix
    aif_padded = xp.zeros(2 * n, dtype=aif.dtype)
    aif_padded[:n] = aif

    # Build the full 2n x 2n circulant matrix: A[i,j] = aif_padded[(i-j) mod 2n].
    # The wraparound terms (upper-right block) are what make deconvolution
    # insensitive to bolus arrival delay.
    idx = (xp.arange(2 * n)[:, None] - xp.arange(2 * n)[None, :]) % (2 * n)
    return aif_padded[idx]


def _build_toeplitz_matrix_xp(
    aif: "NDArray[np.floating[Any]]",
    n: int,
    xp: Any,
) -> "NDArray[np.floating[Any]]":
    """Build lower-triangular Toeplitz convolution matrix (xp-compatible).

    GPU/CPU agnostic implementation.

    Parameters
    ----------
    aif : NDArray
        Arterial input function.
    n : int
        Matrix size.
    xp : module
        Array module (numpy or cupy).

    Returns
    -------
    NDArray
        Lower-triangular Toeplitz matrix.
    """
    # A[i,j] = aif[i-j] for j <= i, else 0 (lower-triangular Toeplitz)
    row_idx = xp.arange(n)[:, None]
    col_idx = xp.arange(n)[None, :]
    diff = row_idx - col_idx
    A = xp.where(diff >= 0, aif[diff], 0.0)
    return A


# --- Strategy classes wrapping existing functions ---

from osipy.dsc.deconvolution.base import BaseDeconvolver
from osipy.dsc.deconvolution.registry import register_deconvolver


@register_deconvolver("sSVD")
class StandardSVDDeconvolver(BaseDeconvolver):
    """Standard SVD (sSVD) deconvolution strategy.

    Uses a lower-triangular Toeplitz matrix for causal convolution.
    Sensitive to bolus arrival time delays.
    """

    @property
    def name(self) -> str:
        """Return human-readable component name."""
        return "Standard SVD"

    @property
    def reference(self) -> str:
        """Return primary literature citation."""
        return "Ostergaard L et al. MRM 1996;36(5):715-725."

    def deconvolve(self, concentration, aif, time, mask=None, **kwargs):
        """Perform sSVD deconvolution to recover the residue function."""
        params = kwargs.get("params") or SVDDeconvolutionParams(method="sSVD")
        return _deconvolve_sSVD(concentration, aif, time, mask, params)


@register_deconvolver("cSVD")
class CircularSVDDeconvolver(BaseDeconvolver):
    """Circular SVD (cSVD) deconvolution strategy.

    Uses a block-circulant matrix to make deconvolution insensitive
    to bolus arrival time delays (OSIPI: Q.PH1.007).
    """

    @property
    def name(self) -> str:
        """Return human-readable component name."""
        return "Circular SVD"

    @property
    def reference(self) -> str:
        """Return primary literature citation."""
        return "Wu O et al. MRM 2003;50(1):164-174."

    def deconvolve(self, concentration, aif, time, mask=None, **kwargs):
        """Perform cSVD deconvolution to recover the residue function."""
        params = kwargs.get("params") or SVDDeconvolutionParams(method="cSVD")
        return deconvolve_cSVD(concentration, aif, time, mask, params)


@register_deconvolver("oSVD")
class OscillationSVDDeconvolver(BaseDeconvolver):
    """Oscillation-index SVD (oSVD) deconvolution strategy.

    Automatically selects the SVD truncation threshold to minimize
    oscillations in the residue function while preserving CBF
    (OSIPI: Q.PH1.003) estimates.
    """

    @property
    def name(self) -> str:
        """Return human-readable component name."""
        return "Oscillation-index SVD"

    @property
    def reference(self) -> str:
        """Return primary literature citation."""
        return "Wu O et al. MRM 2003;50(1):164-174."

    def deconvolve(self, concentration, aif, time, mask=None, **kwargs):
        """Perform oSVD deconvolution to recover the residue function."""
        params = kwargs.get("params") or SVDDeconvolutionParams(method="oSVD")
        return deconvolve_oSVD(concentration, aif, time, mask, params)
