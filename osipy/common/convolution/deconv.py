"""Matrix-based deconvolution with regularization.

This module provides deconvolution operations using convolution matrix
construction and regularized pseudo-inverse. Supports both TSVD
(Truncated SVD) and Tikhonov regularization methods.

Also includes batch SVD deconvolution functions for efficient
multi-voxel processing.

References
----------
    - dcmri (https://github.com/dcmri/dcmri) utils.deconv()
    - Wu et al. (2003). Tracer arrival timing-insensitive technique.
      Magn Reson Med. 50:164-174.
    - Hansen PC (1998). Rank-Deficient and Discrete Ill-Posed Problems.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from osipy.common.backend import get_array_module
from osipy.common.convolution.matrix import circulant_convmat, convmat, invconvmat
from osipy.common.exceptions import DataValidationError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def deconv(
    c: NDArray[np.floating],
    aif: NDArray[np.floating],
    t: NDArray[np.floating],
    *,
    method: Literal["tsvd", "tikhonov"] = "tsvd",
    tol: float = 0.1,
    lambda_reg: float | None = None,
    circulant: bool = False,
) -> NDArray[np.floating]:
    """Deconvolve a tissue concentration curve by an arterial input function.

    Computes the residue function R(t) such that:
        C(t) = AIF(t) * R(t)

    where * denotes convolution. This is the inverse problem of
    pharmacokinetic modeling.

    Parameters
    ----------
    c : ndarray
        Tissue concentration curve. Shape: (n_times,).
    aif : ndarray
        Arterial input function. Shape: (n_times,).
    t : ndarray
        Time points in seconds. Shape: (n_times,). Must be monotonically
        increasing.
    method : {"tsvd", "tikhonov"}, default "tsvd"
        Regularization method:
        - "tsvd": Truncated SVD. Robust and widely used in DSC-MRI.
        - "tikhonov": Tikhonov regularization. Smoother but may over-smooth.
    tol : float, default 0.1
        Regularization tolerance. For TSVD, singular values below
        tol * max(s) are truncated. For Tikhonov, controls the
        regularization strength if lambda_reg is not provided.
    lambda_reg : float, optional
        Tikhonov regularization parameter. Only used if method="tikhonov".
        If not provided, computed as tol * max(singular_values).
    circulant : bool, default False
        If True, use block-circulant matrix for delay-insensitive
        deconvolution (cSVD/oSVD approach). This makes the result
        insensitive to timing differences between AIF and tissue curves.

    Returns
    -------
    ndarray
        Residue function R(t). Shape: (n_times,).
        The residue function satisfies R(0) = 1 for a single-compartment
        model with no delay.

    Notes
    -----
    The deconvolution problem is ill-posed due to noise amplification.
    Regularization is essential for stable solutions.

    **TSVD (Truncated SVD)**:
    - Sets small singular values to zero
    - Simple and robust
    - May introduce ringing artifacts (Gibbs phenomenon)

    **Tikhonov regularization**:
    - Damps small singular values smoothly
    - Produces smoother solutions
    - May over-smooth genuine features

    **Circulant matrix (cSVD/oSVD)**:
    - Makes deconvolution insensitive to AIF timing
    - Standard approach in DSC-MRI for CBF estimation
    - Requires uniform time sampling

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.convolution import deconv
    >>> t = np.linspace(0, 60, 61)
    >>> aif = np.exp(-t / 10) * (1 - np.exp(-t / 2))  # Gamma variate AIF
    >>> # Create synthetic tissue curve with known residue function
    >>> from osipy.common.convolution import conv
    >>> true_irf = np.exp(-t / 20)  # Exponential residue
    >>> c = conv(aif, true_irf, t)
    >>> # Deconvolve to recover residue function
    >>> recovered_irf = deconv(c, aif, t, method="tsvd", tol=0.1)

    References
    ----------
    .. [1] dcmri library: https://github.com/dcmri/dcmri
    .. [2] Wu et al. (2003). Magn Reson Med. 50:164-174.
    .. [3] Ostergaard et al. (1996). Magn Reson Med. 36:715-725.
    """
    xp = get_array_module(c)

    c = xp.asarray(c, dtype=xp.float64)
    aif = xp.asarray(aif, dtype=xp.float64)
    t = xp.asarray(t, dtype=xp.float64)

    n = len(t)
    if len(c) != n or len(aif) != n:
        raise DataValidationError("c, aif, and t must have the same length")

    if n == 0:
        return xp.array([], dtype=xp.float64)

    if n == 1:
        return xp.array([0.0], dtype=xp.float64)

    # Build convolution matrix
    A = circulant_convmat(aif, t) if circulant else convmat(aif, t)

    # Compute regularized pseudo-inverse
    A_inv = invconvmat(A, method=method, tol=tol, lambda_reg=lambda_reg)

    # Deconvolution: R = A_inv @ C
    R = A_inv @ c

    return R


def deconv_osvd(
    c: NDArray[np.floating],
    aif: NDArray[np.floating],
    t: NDArray[np.floating],
    *,
    osc_threshold: float = 0.035,
) -> NDArray[np.floating]:
    """Oscillation-index SVD deconvolution.

    Implements the oSVD algorithm which selects the optimal truncation
    threshold based on minimizing oscillations in the residue function.

    Parameters
    ----------
    c : ndarray
        Tissue concentration curve. Shape: (n_times,).
    aif : ndarray
        Arterial input function. Shape: (n_times,).
    t : ndarray
        Time points in seconds. Shape: (n_times,).
    osc_threshold : float, default 0.035
        Target oscillation index threshold. The algorithm finds
        the smallest truncation that keeps oscillations below this.

    Returns
    -------
    ndarray
        Residue function with minimized oscillations. Shape: (n_times,).

    Notes
    -----
    The oscillation index (OI) is defined as:
        OI = (1/N) * sum(|R[i] - R[i-1]|) / max(R)

    oSVD iteratively increases the truncation threshold until OI < osc_threshold.

    References
    ----------
    .. [1] Wu et al. (2003). Magn Reson Med. 50:164-174.
    """
    xp = get_array_module(c)

    # Try different truncation thresholds
    tol_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

    best_R = None
    best_oi = float("inf")

    for tol_try in tol_values:
        R = deconv(c, aif, t, method="tsvd", tol=tol_try, circulant=True)

        # Compute oscillation index
        oi = _oscillation_index(R)

        if oi < osc_threshold:
            return R

        if oi < best_oi:
            best_oi = oi
            best_R = R

    # Return best result if threshold not achieved
    return best_R if best_R is not None else xp.zeros(len(t), dtype=xp.float64)


def _oscillation_index(R: NDArray[np.floating]) -> float:
    """Compute oscillation index of a residue function."""
    xp = get_array_module(R)

    if len(R) < 2:
        return 0.0

    R_max = float(xp.max(xp.abs(R)))
    if R_max < 1e-10:
        return 0.0

    # Sum of absolute differences normalized by max
    diff_sum = float(xp.sum(xp.abs(xp.diff(R))))
    oi = diff_sum / (len(R) * R_max)

    return oi


def deconvolve_svd(
    ct: NDArray[Any],
    aif: NDArray[Any],
    dt: float = 1.0,
    threshold: float = 0.1,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Deconvolve tissue curve with AIF using SVD.

    Solves the inverse problem to recover the impulse response function
    from the measured tissue curve and AIF.

    Parameters
    ----------
    ct : NDArray
        Tissue concentration curve, shape (n_timepoints,) or (n_timepoints, n_voxels).
    aif : NDArray
        Arterial input function, shape (n_timepoints,).
    dt : float, optional
        Time step in seconds. Default is 1.0.
    threshold : float, optional
        SVD truncation threshold as fraction of maximum singular value.
        Default is 0.1 (10%).

    Returns
    -------
    irf : NDArray
        Recovered impulse response function.
    residue : NDArray
        Residue function (cumulative integral of IRF).
    """
    xp = get_array_module(ct, aif)

    n_time = ct.shape[0]

    conv_matrix = xp.zeros((n_time, n_time), dtype=ct.dtype)
    for i in range(n_time):
        conv_matrix[i, : i + 1] = aif[i::-1]
    conv_matrix *= dt

    u, s, vh = xp.linalg.svd(conv_matrix, full_matrices=False)

    s_max = xp.max(s)
    s_thresh = s_max * threshold
    s_inv = xp.where(s > s_thresh, 1.0 / s, 0.0)

    conv_inv = vh.T @ xp.diag(s_inv) @ u.T

    irf = conv_inv @ ct

    residue = xp.cumsum(irf, axis=0) * dt

    return irf, residue


def deconvolve_svd_batch(
    ct: NDArray[Any],
    aif: NDArray[Any],
    dt: float = 1.0,
    threshold: float = 0.1,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Deconvolve multiple tissue curves with a single AIF.

    Optimized batch deconvolution: the convolution matrix SVD is
    computed only once and applied to all voxels.

    Parameters
    ----------
    ct : NDArray
        Tissue concentration curves, shape (n_timepoints, n_voxels).
    aif : NDArray
        Arterial input function, shape (n_timepoints,).
    dt : float, optional
        Time step in seconds. Default is 1.0.
    threshold : float, optional
        SVD truncation threshold. Default is 0.1.

    Returns
    -------
    irf : NDArray
        Recovered impulse response functions, shape (n_timepoints, n_voxels).
    residue : NDArray
        Residue functions, shape (n_timepoints, n_voxels).
    """
    xp = get_array_module(ct, aif)

    n_time = ct.shape[0]

    conv_matrix = xp.zeros((n_time, n_time), dtype=ct.dtype)
    for i in range(n_time):
        conv_matrix[i, : i + 1] = aif[i::-1]
    conv_matrix *= dt

    u, s, vh = xp.linalg.svd(conv_matrix, full_matrices=False)

    s_max = xp.max(s)
    s_thresh = s_max * threshold
    s_inv = xp.where(s > s_thresh, 1.0 / s, 0.0)

    conv_inv = vh.T @ xp.diag(s_inv) @ u.T

    irf = conv_inv @ ct
    residue = xp.cumsum(irf, axis=0) * dt

    return irf, residue
