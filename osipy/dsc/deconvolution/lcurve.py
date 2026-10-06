"""L-curve choice of the Tikhonov regularization parameter for DSC deconvolution.

The corner is a curvature maximum of (log rho, log eta), using Hansen's closed
form, in which the logarithms cancel analytically. It is taken as the first
peak met from large lambda: at high noise the curve can develop a sharper bend
at small lambda, where the global maximum lands and the solution blows up.

References
----------
.. [1] Hansen PC, O'Leary DP. The use of the L-curve in the regularization of
   discrete ill-posed problems. SIAM J Sci Comput 1993;14(6):1487-1503.
.. [2] Calamante F, Gadian DG, Connelly A. MRM 2003;50(6):1237-1247.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    Arr = NDArray[np.floating[Any]]

DEFAULT_N_LAMBDA = 400
DEFAULT_LAMBDA_MIN = 1e-9
DEFAULT_LAMBDA_MAX = 10.0


def lcurve_curve(
    S: Arr,
    UtC: Arr,
    xp: Any,
    n_lambda: int = DEFAULT_N_LAMBDA,
    lambda_min: float = DEFAULT_LAMBDA_MIN,
    lambda_max: float = DEFAULT_LAMBDA_MAX,
) -> tuple[Arr, Arr, Arr, Arr, Arr]:
    """Evaluate the L-curve over a log grid of lambda.

    Parameters
    ----------
    S : NDArray
        Singular values of the AIF matrix, shape ``(n_sv,)``.
    UtC : NDArray
        ``U.T @ C``, shape ``(n_sv, n_voxels)``.
    xp : module
        Array module (``numpy`` or ``cupy``).
    n_lambda, lambda_min, lambda_max : int, float, float
        Grid size and bounds relative to ``max(S)``.

    Returns
    -------
    lam : NDArray
        Grid, shape ``(n_lambda,)``.
    rho, eta, deta, g : NDArray
        Squared residual norm, squared solution norm, d(eta)/d(lambda) and
        negative curvature, each shape ``(n_lambda, n_voxels)``.
    """
    s_max = float(xp.max(S))
    k = xp.arange(n_lambda, dtype=UtC.dtype)
    lam = s_max * lambda_min * (lambda_max / lambda_min) ** (k / (n_lambda - 1))

    u = lam[:, None]
    d = S[None, :] ** 2 + u**2
    b2 = UtC**2
    rho = ((u**2 / d) ** 2) @ b2
    eta = ((S[None, :] / d) ** 2) @ b2
    deta = (-4 * u * S[None, :] ** 2 / d**3) @ b2

    ok = deta != 0  # false only for all-zero voxels
    num = u**2 * deta * rho + 2 * u * eta * rho + u**4 * eta * deta
    den = (u**4 * eta**2 + rho**2) ** 1.5
    g = xp.where(
        ok, 2 * eta * rho / xp.where(ok, deta, 1) * num / xp.where(ok, den, 1), 0.0
    )
    return lam, rho, eta, deta, g


def _corner_index(g: Arr, xp: Any) -> NDArray[np.integer[Any]]:
    """Last strict local minimum of ``g``, else ``argmin(g)``."""
    is_min = (g[1:-1] < g[:-2]) & (g[1:-1] < g[2:])
    k = is_min.shape[0] - xp.argmax(is_min[::-1], axis=0)
    return xp.where(xp.any(is_min, axis=0), k, xp.argmin(g, axis=0))


def select_lambda(
    S: Arr,
    UtC: Arr,
    xp: Any,
    n_lambda: int = DEFAULT_N_LAMBDA,
    lambda_min: float = DEFAULT_LAMBDA_MIN,
    lambda_max: float = DEFAULT_LAMBDA_MAX,
) -> Arr:
    """Per-voxel lambda at the L-curve corner, shape ``(n_voxels,)``.

    ``g`` is the negative curvature, so the corner is a minimum of ``g``.
    Parameters are as for :func:`lcurve_curve`.
    """
    lam, *_, g = lcurve_curve(S, UtC, xp, n_lambda, lambda_min, lambda_max)
    return lam[_corner_index(g, xp)]


def tikhonov_filter_factors(S: Arr, lambdas: Arr, xp: Any) -> Arr:
    """Filter factors ``s / (s**2 + lam**2)``, shape ``(n_sv, n_voxels)``."""
    return S[:, None] / (S[:, None] ** 2 + lambdas[None, :] ** 2)


def solve_tikhonov_lcurve(
    U: Arr,
    S: Arr,
    Vh: Arr,
    concentration: Arr,
    xp: Any,
    n_lambda: int = DEFAULT_N_LAMBDA,
    lambda_min: float = DEFAULT_LAMBDA_MIN,
    lambda_max: float = DEFAULT_LAMBDA_MAX,
) -> tuple[Arr, Arr]:
    """Tikhonov deconvolution with lambda chosen per voxel by the L-curve.

    Parameters
    ----------
    U, S, Vh : NDArray
        SVD of the AIF convolution matrix.
    concentration : NDArray
        Tissue curves, shape ``(n_timepoints, n_voxels)``.
    xp : module
        Array module (``numpy`` or ``cupy``).
    n_lambda, lambda_min, lambda_max : int, float, float
        As for :func:`lcurve_curve`.

    Returns
    -------
    residue : NDArray
        Shape ``(Vh.shape[1], n_voxels)``.
    lambdas : NDArray
        Shape ``(n_voxels,)``.
    """
    UtC = U.T @ concentration
    lambdas = select_lambda(S, UtC, xp, n_lambda, lambda_min, lambda_max)
    return Vh.T @ (tikhonov_filter_factors(S, lambdas, xp) * UtC), lambdas
