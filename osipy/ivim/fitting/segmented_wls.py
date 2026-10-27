"""Segmented IVIM fitting with iterative weighted least squares.

Vectorized implementation of the OSIPI TF2.4 segmented reference method
(``segmented_IVIMfit.py`` in the TF2.4 IVIM-MRI CodeCollection):

1. D and the diffusion intercept are estimated from the signal at
   ``b >= b_threshold`` with an iterative weighted linear least squares
   fit of ``log(S)``. Weights are ``S_pred**2``, initialised from an
   ordinary linear fit.
2. f follows from the b=0 signal and the intercept:
   ``f = (S(0) - intercept) / S(0)``.
3. D* is fitted on all b-values with D and f held fixed.

Works for any array module (NumPy/CuPy); no SciPy or statsmodels needed.

References
----------
.. [1] Veraart J, Sijbers J, Sunaert S, Leemans A, Jeurissen B. Weighted
       linear least squares estimation of diffusion MRI parameters:
       strengths, limitations, and pitfalls. NeuroImage 2013;81:335-346.
       doi:10.1016/j.neuroimage.2013.05.028
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from osipy.common.backend.array_module import get_array_module
from osipy.common.fitting.base import BaseFitter
from osipy.common.fitting.least_squares import LevenbergMarquardtFitter

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from osipy.common.models.fittable import FittableModel

# Smallest signal allowed before taking the log (matches the reference)
_MIN_SIGNAL = 1e-6


class SegmentedWLSIVIMFitter(BaseFitter):
    """Segmented IVIM fitter using iterative WLS for D (TF2.4 reference).

    Parameters
    ----------
    wls_iterations : int
        Number of reweighting iterations for the D fit.
    dstar_iterations : int
        Maximum Levenberg-Marquardt iterations for the D* fit.
    chunk_size : int
        Number of voxels per batch.
    r2_threshold : float
        Minimum R-squared for a voxel to be flagged as good.
    """

    fitting_method_name = "segmented_wls"

    def __init__(
        self,
        wls_iterations: int = 50,
        dstar_iterations: int = 50,
        chunk_size: int = 10000,
        r2_threshold: float = 0.5,
    ) -> None:
        self.wls_iterations = wls_iterations
        self.dstar_iterations = dstar_iterations
        self.chunk_size = chunk_size
        self.r2_threshold = r2_threshold

    def fit_batch(
        self,
        model: FittableModel,
        observed_batch: NDArray[np.floating[Any]],
        bounds_override: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Fit a batch of voxels.

        Parameters
        ----------
        model : FittableModel
            ``BoundIVIMModel`` (biexponential, free D, f, D*, optional S0).
        observed_batch : NDArray
            Observed data, shape ``(n_b, n_voxels)``.
        bounds_override : dict, optional
            Per-parameter bound overrides.

        Returns
        -------
        params : NDArray
            Fitted parameters, shape ``(n_params, n_voxels)``.
        r2 : NDArray
            R-squared values, shape ``(n_voxels,)``.
        converged : NDArray
            Convergence flags, shape ``(n_voxels,)``.
        """
        xp = get_array_module(observed_batch)
        dtype = observed_batch.dtype

        names = list(model.parameters)
        bounds = self._merge_bounds(model.get_bounds(), bounds_override)
        d_lo, d_hi = bounds["D"]
        f_lo, f_hi = bounds["f"]
        ds_lo, ds_hi = bounds["D*"]

        b = xp.asarray(model._b_values, dtype=dtype)
        b_cutoff = model._b_threshold
        high = b >= b_cutoff
        if int(xp.sum(high)) < 2:
            msg = f"Segmented WLS needs at least 2 b-values >= {b_cutoff} to estimate D"
            raise ValueError(msg)

        b0_idx = int(xp.argmin(b))
        s0 = observed_batch[b0_idx, :]

        # --- Step 1: iterative WLS for D and the b=0 intercept ---
        log_s = xp.log(xp.clip(observed_batch[high, :], _MIN_SIGNAL, None))
        d, intercept = self._iterative_wls(b[high], log_s, xp)
        d = xp.clip(d, d_lo, d_hi)

        # --- Step 2: perfusion fraction from the b=0 deficit ---
        safe_s0 = xp.where(xp.abs(s0) > 1e-10, s0, 1.0)
        f = xp.where(xp.abs(s0) > 1e-10, (s0 - intercept) / safe_s0, 0.0)
        f = xp.clip(f, f_lo, f_hi)

        # --- Step 3: D* with D and f fixed ---
        d_star = self._fit_d_star(b, observed_batch, s0, d, f, (ds_lo, ds_hi), xp)

        # S0 is fixed to the measured b=0 signal as in the reference
        # (S0 is not refitted; the reference works on normalised data).
        values = {"S0": s0, "D": d, "D*": d_star, "f": f}
        params = xp.stack([values[n] for n in names]).astype(dtype)

        pred = model.predict_array_batch(params, xp)
        ss_res = xp.sum((observed_batch - pred) ** 2, axis=0)
        ss_tot = xp.sum((observed_batch - xp.mean(observed_batch, axis=0)) ** 2, axis=0)
        safe_tot = xp.where(ss_tot > 1e-10, ss_tot, 1.0)
        r2 = xp.where(ss_tot > 1e-10, 1.0 - ss_res / safe_tot, 0.0)

        converged = xp.isfinite(params).all(axis=0)
        return params, r2, converged

    def _iterative_wls(
        self,
        b_high: NDArray[np.floating[Any]],
        log_s: NDArray[np.floating[Any]],
        xp: Any,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Fit ``log S = ln(S0') - b*D`` with iteratively reweighted WLS.

        Parameters
        ----------
        b_high : NDArray
            b-values used for D, shape ``(n_high,)``.
        log_s : NDArray
            Log signal, shape ``(n_high, n_voxels)``.
        xp : module
            Array module.

        Returns
        -------
        d, intercept : NDArray
            D and the (linear-scale) b=0 intercept, each ``(n_voxels,)``.
        """
        x = b_high[:, xp.newaxis]
        w = xp.ones_like(log_s)  # first pass is the ordinary LLS fit
        for _ in range(self.wls_iterations + 1):
            sw = xp.sum(w, axis=0)
            swx = xp.sum(w * x, axis=0)
            swxx = xp.sum(w * x * x, axis=0)
            swy = xp.sum(w * log_s, axis=0)
            swxy = xp.sum(w * x * log_s, axis=0)
            det = sw * swxx - swx**2
            safe_det = xp.where(xp.abs(det) > 1e-30, det, 1.0)
            # log S = c - D*b  ->  slope = -D
            slope = (sw * swxy - swx * swy) / safe_det
            c = (swxx * swy - swx * swxy) / safe_det
            # Weights for the next pass: predicted signal squared
            w = xp.exp(2.0 * (c[xp.newaxis, :] + slope[xp.newaxis, :] * x))
        return -slope, xp.exp(c)

    def _fit_d_star(
        self,
        b: NDArray[np.floating[Any]],
        signal: NDArray[np.floating[Any]],
        s0: NDArray[np.floating[Any]],
        d: NDArray[np.floating[Any]],
        f: NDArray[np.floating[Any]],
        bounds: tuple[float, float],
        xp: Any,
    ) -> NDArray[Any]:
        """Fit D* per voxel with D and f fixed, using the shared LM fitter.

        Parameters
        ----------
        b : NDArray
            All b-values, shape ``(n_b,)``.
        signal : NDArray
            Observed data, shape ``(n_b, n_voxels)``.
        s0 : NDArray
            Measured b=0 signal, shape ``(n_voxels,)``.
        d, f : NDArray
            Fixed D and f, shape ``(n_voxels,)``.
        bounds : tuple[float, float]
            Lower/upper bound for D*.
        xp : module
            Array module.

        Returns
        -------
        NDArray
            D*, shape ``(n_voxels,)``.
        """
        model = _DStarModel(b, s0, d, f, bounds)
        fitter = LevenbergMarquardtFitter(max_iterations=self.dstar_iterations)
        params, _, _ = fitter.fit_batch(model, signal)
        return params[0]


class _DStarModel:
    """Single-parameter model: D* with per-voxel fixed D, f and S0.

    Implements the ``FittableModel`` interface for one batch of voxels so
    the shared LM fitter can be reused for the D* step.
    """

    name = "ivim_dstar"
    parameters = ["D*"]  # noqa: RUF012
    parameter_units = {"D*": "mm^2/s"}  # noqa: RUF012
    reference = "OSIPI TF2.4 segmented IVIM"

    def __init__(
        self,
        b: NDArray[np.floating[Any]],
        s0: NDArray[np.floating[Any]],
        d: NDArray[np.floating[Any]],
        f: NDArray[np.floating[Any]],
        bounds: tuple[float, float],
    ) -> None:
        self._b = b
        self._s0 = s0
        self._d = d
        self._f = f
        self._bounds = bounds

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        return {"D*": self._bounds}

    def get_initial_guess_batch(
        self, observed_batch: NDArray[np.floating[Any]], xp: Any
    ) -> NDArray[np.floating[Any]]:
        return xp.clip(10.0 * self._d, *self._bounds)[xp.newaxis, :]

    def predict_array_batch(
        self, params_batch: NDArray[np.floating[Any]], xp: Any
    ) -> NDArray[np.floating[Any]]:
        b = self._b[:, xp.newaxis]
        s0, d, f = self._s0, self._d, self._f
        return s0 * (
            (1.0 - f) * xp.exp(-b * d) + f * xp.exp(-b * params_batch[0][xp.newaxis])
        )

    def compute_jacobian_batch(
        self,
        params_batch: NDArray[np.floating[Any]],
        predicted: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]]:
        b = self._b[:, xp.newaxis]
        e = xp.exp(-b * params_batch[0][xp.newaxis])
        return (-self._s0 * self._f * b * e)[xp.newaxis]
