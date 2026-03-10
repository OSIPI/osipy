"""Levenberg-Marquardt model fitting for osipy.

This module provides non-linear least squares optimization using
xp-compatible operations. Works on any ``FittableModel`` — the fitter
never knows about modality-specific context (time, AIF, b-values).

NO scipy dependency - uses custom vectorized Levenberg-Marquardt
implementation that processes all voxels in parallel via batch
operations. Works identically on CPU (NumPy) and GPU (CuPy).

References
----------
Marquardt (1963). An Algorithm for Least-Squares Estimation of
Nonlinear Parameters. SIAM Journal on Applied Mathematics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from osipy.common.backend.array_module import get_array_module
from osipy.common.backend.config import get_backend
from osipy.common.fitting.base import BaseFitter
from osipy.common.fitting.registry import register_fitter

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from osipy.common.models.fittable import FittableModel

logger = logging.getLogger(__name__)


@register_fitter("lm")
class LevenbergMarquardtFitter(BaseFitter):
    """Vectorized Levenberg-Marquardt fitter.

    Processes all voxels simultaneously using batched array operations.
    GPU/CPU is automatic via ``xp = get_array_module()`` — no manual
    ``use_gpu`` flag needed.

    Supports analytical Jacobians via ``FittableModel.compute_jacobian_batch()``.
    Falls back to numerical finite differences when the model returns ``None``.

    Parameters
    ----------
    max_iterations : int
        Maximum number of LM iterations per batch.
    tolerance : float
        Convergence tolerance for relative cost change.
    chunk_size : int | None
        Number of voxels per processing chunk for memory management.
        Defaults to ``get_backend().default_batch_size``.
    r2_threshold : float
        Minimum R-squared value for a valid fit.

    Examples
    --------
    >>> from osipy.common.fitting.least_squares import LevenbergMarquardtFitter
    >>> fitter = LevenbergMarquardtFitter(max_iterations=100)
    >>> result = fitter.fit_image(bound_model, data_4d, mask=brain_mask)
    """

    fitting_method_name = "levenberg_marquardt"

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        chunk_size: int | None = None,
        r2_threshold: float = 0.5,
    ) -> None:
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.chunk_size = (
            chunk_size if chunk_size is not None else get_backend().default_batch_size
        )
        self.r2_threshold = r2_threshold

    # ------------------------------------------------------------------
    # Core batch LM algorithm
    # ------------------------------------------------------------------

    def fit_batch(
        self,
        model: FittableModel,
        observed_batch: NDArray[np.floating[Any]],
        bounds_override: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Fit a batch of voxels using vectorized Levenberg-Marquardt.

        Parameters
        ----------
        model : FittableModel
            Model with independent variables bound.
        observed_batch : NDArray
            Observed data, shape ``(n_observations, n_voxels)``.
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

        _n_obs, n_voxels = observed_batch.shape
        param_names = model.parameters
        n_params = len(param_names)
        bounds = self._merge_bounds(model.get_bounds(), bounds_override)

        # Pre-compute bound arrays for vectorized clipping
        lower_bounds = xp.full((n_params, 1), -xp.inf, dtype=observed_batch.dtype)
        upper_bounds = xp.full((n_params, 1), xp.inf, dtype=observed_batch.dtype)
        for i, name in enumerate(param_names):
            if name in bounds:
                lower_bounds[i, 0] = bounds[name][0]
                upper_bounds[i, 0] = bounds[name][1]

        # Use model's data-dependent initial guesses
        params = model.get_initial_guess_batch(observed_batch, xp)
        params = xp.clip(params, lower_bounds, upper_bounds)

        # Levenberg-Marquardt damping parameter
        lambda_lm = 1e-3 * xp.ones(n_voxels, dtype=observed_batch.dtype)

        converged = xp.zeros(n_voxels, dtype=bool)
        prev_cost = xp.full(n_voxels, xp.inf, dtype=observed_batch.dtype)

        # Vectorized diagonal index for damping
        diag_idx = xp.arange(n_params)

        # Working-set arrays — compacted each iteration as voxels converge.
        # active_idx maps working-set columns back to the full batch.
        active_idx = xp.arange(n_voxels)
        act_obs = observed_batch
        act_params = params.copy()
        act_lambda = lambda_lm.copy()
        act_prev_cost = prev_cost.copy()

        for _iteration in range(self.max_iterations):
            # --- Forward pass (active voxels only) ---
            pred = model.predict_array_batch(act_params, xp)
            residuals = act_obs - pred
            cost = xp.sum(residuals**2, axis=0)

            # --- Convergence check ---
            has_prev = xp.isfinite(act_prev_cost)
            safe_prev = xp.where(has_prev, act_prev_cost, xp.ones_like(cost))
            safe_cost = xp.where(has_prev, cost, xp.zeros_like(cost))
            cost_change = xp.abs(safe_prev - safe_cost) / (xp.abs(safe_prev) + 1e-10)
            newly_converged = cost_change < self.tolerance

            # Integer indices of still-active voxels — 1 sync (index count)
            still_active = xp.nonzero(~newly_converged)[0]

            # All converged? (.shape[0] is host-side metadata, no sync)
            if still_active.shape[0] == 0:
                # Write back final params for all remaining active voxels
                params[:, active_idx] = act_params
                converged[active_idx] = True
                break

            # Compact working set if any voxels converged this iteration
            if still_active.shape[0] < act_params.shape[1]:
                # Write back converged voxels' params to the master array
                params[:, active_idx] = act_params
                converged[active_idx] = converged[active_idx] | newly_converged

                # Compact all working-set arrays via integer indexing (no sync)
                active_idx = active_idx[still_active]
                act_obs = act_obs[:, still_active]
                act_params = act_params[:, still_active]
                act_lambda = act_lambda[still_active]
                act_prev_cost = act_prev_cost[still_active]

                # Also compact already-computed arrays for this iteration
                pred = pred[:, still_active]
                residuals = residuals[:, still_active]
                cost = cost[still_active]

            # --- Jacobian (active voxels only) ---
            jacobian = model.compute_jacobian_batch(act_params, pred, xp)
            if jacobian is None:
                jacobian = self._compute_jacobian_numerical(model, act_params, pred, xp)

            # --- LM update ---
            jtr = xp.einsum("ptn,tn->pn", jacobian, residuals)
            jtj = xp.einsum("ptn,qtn->pqn", jacobian, jacobian)

            # Vectorized damping
            jtj[diag_idx, diag_idx, :] += act_lambda * (
                jtj[diag_idx, diag_idx, :] + 1e-10
            )

            # Solve — all voxels in working set are active, pass all-False mask
            no_conv = xp.zeros(act_params.shape[1], dtype=bool)
            delta_params = self._batch_solve(jtj, jtr, no_conv, xp)

            # Bounds clipping
            new_params = xp.clip(act_params + delta_params, lower_bounds, upper_bounds)

            # Cost check for acceptance
            pred_new = model.predict_array_batch(new_params, xp)
            new_cost = xp.sum((act_obs - pred_new) ** 2, axis=0)

            # Accept/reject per voxel
            improved = new_cost < cost
            act_params = xp.where(improved, new_params, act_params)
            act_prev_cost = xp.where(improved, cost, act_prev_cost)
            act_lambda = xp.where(improved, act_lambda * 0.5, act_lambda * 2.0)
            act_lambda = xp.clip(act_lambda, 1e-10, 1e10)

        # Write back any remaining active (unconverged) voxels after loop exits
        if active_idx.shape[0] > 0:
            params[:, active_idx] = act_params

        # Compute final R-squared values
        pred_final = model.predict_array_batch(params, xp)
        ss_res = xp.sum((observed_batch - pred_final) ** 2, axis=0)
        ss_tot = xp.sum((observed_batch - xp.mean(observed_batch, axis=0)) ** 2, axis=0)
        if hasattr(xp, "errstate"):
            with xp.errstate(divide="ignore", invalid="ignore"):
                r2 = xp.where(ss_tot > 1e-10, 1.0 - ss_res / ss_tot, 0.0)
        else:
            r2 = xp.where(ss_tot > 1e-10, 1.0 - ss_res / ss_tot, 0.0)

        return params, r2, converged

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def _compute_jacobian_numerical(
        self,
        model: FittableModel,
        params: NDArray[Any],
        pred: NDArray[Any],
        xp: Any,
    ) -> NDArray[Any]:
        """Compute Jacobian via numerical finite differences.

        Parameters
        ----------
        model : FittableModel
            Model.
        params : NDArray
            Current parameters, shape ``(n_params, n_voxels)``.
        pred : NDArray
            Current predictions, shape ``(n_obs, n_voxels)``.
        xp : module
            Array module.

        Returns
        -------
        NDArray
            Jacobian, shape ``(n_params, n_obs, n_voxels)``.
        """
        n_params = params.shape[0]
        n_obs, n_voxels = pred.shape

        jacobian = xp.zeros((n_params, n_obs, n_voxels), dtype=params.dtype)

        eps = 1e-7
        for p in range(n_params):
            params_plus = params.copy()
            params_plus[p, :] += eps

            pred_plus = model.predict_array_batch(params_plus, xp)
            jacobian[p, :, :] = (pred_plus - pred) / eps

        return jacobian

    def _batch_solve(
        self,
        jtj: NDArray[Any],
        jtr: NDArray[Any],
        converged: NDArray[Any],
        xp: Any,
    ) -> NDArray[Any]:
        """Solve JtJ @ delta = Jtr for all voxels.

        Uses analytical matrix inversion for small matrices (2x2, 3x3)
        which is more efficient than batched solve on GPU.

        Parameters
        ----------
        jtj : NDArray
            J^T @ J matrices, shape ``(n_params, n_params, n_voxels)``.
        jtr : NDArray
            J^T @ residuals, shape ``(n_params, n_voxels)``.
        converged : NDArray
            Boolean mask of converged voxels.
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        NDArray
            Parameter updates, shape ``(n_params, n_voxels)``.
        """
        n_params = jtj.shape[0]
        delta = xp.zeros_like(jtr)

        if n_params == 2:
            # Analytical 2x2 inverse
            a = jtj[0, 0, :]
            b = jtj[0, 1, :]
            c = jtj[1, 0, :]
            d = jtj[1, 1, :]

            det = a * d - b * c
            det_safe = xp.where(xp.abs(det) > 1e-15, det, 1e-15)

            delta[0, :] = (d * jtr[0, :] - b * jtr[1, :]) / det_safe
            delta[1, :] = (-c * jtr[0, :] + a * jtr[1, :]) / det_safe
            delta[:, converged] = 0.0

        elif n_params == 3:
            # Analytical 3x3 inverse using cofactor expansion
            a = jtj[0, 0, :]
            b = jtj[0, 1, :]
            c = jtj[0, 2, :]
            d = jtj[1, 0, :]
            e = jtj[1, 1, :]
            f = jtj[1, 2, :]
            g = jtj[2, 0, :]
            h = jtj[2, 1, :]
            i = jtj[2, 2, :]

            det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
            det_safe = xp.where(xp.abs(det) > 1e-15, det, 1e-15)

            # Cofactor matrix (transpose = adjugate)
            A = e * i - f * h
            B = -(d * i - f * g)
            C = d * h - e * g
            D = -(b * i - c * h)
            E = a * i - c * g
            F = -(a * h - b * g)
            G = b * f - c * e
            H = -(a * f - c * d)
            II = a * e - b * d

            delta[0, :] = (A * jtr[0, :] + D * jtr[1, :] + G * jtr[2, :]) / det_safe
            delta[1, :] = (B * jtr[0, :] + E * jtr[1, :] + H * jtr[2, :]) / det_safe
            delta[2, :] = (C * jtr[0, :] + F * jtr[1, :] + II * jtr[2, :]) / det_safe
            delta[:, converged] = 0.0

        else:
            # Vectorized batched solve for 4+ parameter models.
            # Transpose from (n_params, n_params, n_voxels) to
            # (n_voxels, n_params, n_params) for batched linalg.solve.
            active = ~converged
            if xp.any(active):
                jtj_act = jtj[:, :, active].transpose((2, 0, 1))
                # (n_active, n_params, 1) — batched solve requires 3-D RHS
                jtr_act = jtr[:, active].T[:, :, None]

                # Add small regularization to diagonal for numerical
                # stability (handles near-singular matrices).
                reg = 1e-12 * xp.eye(n_params, dtype=jtj.dtype)
                jtj_act = jtj_act + reg

                try:
                    delta_act = xp.linalg.solve(jtj_act, jtr_act)
                    delta[:, active] = delta_act[:, :, 0].T
                except xp.linalg.LinAlgError:
                    # If batched solve fails, increase regularization
                    # and retry once.
                    reg_large = 1e-6 * xp.eye(n_params, dtype=jtj.dtype)
                    jtj_act = jtj_act + reg_large
                    try:
                        delta_act = xp.linalg.solve(jtj_act, jtr_act)
                        delta[:, active] = delta_act[:, :, 0].T
                    except xp.linalg.LinAlgError:
                        pass  # delta stays zero for all active voxels

        return delta
