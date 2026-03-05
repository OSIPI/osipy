"""Bayesian model fitting for osipy.

This module provides Bayesian inference for model fitting with
uncertainty estimation via maximum a posteriori (MAP) estimation.

GPU/CPU agnostic using the xp array module pattern.
NO scipy dependency - uses custom bounded optimization implementation.
"""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.backend.array_module import get_array_module, to_numpy
from osipy.common.fitting.base import BaseFitter
from osipy.common.fitting.registry import register_fitter

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from osipy.common.models.fittable import FittableModel

logger = logging.getLogger(__name__)


@register_fitter("bayesian")
class BayesianFitter(BaseFitter):
    """Bayesian fitter with uncertainty estimation.

    Uses maximum a posteriori (MAP) estimation with Laplace
    approximation for posterior uncertainty estimation.

    GPU/CPU is automatic via ``xp = get_array_module()``.

    Parameters
    ----------
    n_samples : int
        Maximum number of MAP iterations.
    prior_std : NDArray
        Per-parameter standard deviations for Gaussian priors,
        shape ``(n_params,)``.
    noise_std : float | None
        Assumed noise standard deviation. If None, estimated from data.
    chunk_size : int | None
        Number of voxels per processing chunk.
    compute_uncertainty : bool
        If True, add ``{param}_std`` ParameterMap objects to the
        ``fit_image()`` result via Laplace approximation at the MAP
        estimate.
    tolerance : float
        Convergence tolerance for relative cost change.
    """

    fitting_method_name = "bayesian"

    def __init__(
        self,
        n_samples: int = 100,
        prior_std: "NDArray[np.floating[Any]] | None" = None,
        noise_std: float | None = None,
        chunk_size: int | None = None,
        compute_uncertainty: bool = False,
        tolerance: float = 1e-6,
    ) -> None:
        from osipy.common.backend.config import get_backend

        self.n_samples = n_samples
        self.tolerance = tolerance
        if prior_std is None:
            prior_std = np.ones(1)
        self.prior_std = np.atleast_1d(np.asarray(prior_std, dtype=np.float64))
        self.noise_std = noise_std
        self.chunk_size = (
            chunk_size if chunk_size is not None else get_backend().default_batch_size
        )
        self.r2_threshold = 0.5
        self.compute_uncertainty = compute_uncertainty

    # ------------------------------------------------------------------
    # Batch fitting (MAP via damped LM with prior)
    # ------------------------------------------------------------------

    def fit_batch(
        self,
        model: "FittableModel",
        observed_batch: "NDArray[np.floating[Any]]",
        bounds_override: dict[str, tuple[float, float]] | None = None,
        initial_params: "NDArray[np.floating[Any]] | None" = None,
    ) -> tuple["NDArray[Any]", "NDArray[Any]", "NDArray[Any]"]:
        """Fit a batch of voxels using vectorized MAP estimation.

        Same structure as LM ``fit_batch()`` but with a prior penalty
        in the cost function.

        Parameters
        ----------
        model : FittableModel
            Model with independent variables bound.
        observed_batch : NDArray
            Observed data, shape ``(n_observations, n_voxels)``.
        bounds_override : dict, optional
            Per-parameter bound overrides.
        initial_params : NDArray, optional
            Initial parameter values, shape ``(n_params, n_voxels)``.
            If provided, used instead of ``model.get_initial_guess_batch()``.
            Also serves as the prior mean.

        Returns
        -------
        params : NDArray
            Fitted parameters, shape ``(n_params, n_voxels)``.
        r2 : NDArray
            R-squared values, shape ``(n_voxels,)``.
        converged : NDArray
            Convergence flags, shape ``(n_voxels,)``.
        """
        from osipy.common.fitting.least_squares import (
            LevenbergMarquardtFitter,
        )

        xp = get_array_module(observed_batch)

        n_obs, n_voxels = observed_batch.shape
        param_names = model.parameters
        n_params = len(param_names)
        bounds = self._merge_bounds(model.get_bounds(), bounds_override)

        # Pre-compute vectorized bound arrays for single-call clipping
        lower_bounds = xp.full((n_params, 1), -xp.inf, dtype=observed_batch.dtype)
        upper_bounds = xp.full((n_params, 1), xp.inf, dtype=observed_batch.dtype)
        for i, name in enumerate(param_names):
            if name in bounds:
                lower_bounds[i, 0] = bounds[name][0]
                upper_bounds[i, 0] = bounds[name][1]

        # Initial guess (also serves as prior mean)
        if initial_params is not None:
            params = xp.asarray(initial_params)
        else:
            params = model.get_initial_guess_batch(observed_batch, xp)
        p0 = params.copy()
        params = xp.clip(params, lower_bounds, upper_bounds)

        # Estimate noise from baseline frames
        n_baseline = min(5, n_obs // 4)
        noise_std = self.noise_std
        if noise_std is None:
            baseline_std = xp.std(observed_batch[:n_baseline, :], axis=0)
            noise_est = xp.mean(baseline_std)
            noise_std = float(to_numpy(noise_est))
            if noise_std < 1e-10:
                noise_std = 0.01 * float(to_numpy(xp.max(xp.abs(observed_batch))))

        # Per-parameter prior weights: (n_params, 1) for broadcasting
        prior_std_arr = xp.asarray(self.prior_std, dtype=observed_batch.dtype)
        if prior_std_arr.shape[0] == 1 and n_params > 1:
            prior_std_arr = xp.broadcast_to(prior_std_arr, (n_params,))
        prior_std_col = prior_std_arr.reshape(n_params, 1)
        prior_weights = (noise_std / prior_std_col) ** 2  # (n_params, 1)

        # Reuse LM infrastructure for Jacobian / solve
        lm = LevenbergMarquardtFitter.__new__(LevenbergMarquardtFitter)

        converged = xp.zeros(n_voxels, dtype=bool)

        # Vectorized diagonal index for damping
        diag_idx = xp.arange(n_params)

        # Working-set arrays — compacted each iteration as voxels converge.
        # active_idx maps working-set columns back to the full batch.
        active_idx = xp.arange(n_voxels)
        act_obs = observed_batch
        act_params = params.copy()
        act_p0 = p0.copy()
        act_lambda = 1e-3 * xp.ones(n_voxels, dtype=observed_batch.dtype)
        act_prev_cost = xp.full(n_voxels, xp.inf, dtype=observed_batch.dtype)

        for _iteration in range(self.n_samples):
            # --- Forward pass (active voxels only) ---
            pred = model.predict_array_batch(act_params, xp)
            residuals = act_obs - pred

            # MAP cost: likelihood + prior (per-parameter weights)
            likelihood_cost = xp.sum(residuals**2, axis=0)
            prior_cost = xp.sum(prior_weights * (act_params - act_p0) ** 2, axis=0)
            cost = likelihood_cost + prior_cost

            # --- Convergence check ---
            has_prev = xp.isfinite(act_prev_cost)
            safe_prev = xp.where(has_prev, act_prev_cost, xp.ones_like(cost))
            safe_cost = xp.where(has_prev, cost, xp.zeros_like(cost))
            cost_change = xp.abs(safe_prev - safe_cost) / (xp.abs(safe_prev) + 1e-10)
            newly_converged = cost_change < self.tolerance

            still_active = xp.nonzero(~newly_converged)[0]

            # All converged? Write back and break.
            if still_active.shape[0] == 0:
                params[:, active_idx] = act_params
                converged[active_idx] = True
                break

            # Compact working set if any voxels converged this iteration
            if still_active.shape[0] < act_params.shape[1]:
                params[:, active_idx] = act_params
                converged[active_idx] = converged[active_idx] | newly_converged

                active_idx = active_idx[still_active]
                act_obs = act_obs[:, still_active]
                act_params = act_params[:, still_active]
                act_p0 = act_p0[:, still_active]
                act_lambda = act_lambda[still_active]
                act_prev_cost = act_prev_cost[still_active]

                pred = pred[:, still_active]
                residuals = residuals[:, still_active]
                cost = cost[still_active]

            # --- Jacobian (active voxels only) ---
            jacobian = model.compute_jacobian_batch(act_params, pred, xp)
            if jacobian is None:
                jacobian = lm._compute_jacobian_numerical(model, act_params, pred, xp)

            # J^T @ residuals with prior gradient
            jtr = xp.einsum("ptn,tn->pn", jacobian, residuals)
            jtr -= prior_weights * (act_params - act_p0)

            # J^T @ J with prior Hessian contribution + LM damping
            jtj = xp.einsum("ptn,qtn->pqn", jacobian, jacobian)
            jtj[diag_idx, diag_idx, :] += prior_weights[:, 0].reshape(n_params, 1)
            jtj[diag_idx, diag_idx, :] += act_lambda * (
                jtj[diag_idx, diag_idx, :] + 1e-10
            )

            # Solve — all voxels in working set are active
            no_conv = xp.zeros(act_params.shape[1], dtype=bool)
            delta_params = lm._batch_solve(jtj, jtr, no_conv, xp)

            # Vectorized bound clipping
            new_params = xp.clip(act_params + delta_params, lower_bounds, upper_bounds)

            # Trial-point cost
            pred_new = model.predict_array_batch(new_params, xp)
            new_likelihood = xp.sum((act_obs - pred_new) ** 2, axis=0)
            new_prior = xp.sum(prior_weights * (new_params - act_p0) ** 2, axis=0)
            new_cost = new_likelihood + new_prior

            # Accept/reject per voxel
            improved = new_cost < cost
            act_params = xp.where(improved, new_params, act_params)
            act_prev_cost = xp.where(improved, cost, act_prev_cost)
            act_lambda = xp.where(improved, act_lambda * 0.5, act_lambda * 2.0)
            act_lambda = xp.clip(act_lambda, 1e-10, 1e10)

        # Write back any remaining active (unconverged) voxels
        if active_idx.shape[0] > 0:
            params[:, active_idx] = act_params

        # Final R-squared (likelihood only, not MAP cost)
        pred_final = model.predict_array_batch(params, xp)
        ss_res = xp.sum((observed_batch - pred_final) ** 2, axis=0)
        ss_tot = xp.sum(
            (observed_batch - xp.mean(observed_batch, axis=0)) ** 2,
            axis=0,
        )
        r2 = xp.where(ss_tot > 1e-10, 1.0 - ss_res / ss_tot, 0.0)

        return params, r2, converged

    # ------------------------------------------------------------------
    # Image-level fitting with optional uncertainty maps
    # ------------------------------------------------------------------

    def fit_image(
        self,
        model: "FittableModel",
        data: "NDArray[np.floating[Any]]",
        mask: "NDArray[np.bool_] | None" = None,
        bounds_override: (dict[str, tuple[float, float]] | None) = None,
        progress_callback: "Callable[[float], None] | None" = None,
    ) -> dict[str, Any]:
        """Fit model to image with optional uncertainty maps.

        Calls ``super().fit_image()`` for MAP fitting, then adds
        ``{param}_std`` ParameterMap objects via Laplace approximation
        when ``compute_uncertainty`` is True.

        Parameters
        ----------
        model : FittableModel
            Model with independent variables bound.
        data : NDArray
            Image data, shape ``(x, y, z, n_observations)``.
        mask : NDArray[np.bool_] | None
            Boolean mask of voxels to fit.
        bounds_override : dict, optional
            Per-parameter bound overrides.
        progress_callback : Callable, optional
            Progress callback.

        Returns
        -------
        dict[str, ParameterMap]
            Parameter maps, plus ``{param}_std`` maps if uncertainty
            estimation is enabled.
        """
        param_maps = super().fit_image(
            model, data, mask, bounds_override, progress_callback
        )

        if self.compute_uncertainty:
            self._compute_uncertainty_maps(model, data, mask, param_maps)

        return param_maps

    def _compute_uncertainty_maps(
        self,
        model: "FittableModel",
        data: "NDArray[np.floating[Any]]",
        mask: "NDArray[np.bool_] | None",
        param_maps: dict[str, Any],
    ) -> None:
        """Add per-parameter std maps via Laplace approximation.

        Computes H = J^T J + prior_weight * I at the MAP estimate,
        inverts per-voxel, and extracts sqrt(diag) for each parameter.
        Results are added in-place as ``{name}_std`` ParameterMap
        entries.
        """
        from osipy.common.fitting.least_squares import (
            LevenbergMarquardtFitter,
        )
        from osipy.common.parameter_map import ParameterMap

        nx, ny, nz, nt = data.shape

        if mask is None:
            mask = np.ones((nx, ny, nz), dtype=bool)

        voxel_indices = np.argwhere(to_numpy(mask))
        n_voxels = len(voxel_indices)

        if n_voxels == 0:
            return

        param_names = model.parameters
        n_params = len(param_names)
        param_units = model.parameter_units

        # Extract fitted parameter values at masked voxels
        fitted = np.zeros((n_params, n_voxels), dtype=np.float64)
        for i, name in enumerate(param_names):
            vol = param_maps[name].values
            fitted[i, :] = vol[
                voxel_indices[:, 0],
                voxel_indices[:, 1],
                voxel_indices[:, 2],
            ]

        # Determine device
        from osipy.common.backend.config import (
            get_backend,
            is_gpu_available,
        )

        use_gpu = is_gpu_available() and not get_backend().force_cpu
        if use_gpu:
            from osipy.common.backend.array_module import to_gpu

            fitted_dev = to_gpu(fitted)
        else:
            fitted_dev = fitted
        xp = get_array_module(fitted_dev)

        if hasattr(model, "ensure_device"):
            model.ensure_device(xp)

        # Compute per-parameter prior weights (same logic as fit_batch)
        observed_masked = data[mask].T  # (nt, n_voxels)
        if use_gpu:
            from osipy.common.backend.array_module import to_gpu

            observed_masked = to_gpu(observed_masked)
        n_baseline = min(5, nt // 4)
        noise_std = self.noise_std
        if noise_std is None:
            baseline_std = xp.std(observed_masked[:n_baseline, :], axis=0)
            noise_est = xp.mean(baseline_std)
            noise_std = float(to_numpy(noise_est))
            if noise_std < 1e-10:
                noise_std = 0.01 * float(to_numpy(xp.max(xp.abs(observed_masked))))
        prior_std_arr = xp.asarray(self.prior_std, dtype=fitted_dev.dtype)
        if prior_std_arr.shape[0] == 1 and n_params > 1:
            prior_std_arr = xp.broadcast_to(prior_std_arr, (n_params,))

        # Compute uncertainty in chunks
        lm = LevenbergMarquardtFitter.__new__(LevenbergMarquardtFitter)
        std_values = np.zeros((n_params, n_voxels), dtype=np.float64)

        chunk_size = self.chunk_size
        for start in range(0, n_voxels, chunk_size):
            end = min(start + chunk_size, n_voxels)
            params_chunk = fitted_dev[:, start:end]

            pred = model.predict_array_batch(params_chunk, xp)

            jacobian = model.compute_jacobian_batch(params_chunk, pred, xp)
            if jacobian is None:
                jacobian = lm._compute_jacobian_numerical(model, params_chunk, pred, xp)

            # H = J^T J + diag(prior_weights)
            # jacobian: (n_params, n_obs, n_chunk)
            jtj = xp.einsum("ptn,qtn->pqn", jacobian, jacobian)
            for p in range(n_params):
                pw = (noise_std / float(to_numpy(prior_std_arr[p]))) ** 2
                jtj[p, p, :] += pw

            # (n_chunk, n_params, n_params) for batched inv
            h_batch = xp.transpose(jtj, (2, 0, 1))

            try:
                h_inv = xp.linalg.inv(h_batch)
            except Exception:
                jitter = 1e-6 * xp.eye(n_params, dtype=h_batch.dtype)
                h_inv = xp.linalg.inv(h_batch + jitter[xp.newaxis, :, :])

            for p in range(n_params):
                diag_vals = xp.maximum(h_inv[:, p, p], 0.0)
                std_values[p, start:end] = to_numpy(xp.sqrt(diag_vals))

        # Create _std ParameterMap objects
        quality_mask_vol = param_maps[param_names[0]].quality_mask
        affine = param_maps[param_names[0]].affine
        ref = model.reference if hasattr(model, "reference") else ""

        for i, name in enumerate(param_names):
            std_vol = np.zeros((nx, ny, nz), dtype=np.float64)
            std_vol[
                voxel_indices[:, 0],
                voxel_indices[:, 1],
                voxel_indices[:, 2],
            ] = std_values[i, :]

            std_name = f"{name}_std"
            qm = quality_mask_vol.copy() if quality_mask_vol is not None else None
            param_maps[std_name] = ParameterMap(
                name=std_name,
                symbol=f"std({name})",
                values=std_vol,
                affine=affine,
                units=param_units.get(name, ""),
                quality_mask=qm,
                model_name=model.name,
                fitting_method=self.fitting_method_name,
                literature_reference=ref,
            )
