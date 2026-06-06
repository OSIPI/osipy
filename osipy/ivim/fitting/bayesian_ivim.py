"""Two-stage Bayesian IVIM fitting with empirical priors.

Implements the Barbieri et al. (MRM 2016) approach:
1. Stage 1: Least-squares fit on all voxels
2. Build empirical per-parameter priors from the population of Stage 1 fits
3. Stage 2: MAP fitting with empirical priors and per-voxel initial guesses

References
----------
.. [1] Barbieri S, Donati OF, Froehlich JM, Thoeny HC. Impact of the
       calculation algorithm on biexponential fitting of diffusion-weighted
       MRI in upper abdominal organs. Magn Reson Med 2016;75(5):2175-2184.
       doi:10.1002/mrm.25765
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.backend.array_module import get_array_module, to_gpu, to_numpy
from osipy.common.backend.config import (
    get_backend,
    get_gpu_batch_size,
    is_gpu_available,
)
from osipy.common.fitting.batch import create_empty_maps, create_parameter_maps
from osipy.common.fitting.bayesian import BayesianFitter
from osipy.common.fitting.least_squares import LevenbergMarquardtFitter

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from osipy.common.models.fittable import FittableModel

logger = logging.getLogger(__name__)

# Parameters best modeled in log-space (lognormal distribution)
_LOG_SPACE_PARAMS = {"D", "D*"}


class TwoStageBayesianIVIMFitter:
    """Two-stage Bayesian IVIM fitter with empirical priors.

    Stage 1 runs a standard Levenberg-Marquardt fit to obtain
    point estimates. Those estimates are used to build per-parameter
    empirical priors (mean + std). Stage 2 runs a Bayesian MAP fit
    using those priors as regularization.

    Parameters
    ----------
    noise_std : float | None
        Assumed noise standard deviation. If None, estimated from data.
    compute_uncertainty : bool
        If True, compute posterior uncertainty maps via Laplace
        approximation after Stage 2.
    prior_scale : float
        Scale factor applied to empirical prior standard deviations.
        Larger values create weaker priors. Default 1.5.
    stage1_max_iter : int
        Maximum iterations for Stage 1 LM fitting.
    """

    fitting_method_name = "two_stage_bayesian"

    def __init__(
        self,
        noise_std: float | None = None,
        compute_uncertainty: bool = False,
        prior_scale: float = 1.5,
        stage1_max_iter: int = 100,
    ) -> None:
        self.noise_std = noise_std
        self.compute_uncertainty = compute_uncertainty
        self.prior_scale = prior_scale
        self.stage1_max_iter = stage1_max_iter
        self.r2_threshold = 0.5

    def fit_image(
        self,
        model: FittableModel,
        data: NDArray[np.floating[Any]],
        mask: NDArray[np.bool_] | None = None,
        bounds_override: dict[str, tuple[float, float]] | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> dict[str, Any]:
        """Fit IVIM model using two-stage Bayesian approach.

        Parameters
        ----------
        model : FittableModel
            Bound IVIM model.
        data : NDArray
            Image data, shape ``(x, y, z, n_observations)``.
        mask : NDArray[np.bool_] | None
            Boolean mask.
        bounds_override : dict, optional
            Per-parameter bound overrides.
        progress_callback : Callable, optional
            Progress callback (0.0 to 1.0).

        Returns
        -------
        dict[str, ParameterMap]
            Parameter maps.
        """
        nx, ny, nz, _nt = data.shape

        if mask is None:
            mask = np.ones((nx, ny, nz), dtype=bool)

        voxel_indices = np.argwhere(to_numpy(mask))
        n_voxels = len(voxel_indices)

        if n_voxels == 0:
            logger.warning("No voxels in mask to fit")
            return create_empty_maps(model, (nx, ny, nz))

        # Determine device
        use_gpu = is_gpu_available() and not get_backend().force_cpu
        logger.info(
            "Two-stage Bayesian: fitting %d voxels (%s)",
            n_voxels,
            "GPU" if use_gpu else "CPU",
        )

        # Extract masked voxels: (nt, n_voxels)
        observed_masked = data[mask].T

        if use_gpu:
            observed_masked = to_gpu(observed_masked)

        xp = get_array_module(observed_masked)

        # Resolve working dtype
        config = get_backend()
        if use_gpu and config.gpu_dtype == "float32":
            working_dtype = xp.float32
        else:
            working_dtype = xp.float64
        observed_masked = observed_masked.astype(working_dtype)

        if hasattr(model, "ensure_device"):
            model.ensure_device(xp)

        n_params = len(model.parameters)

        # Select chunk size
        if use_gpu:
            gpu_threads = get_gpu_batch_size()
            chunk_size = gpu_threads if gpu_threads > 0 else 10000
        else:
            chunk_size = 10000

        # ==================================================================
        # Stage 1: LM fit
        # ==================================================================
        logger.info("Stage 1: Levenberg-Marquardt fit")
        lm_fitter = LevenbergMarquardtFitter()

        stage1_params = xp.zeros((n_params, n_voxels), dtype=working_dtype)
        stage1_r2 = xp.zeros(n_voxels, dtype=working_dtype)
        stage1_converged = xp.zeros(n_voxels, dtype=bool)

        total_chunks = (n_voxels + chunk_size - 1) // chunk_size

        for chunk_idx, start in enumerate(range(0, n_voxels, chunk_size)):
            end = min(start + chunk_size, n_voxels)
            batch = observed_masked[:, start:end]

            bp, br, bc = lm_fitter.fit_batch(model, batch, bounds_override)
            stage1_params[:, start:end] = bp
            stage1_r2[start:end] = br
            stage1_converged[start:end] = bc

            if progress_callback is not None:
                progress_callback(0.4 * (chunk_idx + 1) / total_chunks)

        # ==================================================================
        # Stage 2: Empirical priors + Bayesian MAP
        # ==================================================================
        logger.info("Stage 2: Computing empirical priors")
        prior_stds = self._compute_empirical_priors(
            stage1_params,
            stage1_r2,
            stage1_converged,
            model.parameters,
            model.get_bounds(),
            xp,
        )
        logger.info(
            "Empirical prior stds: %s",
            dict(zip(model.parameters, to_numpy(prior_stds).tolist(), strict=False)),
        )

        bayesian_fitter = BayesianFitter(
            prior_std=to_numpy(prior_stds),
            noise_std=self.noise_std,
            compute_uncertainty=False,  # handled separately below
        )

        logger.info("Stage 2: Bayesian MAP fit")
        stage2_params = xp.zeros((n_params, n_voxels), dtype=working_dtype)
        stage2_r2 = xp.zeros(n_voxels, dtype=working_dtype)
        stage2_converged = xp.zeros(n_voxels, dtype=bool)

        for chunk_idx, start in enumerate(range(0, n_voxels, chunk_size)):
            end = min(start + chunk_size, n_voxels)
            batch = observed_masked[:, start:end]

            bp, br, bc = bayesian_fitter.fit_batch(
                model,
                batch,
                bounds_override,
                initial_params=stage1_params[:, start:end],
            )
            stage2_params[:, start:end] = bp
            stage2_r2[start:end] = br
            stage2_converged[start:end] = bc

            if progress_callback is not None:
                progress_callback(0.4 + 0.55 * (chunk_idx + 1) / total_chunks)

        # ==================================================================
        # Assemble parameter maps
        # ==================================================================
        fitted_np = to_numpy(stage2_params)
        r2_np = to_numpy(stage2_r2)
        converged_np = to_numpy(stage2_converged)

        param_maps = create_parameter_maps(
            model,
            fitted_np,
            r2_np,
            converged_np,
            voxel_indices,
            (nx, ny, nz),
            r2_threshold=self.r2_threshold,
            fitting_method=self.fitting_method_name,
        )

        if progress_callback is not None:
            progress_callback(0.95)

        # Optional uncertainty maps
        if self.compute_uncertainty:
            bayesian_fitter.compute_uncertainty = True
            bayesian_fitter._compute_uncertainty_maps(model, data, mask, param_maps)

        if progress_callback is not None:
            progress_callback(1.0)

        return param_maps

    def _compute_empirical_priors(
        self,
        params: NDArray[np.floating[Any]],
        r2: NDArray[np.floating[Any]],
        converged: NDArray[np.bool_],
        param_names: list[str],
        bounds: dict[str, tuple[float, float]],
        xp: Any,
    ) -> NDArray[np.floating[Any]]:
        """Compute empirical per-parameter prior standard deviations.

        For D and D*: uses log-space statistics (lognormal approx).
        For f and S0: uses direct statistics.
        Falls back to bounds-based priors when fewer than 10 valid voxels.

        Parameters
        ----------
        params : NDArray
            Stage 1 fitted parameters, shape ``(n_params, n_voxels)``.
        r2 : NDArray
            R-squared values from Stage 1.
        converged : NDArray
            Convergence flags from Stage 1.
        param_names : list[str]
            Parameter names.
        bounds : dict
            Parameter bounds.
        xp : module
            Array module.

        Returns
        -------
        NDArray
            Prior standard deviations, shape ``(n_params,)``.
        """
        n_params = len(param_names)

        # Valid voxels: converged and reasonable R²
        valid = converged & (r2 >= 0.3)
        n_valid = int(to_numpy(xp.sum(valid)))

        prior_stds = xp.zeros(n_params, dtype=params.dtype)

        if n_valid < 10:
            # Fallback: use bounds range / 4
            logger.warning(
                "Only %d valid Stage 1 voxels; using bounds-based priors",
                n_valid,
            )
            for i, name in enumerate(param_names):
                if name in bounds:
                    lo, hi = bounds[name]
                    prior_stds[i] = (hi - lo) / 4.0
                else:
                    prior_stds[i] = 1.0
            return xp.maximum(prior_stds * self.prior_scale, 1e-10)

        for i, name in enumerate(param_names):
            values = params[i, valid]

            if name in _LOG_SPACE_PARAMS:
                # Lognormal: compute std in log-space
                safe_vals = xp.maximum(values, 1e-10)
                log_vals = xp.log(safe_vals)
                std = float(to_numpy(xp.std(log_vals)))
                # Convert back: for lognormal, the "spread" in
                # original space is roughly mean * log_std
                mean_val = float(to_numpy(xp.mean(safe_vals)))
                prior_stds[i] = max(mean_val * std, 1e-10)
            else:
                # Direct statistics
                std = float(to_numpy(xp.std(values)))
                prior_stds[i] = max(std, 1e-10)

        return xp.maximum(prior_stds * self.prior_scale, 1e-10)
