"""IVIM binding adapter for the shared fitting infrastructure.

``BoundIVIMModel`` wraps an ``IVIMModel`` together with b-values,
producing a ``FittableModel`` that the shared fitter can use without
knowing about IVIM-specific context.

Provides an analytical Jacobian for efficient Levenberg-Marquardt
optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from osipy.common.backend.array_module import get_array_module
from osipy.common.models.fittable import BaseBoundModel

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from osipy.ivim.models.biexponential import IVIMModel


class BoundIVIMModel(BaseBoundModel):
    """IVIM model with b-values bound.

    Wraps an ``IVIMModel`` so the fitter only sees
    ``predict_array_batch(free_params) -> output``.

    Parameters
    ----------
    model : IVIMModel
        IVIM signal model.
    b_values : NDArray
        Diffusion weighting values in s/mm^2.
    fixed : dict[str, float] | None
        Parameters to fix at constant values during fitting.
    b_threshold : float
        b-value threshold for segmented initial guess estimation.
    initial_guess : dict[str, float] | None
        Per-parameter overrides for the optimizer's starting values
        (e.g. ``{"D": 1e-3, "f": 0.1}``). Any free parameter named here
        seeds the fit with the given value in place of the data-driven
        estimate; unspecified parameters keep their data-driven guess.
    """

    def __init__(
        self,
        model: IVIMModel,
        b_values: NDArray[np.floating[Any]],
        fixed: dict[str, float] | None = None,
        b_threshold: float = 200.0,
        initial_guess: dict[str, float] | None = None,
    ) -> None:
        super().__init__(model, fixed)
        xp = get_array_module(b_values)
        self._b_values = xp.asarray(b_values, dtype=xp.float64)
        self._ivim_model: IVIMModel = model
        self._b_threshold = b_threshold
        self._initial_guess = initial_guess or {}

    def ensure_device(self, xp: Any) -> None:
        """Transfer b-values array to the target device."""
        super().ensure_device(xp)
        self._b_values = xp.asarray(self._b_values)

    def predict_array_batch(
        self, free_params_batch: NDArray[np.floating[Any]], xp: Any
    ) -> NDArray[np.floating[Any]]:
        """Predict signal for a batch of voxels.

        Parameters
        ----------
        free_params_batch : NDArray
            Free parameter values, shape ``(n_free, n_voxels)``.
        xp : module
            Array module.

        Returns
        -------
        NDArray
            Predicted signal, shape ``(n_b, n_voxels)``.
        """
        full_params = self._expand_params(free_params_batch, xp)
        return self._ivim_model.predict_batch(self._b_values, full_params, xp)

    def get_initial_guess_batch(
        self, observed_batch: NDArray[np.floating[Any]], xp: Any
    ) -> NDArray[np.floating[Any]]:
        """Get initial parameter guesses using segmented approach.

        Uses log-linear fit at high b-values for D, heuristic for f
        and D*, and b=0 signal for S0.

        Parameters
        ----------
        observed_batch : NDArray
            Observed data, shape ``(n_b, n_voxels)``.
        xp : module
            Array module.

        Returns
        -------
        NDArray
            Initial guesses, shape ``(n_free, n_voxels)``.
        """
        _n_b, n_voxels = observed_batch.shape
        b = self._b_values

        # All model parameters (before fixed filtering)
        all_params = self._ivim_model.parameters
        n_all = len(all_params)
        full_guess = xp.zeros((n_all, n_voxels), dtype=observed_batch.dtype)

        # S0 from minimum b-value (typically b=0)
        b0_idx = int(xp.argmin(b))
        s0 = xp.maximum(observed_batch[b0_idx, :], 1e-10)
        s0_idx = all_params.index("S0") if "S0" in all_params else None
        if s0_idx is not None:
            full_guess[s0_idx, :] = s0

        # Normalize signal
        signal_norm = observed_batch / s0[xp.newaxis, :]
        signal_norm = xp.maximum(signal_norm, 1e-10)

        # Log-linear fit at high b-values for D
        high_b_mask = b >= self._b_threshold
        n_high = int(xp.sum(high_b_mask))

        if n_high >= 2:
            b_high = b[high_b_mask]
            s_high = signal_norm[high_b_mask, :]  # (n_high, n_voxels)
            log_s = xp.log(s_high)

            # Normal equations for log(S/S0) = -b*D + log(1-f)
            sum_b = xp.sum(b_high)
            sum_b2 = xp.sum(b_high**2)
            sum_log_s = xp.sum(log_s, axis=0)
            sum_b_log_s = xp.sum(b_high[:, xp.newaxis] * log_s, axis=0)

            denom = n_high * sum_b2 - sum_b**2
            denom = xp.maximum(xp.abs(denom), 1e-10) * xp.sign(denom + 1e-20)
            d_init = -(n_high * sum_b_log_s - sum_b * sum_log_s) / denom

            intercept = (sum_b2 * sum_log_s - sum_b * sum_b_log_s) / denom
            intercept = xp.clip(intercept, -20.0, 20.0)
            f_init = 1.0 - xp.exp(intercept)
        else:
            d_init = xp.full(n_voxels, 1.0e-3, dtype=observed_batch.dtype)
            f_init = xp.full(n_voxels, 0.1, dtype=observed_batch.dtype)

        # Clamp to physiological bounds
        bounds = self._ivim_model.get_bounds()
        d_init = xp.clip(d_init, bounds["D"][0], bounds["D"][1])
        f_init = xp.clip(f_init, bounds["f"][0], bounds["f"][1])

        # D* initial guess: log-linear fit on low b-value perfusion signal
        ds_bounds = bounds.get("D*", (2e-3, 100e-3))
        low_b_mask = b < self._b_threshold
        n_low = int(xp.sum(low_b_mask))

        if n_low >= 2:
            b_low = b[low_b_mask]  # (n_low,)
            s_low = signal_norm[low_b_mask, :]  # (n_low, n_voxels)

            # Isolate perfusion component:
            #   S/S0 = (1-f)*exp(-b*D) + f*exp(-b*D*)
            #   perfusion = [S/S0 - (1-f)*exp(-b*D)] / f
            diff_component = (1 - f_init[xp.newaxis, :]) * xp.exp(
                -b_low[:, xp.newaxis] * d_init[xp.newaxis, :]
            )
            safe_f = xp.maximum(f_init[xp.newaxis, :], 1e-6)
            perf_signal = (s_low - diff_component) / safe_f
            perf_signal = xp.maximum(perf_signal, 1e-10)

            log_perf = xp.log(perf_signal)

            # Log-linear fit: log(perf) = -b*D* + intercept
            sum_b = xp.sum(b_low)
            sum_b2 = xp.sum(b_low**2)
            sum_lp = xp.sum(log_perf, axis=0)
            sum_b_lp = xp.sum(b_low[:, xp.newaxis] * log_perf, axis=0)

            denom = n_low * sum_b2 - sum_b**2
            denom = xp.maximum(xp.abs(denom), 1e-10) * xp.sign(denom + 1e-20)
            d_star_init = -(n_low * sum_b_lp - sum_b * sum_lp) / denom

            # Enforce D* > 2*D, clip to bounds
            d_star_init = xp.maximum(d_star_init, 2.0 * d_init)
            d_star_init = xp.clip(d_star_init, ds_bounds[0], ds_bounds[1])
        else:
            # Fallback: 10x D heuristic
            d_star_init = xp.clip(d_init * 10, ds_bounds[0], ds_bounds[1])

        # Fill full guess
        d_idx = all_params.index("D")
        f_idx = all_params.index("f")
        full_guess[d_idx, :] = d_init
        full_guess[f_idx, :] = f_init
        if "D*" in all_params:
            d_star_idx = all_params.index("D*")
            full_guess[d_star_idx, :] = d_star_init

        if not self._fixed:
            free_guess = full_guess
        else:
            # Filter to free params only
            free_guess = xp.zeros((self._n_free, n_voxels), dtype=full_guess.dtype)
            for free_idx, all_idx in enumerate(self._free_indices):
                free_guess[free_idx, :] = full_guess[all_idx, :]

        # Apply user-supplied initial-guess overrides (keyed by free param
        # name): seed the optimizer with the given constant in place of the
        # data-driven estimate. Unspecified parameters are left untouched.
        if self._initial_guess:
            for free_idx, name in enumerate(self._free_params):
                if name in self._initial_guess:
                    free_guess[free_idx, :] = self._initial_guess[name]
        return free_guess

    def compute_jacobian_batch(
        self,
        params_batch: NDArray[np.floating[Any]],
        predicted: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]] | None:
        """Compute analytical Jacobian for IVIM bi-exponential model.

        Only computes columns for free (non-fixed) parameters.

        Parameters
        ----------
        params_batch : NDArray
            Free parameter values, shape ``(n_free, n_voxels)``.
        predicted : NDArray
            Predicted signal, shape ``(n_b, n_voxels)``.
        xp : module
            Array module.

        Returns
        -------
        NDArray
            Jacobian, shape ``(n_free, n_b, n_voxels)``.
        """
        full_params = self._expand_params(params_batch, xp)
        all_params = self._ivim_model.parameters
        b = self._b_values[:, xp.newaxis]  # (n_b, 1)

        # Extract all parameters
        param_dict = {}
        for i, name in enumerate(all_params):
            param_dict[name] = full_params[i, :]  # (n_voxels,)

        s0 = param_dict.get(
            "S0", xp.ones(full_params.shape[1], dtype=full_params.dtype)
        )
        d = param_dict["D"]
        f = param_dict["f"]

        exp_d = xp.exp(-b * d[xp.newaxis, :])  # (n_b, n_voxels)

        # Build Jacobian columns for each parameter
        all_cols: dict[str, NDArray[np.floating[Any]]] = {}

        # dS/dS0 = (1-f)*exp(-b*D) + f*exp(-b*D*)
        all_cols["S0"] = predicted / (s0[xp.newaxis, :] + 1e-30)

        # dS/dD = S0 * -(1-f) * b * exp(-b*D)
        all_cols["D"] = s0[xp.newaxis, :] * (-(1 - f[xp.newaxis, :]) * b * exp_d)

        if "D*" in param_dict:
            d_star = param_dict["D*"]
            exp_ds = xp.exp(-b * d_star[xp.newaxis, :])  # (n_b, n_voxels)

            # dS/dD* = S0 * -f * b * exp(-b*D*)
            all_cols["D*"] = s0[xp.newaxis, :] * (-f[xp.newaxis, :] * b * exp_ds)

            # dS/df = S0 * (-exp(-b*D) + exp(-b*D*))
            all_cols["f"] = s0[xp.newaxis, :] * (-exp_d + exp_ds)
        else:
            # Simplified model: no D*. The forward model is
            #   b >  threshold:  S = S0 * (1-f) * exp(-b*D)
            #   b <= threshold:  S = S0 * ((1-f) * exp(-b*D) + f)
            # so dS/df = S0 * (-exp(-b*D))            for b >  threshold
            #    dS/df = S0 * (-exp(-b*D) + 1)        for b <= threshold
            threshold = getattr(self._ivim_model, "b_threshold", self._b_threshold)
            low_b = (self._b_values <= threshold)[:, xp.newaxis]  # (n_b, 1)
            df = -exp_d + xp.where(low_b, 1.0, 0.0)
            all_cols["f"] = s0[xp.newaxis, :] * df

        # Select only free parameter columns
        return xp.stack([all_cols[p] for p in self._free_params])
