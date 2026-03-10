"""DSC gamma-variate binding adapter for the shared fitting infrastructure.

``BoundGammaVariateModel`` wraps the gamma-variate function with
time points, producing a ``FittableModel`` that the shared fitter
can use for recirculation removal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from osipy.common.backend.array_module import get_array_module
from osipy.common.models.base import BaseSignalModel
from osipy.common.models.fittable import BaseBoundModel

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class _GammaVariateSignalModel(BaseSignalModel):
    """Minimal signal model for the gamma-variate function.

    Parameters are: k (amplitude), t0 (arrival), alpha (shape), beta (scale).
    """

    @property
    def name(self) -> str:
        return "Gamma Variate"

    @property
    def parameters(self) -> list[str]:
        return ["k", "t0", "alpha", "beta"]

    @property
    def parameter_units(self) -> dict[str, str]:
        return {"k": "a.u.", "t0": "s", "alpha": "", "beta": "s"}

    @property
    def reference(self) -> str:
        return "Thompson HK et al. Circ Res 1964;14:502-515."

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "k": (0.0, 1e6),
            "t0": (0.0, 300.0),
            "alpha": (0.1, 20.0),
            "beta": (0.1, 50.0),
        }


class BoundGammaVariateModel(BaseBoundModel):
    """Gamma-variate model with time points bound.

    Wraps the gamma-variate function so the fitter only sees
    ``predict_array_batch(free_params) -> output``.

    Parameters
    ----------
    time : NDArray
        Time points in seconds.
    fixed : dict[str, float] | None
        Parameters to fix at constant values during fitting.
    peak_time : float | None
        Approximate peak time for setting t0 upper bound.
    """

    def __init__(
        self,
        time: NDArray[np.floating[Any]],
        fixed: dict[str, float] | None = None,
        peak_time: float | None = None,
    ) -> None:
        model = _GammaVariateSignalModel()
        super().__init__(model, fixed)
        xp = get_array_module(time)
        self._time = xp.asarray(time)
        self._peak_time = peak_time

    def ensure_device(self, xp: Any) -> None:
        """Transfer time array to the target device."""
        super().ensure_device(xp)
        self._time = xp.asarray(self._time)

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Return bounds, optionally restricting t0 to before peak."""
        bounds = super().get_bounds()
        if self._peak_time is not None and "t0" in bounds:
            bounds["t0"] = (bounds["t0"][0], self._peak_time)
        return bounds

    def predict_array_batch(
        self,
        free_params_batch: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]]:
        """Predict gamma-variate concentration for a batch.

        Parameters
        ----------
        free_params_batch : NDArray
            Free parameter values, shape ``(n_free, n_voxels)``.
        xp : module
            Array module.

        Returns
        -------
        NDArray
            Predicted concentration, shape ``(n_time, n_voxels)``.
        """
        full_params = self._expand_params(free_params_batch, xp)
        all_names = self._model.parameters

        k = full_params[all_names.index("k"), :]
        t0 = full_params[all_names.index("t0"), :]
        alpha = full_params[all_names.index("alpha"), :]
        beta = full_params[all_names.index("beta"), :]

        # (n_time, 1) - (1, n_voxels) -> (n_time, n_voxels)
        dt = self._time[:, xp.newaxis] - t0[xp.newaxis, :]
        valid = dt > 0

        result = xp.where(
            valid,
            k[xp.newaxis, :]
            * xp.power(xp.maximum(dt, 1e-10), alpha[xp.newaxis, :])
            * xp.exp(-dt / (beta[xp.newaxis, :] + 1e-10)),
            0.0,
        )
        return result

    def get_initial_guess_batch(
        self,
        observed_batch: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]]:
        """Get initial parameter guesses from curve shape.

        Parameters
        ----------
        observed_batch : NDArray
            Observed data, shape ``(n_time, n_voxels)``.
        xp : module
            Array module.

        Returns
        -------
        NDArray
            Initial guesses, shape ``(n_free, n_voxels)``.
        """
        import math

        _n_time, n_voxels = observed_batch.shape
        all_names = self._model.parameters

        full_guess = xp.zeros((len(all_names), n_voxels), dtype=observed_batch.dtype)

        # Peak detection per voxel
        peak_idx = xp.argmax(observed_batch, axis=0)
        peak_conc = xp.max(observed_batch, axis=0)

        # t0: estimate from first significant rise
        # Simple heuristic: t0 is a few frames before peak
        t0_idx = xp.maximum(peak_idx - 3, 0)
        t0_init = self._time[t0_idx]

        alpha_init = xp.full(n_voxels, 3.0, dtype=observed_batch.dtype)
        peak_time = self._time[peak_idx]
        beta_init = xp.maximum((peak_time - t0_init) / 3.0, 0.5)

        # k from peak value
        k_init = peak_conc / (
            xp.power(alpha_init * beta_init, alpha_init) * math.exp(-3.0) + 1e-10
        )

        full_guess[all_names.index("k"), :] = k_init
        full_guess[all_names.index("t0"), :] = t0_init
        full_guess[all_names.index("alpha"), :] = alpha_init
        full_guess[all_names.index("beta"), :] = beta_init

        if not self._fixed:
            return full_guess

        free_guess = xp.zeros((self._n_free, n_voxels), dtype=full_guess.dtype)
        for free_idx, all_idx in enumerate(self._free_indices):
            free_guess[free_idx, :] = full_guess[all_idx, :]
        return free_guess
