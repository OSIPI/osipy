"""DCE binding adapter for the shared fitting infrastructure.

``BoundDCEModel`` wraps a ``BasePerfusionModel`` together with time and
AIF arrays, producing a ``FittableModel`` that the shared fitter can
use without knowing about DCE-specific context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from osipy.common.backend.array_module import get_array_module
from osipy.common.models.fittable import BaseBoundModel

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from osipy.dce.models.base import BasePerfusionModel


class BoundDCEModel(BaseBoundModel):
    """DCE model with time and AIF bound.

    Wraps a ``BasePerfusionModel`` so the fitter only sees
    ``predict_array_batch(free_params) -> output``.

    Parameters
    ----------
    model : BasePerfusionModel
        DCE pharmacokinetic model.
    t : NDArray
        Time points in seconds.
    aif : NDArray
        Arterial input function concentration.
    fixed : dict[str, float] | None
        Parameters to fix at constant values during fitting.
    """

    def __init__(
        self,
        model: BasePerfusionModel[Any],
        t: NDArray[np.floating[Any]],
        aif: NDArray[np.floating[Any]],
        fixed: dict[str, float] | None = None,
    ) -> None:
        super().__init__(model, fixed)
        xp = get_array_module(t, aif)
        self._t = xp.asarray(t)
        self._aif = xp.asarray(aif)
        self._dce_model: BasePerfusionModel[Any] = model

    def ensure_device(self, xp: Any) -> None:
        """Transfer time and AIF arrays to the target device."""
        super().ensure_device(xp)
        self._t = xp.asarray(self._t)
        self._aif = xp.asarray(self._aif)

    def predict_array_batch(
        self,
        free_params_batch: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]]:
        """Predict tissue concentration for a batch of voxels.

        Parameters
        ----------
        free_params_batch : NDArray
            Free parameter values, shape ``(n_free, n_voxels)``.
        xp : module
            Array module.

        Returns
        -------
        NDArray
            Predicted concentrations, shape ``(n_time, n_voxels)``.
        """
        full_params = self._expand_params(free_params_batch, xp)
        return self._dce_model.predict_batch(self._t, self._aif, full_params, xp)

    def get_initial_guess_batch(
        self,
        observed_batch: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]]:
        """Get initial parameter guesses for a batch of voxels.

        Delegates to the DCE model's ``get_initial_guess_batch`` and
        filters out fixed parameters.

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
        # Get full initial guesses from model
        full_guess = self._dce_model.get_initial_guess_batch(
            observed_batch,
            self._aif,
            self._t,
            xp,
        )

        if not self._fixed:
            return full_guess

        # Filter to free params only
        n_voxels = observed_batch.shape[1]
        free_guess = xp.zeros((self._n_free, n_voxels), dtype=full_guess.dtype)
        for free_idx, all_idx in enumerate(self._free_indices):
            free_guess[free_idx, :] = full_guess[all_idx, :]
        return free_guess

    def compute_jacobian_batch(
        self,
        params_batch: NDArray[np.floating[Any]],
        predicted: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]] | None:
        """DCE uses numerical Jacobian (convolution models)."""
        return None
