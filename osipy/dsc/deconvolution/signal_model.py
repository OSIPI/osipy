"""DSC perfusion forward model and binding adapter.

Implements ``C(t) = CBF * AIF(t) * R(t)`` (convolution) as a
``BaseSignalModel`` with pre-computed SVD via ``BoundDSCModel``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from osipy.common.backend.array_module import get_array_module
from osipy.common.models.base import BaseSignalModel
from osipy.common.models.fittable import BaseBoundModel

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class DSCConvolutionModel(BaseSignalModel):
    """DSC perfusion forward model: C(t) = CBF * AIF(t) * R(t).

    Parameters are CBF (mL/100g/min), MTT (s), and Ta (s).

    References
    ----------
    .. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    .. [2] Ostergaard L et al. MRM 1996;36(5):715-725.
    """

    @property
    def name(self) -> str:
        """Return human-readable component name."""
        return "DSC Convolution"

    @property
    def parameters(self) -> list[str]:
        """Return list of parameter names: CBF, MTT, Ta."""
        return ["CBF", "MTT", "Ta"]

    @property
    def parameter_units(self) -> dict[str, str]:
        """Return mapping of parameter names to OSIPI-compliant units."""
        return {"CBF": "mL/100g/min", "MTT": "s", "Ta": "s"}

    @property
    def reference(self) -> str:
        """Return primary literature citation."""
        return "Ostergaard L et al. MRM 1996;36(5):715-725."

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Return default parameter bounds as {name: (lower, upper)}."""
        return {
            "CBF": (0.0, 1000.0),
            "MTT": (0.0, 60.0),
            "Ta": (0.0, 30.0),
        }


class BoundDSCModel(BaseBoundModel):
    """DSC model with AIF and time bound, SVD pre-computed.

    Pre-computes the SVD of the AIF convolution matrix once. The
    fitter uses the pre-computed SVD components to recover R(t)
    via regularized inversion.

    Parameters
    ----------
    model : DSCConvolutionModel
        The signal model.
    aif : NDArray
        Arterial input function (delta-R2*), shape ``(n_timepoints,)``.
    time : NDArray
        Time points in seconds, shape ``(n_timepoints,)``.
    matrix_type : str
        ``'circulant'`` (default, delay-insensitive) or ``'toeplitz'`` (causal).
    fixed : dict[str, float] | None
        Parameters to fix.
    """

    def __init__(
        self,
        model: DSCConvolutionModel,
        aif: NDArray[np.floating[Any]],
        time: NDArray[np.floating[Any]],
        matrix_type: str = "circulant",
        fixed: dict[str, float] | None = None,
    ) -> None:
        super().__init__(model, fixed)
        from osipy.dsc.deconvolution.svd import (
            _build_circulant_matrix_xp,
            _build_toeplitz_matrix_xp,
        )

        xp = get_array_module(aif)
        self._aif = xp.asarray(aif)
        self._time = xp.asarray(time)
        n = len(time)
        dt = float(time[1] - time[0]) if n > 1 else 1.0
        self._dt = dt

        if matrix_type == "circulant":
            A = _build_circulant_matrix_xp(self._aif, n, xp) * dt
        else:
            A = _build_toeplitz_matrix_xp(self._aif, n, xp) * dt

        self._U, self._S, self._Vh = xp.linalg.svd(A, full_matrices=False)

    def ensure_device(self, xp: Any) -> None:
        """Transfer arrays to the target device."""
        super().ensure_device(xp)
        self._aif = xp.asarray(self._aif)
        self._time = xp.asarray(self._time)
        self._U = xp.asarray(self._U)
        self._S = xp.asarray(self._S)
        self._Vh = xp.asarray(self._Vh)

    def predict_array_batch(
        self,
        free_params_batch: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]]:
        """Forward model: reconstruct C(t) from CBF, MTT, Ta.

        Parameters
        ----------
        free_params_batch : NDArray
            Free parameter values, shape ``(n_free, n_voxels)``.
        xp : module
            Array module.

        Returns
        -------
        NDArray
            Predicted concentration, shape ``(n_timepoints, n_voxels)``.
        """
        full_params = self._expand_params(free_params_batch, xp)
        all_names = self._model.parameters

        cbf = full_params[all_names.index("CBF"), :]
        mtt = full_params[all_names.index("MTT"), :]
        ta = full_params[all_names.index("Ta"), :]

        # Build R(t) = exp(-t/MTT) shifted by Ta
        t = self._time[:, xp.newaxis]  # (n_t, 1)
        dt_arr = t - ta[xp.newaxis, :]  # (n_t, n_voxels)
        mtt_safe = xp.maximum(mtt[xp.newaxis, :], 1e-10)
        R = xp.where(dt_arr >= 0, xp.exp(-dt_arr / mtt_safe), 0.0)

        # C(t) = CBF * A @ R (convolution via pre-computed matrix)
        # A is (n_t, n_t), R is (n_t, n_voxels)
        A = self._U @ (self._S[:, xp.newaxis] * (self._Vh @ R))
        return cbf[xp.newaxis, :] * A

    def get_initial_guess_batch(
        self,
        observed_batch: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]]:
        """Compute initial guesses from signal shape.

        Parameters
        ----------
        observed_batch : NDArray
            Observed data, shape ``(n_timepoints, n_voxels)``.
        xp : module
            Array module.

        Returns
        -------
        NDArray
            Initial guesses, shape ``(n_free, n_voxels)``.
        """
        _n_t, n_voxels = observed_batch.shape
        all_names = self._model.parameters

        full_guess = xp.zeros((len(all_names), n_voxels), dtype=observed_batch.dtype)

        # CBF initial: from peak of deconvolved signal
        peak_conc = xp.max(observed_batch, axis=0)
        aif_peak = xp.max(self._aif)
        cbf_init = xp.where(aif_peak > 0, peak_conc / (aif_peak + 1e-10), 10.0)
        full_guess[all_names.index("CBF"), :] = cbf_init

        # MTT initial: 4 seconds
        full_guess[all_names.index("MTT"), :] = 4.0

        # Ta initial: from TTP
        peak_idx = xp.argmax(observed_batch, axis=0)
        ta_init = self._time[peak_idx]
        full_guess[all_names.index("Ta"), :] = ta_init

        if not self._fixed:
            return full_guess

        free_guess = xp.zeros((self._n_free, n_voxels), dtype=full_guess.dtype)
        for free_idx, all_idx in enumerate(self._free_indices):
            free_guess[free_idx, :] = full_guess[all_idx, :]
        return free_guess
