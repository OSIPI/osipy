"""T1 mapping binding adapters for the shared fitting infrastructure.

``BoundSPGRModel`` and ``BoundLookLockerModel`` wrap the corresponding
signal models together with their fixed independent variables (flip
angles / TR, inversion times), producing ``FittableModel`` instances
that the shared ``LevenbergMarquardtFitter`` can use.

Both provide analytical Jacobians for efficient convergence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from osipy.common.backend.array_module import get_array_module
from osipy.common.models.fittable import BaseBoundModel

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from osipy.dce.t1_mapping.models import LookLockerSignalModel, SPGRSignalModel


class BoundSPGRModel(BaseBoundModel):
    """SPGR model with flip angles and TR bound.

    Wraps an ``SPGRSignalModel`` so the fitter only sees
    free parameters [T1, M0].

    Parameters
    ----------
    model : SPGRSignalModel
        SPGR signal model.
    flip_angles_rad : NDArray
        Flip angles in radians.
    tr : float
        Repetition time in milliseconds.
    fixed : dict[str, float] | None
        Parameters to fix at constant values during fitting.
    """

    def __init__(
        self,
        model: SPGRSignalModel,
        flip_angles_rad: NDArray[np.floating[Any]],
        tr: float,
        fixed: dict[str, float] | None = None,
    ) -> None:
        super().__init__(model, fixed)
        xp = get_array_module(flip_angles_rad)
        self._flip_angles_rad = xp.asarray(flip_angles_rad)
        self._tr = tr
        # Precompute trig values
        self._sin_fa = xp.sin(self._flip_angles_rad)
        self._cos_fa = xp.cos(self._flip_angles_rad)

    def ensure_device(self, xp: Any) -> None:
        """Transfer flip angle arrays to the target device."""
        super().ensure_device(xp)
        self._flip_angles_rad = xp.asarray(self._flip_angles_rad)
        self._sin_fa = xp.asarray(self._sin_fa)
        self._cos_fa = xp.asarray(self._cos_fa)

    def predict_array_batch(
        self,
        free_params_batch: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]]:
        """Predict SPGR signal for a batch of voxels.

        Parameters
        ----------
        free_params_batch : NDArray
            Free parameter values, shape ``(n_free, n_voxels)``.
        xp : module
            Array module.

        Returns
        -------
        NDArray
            Predicted signal, shape ``(n_fa, n_voxels)``.
        """
        full_params = self._expand_params(free_params_batch, xp)

        # Extract parameters: full_params shape (n_all, n_voxels)
        all_params = self._model.parameters
        t1_idx = all_params.index("T1")
        m0_idx = all_params.index("M0")

        t1 = full_params[t1_idx, :]  # (n_voxels,)
        m0 = full_params[m0_idx, :]  # (n_voxels,)

        # Clamp T1 to avoid numerical issues
        t1_safe = xp.clip(t1, 1.0, 50000.0)

        # E1 = exp(-TR / T1): (n_voxels,)
        e1 = xp.exp(-self._tr / t1_safe)

        # S = M0 * sin(a) * (1 - E1) / (1 - E1 * cos(a))
        # Broadcasting: (1, n_voxels) and (n_fa, 1)
        sin_fa = self._sin_fa[:, xp.newaxis]  # (n_fa, 1)
        cos_fa = self._cos_fa[:, xp.newaxis]  # (n_fa, 1)
        e1_row = e1[xp.newaxis, :]  # (1, n_voxels)
        m0_row = m0[xp.newaxis, :]  # (1, n_voxels)

        one_minus_e1 = 1.0 - e1_row
        denom = 1.0 - e1_row * cos_fa
        denom = xp.where(xp.abs(denom) < 1e-10, 1e-10, denom)

        return m0_row * sin_fa * one_minus_e1 / denom

    def get_initial_guess_batch(
        self,
        observed_batch: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]]:
        """Get initial parameter guesses from data.

        Uses T1=1000 ms and M0=2*max(signal) as defaults.

        Parameters
        ----------
        observed_batch : NDArray
            Observed data, shape ``(n_fa, n_voxels)``.
        xp : module
            Array module.

        Returns
        -------
        NDArray
            Initial guesses, shape ``(n_free, n_voxels)``.
        """
        n_voxels = observed_batch.shape[1]
        all_params = self._model.parameters
        n_all = len(all_params)

        full_guess = xp.zeros((n_all, n_voxels), dtype=observed_batch.dtype)

        t1_idx = all_params.index("T1")
        m0_idx = all_params.index("M0")

        full_guess[t1_idx, :] = 1000.0
        full_guess[m0_idx, :] = 2.0 * xp.max(observed_batch, axis=0)

        if not self._fixed:
            return full_guess

        # Filter to free params only
        free_guess = xp.zeros((self._n_free, n_voxels), dtype=full_guess.dtype)
        for free_idx, all_idx in enumerate(self._free_indices):
            free_guess[free_idx, :] = full_guess[all_idx, :]
        return free_guess

    def compute_jacobian_batch(
        self,
        params_batch: NDArray[np.floating[Any]],
        _predicted: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]]:
        """Compute analytical Jacobian for SPGR model.

        Parameters
        ----------
        params_batch : NDArray
            Free parameter values, shape ``(n_free, n_voxels)``.
        _predicted : NDArray
            Predicted signal (unused, Jacobian computed from params).
        xp : module
            Array module.

        Returns
        -------
        NDArray
            Jacobian, shape ``(n_free, n_fa, n_voxels)``.
        """
        full_params = self._expand_params(params_batch, xp)
        all_params = self._model.parameters
        t1_idx = all_params.index("T1")
        m0_idx = all_params.index("M0")

        t1 = full_params[t1_idx, :]  # (n_voxels,)
        m0 = full_params[m0_idx, :]  # (n_voxels,)
        t1_safe = xp.clip(t1, 1.0, 50000.0)

        e1 = xp.exp(-self._tr / t1_safe)  # (n_voxels,)

        sin_fa = self._sin_fa[:, xp.newaxis]  # (n_fa, 1)
        cos_fa = self._cos_fa[:, xp.newaxis]  # (n_fa, 1)
        e1_row = e1[xp.newaxis, :]  # (1, n_voxels)
        m0_row = m0[xp.newaxis, :]  # (1, n_voxels)

        one_minus_e1 = 1.0 - e1_row
        denom = 1.0 - e1_row * cos_fa
        denom = xp.where(xp.abs(denom) < 1e-10, 1e-10, denom)

        # dS/dM0 = sin(a) * (1 - E1) / (1 - E1 * cos(a))
        dS_dM0 = sin_fa * one_minus_e1 / denom  # (n_fa, n_voxels)

        # dS/dT1 = M0 * sin(a) * dE1/dT1 * d/dE1[(1-E1)/(1-E1*cos)]
        # dE1/dT1 = E1 * TR / T1^2
        # d/dE1[(1-E1)/(1-E1*cos)] = (-v + u*cos) / v^2
        #   where u = 1-E1, v = 1-E1*cos
        #   = (-(1-E1*cos) + (1-E1)*cos) / v^2
        #   = (-1 + cos) / v^2  (simplified: cos - 1 terms cancel)
        # Full: dS/dT1 = M0 * sin(a) * E1 * TR / T1^2 * (cos - 1) / v^2
        #              = -M0 * sin(a) * E1 * TR / T1^2 * (1 - cos) / v^2
        t1_sq = t1_safe**2
        one_minus_cos = 1.0 - cos_fa  # (n_fa, 1)
        denom_sq = denom**2

        dS_dT1 = (
            -m0_row
            * sin_fa
            * e1_row
            * self._tr
            / t1_sq[xp.newaxis, :]
            * one_minus_cos
            / denom_sq
        )  # (n_fa, n_voxels)

        # Build Jacobian: select only free parameter columns
        all_cols = {
            "T1": dS_dT1,
            "M0": dS_dM0,
        }
        return xp.stack([all_cols[p] for p in self._free_params])


class BoundLookLockerModel(BaseBoundModel):
    """Look-Locker model with inversion times bound.

    Wraps a ``LookLockerSignalModel`` so the fitter only sees
    free parameters [T1_star, A, B].

    Parameters
    ----------
    model : LookLockerSignalModel
        Look-Locker signal model.
    ti_times : NDArray
        Inversion times in milliseconds.
    fixed : dict[str, float] | None
        Parameters to fix at constant values during fitting.
    """

    def __init__(
        self,
        model: LookLockerSignalModel,
        ti_times: NDArray[np.floating[Any]],
        fixed: dict[str, float] | None = None,
    ) -> None:
        super().__init__(model, fixed)
        xp = get_array_module(ti_times)
        self._ti_times = xp.asarray(ti_times)

    def ensure_device(self, xp: Any) -> None:
        """Transfer TI times array to the target device."""
        super().ensure_device(xp)
        self._ti_times = xp.asarray(self._ti_times)

    def predict_array_batch(
        self,
        free_params_batch: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]]:
        """Predict Look-Locker signal for a batch of voxels.

        S(TI) = A - B * exp(-TI / T1*)

        Parameters
        ----------
        free_params_batch : NDArray
            Free parameter values, shape ``(n_free, n_voxels)``.
        xp : module
            Array module.

        Returns
        -------
        NDArray
            Predicted signal, shape ``(n_ti, n_voxels)``.
        """
        full_params = self._expand_params(free_params_batch, xp)
        all_params = self._model.parameters
        t1s_idx = all_params.index("T1_star")
        a_idx = all_params.index("A")
        b_idx = all_params.index("B")

        t1_star = full_params[t1s_idx, :]  # (n_voxels,)
        a = full_params[a_idx, :]  # (n_voxels,)
        b = full_params[b_idx, :]  # (n_voxels,)

        t1_star_safe = xp.clip(t1_star, 1.0, 50000.0)

        # Broadcasting: (n_ti, 1) and (1, n_voxels)
        ti = self._ti_times[:, xp.newaxis]  # (n_ti, 1)
        exp_term = xp.exp(-ti / t1_star_safe[xp.newaxis, :])

        return a[xp.newaxis, :] - b[xp.newaxis, :] * exp_term

    def get_initial_guess_batch(
        self,
        observed_batch: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]]:
        """Get initial parameter guesses from data.

        Uses T1*=median(TI), A=max(S), B=max(S)+min(S).

        Parameters
        ----------
        observed_batch : NDArray
            Observed data, shape ``(n_ti, n_voxels)``.
        xp : module
            Array module.

        Returns
        -------
        NDArray
            Initial guesses, shape ``(n_free, n_voxels)``.
        """
        n_voxels = observed_batch.shape[1]
        all_params = self._model.parameters
        n_all = len(all_params)

        full_guess = xp.zeros((n_all, n_voxels), dtype=observed_batch.dtype)

        t1s_idx = all_params.index("T1_star")
        a_idx = all_params.index("A")
        b_idx = all_params.index("B")

        # T1* ~ median of TI times
        median_ti = float(xp.median(self._ti_times))
        full_guess[t1s_idx, :] = median_ti

        # A ~ max signal (steady state)
        a_init = xp.max(observed_batch, axis=0)
        full_guess[a_idx, :] = a_init

        # B ~ A - min(signal) (at TI→0, S≈A-B, so B≈A-min(S))
        b_init = a_init - xp.min(observed_batch, axis=0)
        full_guess[b_idx, :] = xp.maximum(b_init, 1.0)

        if not self._fixed:
            return full_guess

        # Filter to free params only
        free_guess = xp.zeros((self._n_free, n_voxels), dtype=full_guess.dtype)
        for free_idx, all_idx in enumerate(self._free_indices):
            free_guess[free_idx, :] = full_guess[all_idx, :]
        return free_guess

    def compute_jacobian_batch(
        self,
        params_batch: NDArray[np.floating[Any]],
        _predicted: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]]:
        """Compute analytical Jacobian for Look-Locker model.

        Parameters
        ----------
        params_batch : NDArray
            Free parameter values, shape ``(n_free, n_voxels)``.
        _predicted : NDArray
            Predicted signal (unused, Jacobian computed from params).
        xp : module
            Array module.

        Returns
        -------
        NDArray
            Jacobian, shape ``(n_free, n_ti, n_voxels)``.
        """
        full_params = self._expand_params(params_batch, xp)
        all_params = self._model.parameters
        t1s_idx = all_params.index("T1_star")
        b_idx = all_params.index("B")

        t1_star = full_params[t1s_idx, :]  # (n_voxels,)
        b = full_params[b_idx, :]  # (n_voxels,)

        t1_star_safe = xp.clip(t1_star, 1.0, 50000.0)

        ti = self._ti_times[:, xp.newaxis]  # (n_ti, 1)
        exp_term = xp.exp(-ti / t1_star_safe[xp.newaxis, :])  # (n_ti, n_voxels)

        # dS/dA = 1
        n_ti = self._ti_times.shape[0]
        n_voxels = params_batch.shape[1]
        dS_dA = xp.ones((n_ti, n_voxels), dtype=params_batch.dtype)

        # dS/dB = -exp(-TI / T1*)
        dS_dB = -exp_term

        # dS/dT1* = -B * TI / T1*^2 * exp(-TI / T1*)
        t1_star_sq = t1_star_safe**2
        dS_dT1star = -b[xp.newaxis, :] * ti / t1_star_sq[xp.newaxis, :] * exp_term

        # Build Jacobian: select only free parameter columns
        all_cols = {
            "T1_star": dS_dT1star,
            "A": dS_dA,
            "B": dS_dB,
        }
        return xp.stack([all_cols[p] for p in self._free_params])
