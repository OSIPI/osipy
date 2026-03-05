"""Two-Compartment Uptake Model (2CUM).

This module implements the Two-Compartment Uptake Model for DCE-MRI,
a simplification of 2CXM with unidirectional uptake (no backflux from EES).
3 parameters: Fp, PS, vp (no ve).

GPU/CPU agnostic using the xp array module pattern.
NO scipy dependency - uses xp.linalg operations.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.30101
.. [3] Sourbron SP, Buckley DL. MRM 2011;66(3):735-745.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from osipy.common.backend.array_module import get_array_module, to_numpy
from osipy.dce.models.base import BasePerfusionModel, ModelParameters
from osipy.dce.models.registry import register_model

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass
class TwoCompartmentUptakeParams(ModelParameters):
    """Parameters for Two-Compartment Uptake Model (2CUM).

    Attributes
    ----------
    fp : float
        Plasma flow (OSIPI: Q.PH1.002) in mL/min/100mL.
    ps : float
        Permeability-surface area product (OSIPI: Q.PH1.004) in mL/min/100mL.
    vp : float
        Plasma volume fraction (OSIPI: Q.PH1.001), mL/100mL.
    """

    fp: float = 15.0  # ml/100ml/min
    ps: float = 2.0  # ml/100ml/min
    vp: float = 0.02


@register_model("2cum")
class TwoCompartmentUptakeModel(BasePerfusionModel[TwoCompartmentUptakeParams]):
    """Two-Compartment Uptake Model (2CUM).

    Implements the 2CUM with unidirectional uptake from plasma to EES
    (no backflux). This is a simplification of the 2CXM with 3 parameters
    instead of 4.

    Attributes
    ----------
    Fp : float
        Plasma flow (OSIPI: Q.PH1.002), mL/min/100mL.
    PS : float
        Permeability-surface area product (OSIPI: Q.PH1.004), mL/min/100mL.
    vp : float
        Plasma volume fraction (OSIPI: Q.PH1.001), mL/100mL.

    Notes
    -----
    The tissue concentration is:
        E  = PS / (Fp + PS)           (extraction fraction)
        T  = vp / (Fp + PS)           (time constant)
        Ct(t) = Fp * [(1-E) * exp(-t/T) ⊛ Ca(t) + E * ∫₀ᵗ Ca(τ)dτ]

    where:
        - Ca is arterial concentration (AIF)
        - Fp is plasma flow (OSIPI: Q.PH1.002)
        - PS is permeability-surface area product (OSIPI: Q.PH1.004)
        - vp is plasma volume fraction (OSIPI: Q.PH1.001)
        - ⊛ denotes convolution

    GPU/CPU agnostic - operates on same device as input arrays.

    References
    ----------
    .. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    .. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.30101
    .. [3] Sourbron SP, Buckley DL. MRM 2011;66(3):735-745.
    """

    @property
    def time_unit(self) -> str:
        """Return time unit (minutes for 2CUM)."""
        return "minutes"

    @property
    def name(self) -> str:
        """Return model name."""
        return "Two-Compartment Uptake (2CUM)"

    @property
    def parameters(self) -> list[str]:
        """Return parameter names."""
        return ["Fp", "PS", "vp"]

    @property
    def parameter_units(self) -> dict[str, str]:
        """Return parameter units."""
        return {
            "Fp": "mL/min/100mL",
            "PS": "mL/min/100mL",
            "vp": "mL/100mL",
        }

    @property
    def reference(self) -> str:
        """Return literature citation."""
        return "Sourbron SP, Buckley DL (2011). Magn Reson Med 66(3):735-745."

    def _predict(
        self,
        t: "NDArray[np.floating[Any]]",
        aif: "NDArray[np.floating[Any]]",
        params: "NDArray[np.floating[Any]]",
        xp: Any,
    ) -> "NDArray[np.floating[Any]]":
        """Predict tissue concentration.

        Works for both single-voxel and batch via broadcasting.

        Parameters
        ----------
        t : NDArray[np.floating]
            Time points in seconds, ``(n_time,)`` or ``(n_time, 1)``.
        aif : NDArray[np.floating]
            Arterial input function, ``(n_time,)`` or ``(n_time, 1)``.
        params : NDArray[np.floating]
            ``[Fp, PS, vp]`` — shape ``(3,)`` or ``(3, n_voxels)``.
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        NDArray[np.floating]
            Predicted tissue concentration.
        """
        fp = params[0]
        ps = params[1]
        vp = params[2]

        # Convert time to model units (minutes)
        t_min = self._convert_time(t, xp)

        # Convert from ml/100ml/min to ml/ml/min
        fp_frac = fp / 100.0
        ps_frac = ps / 100.0

        # Extraction fraction and time constant
        sum_fp_ps = fp_frac + ps_frac
        E = xp.where(xp.abs(sum_fp_ps) > 1e-15, ps_frac / sum_fp_ps, xp.asarray(0.0))
        T = xp.where(xp.abs(sum_fp_ps) > 1e-15, vp / sum_fp_ps, xp.asarray(1e15))

        # Exponential convolution (handles scalar and array T)
        from osipy.common.convolution.expconv import expconv

        conv_term = expconv(aif, T, t_min)

        # Cumulative integral of AIF (handles 1-D and 2-D via _cumulative_aif_integral)
        from osipy.dce.models.patlak import _cumulative_aif_integral

        cum_aif = _cumulative_aif_integral(aif, t_min, xp)

        # Ct = Fp * [(1-E) * expconv(Ca,T,t) + E * ∫Ca dτ]
        return fp_frac * ((1.0 - E) * conv_term + E * cum_aif)

    def get_initial_guess_batch(
        self,
        ct_batch: "NDArray[np.floating[Any]]",
        aif: "NDArray[np.floating[Any]]",
        t: "NDArray[np.floating[Any]]",
        xp: Any,
    ) -> "NDArray[np.floating[Any]]":
        """Compute vectorized initial guesses for a batch of voxels.

        Parameters
        ----------
        ct_batch : NDArray[np.floating]
            Tissue concentration curves, shape (n_timepoints, n_voxels).
        aif : NDArray[np.floating]
            Arterial input function, shape (n_timepoints,).
        t : NDArray[np.floating]
            Time points in seconds, shape (n_timepoints,).
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        NDArray[np.floating]
            Initial parameter guesses, shape (3, n_voxels).
        """
        n_voxels = ct_batch.shape[1]

        # vp from first-pass peak ratio
        aif_peak_idx = xp.argmax(aif)
        aif_peak_val = aif[aif_peak_idx]
        ct_at_peak = ct_batch[aif_peak_idx, :]  # (n_voxels,)
        safe_aif_peak = xp.where(
            aif_peak_val > 0,
            aif_peak_val,
            xp.ones_like(aif_peak_val),
        )
        vp_batch = xp.where(
            aif_peak_val > 0,
            ct_at_peak / safe_aif_peak,
            xp.full(n_voxels, 0.02, dtype=ct_batch.dtype),
        )
        vp_batch = xp.clip(vp_batch, 0.01, 0.1)

        fp_batch = xp.full(n_voxels, 15.0, dtype=ct_batch.dtype)
        ps_batch = xp.full(n_voxels, 2.0, dtype=ct_batch.dtype)

        return xp.stack([fp_batch, ps_batch, vp_batch])

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Return physiological parameter bounds."""
        return {
            "Fp": (0.1, 200.0),  # ml/100ml/min
            "PS": (0.0, 50.0),  # ml/100ml/min
            "vp": (0.001, 0.3),
        }

    def get_initial_guess(
        self,
        ct: "NDArray[np.floating[Any]]",
        aif: "NDArray[np.floating[Any]]",
        t: "NDArray[np.floating[Any]]",
    ) -> TwoCompartmentUptakeParams:
        """Compute initial parameter guess.

        GPU/CPU agnostic - converts to numpy for scalar extraction.

        Parameters
        ----------
        ct : NDArray[np.floating]
            Tissue concentration curve.
        aif : NDArray[np.floating]
            Arterial input function.
        t : NDArray[np.floating]
            Time points in seconds.

        Returns
        -------
        TwoCompartmentUptakeParams
            Initial parameter estimates.
        """
        xp = get_array_module(ct, aif)

        # Estimate vp from first-pass
        aif_peak_idx = int(xp.argmax(aif))
        aif_peak = float(to_numpy(aif[aif_peak_idx]))

        if aif_peak > 0:
            ct_at_peak = float(to_numpy(ct[aif_peak_idx]))
            vp_init = ct_at_peak / aif_peak
            vp_init = max(0.01, min(vp_init, 0.1))
        else:
            vp_init = 0.02

        return TwoCompartmentUptakeParams(
            fp=15.0,
            ps=2.0,
            vp=float(vp_init),
        )
