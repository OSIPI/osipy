"""Extended Tofts-Kety pharmacokinetic model (OSIPI: M.IC1.005).

This module implements the Extended Tofts model for DCE-MRI analysis,
which includes both the extravascular extracellular space (EES) and
a vascular plasma compartment.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.30101
.. [3] Tofts PS et al. J Magn Reson Imaging 1999;10(3):223-232.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from osipy.common.backend.array_module import get_array_module
from osipy.common.convolution import convolve_aif
from osipy.dce.models.base import BasePerfusionModel, ModelParameters
from osipy.dce.models.registry import register_model

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass
class ExtendedToftsParams(ModelParameters):
    """Parameters for Extended Tofts model (OSIPI: M.IC1.005).

    Attributes
    ----------
    ktrans : float
        Volume transfer constant (OSIPI: Q.PH1.008) in 1/min.
    ve : float
        Extravascular extracellular volume fraction (OSIPI: Q.PH1.001),
        mL/100mL.
    vp : float
        Plasma volume fraction (OSIPI: Q.PH1.001), mL/100mL.
    """

    ktrans: float = 0.1
    ve: float = 0.2
    vp: float = 0.02

    @property
    def kep(self) -> float:
        """Rate constant kep = Ktrans/ve (OSIPI: Q.PH1.009) in 1/min."""
        if self.ve > 0:
            return self.ktrans / self.ve
        return 0.0


@register_model("extended_tofts")
class ExtendedToftsModel(BasePerfusionModel[ExtendedToftsParams]):
    """Extended Tofts-Kety pharmacokinetic model (OSIPI: M.IC1.005).

    Implements the Extended Tofts model for indicator concentration
    in tissue as a function of arterial input, including a vascular
    plasma compartment.

    Attributes
    ----------
    Ktrans : float
        Volume transfer constant (OSIPI: Q.PH1.008), 1/min.
    ve : float
        Extravascular extracellular volume fraction (OSIPI: Q.PH1.001),
        mL/100mL.
    vp : float
        Plasma volume fraction (OSIPI: Q.PH1.001), mL/100mL.

    Notes
    -----
    Model equation:
        Ct(t) = vp*Cp(t) + Ktrans * integral_0^t Cp(tau) * exp(-kep*(t-tau)) dtau

    where:
        - Ct(t) is tissue concentration (OSIPI: Q.IC1.001) at time t
        - Cp(t) is plasma concentration (AIF)
        - vp is plasma volume fraction (OSIPI: Q.PH1.001)
        - Ktrans is volume transfer constant (OSIPI: Q.PH1.008)
        - kep = Ktrans/ve is the rate constant (OSIPI: Q.PH1.009)

    Assumptions:
        - Fast exchange between plasma and EES
        - Well-mixed compartments
        - First-order kinetics

    References
    ----------
    .. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    .. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.30101
    .. [3] Tofts PS et al. J Magn Reson Imaging 1999;10(3):223-232.
    """

    @property
    def time_unit(self) -> str:
        """Return time unit (minutes for Extended Tofts)."""
        return "minutes"

    @property
    def name(self) -> str:
        """Return model name."""
        return "Extended Tofts"

    @property
    def parameters(self) -> list[str]:
        """Return parameter names following OSIPI terminology."""
        return ["Ktrans", "ve", "vp"]

    @property
    def parameter_units(self) -> dict[str, str]:
        """Return parameter units."""
        return {"Ktrans": "1/min", "ve": "mL/100mL", "vp": "mL/100mL"}

    @property
    def reference(self) -> str:
        """Return literature citation."""
        return "Tofts PS et al. (1999). J Magn Reson Imaging 10(3):223-232."

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
            ``[Ktrans, ve, vp]`` — shape ``(3,)`` or ``(3, n_voxels)``.
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        NDArray[np.floating]
            Predicted tissue concentration.
        """
        ktrans = params[0]
        ve = params[1]
        vp = params[2]

        # Convert time to model units (minutes)
        t_min = self._convert_time(t, xp)

        ve_safe = xp.where(ve > 0, ve, xp.asarray(1e-10))
        kep = ktrans / ve_safe

        # Vascular contribution (broadcasting handles scalar and array vp)
        ct_vascular = vp * aif

        # Time step in minutes
        dt = float(t_min.ravel()[1] - t_min.ravel()[0]) if t_min.size > 1 else 1.0

        # Impulse response
        irf = ktrans * xp.exp(-kep * t_min)

        ct_ees = convolve_aif(aif, irf, dt=dt)

        return ct_vascular + ct_ees

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

        # vp from first-pass peak ratio (matching per-voxel)
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
        vp_batch = xp.clip(vp_batch, 0.001, 0.1)

        # Ktrans from peak ratio
        aif_max = xp.max(aif)
        ct_max = xp.max(ct_batch, axis=0)
        safe_aif_max = xp.where(aif_max > 0, aif_max, xp.ones_like(aif_max))
        ktrans_batch = xp.where(
            aif_max > 0,
            0.1 * ct_max / safe_aif_max,
            xp.full(n_voxels, 0.1, dtype=ct_batch.dtype),
        )
        ktrans_batch = xp.clip(ktrans_batch, 0.0, 5.0)

        # ve = 0.2 constant
        ve_batch = xp.full(n_voxels, 0.2, dtype=ct_batch.dtype)

        return xp.stack([ktrans_batch, ve_batch, vp_batch])

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Return physiological parameter bounds.

        Returns
        -------
        dict[str, tuple[float, float]]
            Parameter bounds based on QIBA DCE Profile.
        """
        return {
            "Ktrans": (0.0, 5.0),  # min⁻¹
            "ve": (0.001, 1.0),  # fraction
            "vp": (0.0, 1.0),  # fraction
        }

    def get_initial_guess(
        self,
        ct: "NDArray[np.floating[Any]]",
        aif: "NDArray[np.floating[Any]]",
        t: "NDArray[np.floating[Any]]",
    ) -> ExtendedToftsParams:
        """Compute initial parameter guess.

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
        ExtendedToftsParams
            Initial parameter estimates.
        """
        xp = get_array_module(ct, aif)

        # Estimate vp from first-pass peak
        # At bolus arrival, ct ≈ vp * aif
        aif_peak_idx = int(xp.argmax(aif))
        if aif[aif_peak_idx] > 0:
            vp_init = float(ct[aif_peak_idx] / aif[aif_peak_idx])
            vp_init = max(0.001, min(vp_init, 0.1))
        else:
            vp_init = 0.02

        # Estimate Ktrans from peak ratio
        ct_max = float(xp.max(ct))
        aif_max = float(xp.max(aif))
        ktrans_init = 0.1 * ct_max / aif_max if aif_max > 0 else 0.1

        # Default ve
        ve_init = 0.2

        return ExtendedToftsParams(
            ktrans=float(ktrans_init),
            ve=float(ve_init),
            vp=float(vp_init),
        )
