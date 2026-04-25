"""Standard Tofts pharmacokinetic model (OSIPI: M.IC1.004).

This module implements the Standard Tofts model for DCE-MRI analysis.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.29840
.. [3] Tofts PS, Kermode AG. MRM 1991;17(2):357-367.
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
class ToftsParams(ModelParameters):
    """Parameters for Standard Tofts model (OSIPI: M.IC1.004).

    Attributes
    ----------
    ktrans : float
        Volume transfer constant (OSIPI: Q.PH1.008) in 1/min.
        Represents transfer from plasma to EES.
    ve : float
        Extravascular extracellular volume fraction (OSIPI: Q.PH1.001),
        unitless fraction [0, 1] (CAPLEX lists mL/100mL; osipy uses the
        equivalent fraction convention).
    """

    ktrans: float = 0.1
    ve: float = 0.2

    @property
    def kep(self) -> float:
        """Rate constant kep = Ktrans/ve (OSIPI: Q.PH1.009) in 1/min."""
        if self.ve > 0:
            return self.ktrans / self.ve
        return 0.0


@register_model("tofts")
class ToftsModel(BasePerfusionModel[ToftsParams]):
    """Standard Tofts pharmacokinetic model (OSIPI: M.IC1.004).

    Implements the standard Tofts model for indicator concentration
    in tissue as a function of arterial input.

    Attributes
    ----------
    Ktrans : float
        Volume transfer constant (OSIPI: Q.PH1.008), 1/min.
    ve : float
        Extravascular extracellular volume fraction (OSIPI: Q.PH1.001),
        unitless fraction [0, 1].

    Notes
    -----
    Model equation:
        Ct(t) = Ktrans * integral_0^t Cp(tau) * exp(-kep*(t-tau)) dtau

    where:
        - Ct(t) is tissue concentration (OSIPI: Q.IC1.001) at time t
        - Cp(t) is plasma concentration (AIF)
        - Ktrans is volume transfer constant (OSIPI: Q.PH1.008)
        - kep = Ktrans/ve is the rate constant (OSIPI: Q.PH1.009)

    Assumptions:
        - Negligible vascular contribution (vp ~ 0)
        - Well-mixed compartments
        - First-order kinetics

    References
    ----------
    .. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    .. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.29840
    .. [3] Tofts PS, Kermode AG. MRM 1991;17(2):357-367.
    """

    @property
    def time_unit(self) -> str:
        """Return time unit (minutes for Tofts)."""
        return "minutes"

    @property
    def name(self) -> str:
        """Return model name."""
        return "Standard Tofts"

    @property
    def parameters(self) -> list[str]:
        """Return parameter names."""
        return ["Ktrans", "ve"]

    @property
    def parameter_units(self) -> dict[str, str]:
        """Return parameter units."""
        return {"Ktrans": "1/min", "ve": "fraction"}

    @property
    def reference(self) -> str:
        """Return literature citation."""
        return "Tofts PS, Kermode AG (1991). Magn Reson Med 17(2):357-367."

    def _predict(
        self,
        t: "NDArray[np.floating[Any]]",
        aif: "NDArray[np.floating[Any]]",
        params: "NDArray[np.floating[Any]]",
        xp: Any,
    ) -> "NDArray[np.floating[Any]]":
        """Predict tissue concentration.

        Works for both single-voxel and batch via broadcasting
        (base class ``predict_batch`` reshapes *t* and *aif* to columns).

        Parameters
        ----------
        t : NDArray[np.floating]
            Time points in seconds, ``(n_time,)`` or ``(n_time, 1)``.
        aif : NDArray[np.floating]
            Arterial input function, ``(n_time,)`` or ``(n_time, 1)``.
        params : NDArray[np.floating]
            ``[Ktrans, ve]`` — shape ``(2,)`` or ``(2, n_voxels)``.
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        NDArray[np.floating]
            Predicted tissue concentration.
        """
        ktrans = params[0]
        ve = params[1]

        # Convert time to model units (minutes)
        t_min = self._convert_time(t, xp)

        # Avoid division by zero (works for scalar and array)
        ve_safe = xp.where(ve > 0, ve, xp.asarray(1e-10))
        kep = ktrans / ve_safe

        # Time step in minutes (use ravel since t may be (n_time, 1))
        dt = float(t_min.ravel()[1] - t_min.ravel()[0]) if t_min.size > 1 else 1.0

        # Impulse response: h(t) = Ktrans * exp(-kep * t)
        irf = ktrans * xp.exp(-kep * t_min)

        # FFT convolution handles 1-D and 2-D via broadcasting
        return convolve_aif(aif, irf, dt=dt)

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
            Initial parameter guesses, shape (2, n_voxels).
        """
        n_voxels = ct_batch.shape[1]
        aif_max = xp.max(aif)
        ct_max = xp.max(ct_batch, axis=0)  # (n_voxels,)

        # Ktrans from peak ratio, matching per-voxel heuristic
        safe_aif_max = xp.where(aif_max > 0, aif_max, xp.ones_like(aif_max))
        ktrans_batch = xp.where(
            aif_max > 0,
            0.1 * ct_max / safe_aif_max,
            xp.full(n_voxels, 0.1, dtype=ct_batch.dtype),
        )
        ktrans_batch = xp.clip(ktrans_batch, 0.0, 5.0)

        # ve = 0.2 constant (matching per-voxel)
        ve_batch = xp.full(n_voxels, 0.2, dtype=ct_batch.dtype)

        return xp.stack([ktrans_batch, ve_batch])

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Return physiological parameter bounds.

        Returns
        -------
        dict[str, tuple[float, float]]
            Parameter bounds.
        """
        return {
            "Ktrans": (0.0, 5.0),  # min⁻¹
            "ve": (0.001, 1.0),  # fraction
        }

    def get_initial_guess(
        self,
        ct: "NDArray[np.floating[Any]]",
        aif: "NDArray[np.floating[Any]]",
        t: "NDArray[np.floating[Any]]",
    ) -> ToftsParams:
        """Compute initial parameter guess.

        Uses simple heuristics based on signal characteristics.

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
        ToftsParams
            Initial parameter estimates.
        """
        xp = get_array_module(ct, aif)

        # Estimate Ktrans from peak ratio
        ct_max = float(xp.max(ct))
        aif_max = float(xp.max(aif))
        ktrans_init = 0.1 * ct_max / aif_max if aif_max > 0 else 0.1

        # Estimate ve from tail behavior
        # Higher ve means slower washout
        ve_init = 0.2

        return ToftsParams(ktrans=ktrans_init, ve=ve_init)
