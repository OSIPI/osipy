"""Patlak pharmacokinetic model (OSIPI: M.IC1.006).

This module implements the Patlak graphical analysis model for DCE-MRI,
which assumes unidirectional transfer from plasma to tissue.

GPU/CPU agnostic using the xp array module pattern.
NO scipy dependency - uses xp.linalg operations.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.29840
.. [3] Patlak CS et al. J Cereb Blood Flow Metab 1983;3(1):1-7.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.backend.array_module import get_array_module, to_numpy
from osipy.dce.models.base import BasePerfusionModel, ModelParameters
from osipy.dce.models.registry import register_model

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class PatlakParams(ModelParameters):
    """Parameters for Patlak model (OSIPI: M.IC1.006).

    Attributes
    ----------
    ktrans : float
        Volume transfer constant (OSIPI: Q.PH1.008) in 1/min.
        Represents unidirectional influx rate.
    vp : float
        Plasma volume fraction (OSIPI: Q.PH1.001), unitless fraction [0, 1].
    """

    ktrans: float = 0.1
    vp: float = 0.02


def _cumulative_aif_integral(
    aif: "NDArray[np.floating[Any]]",
    t_min: "NDArray[np.floating[Any]]",
    xp: Any,
) -> "NDArray[np.floating[Any]]":
    """Compute cumulative integral of AIF using trapezoidal rule.

    Uses O(n) cumsum instead of O(n²) per-timepoint integration.

    Parameters
    ----------
    aif : NDArray[np.floating]
        Arterial input function.
    t_min : NDArray[np.floating]
        Time points in minutes.
    xp : module
        Array module (numpy or cupy).

    Returns
    -------
    NDArray[np.floating]
        Cumulative integral of AIF at each time point.
    """
    t_flat = t_min.ravel()  # time is always 1D logically
    dt = t_flat[1:] - t_flat[:-1]
    avg_aif = (aif[:-1] + aif[1:]) / 2.0
    # dt is 1D; if aif is 2D (n_time, n_voxels), reshape dt for broadcasting
    if aif.ndim > 1:
        dt = dt[:, xp.newaxis]
    aif_integral = xp.zeros_like(aif)
    aif_integral[1:] = xp.cumsum(avg_aif * dt, axis=0)
    return aif_integral


@register_model("patlak")
class PatlakModel(BasePerfusionModel[PatlakParams]):
    """Patlak graphical analysis model (OSIPI: M.IC1.006).

    Implements the Patlak model for indicator concentration in tissue
    assuming unidirectional transfer from blood to tissue.

    Attributes
    ----------
    Ktrans : float
        Volume transfer constant (OSIPI: Q.PH1.008), 1/min.
    vp : float
        Plasma volume fraction (OSIPI: Q.PH1.001), unitless fraction [0, 1].

    Notes
    -----
    Model equation:
        Ct(t) = vp*Cp(t) + Ktrans * integral_0^t Cp(tau) dtau

    Assumptions:
        - No backflux from tissue to plasma (kep = 0)
        - Valid for early time points only
        - Linear accumulation over time

    GPU/CPU agnostic - operates on same device as input arrays.

    References
    ----------
    .. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    .. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.29840
    .. [3] Patlak CS et al. J Cereb Blood Flow Metab 1983;3(1):1-7.
    """

    @property
    def time_unit(self) -> str:
        """Return time unit (minutes for Patlak)."""
        return "minutes"

    @property
    def name(self) -> str:
        """Return model name."""
        return "Patlak"

    @property
    def parameters(self) -> list[str]:
        """Return parameter names."""
        return ["Ktrans", "vp"]

    @property
    def parameter_units(self) -> dict[str, str]:
        """Return parameter units."""
        return {"Ktrans": "1/min", "vp": "fraction"}

    @property
    def reference(self) -> str:
        """Return literature citation."""
        return "Patlak CS et al. (1983). J Cereb Blood Flow Metab 3(1):1-7."

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
            ``[Ktrans, vp]`` — shape ``(2,)`` or ``(2, n_voxels)``.
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        NDArray[np.floating]
            Predicted tissue concentration.
        """
        ktrans = params[0]
        vp = params[1]

        # Convert time to model units (minutes)
        t_min = self._convert_time(t, xp)

        # Vascular contribution (broadcasting handles scalar and array)
        ct_vascular = vp * aif

        # Cumulative integral of AIF (handles 1-D and 2-D)
        aif_integral = _cumulative_aif_integral(aif, t_min, xp)

        return ct_vascular + ktrans * aif_integral

    def get_initial_guess_batch(
        self,
        ct_batch: "NDArray[np.floating[Any]]",
        aif: "NDArray[np.floating[Any]]",
        t: "NDArray[np.floating[Any]]",
        xp: Any,
    ) -> "NDArray[np.floating[Any]]":
        """Compute vectorized initial guesses using Patlak graphical analysis.

        Falls back to the base class per-voxel loop since the graphical
        analysis involves per-voxel valid-point masks that are difficult
        to vectorize efficiently.

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
        # Use base class per-voxel loop for correctness
        return super().get_initial_guess_batch(ct_batch, aif, t, xp)

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Return physiological parameter bounds."""
        return {
            "Ktrans": (0.0, 5.0),
            "vp": (0.0, 1.0),
        }

    def get_initial_guess(
        self,
        ct: "NDArray[np.floating[Any]]",
        aif: "NDArray[np.floating[Any]]",
        t: "NDArray[np.floating[Any]]",
    ) -> PatlakParams:
        """Compute initial parameter guess using graphical analysis.

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
        PatlakParams
            Initial parameter estimates.
        """
        xp = get_array_module(ct, aif, t)

        t_min = self._convert_time(t, xp)

        # Vectorized cumulative integral of AIF
        aif_integral = _cumulative_aif_integral(aif, t_min, xp)

        # Patlak y-axis: Ct(t) / Cp(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = aif_integral / aif
            y = ct / aif

        # Find linear region (typically after peak)
        peak_idx = int(to_numpy(xp.argmax(aif)))
        start_idx = min(peak_idx + 2, len(t) - 5)

        if start_idx < len(t) - 2:
            x_linear = x[start_idx:]
            y_linear = y[start_idx:]

            # Remove invalid points
            valid = xp.isfinite(x_linear) & xp.isfinite(y_linear)
            if int(to_numpy(xp.sum(valid))) > 2:
                x_valid = x_linear[valid]
                y_valid = y_linear[valid]

                # Linear fit: y = Ktrans * x + vp (using lstsq instead of polyfit)
                A = xp.stack([x_valid, xp.ones_like(x_valid)], axis=1)
                result = xp.linalg.lstsq(A, y_valid, rcond=None)
                coeffs = result[0]
                ktrans_init = float(to_numpy(coeffs[0]))
                vp_init = float(to_numpy(coeffs[1]))

                return PatlakParams(
                    ktrans=float(np.clip(ktrans_init, 0.01, 1.0)),
                    vp=float(np.clip(vp_init, 0.001, 0.1)),
                )

        return PatlakParams(ktrans=0.1, vp=0.02)
