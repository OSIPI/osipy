"""Two-Compartment Exchange Model (2CXM) (OSIPI: M.IC1.009).

This module implements the Two-Compartment Exchange Model for DCE-MRI,
which provides a more complete description of tracer kinetics with
separate plasma and EES compartments.

GPU/CPU agnostic using the xp array module pattern.
NO scipy dependency - uses xp.linalg operations.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.30101
.. [3] Sourbron SP, Buckley DL. MRM 2011;66(3):735-745.
.. [4] Brix G et al. MRM 2004;52(2):420-429.
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
class TwoCompartmentParams(ModelParameters):
    """Parameters for Two-Compartment Exchange Model (OSIPI: M.IC1.009).

    Attributes
    ----------
    fp : float
        Plasma flow (OSIPI: Q.PH1.002) in mL/min/100mL.
    ps : float
        Permeability-surface area product (OSIPI: Q.PH1.004) in mL/min/100mL.
    ve : float
        Extravascular extracellular volume fraction (OSIPI: Q.PH1.001),
        mL/100mL.
    vp : float
        Plasma volume fraction (OSIPI: Q.PH1.001), mL/100mL.
    """

    fp: float = 15.0  # ml/100ml/min
    ps: float = 8.0  # ml/100ml/min
    ve: float = 0.2
    vp: float = 0.02


@register_model("2cxm")
class TwoCompartmentModel(BasePerfusionModel[TwoCompartmentParams]):
    """Two-Compartment Exchange Model (OSIPI: M.IC1.009).

    Implements the 2CXM with explicit plasma and EES compartments
    connected by bidirectional exchange.

    Attributes
    ----------
    Fp : float
        Plasma flow (OSIPI: Q.PH1.002), mL/min/100mL.
    PS : float
        Permeability-surface area product (OSIPI: Q.PH1.004), mL/min/100mL.
    ve : float
        Extravascular extracellular volume fraction (OSIPI: Q.PH1.001),
        mL/100mL.
    vp : float
        Plasma volume fraction (OSIPI: Q.PH1.001), mL/100mL.

    Notes
    -----
    The tissue concentration is the bi-exponential convolution:
        Ct(t) = Fp * [A1*exp(λ1*t) + A2*exp(λ2*t)] ⊛ Ca(t)

    where λ1, λ2 are eigenvalues of the rate matrix and the
    amplitudes A_i = Fp*(λ_i + β)/(λ_i - λ_j) with β = PS/ve + PS/vp
    account for tracer in both plasma (vp*Cp) and EES (ve*Ce).

    Derived from the coupled ODEs:
        vp*dCp/dt = Fp*(Ca - Cp) - PS*(Cp - Ce)
        ve*dCe/dt = PS*(Cp - Ce)

    where:
        - Cp, Ce are plasma and EES concentrations
        - Ca is arterial concentration (AIF)
        - Fp is plasma flow (OSIPI: Q.PH1.002)
        - PS is permeability-surface area product (OSIPI: Q.PH1.004)
        - vp, ve are volume fractions (OSIPI: Q.PH1.001)

    GPU/CPU agnostic - operates on same device as input arrays.

    References
    ----------
    .. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    .. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.30101
    .. [3] Sourbron SP, Buckley DL. MRM 2011;66(3):735-745.
    """

    @property
    def time_unit(self) -> str:
        """Return time unit (minutes for 2CXM)."""
        return "minutes"

    @property
    def name(self) -> str:
        """Return model name."""
        return "Two-Compartment Exchange (2CXM)"

    @property
    def parameters(self) -> list[str]:
        """Return parameter names."""
        return ["Fp", "PS", "ve", "vp"]

    @property
    def parameter_units(self) -> dict[str, str]:
        """Return parameter units."""
        return {
            "Fp": "mL/min/100mL",
            "PS": "mL/min/100mL",
            "ve": "mL/100mL",
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
            ``[Fp, PS, ve, vp]`` — shape ``(4,)`` or ``(4, n_voxels)``.
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        NDArray[np.floating]
            Predicted tissue concentration.
        """
        fp = params[0]
        ps = params[1]
        ve = params[2]
        vp = params[3]

        # Convert time to model units (minutes)
        t_min = self._convert_time(t, xp)

        # Convert from ml/100ml/min to ml/ml/min
        fp_frac = fp / 100.0
        ps_frac = ps / 100.0

        # Rate constants
        k_pe = ps_frac / vp
        k_ep = ps_frac / ve
        k_in = fp_frac / vp

        # Eigenvalue computation
        a = k_in + k_pe
        d = k_ep

        trace = -(a + d)
        det = a * d - k_ep * k_pe

        discriminant = trace**2 - 4 * det
        sqrt_disc = xp.sqrt(xp.maximum(discriminant, xp.asarray(0.0)))

        lambda1 = (trace - sqrt_disc) / 2
        lambda2 = (trace + sqrt_disc) / 2

        # Amplitudes
        beta = k_ep + k_pe
        denom = lambda1 - lambda2
        denom = xp.where(xp.abs(denom) < 1e-10, xp.asarray(1e-10), denom)
        A1 = fp_frac * (lambda1 + beta) / denom
        A2 = fp_frac * (lambda2 + beta) / (-denom)

        # Exponential convolution (handles scalar and array T)
        from osipy.common.convolution.expconv import expconv

        T1 = xp.where(xp.abs(lambda1) > 1e-15, -1.0 / lambda1, xp.asarray(1e15))
        T2 = xp.where(xp.abs(lambda2) > 1e-15, -1.0 / lambda2, xp.asarray(1e15))

        conv1 = expconv(aif, T1, t_min)
        conv2 = expconv(aif, T2, t_min)

        return A1 * conv1 + A2 * conv2

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
            Initial parameter guesses, shape (4, n_voxels).
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
        vp_batch = xp.clip(vp_batch, 0.01, 0.2)

        fp_batch = xp.full(n_voxels, 20.0, dtype=ct_batch.dtype)
        ps_batch = xp.full(n_voxels, 8.0, dtype=ct_batch.dtype)
        ve_batch = xp.full(n_voxels, 0.2, dtype=ct_batch.dtype)

        return xp.stack([fp_batch, ps_batch, ve_batch, vp_batch])

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Return physiological parameter bounds."""
        return {
            "Fp": (0.1, 200.0),  # ml/100ml/min
            "PS": (0.0, 50.0),  # ml/100ml/min
            "ve": (0.001, 1.0),
            "vp": (0.001, 0.3),
        }

    def get_initial_guess(
        self,
        ct: "NDArray[np.floating[Any]]",
        aif: "NDArray[np.floating[Any]]",
        t: "NDArray[np.floating[Any]]",
    ) -> TwoCompartmentParams:
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
        TwoCompartmentParams
            Initial parameter estimates.
        """
        xp = get_array_module(ct, aif)

        # Estimate vp from first-pass
        aif_peak_idx = int(xp.argmax(aif))
        aif_peak = float(to_numpy(aif[aif_peak_idx]))

        if aif_peak > 0:
            ct_at_peak = float(to_numpy(ct[aif_peak_idx]))
            vp_init = ct_at_peak / aif_peak
            vp_init = max(0.01, min(vp_init, 0.2))
        else:
            vp_init = 0.02

        return TwoCompartmentParams(
            fp=20.0,
            ps=8.0,
            ve=0.2,
            vp=float(vp_init),
        )
