"""DSC signal-to-concentration conversion.

This module converts DSC-MRI T2*-weighted signal to contrast agent
concentration via delta-R2* (OSIPI: Q.EL1.009), the change in transverse
relaxation rate.

The TE (OSIPI: Q.MS1.005) is required for the conversion from signal
intensity to delta-R2*.

GPU/CPU agnostic using the xp array module pattern.
NO scipy dependency - uses custom Levenberg-Marquardt implementation.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Rosen BR et al. (1990). Contrast agents and cerebral hemodynamics.
   Magn Reson Med 14(2):249-265.
.. [3] Ostergaard L et al. (1996). High resolution measurement of cerebral blood
   flow using intravascular tracer bolus passages. Magn Reson Med 36(5):715-725.
.. [4] Dickie BR et al. MRM 2024. doi:10.1002/mrm.30101
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from osipy.common.backend.array_module import get_array_module, to_numpy
from osipy.common.exceptions import DataValidationError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass
class DSCAcquisitionParams:
    """Acquisition parameters for DSC-MRI.

    Attributes
    ----------
    te : float
        Echo time TE (OSIPI: Q.MS1.005) in milliseconds.
    tr : float
        Repetition time (TR) in milliseconds.
    field_strength : float
        Main magnetic field strength in Tesla.
    r2_star : float
        T2* relaxivity of contrast agent (1/s/mM).
        Default is for Gd-DTPA at 1.5T.
    """

    te: float = 30.0  # ms
    tr: float = 1500.0  # ms
    field_strength: float = 1.5  # T
    r2_star: float = 32.0  # s⁻¹ mM⁻¹ at 1.5T


def signal_to_delta_r2(
    signal: "NDArray[np.floating[Any]]",
    te: float,
    baseline_frames: int = 10,
    baseline_indices: "NDArray[np.intp] | None" = None,
    baseline_end: int | None = None,
) -> "NDArray[np.floating[Any]]":
    """Convert DSC signal intensity to delta-R2* (OSIPI: Q.EL1.009).

    The conversion uses the relationship:
        S(t) = S0 * exp(-TE * dR2*(t))

    Therefore:
        dR2*(t) = -ln(S(t)/S0) / TE

    where TE is the echo time (OSIPI: Q.MS1.005).

    Parameters
    ----------
    signal : NDArray[np.floating]
        DSC signal intensity, shape (..., n_timepoints).
        Last dimension is time.
    te : float
        Echo time (OSIPI: Q.MS1.005) in milliseconds.
    baseline_frames : int
        Number of initial frames to use for baseline estimation.
        Ignored if baseline_indices or baseline_end is provided.
    baseline_indices : NDArray[np.intp] | None
        Specific indices to use for baseline calculation.
        Overrides baseline_frames if provided.
    baseline_end : int | None
        End index for baseline (uses frames 0 to baseline_end).
        Alias for baseline_frames. Overrides baseline_frames if provided.

    Returns
    -------
    NDArray[np.floating]
        Delta-R2* values in 1/s, same shape as input.

    Raises
    ------
    DataValidationError
        If signal contains invalid values.

    References
    ----------
    .. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.dsc.concentration import signal_to_delta_r2
    >>> signal = np.random.rand(64, 64, 20, 60) * 1000
    >>> delta_r2 = signal_to_delta_r2(signal, te=30.0)
    """
    # Handle baseline_end as alias for baseline_frames
    if baseline_end is not None:
        baseline_frames = baseline_end

    xp = get_array_module(signal)

    # Convert TE to seconds
    te_sec = te / 1000.0

    if te_sec <= 0:
        msg = f"TE must be positive, got {te}"
        raise DataValidationError(msg)

    # Determine baseline indices
    if baseline_indices is not None:
        bl_idx = baseline_indices
    else:
        n_timepoints = signal.shape[-1]
        baseline_frames = min(baseline_frames, n_timepoints // 4)
        bl_idx = xp.arange(baseline_frames)

    # Calculate baseline signal (mean over baseline frames)
    s0 = xp.mean(signal[..., bl_idx], axis=-1, keepdims=True)

    # Avoid division by zero or log of non-positive values
    s0 = xp.maximum(s0, 1e-10)
    signal_clipped = xp.maximum(signal, 1e-10)

    # Calculate ΔR2*: -ln(S/S0) / TE
    if hasattr(xp, "errstate"):
        with xp.errstate(divide="ignore", invalid="ignore"):
            delta_r2 = -xp.log(signal_clipped / s0) / te_sec
    else:
        delta_r2 = -xp.log(signal_clipped / s0) / te_sec

    # Handle invalid values
    delta_r2 = xp.nan_to_num(delta_r2, nan=0.0, posinf=0.0, neginf=0.0)

    return delta_r2


def delta_r2_to_concentration(
    delta_r2: "NDArray[np.floating[Any]]",
    r2_star: float = 32.0,
    field_strength: float = 1.5,
) -> "NDArray[np.floating[Any]]":
    """Convert ΔR2* to contrast agent concentration.

    Assumes a linear relationship:
        C(t) = ΔR2*(t) / r2*

    where r2* is the transverse relaxivity of the contrast agent.

    Parameters
    ----------
    delta_r2 : NDArray[np.floating]
        ΔR2* values in s⁻¹.
    r2_star : float
        T2* relaxivity in s⁻¹ mM⁻¹. Default is 32 s⁻¹ mM⁻¹
        (typical for Gd-DTPA at 1.5T).
    field_strength : float
        Field strength in Tesla. Used to scale relaxivity
        if significantly different from 1.5T.

    Returns
    -------
    NDArray[np.floating]
        Contrast agent concentration in mM.

    Notes
    -----
    The relaxivity scales approximately linearly with field strength
    for T2* effects:
        r2*(B) ≈ r2*(1.5T) × (B / 1.5)

    Examples
    --------
    >>> import numpy as np
    >>> delta_r2 = np.random.rand(64, 64, 20, 60) * 10
    >>> concentration = delta_r2_to_concentration(delta_r2, r2_star=32.0)
    """
    # Scale relaxivity for field strength
    # r2* scales approximately with B0
    r2_star_scaled = r2_star * (field_strength / 1.5)

    if r2_star_scaled <= 0:
        msg = f"r2* must be positive, got {r2_star_scaled}"
        raise DataValidationError(msg)

    # Convert to concentration
    concentration = delta_r2 / r2_star_scaled

    return concentration


def gamma_variate_fit(
    concentration: "NDArray[np.floating[Any]]",
    time: "NDArray[np.floating[Any]]",
    baseline_end: int | None = None,
) -> tuple["NDArray[np.floating[Any]]", dict[str, float]]:
    """Fit gamma-variate function for recirculation removal.

    The gamma-variate function models the first-pass bolus:
        C(t) = K × (t - t0)^α × exp(-(t - t0)/β)  for t > t0
        C(t) = 0  for t ≤ t0

    This allows separation of the first-pass bolus from recirculation.

    Uses the shared ``LevenbergMarquardtFitter`` via ``BoundGammaVariateModel``.

    Parameters
    ----------
    concentration : NDArray[np.floating]
        Concentration-time curve, shape (n_timepoints,).
    time : NDArray[np.floating]
        Time points in seconds.
    baseline_end : int | None
        Index of baseline end. If None, auto-detected.

    Returns
    -------
    tuple[NDArray[np.floating], dict[str, float]]
        First-pass concentration curve and fitted parameters dict.

    References
    ----------
    Thompson HK et al. (1964). Indicator Transit Time Considered as a
    Gamma Variate. Circ Res 14:502-515.

    Examples
    --------
    >>> import numpy as np
    >>> time = np.arange(60) * 1.5  # seconds
    >>> concentration = np.random.rand(60) * 5
    >>> first_pass, params = gamma_variate_fit(concentration, time)
    """
    from osipy.common.fitting.least_squares import LevenbergMarquardtFitter
    from osipy.dsc.concentration.gamma_model import BoundGammaVariateModel

    xp = get_array_module(concentration, time)

    # Auto-detect baseline end (bolus arrival)
    if baseline_end is None:
        threshold = xp.mean(concentration[:5]) + 2 * xp.std(concentration[:5])
        above_threshold = xp.where(concentration > threshold)[0]
        baseline_end = (
            int(to_numpy(above_threshold[0])) if len(above_threshold) > 0 else 5
        )

    # Find peak time for bounds
    peak_idx = int(to_numpy(xp.argmax(concentration)))
    peak_time = float(to_numpy(time[peak_idx]))

    try:
        # Create bound model and fit as single-voxel batch
        bound_model = BoundGammaVariateModel(time, peak_time=peak_time)
        fitter = LevenbergMarquardtFitter()

        # Shape for batch: (n_time, 1)
        observed = concentration[:, xp.newaxis]
        fitted_params, _r2, _converged = fitter.fit_batch(bound_model, observed)

        # Extract fitted parameters
        param_names = bound_model.parameters
        k_fit = float(to_numpy(fitted_params[param_names.index("k"), 0]))
        t0_fit = float(to_numpy(fitted_params[param_names.index("t0"), 0]))
        alpha_fit = float(to_numpy(fitted_params[param_names.index("alpha"), 0]))
        beta_fit = float(to_numpy(fitted_params[param_names.index("beta"), 0]))

        # Generate first-pass curve
        first_pass = bound_model.predict_array_batch(fitted_params, xp)[:, 0]

        params_dict = {
            "k": k_fit,
            "t0": t0_fit,
            "alpha": alpha_fit,
            "beta": beta_fit,
            "peak_time": t0_fit + alpha_fit * beta_fit,
            "mtt": alpha_fit * beta_fit,
        }

    except (RuntimeError, ValueError):
        # If fitting fails, return original curve
        first_pass = concentration.copy()
        peak_conc = float(to_numpy(concentration[peak_idx]))
        t0_init = float(to_numpy(time[baseline_end]))
        alpha_init = 3.0
        beta_init = (peak_time - t0_init) / alpha_init if peak_time > t0_init else 1.0
        params_dict = {
            "k": peak_conc,
            "t0": t0_init,
            "alpha": alpha_init,
            "beta": beta_init,
            "peak_time": peak_time,
            "mtt": alpha_init * beta_init,
            "fit_failed": True,
        }

    return to_numpy(first_pass), params_dict
