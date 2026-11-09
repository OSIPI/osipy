"""
Code to calculate IVIM maps

Segmented fitting approach
First fit D using a WLLS

"""

import logging

import numpy as np
import scipy
import statsmodels.api as sm

from osipy.common.backend.array_module import get_array_module, to_numpy
from osipy.common.fitting import LevenbergMarquardtFitter
from osipy.common.fitting.batch import create_empty_maps, create_parameter_maps
from osipy.ivim.models.binding import BoundIVIMModel
from osipy.ivim.models.registry import get_ivim_model

logger = logging.getLogger(__name__)


def ivim_biexp(bvalues, D, f, Dp, S0=1):
    return (S0 * (f * np.exp(-bvalues * Dp) + (1 - f) * np.exp(-bvalues * D)))


def fit_segmented_IVIM_wrapper(
    signal, b_values, mask=None, params=None, use_gpu=False, progress_callback=None
):
    """Fit IVIM per voxel with the TF2.4 segmented WLS reference algorithm.

    Matches the registered IVIM fitter signature and returns the same
    ``dict[str, ParameterMap]`` (S0, D, D*, f, r_squared) as the other fitters.
    ``use_gpu`` is ignored: the reference implementation is CPU only.
    """
    xp = get_array_module(signal)
    signal = np.asarray(to_numpy(signal), dtype=np.float64)
    b_values = np.asarray(to_numpy(b_values), dtype=np.float64)
    mask = np.asarray(to_numpy(mask), dtype=bool) if mask is not None else np.ones(signal.shape[:-1], dtype=bool)

    # Normalise to (nx, ny, nz, n_b) / (nx, ny, nz), as the other fitters do.
    if signal.ndim == 3:
        signal = signal[:, :, np.newaxis, :]
        mask = mask[:, :, np.newaxis]
    elif signal.ndim == 2:
        signal = signal[:, np.newaxis, np.newaxis, :]
        mask = mask[:, np.newaxis, np.newaxis]
    shape = mask.shape

    model = get_ivim_model("biexponential")
    voxel_indices = np.argwhere(mask)
    n_voxels = len(voxel_indices)
    if n_voxels == 0:
        return create_empty_maps(model, shape)

    names = model.parameters  # ["S0", "D", "D*", "f"]
    fitted = np.zeros((len(names), n_voxels))
    r2 = np.zeros(n_voxels)
    converged = np.zeros(n_voxels, dtype=bool)

    for i, (x, y, z) in enumerate(voxel_indices):
        voxel_signal = signal[x, y, z, :]
        try:
            D, f, Dp = segmented_IVIM_fit(b_values, voxel_signal)
        except Exception:  # noqa: BLE001 - leave failed voxels at zero
            logger.debug("Segmented WLS fit failed for voxel %s", (x, y, z))
            continue
        S0 = voxel_signal[b_values == 0].item()
        fitted[names.index("S0"), i] = S0
        fitted[names.index("D"), i] = D
        fitted[names.index("D*"), i] = Dp
        fitted[names.index("f"), i] = f

        pred = ivim_biexp(b_values, D, f, Dp, S0)
        ss_res = np.sum((voxel_signal - pred) ** 2)
        ss_tot = np.sum((voxel_signal - voxel_signal.mean()) ** 2)
        r2[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        converged[i] = np.all(np.isfinite([D, f, Dp]))

        if progress_callback is not None and i % 100 == 0:
            progress_callback((i + 1) / n_voxels)

    if progress_callback is not None:
        progress_callback(1.0)

    return create_parameter_maps(
        model,
        fitted,
        r2,
        converged,
        voxel_indices,
        shape,
        fitting_method="segmented_wls",
    )


def segmented_IVIM_fit(bvalues, dw_data, b_cutoff=200, bounds=([0.0001, 0.0, 0.001], [0.004, 0.7, 0.01])):
    """
        A segmented fitting implementation for a bi-exponential model.
        First D is fitted using a mono exponential model on all signal above the bvalue cutoff using an iterative WLLSVertrekpassage, Schiphol
        Then f is fitted by using the b=0 intercept from the mono expontential fit and substracting this from the measured signal at b=0
        Then D* is fitted using a bi-exponential model with fixed D and f

        """
    # D and f (WLS) are computed in SegmentedBoundIVIMModel.get_initial_guess_batch
    # and frozen there; the shared LM fitter only fits D*.
    bound_model = SegmentedBoundIVIMModel(
        get_ivim_model("biexponential"), bvalues, b_threshold=b_cutoff
    )
    params, _r2, _converged = LevenbergMarquardtFitter().fit_batch(
        bound_model,
        np.asarray(dw_data, dtype=np.float64)[:, np.newaxis],  # (n_b, 1 voxel)
        bounds_override={"D*": (bounds[0][2], bounds[1][2])},
    )

    names = bound_model.parameters  # ["S0", "D", "D*", "f"]
    D = params[names.index("D"), 0]
    f = params[names.index("f"), 0]
    Dp = params[names.index("D*"), 0]

    return D, f, Dp


def d_fit_iterative_wls(bvalues_D, log_signal, max_iter=50):
    """
    Function to calculate D using an iterative wlls on the log(signal)
    weights for the wlls are initialized from the predicted signal of a lls as described in
    http://dx.doi.org/10.1016/j.neuroimage.2013.05.028 equation (7)

    Parameters:
    log_signal: the log() of the signal above the threshold for segmented fitting

    bvalues_D: all bvalues above the threshold for fitting D

    max_iter: the maximum number of iterations for WLS

    """

    bvals = sm.add_constant(-bvalues_D)
    # First do a LLS to initialize the weights
    beta_lls = np.linalg.lstsq(bvals, log_signal, rcond=None)[0]
    # initialize the weights based on the predicted signals W=diag(exp(2*X*Beta_lls))
    init_weights = np.exp(2 * bvals @ beta_lls)
    weights = init_weights

    for i in range(max_iter):
        # Weighted linear regression
        model = sm.WLS(log_signal, bvals, weights=weights)
        results = model.fit()
        # The weights for the next iteration are based on the predicted signal of this iteration
        new_weights = np.exp(2 * bvals @ results.params)
        weights = new_weights

    ln_b0_intercept, D = results.params
    b0_intercept = np.exp(ln_b0_intercept)
    return D, b0_intercept


# Reference bounds used by the segmented fit, keyed by parameter name.
SEGMENTED_BOUNDS = {"D": (1e-4, 4e-3), "f": (0.0, 0.7), "D*": (1e-3, 1e-2)}


class SegmentedBoundIVIMModel(BoundIVIMModel):
    """Step 2 of the segmented fit: only D* is free, S0/D/f are frozen.

    S0, D and f are estimated per voxel in ``get_initial_guess_batch`` (the
    WLS step) and then kept constant by giving them zero Jacobian columns, so
    the shared LM fitter never moves them. The "step 1" fit therefore runs as
    the initial guess of the LM fit.
    """

    _FROZEN = ("S0", "D", "f")


    def get_initial_guess_batch(self, observed_batch, xp):
        names = self._ivim_model.parameters
        b = np.asarray(to_numpy(self._b_values))
        obs = np.asarray(to_numpy(observed_batch), dtype=np.float64)
        _n_b, n_vox = obs.shape

        high = b >= self._b_threshold
        b0 = int(np.argmin(b))

        guess = np.zeros((len(names), n_vox))
        for v in range(n_vox):
            s = obs[:, v]
            s0 = s[b0]
            try:
                log_s = np.log(np.clip(s[high], 1e-6, None))
                d, intercept = d_fit_iterative_wls(b[high], log_s)
                d = np.clip(d, *SEGMENTED_BOUNDS["D"])
                f = np.clip((s0 - intercept) / s0, *SEGMENTED_BOUNDS["f"])
            except Exception:  # noqa: BLE001 - fall back to a sane guess
                d, f = 1e-3, 0.1
            guess[names.index("S0"), v] = s0
            guess[names.index("D"), v] = d
            guess[names.index("f"), v] = f
            guess[names.index("D*"), v] = np.clip(d * 10, *SEGMENTED_BOUNDS["D*"])

        return xp.asarray(guess)

    def compute_jacobian_batch(self, params_batch, predicted, xp):
        jac = super().compute_jacobian_batch(params_batch, predicted, xp)
        # jac: (n_free, n_b, n_vox). Zero the frozen parameters' columns.
        for name in self._FROZEN:
            jac[self._free_params.index(name)] = 0.0
        return jac
