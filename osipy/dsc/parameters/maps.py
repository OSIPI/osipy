"""DSC perfusion parameter maps.

This module computes cerebral blood volume (CBV), cerebral blood flow (CBF),
mean transit time (MTT), and related perfusion parameters from DSC-MRI data.

Parameters follow OSIPI CAPLEX naming conventions:

- CBV: cerebral blood volume (mL/100g)
- CBF (Q.PH1.003): cerebral blood flow (mL/100g/min)
- MTT (Q.PH1.006): mean transit time (s)
- TTP (Q.CD1.010): time to peak (s)
- Tmax: time to maximum of the residue function (s)
- Delay / Ta (Q.PH1.007): arterial delay (s)

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Ostergaard L et al. (1996). High resolution measurement of cerebral blood
   flow using intravascular tracer bolus passages. Magn Reson Med 36(5):715-725.
.. [3] Calamante F et al. (2010). Measuring cerebral blood flow using magnetic
   resonance imaging techniques. J Cereb Blood Flow Metab 19(7):701-735.
.. [4] Dickie BR et al. MRM 2024. doi:10.1002/mrm.29840
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.backend.array_module import get_array_module, to_numpy
from osipy.common.exceptions import DataValidationError
from osipy.common.parameter_map import ParameterMap

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class DSCPerfusionMaps:
    """Container for DSC perfusion parameter maps.

    Attributes
    ----------
    cbv : ParameterMap
        Cerebral blood volume map (mL/100g).
    cbf : ParameterMap
        Cerebral blood flow map (mL/100g/min) (OSIPI: Q.PH1.003).
    mtt : ParameterMap
        Mean transit time map (s) (OSIPI: Q.PH1.006).
    ttp : ParameterMap | None
        Time to peak map (s) (OSIPI: Q.CD1.010), optional.
    tmax : ParameterMap | None
        Time to maximum of residue function (s), optional.
    delay : ParameterMap | None
        Arterial delay map (s), symbol Ta (OSIPI: Q.PH1.007), optional.
    quality_mask : NDArray[np.bool_]
        Mask indicating valid perfusion values.
    """

    cbv: ParameterMap
    cbf: ParameterMap
    mtt: ParameterMap
    ttp: ParameterMap | None = None
    tmax: ParameterMap | None = None
    delay: ParameterMap | None = None
    quality_mask: "NDArray[np.bool_] | None" = None


def compute_perfusion_maps(
    delta_r2: "NDArray[np.floating[Any]]",
    aif: "NDArray[np.floating[Any]]",
    time: "NDArray[np.floating[Any]]",
    mask: "NDArray[np.bool_] | None" = None,
    deconvolve: bool = True,
    deconvolution_method: str = "oSVD",
    svd_threshold: float = 0.2,
    density: float = 1.04,
    hematocrit_ratio: float = 0.73,
) -> DSCPerfusionMaps:
    """Compute comprehensive perfusion parameter maps.

    Computes CBV, CBF (OSIPI: Q.PH1.003), MTT (OSIPI: Q.PH1.006),
    TTP (OSIPI: Q.CD1.010), Tmax, and arterial delay Ta (OSIPI: Q.PH1.007)
    from DSC-MRI delta-R2* data.

    Parameters
    ----------
    delta_r2 : NDArray[np.floating]
        Delta-R2* data (OSIPI: Q.EL1.009), shape (..., n_timepoints).
    aif : NDArray[np.floating]
        Arterial input function (delta-R2*), shape (n_timepoints,).
    time : NDArray[np.floating]
        Time points in seconds.
    mask : NDArray[np.bool_] | None
        Brain mask.
    deconvolve : bool
        If True, use SVD deconvolution for CBF estimation.
        If False, estimate CBF from peak height.
    deconvolution_method : str
        Deconvolution method name from the deconvolver registry (default "oSVD").
    svd_threshold : float
        SVD truncation threshold for deconvolution.
    density : float
        Brain tissue density in g/ml (default 1.04).
    hematocrit_ratio : float
        Large vessel to small vessel hematocrit ratio (default 0.73).

    Returns
    -------
    DSCPerfusionMaps
        Perfusion parameter maps.

    References
    ----------
    .. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    .. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.29840

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.dsc.parameters import compute_perfusion_maps
    >>> delta_r2 = np.random.rand(64, 64, 20, 60) * 10
    >>> aif = np.random.rand(60) * 20
    >>> time = np.linspace(0, 90, 60)
    >>> maps = compute_perfusion_maps(delta_r2, aif, time)
    """
    xp = get_array_module(delta_r2, aif, time)
    spatial_shape = delta_r2.shape[:-1]
    n_timepoints = delta_r2.shape[-1]

    if mask is None:
        mask = xp.ones(spatial_shape, dtype=bool)

    # Compute CBV
    cbv_data = compute_cbv(delta_r2, aif, time, hematocrit_ratio)

    # Compute CBF and MTT
    if deconvolve:
        from osipy.common.fitting.registry import FITTER_REGISTRY

        if deconvolution_method in FITTER_REGISTRY:
            # New fitter-based path
            from osipy.dsc.deconvolution.signal_model import (
                BoundDSCModel,
                DSCConvolutionModel,
            )

            dsc_model = DSCConvolutionModel()
            matrix_type = "toeplitz" if deconvolution_method == "sSVD" else "circulant"
            bound = BoundDSCModel(dsc_model, aif, time, matrix_type=matrix_type)
            fitter = FITTER_REGISTRY[deconvolution_method]()

            # Reshape to 4D for fit_image: (x, y, z, t)
            original_shape = delta_r2.shape
            spatial_shape_4d = original_shape[:-1]
            # Ensure at least 3 spatial dims for fit_image
            if len(spatial_shape_4d) < 3:
                padded = list(spatial_shape_4d) + [1] * (3 - len(spatial_shape_4d))
                data_4d = delta_r2.reshape(*padded, n_timepoints)
                mask_3d = mask.reshape(*padded) if mask is not None else None
            else:
                data_4d = delta_r2
                mask_3d = mask

            result_maps = fitter.fit_image(bound, data_4d, mask=mask_3d)

            # Extract arrays from ParameterMap results
            cbf_data = xp.asarray(result_maps["CBF"].values)
            mtt_data = xp.asarray(result_maps["MTT"].values)
            delay_data = xp.asarray(result_maps["Ta"].values)

            # Reshape back to original spatial shape
            if len(spatial_shape_4d) < 3:
                cbf_data = cbf_data.reshape(spatial_shape)
                mtt_data = mtt_data.reshape(spatial_shape)
                delay_data = delay_data.reshape(spatial_shape)

            tmax_data = delay_data
        else:
            # Legacy deconvolver path
            from osipy.dsc.deconvolution import get_deconvolver

            deconvolver = get_deconvolver(deconvolution_method)
            deconv_result = deconvolver.deconvolve(delta_r2, aif, time, mask=mask)

            cbf_data = deconv_result.cbf
            mtt_data = deconv_result.mtt
            delay_data = deconv_result.delay
            tmax_data = deconv_result.delay  # Tmax from residue function peak
    else:
        # Simple estimation without deconvolution
        cbf_data = _estimate_cbf_simple(delta_r2, aif, mask)
        mtt_data = compute_mtt(cbv_data, cbf_data)
        delay_data = None
        tmax_data = None

    # Time to peak (from raw signal)
    ttp_data = _compute_time_to_peak(delta_r2, time, mask)

    # Create default affine if not provided
    affine = np.eye(4)

    # Create parameter maps
    cbv_map = ParameterMap(
        values=to_numpy(cbv_data * 100 / density),  # Convert to ml/100g
        name="CBV",
        symbol="CBV",
        units="ml/100g",
        affine=affine,
        quality_mask=to_numpy(mask),
    )

    # CBF in ml/100g/min
    # Scaling factor accounts for density and time units
    cbf_scale = 60.0 / density * 100  # Convert to ml/100g/min
    cbf_map = ParameterMap(
        values=to_numpy(cbf_data * cbf_scale),
        name="CBF",
        symbol="CBF",
        units="ml/100g/min",
        affine=affine,
        quality_mask=to_numpy(mask),
    )

    mtt_map = ParameterMap(
        values=to_numpy(mtt_data),
        name="MTT",
        symbol="MTT",
        units="s",
        affine=affine,
        quality_mask=to_numpy(mask),
    )

    ttp_map = ParameterMap(
        values=to_numpy(ttp_data),
        name="TTP",
        symbol="TTP",
        units="s",
        affine=affine,
        quality_mask=to_numpy(mask),
    )

    tmax_map = None
    if tmax_data is not None:
        tmax_map = ParameterMap(
            values=to_numpy(tmax_data),
            name="Tmax",
            symbol="Tmax",
            units="s",
            affine=affine,
            quality_mask=to_numpy(mask),
        )

    delay_map = None
    if delay_data is not None:
        delay_map = ParameterMap(
            values=to_numpy(delay_data),
            name="Delay",
            symbol="Ta",
            units="s",
            affine=affine,
            quality_mask=to_numpy(mask),
        )

    # Quality mask: valid CBF and CBV
    quality = mask & (cbf_data > 0) & (cbv_data > 0) & xp.isfinite(mtt_data)

    return DSCPerfusionMaps(
        cbv=cbv_map,
        cbf=cbf_map,
        mtt=mtt_map,
        ttp=ttp_map,
        tmax=tmax_map,
        delay=delay_map,
        quality_mask=to_numpy(quality),
    )


def compute_cbv(
    delta_r2: "NDArray[np.floating[Any]]",
    aif: "NDArray[np.floating[Any]]",
    time: "NDArray[np.floating[Any]]",
    hematocrit_ratio: float = 0.73,
) -> "NDArray[np.floating[Any]]":
    """Compute cerebral blood volume from delta-R2* curves.

    CBV is proportional to the ratio of the integrals of tissue
    and arterial delta-R2* (OSIPI: Q.EL1.009) curves:

        CBV = (kH / rho) * integral(dR2*_tissue) / integral(dR2*_AIF)

    where kH is the hematocrit correction factor.

    Parameters
    ----------
    delta_r2 : NDArray[np.floating]
        Tissue delta-R2* data, shape (..., n_timepoints).
    aif : NDArray[np.floating]
        Arterial delta-R2*, shape (n_timepoints,).
    time : NDArray[np.floating]
        Time points in seconds.
    hematocrit_ratio : float
        Large to small vessel hematocrit ratio (default 0.73).

    Returns
    -------
    NDArray[np.floating]
        CBV map (fractional blood volume).

    References
    ----------
    .. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    """
    xp = get_array_module(delta_r2, aif, time)

    # Integrate AIF
    aif_integral = xp.trapezoid(aif, time)

    if aif_integral <= 0:
        msg = "AIF integral is non-positive; cannot compute CBV"
        raise DataValidationError(msg)

    # Integrate tissue curves along time axis
    tissue_integral = xp.trapezoid(delta_r2, time, axis=-1)

    # CBV = kH × tissue_integral / aif_integral
    cbv = hematocrit_ratio * tissue_integral / aif_integral

    # Ensure non-negative
    cbv = xp.maximum(cbv, 0)

    return cbv


def compute_mtt(
    cbv: "NDArray[np.floating[Any]]",
    cbf: "NDArray[np.floating[Any]]",
) -> "NDArray[np.floating[Any]]":
    """Compute mean transit time from CBV and CBF.

    Uses the central volume principle (OSIPI: Q.PH1.006):
        MTT = CBV / CBF

    Parameters
    ----------
    cbv : NDArray[np.floating]
        Cerebral blood volume map.
    cbf : NDArray[np.floating]
        Cerebral blood flow map (OSIPI: Q.PH1.003).

    Returns
    -------
    NDArray[np.floating]
        MTT map in same time units as CBV/CBF ratio.

    References
    ----------
    .. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    """
    xp = get_array_module(cbv, cbf)

    if hasattr(xp, "errstate"):
        with xp.errstate(divide="ignore", invalid="ignore"):
            mtt = cbv / cbf
    else:
        mtt = cbv / cbf

    # Handle invalid values
    mtt = xp.nan_to_num(mtt, nan=0.0, posinf=0.0, neginf=0.0)

    return mtt


def _estimate_cbf_simple(
    delta_r2: "NDArray[np.floating[Any]]",
    aif: "NDArray[np.floating[Any]]",
    mask: "NDArray[np.bool_]",
) -> "NDArray[np.floating[Any]]":
    """Simple CBF estimation from peak height ratio.

    Parameters
    ----------
    delta_r2 : NDArray
        Tissue ΔR2* data.
    aif : NDArray
        Arterial ΔR2*.
    mask : NDArray
        Brain mask.

    Returns
    -------
    NDArray
        CBF estimate (arbitrary units).
    """
    xp = get_array_module(delta_r2, aif, mask)
    spatial_shape = delta_r2.shape[:-1]

    # Peak AIF
    aif_peak = xp.max(aif)
    if aif_peak <= 0:
        return xp.zeros(spatial_shape)

    # Peak tissue ΔR2*
    tissue_peak = xp.max(delta_r2, axis=-1)

    # Simple CBF estimate
    cbf = tissue_peak / aif_peak

    # Apply mask
    cbf = xp.where(mask, cbf, 0)

    return cbf


def _compute_time_to_peak(
    delta_r2: "NDArray[np.floating[Any]]",
    time: "NDArray[np.floating[Any]]",
    mask: "NDArray[np.bool_]",
) -> "NDArray[np.floating[Any]]":
    """Compute time to peak of delta-R2* curve (OSIPI: Q.CD1.010).

    Parameters
    ----------
    delta_r2 : NDArray
        Delta-R2* data, shape (..., n_timepoints).
    time : NDArray
        Time points.
    mask : NDArray
        Brain mask.

    Returns
    -------
    NDArray
        Time to peak map in seconds.
    """
    xp = get_array_module(delta_r2, time, mask)

    # Find index of maximum along time axis
    peak_indices = xp.argmax(delta_r2, axis=-1)

    # Convert to time
    ttp = time[peak_indices]

    # Apply mask
    ttp = xp.where(mask, ttp, 0)

    return ttp
