"""ASL Quality Control — Signal-to-Noise Ratio (SNR) metric.

This module computes voxel-wise and region-wise SNR for ASL cerebral blood
flow (CBF) maps. SNR is a fundamental quality indicator for ASL acquisitions:
low SNR may indicate motion artefacts, insufficient averaging, or hardware
issues.

The SNR is defined as the mean signal in a region of interest divided by the
standard deviation of signal in a noise-only region:

    SNR = mean(signal_ROI) / std(noise_ROI)

A voxel-wise variant divides each voxel's value by the global noise estimate,
producing an SNR map with the same shape as the input.

References
----------
.. [1] Alsop DC et al. (2015). Recommended implementation of arterial
   spin-labeled perfusion MRI for clinical applications: A consensus of the
   ISMRM Perfusion Study Group and the European Consortium for ASL in
   Dementia. Magn Reson Med 73(1):102-116. doi:10.1002/mrm.25197
.. [2] Xie J et al. (2023). Quality control for ASL MRI in multi-centre
   studies. NeuroImage 274:120136. doi:10.1016/j.neuroimage.2023.120136
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.backend.array_module import get_array_module
from osipy.common.exceptions import DataValidationError
from osipy.common.parameter_map import ParameterMap

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ASLSNRParams:
    """Parameters for ASL SNR computation.

    Attributes
    ----------
    min_noise_voxels : int
        Minimum number of noise-region voxels required for a reliable
        noise estimate. Raises ``DataValidationError`` if fewer are found.
        Default 10.
    snr_threshold : float
        Voxels with SNR below this value are flagged as low quality in the
        output quality mask. Default 2.0.
    """

    min_noise_voxels: int = 10
    snr_threshold: float = 2.0


@dataclass
class ASLSNRResult:
    """Result of ASL SNR quality-control computation.

    Attributes
    ----------
    snr_map : ParameterMap
        Voxel-wise SNR map (dimensionless), same spatial shape as input.
    noise_std : float
        Estimated noise standard deviation (a.u.) computed from
        ``noise_mask`` region.
    mean_snr : float
        Mean SNR over all brain voxels in the brain mask.
    quality_mask : NDArray[np.bool_]
        Boolean mask marking voxels with SNR >= ``params.snr_threshold``
        as good quality (``True``).
    """

    snr_map: ParameterMap
    noise_std: float
    mean_snr: float
    quality_mask: "NDArray[np.bool_]"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_snr(
    cbf_map: "NDArray[np.floating[Any]]",
    noise_mask: "NDArray[np.bool_]",
    brain_mask: "NDArray[np.bool_] | None" = None,
    params: ASLSNRParams | None = None,
) -> ASLSNRResult:
    """Compute voxel-wise SNR of an ASL CBF map.

    Estimates noise from a user-supplied background/noise region mask and
    divides every brain voxel by the noise standard deviation to produce a
    dimensionless SNR map.  Voxels below ``params.snr_threshold`` are
    flagged as low quality.

    Parameters
    ----------
    cbf_map : NDArray[np.floating]
        CBF map in mL/100g/min, shape (x, y, z).
    noise_mask : NDArray[np.bool_]
        Boolean mask selecting noise-only voxels (e.g., background air
        outside the head).  Must have the same shape as ``cbf_map``.
    brain_mask : NDArray[np.bool_] | None
        Boolean mask selecting brain voxels.  If ``None``, all voxels not
        in ``noise_mask`` are treated as brain.
    params : ASLSNRParams | None
        SNR computation parameters.  Uses defaults if ``None``.

    Returns
    -------
    ASLSNRResult
        SNR map, noise estimate, mean SNR, and quality mask.

    Raises
    ------
    DataValidationError
        If ``cbf_map`` and ``noise_mask`` shapes differ.
    DataValidationError
        If fewer than ``params.min_noise_voxels`` noise voxels are found.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.asl.qc import compute_snr, ASLSNRParams
    >>> cbf = np.random.rand(16, 16, 8) * 60.0   # mL/100g/min
    >>> noise = np.zeros((16, 16, 8), dtype=bool)
    >>> noise[0, :, :] = True                     # top slice = noise
    >>> result = compute_snr(cbf, noise)
    >>> print(result.snr_map.units)
    dimensionless

    References
    ----------
    .. [1] Alsop DC et al. (2015). MRM 73(1):102-116.
       doi:10.1002/mrm.25197
    .. [2] Xie J et al. (2023). NeuroImage 274:120136.
       doi:10.1016/j.neuroimage.2023.120136
    """
    params = params or ASLSNRParams()
    xp = get_array_module(cbf_map)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if cbf_map.shape != noise_mask.shape:
        msg = (
            f"cbf_map shape {cbf_map.shape} must match "
            f"noise_mask shape {noise_mask.shape}"
        )
        raise DataValidationError(msg)

    n_noise = int(xp.sum(noise_mask))
    if n_noise < params.min_noise_voxels:
        msg = (
            f"noise_mask contains only {n_noise} voxels; "
            f"at least {params.min_noise_voxels} are required for a "
            f"reliable noise estimate"
        )
        raise DataValidationError(msg)

    # ------------------------------------------------------------------
    # Noise estimation
    # ------------------------------------------------------------------
    noise_values = cbf_map[noise_mask]
    noise_std = float(xp.std(noise_values))

    logger.debug(
        "Noise estimate: std=%.4f from %d voxels",
        noise_std,
        n_noise,
    )

    # Guard against zero noise (e.g., synthetic data)
    if noise_std == 0.0:
        logger.warning(
            "Noise standard deviation is zero; SNR map will be inf/nan. "
            "Check that noise_mask selects real background voxels."
        )
        noise_std_safe = 1e-10
    else:
        noise_std_safe = noise_std

    # ------------------------------------------------------------------
    # Voxel-wise SNR map
    # ------------------------------------------------------------------
    snr = cbf_map / noise_std_safe

    # Zero out noise region in SNR map (not a brain measurement)
    snr = xp.where(noise_mask, xp.zeros_like(snr), snr)

    # ------------------------------------------------------------------
    # Brain mask
    # ------------------------------------------------------------------
    if brain_mask is None:
        brain_mask = ~noise_mask

    # ------------------------------------------------------------------
    # Quality mask — voxels meeting SNR threshold
    # ------------------------------------------------------------------
    quality_mask = brain_mask & (snr >= params.snr_threshold)

    # ------------------------------------------------------------------
    # Summary statistic
    # ------------------------------------------------------------------
    brain_snr_values = snr[brain_mask]
    mean_snr = float(xp.mean(brain_snr_values)) if brain_snr_values.size > 0 else 0.0

    logger.debug("Mean SNR over brain mask: %.2f", mean_snr)

    # ------------------------------------------------------------------
    # Package result
    # ------------------------------------------------------------------
    snr_map = ParameterMap(
        name="SNR",
        symbol="SNR",
        units="dimensionless",
        values=snr,
        affine=np.eye(4),
        quality_mask=quality_mask,
    )

    return ASLSNRResult(
        snr_map=snr_map,
        noise_std=noise_std,
        mean_snr=mean_snr,
        quality_mask=quality_mask,
    )
