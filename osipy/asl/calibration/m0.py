"""M0 calibration for ASL quantification.

This module implements M0 calibration methods for absolute CBF quantification
in ASL imaging. The equilibrium magnetization (M0) is reported in arbitrary
units (a.u.) per the OSIPI ASL Lexicon.

GPU/CPU agnostic using the xp array module pattern.
NO scipy dependency - uses custom morphological operations.

References
----------
.. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
.. [2] Alsop DC et al. (2015). Recommended implementation of arterial
   spin-labeled perfusion MRI for clinical applications.
   Magn Reson Med 73(1):102-116.
.. [3] Chappell MA et al. (2009). Separation of macrovascular signal in
   multi-inversion time arterial spin labeling MRI.
   Magn Reson Med 63(5):1357-1365.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.asl.calibration.base import BaseM0Calibration
from osipy.asl.calibration.registry import get_m0_calibration, register_m0_calibration
from osipy.common.backend.array_module import get_array_module, to_numpy
from osipy.common.exceptions import DataValidationError

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class M0CalibrationParams:
    """Parameters for M0 calibration.

    Attributes
    ----------
    method : str
        Calibration method: 'single', 'voxelwise', or 'reference_region'.
    reference_region : str
        Reference region for calibration: 'csf', 'white_matter', or 'custom'.
    t1_tissue : float
        T1 of calibration tissue in milliseconds.
    t2_star_tissue : float | None
        T2* of calibration tissue in milliseconds (for GRE).
    tr_m0 : float
        TR of M0 acquisition in milliseconds.
    te_m0 : float
        TE of M0 acquisition in milliseconds.
    """

    method: str = "single"
    reference_region: str = "csf"
    t1_tissue: float = 1330.0  # ms (gray matter at 3T)
    t2_star_tissue: float | None = None
    tr_m0: float = 6000.0  # ms (long TR for full recovery)
    te_m0: float = 13.0  # ms


# ---------------------------------------------------------------------------
# Internal helpers (used by strategy classes)
# ---------------------------------------------------------------------------


def _correct_m0_t1_recovery(
    m0: "NDArray[np.floating[Any]]",
    params: M0CalibrationParams,
) -> "NDArray[np.floating[Any]]":
    """Correct M0 for incomplete T1 recovery.

    Parameters
    ----------
    m0 : NDArray
        Raw M0 image.
    params : M0CalibrationParams
        Calibration parameters.

    Returns
    -------
    NDArray
        T1-corrected M0.
    """
    xp = get_array_module(m0)

    tr_s = params.tr_m0 / 1000.0
    t1_s = params.t1_tissue / 1000.0

    # Full recovery factor
    recovery_factor = 1.0 / (1.0 - xp.exp(-tr_s / t1_s))

    return m0 * recovery_factor


def _correct_m0_t2_star(
    m0: "NDArray[np.floating[Any]]",
    params: M0CalibrationParams,
) -> "NDArray[np.floating[Any]]":
    """Correct M0 for T2* decay.

    Parameters
    ----------
    m0 : NDArray
        M0 image (already T1 corrected).
    params : M0CalibrationParams
        Calibration parameters.

    Returns
    -------
    NDArray
        T2*-corrected M0.
    """
    if params.t2_star_tissue is None:
        return m0

    xp = get_array_module(m0)

    te_s = params.te_m0 / 1000.0
    t2_star_s = params.t2_star_tissue / 1000.0

    # Correct for T2* decay
    t2_factor = xp.exp(te_s / t2_star_s)

    return m0 * t2_factor


def _get_reference_m0(
    m0: "NDArray[np.floating[Any]]",
    mask: "NDArray[np.bool_]",
    params: M0CalibrationParams,
) -> float:
    """Get reference M0 value from specified region.

    Parameters
    ----------
    m0 : NDArray
        M0 image.
    mask : NDArray
        Brain mask.
    params : M0CalibrationParams
        Calibration parameters.

    Returns
    -------
    float
        Reference M0 value.
    """
    xp = get_array_module(m0)

    # Percentile requires numpy (reduction op, convert if on GPU)
    m0_np = to_numpy(m0)
    mask_np = to_numpy(mask)

    if params.reference_region == "csf":
        # Use high-intensity voxels (CSF has high M0)
        threshold = float(np.percentile(m0_np[mask_np], 95))
        ref_mask = mask & (m0 > threshold)
    elif params.reference_region == "white_matter":
        # Use mid-range intensity voxels
        p25 = float(np.percentile(m0_np[mask_np], 25))
        p50 = float(np.percentile(m0_np[mask_np], 50))
        ref_mask = mask & (m0 > p25) & (m0 < p50)
    else:
        # Use all brain voxels
        ref_mask = mask

    if xp.sum(ref_mask) == 0:
        ref_mask = mask

    return float(to_numpy(xp.mean(m0[ref_mask])))


def _prepare_m0(
    m0_image: "NDArray[np.floating[Any]]",
    params: M0CalibrationParams,
) -> "NDArray[np.floating[Any]]":
    """Apply T1 recovery and optional T2* corrections to M0.

    Parameters
    ----------
    m0_image : NDArray
        Raw M0 image.
    params : M0CalibrationParams
        Calibration parameters.

    Returns
    -------
    NDArray
        Corrected M0 image.
    """
    m0_corrected = _correct_m0_t1_recovery(m0_image, params)
    if params.t2_star_tissue is not None and params.te_m0 > 0:
        m0_corrected = _correct_m0_t2_star(m0_corrected, params)
    return m0_corrected


# ---------------------------------------------------------------------------
# Registered M0 calibration strategies (E6)
# ---------------------------------------------------------------------------


@register_m0_calibration("voxelwise")
class VoxelwiseM0Calibration(BaseM0Calibration):
    """Voxel-by-voxel M0 calibration."""

    @property
    def name(self) -> str:
        """Return calibration method name."""
        return "voxelwise"

    @property
    def reference(self) -> str:
        """Return literature reference for this method."""
        return "Alsop DC et al. MRM 2015;73(1):102-116."

    def calibrate(
        self,
        asl_data: "NDArray[np.floating[Any]]",
        m0_image: "NDArray[np.floating[Any]]",
        params: Any,
        mask: "NDArray[np.bool_] | None" = None,
    ) -> tuple["NDArray[np.floating[Any]]", "NDArray[np.floating[Any]]"]:
        """Calibrate ASL data by dividing each voxel by its corrected M0 value.

        Parameters
        ----------
        asl_data : NDArray
            ASL difference images.
        m0_image : NDArray
            M0 calibration image.
        params : Any
            M0 calibration parameters.
        mask : NDArray[np.bool_] | None
            Brain mask.

        Returns
        -------
        calibrated : NDArray
            Calibrated ASL data (delta-M / M0).
        m0_corrected : NDArray
            M0 after T1 recovery and T2* corrections.
        """
        xp = get_array_module(asl_data, m0_image)
        if mask is None:
            mask = xp.ones(asl_data.shape, dtype=bool)
        m0_corrected = _prepare_m0(m0_image, params)
        m0_safe = xp.where(m0_corrected > 0, m0_corrected, 1.0)
        calibrated = asl_data / m0_safe
        calibrated = xp.where(mask & (m0_corrected > 0), calibrated, 0.0)
        calibrated = xp.nan_to_num(calibrated, nan=0.0, posinf=0.0, neginf=0.0)
        return calibrated, m0_corrected


@register_m0_calibration("reference_region")
class ReferenceRegionM0Calibration(BaseM0Calibration):
    """Reference region M0 calibration."""

    @property
    def name(self) -> str:
        """Return calibration method name."""
        return "reference_region"

    @property
    def reference(self) -> str:
        """Return literature reference for this method."""
        return "Alsop DC et al. MRM 2015;73(1):102-116."

    def calibrate(
        self,
        asl_data: "NDArray[np.floating[Any]]",
        m0_image: "NDArray[np.floating[Any]]",
        params: Any,
        mask: "NDArray[np.bool_] | None" = None,
    ) -> tuple["NDArray[np.floating[Any]]", "NDArray[np.floating[Any]]"]:
        """Calibrate ASL data using a single M0 value from a reference region.

        Parameters
        ----------
        asl_data : NDArray
            ASL difference images.
        m0_image : NDArray
            M0 calibration image.
        params : Any
            M0 calibration parameters.
        mask : NDArray[np.bool_] | None
            Brain mask.

        Returns
        -------
        calibrated : NDArray
            Calibrated ASL data (delta-M / M0_ref).
        m0_corrected : NDArray
            M0 after T1 recovery and T2* corrections.
        """
        xp = get_array_module(asl_data, m0_image)
        if mask is None:
            mask = xp.ones(asl_data.shape, dtype=bool)
        m0_corrected = _prepare_m0(m0_image, params)
        m0_ref = _get_reference_m0(m0_corrected, mask, params)
        calibrated = asl_data / m0_ref
        calibrated = xp.nan_to_num(calibrated, nan=0.0, posinf=0.0, neginf=0.0)
        return calibrated, m0_corrected


@register_m0_calibration("single")
class SingleM0Calibration(BaseM0Calibration):
    """Single (mean) M0 calibration."""

    @property
    def name(self) -> str:
        """Return calibration method name."""
        return "single"

    @property
    def reference(self) -> str:
        """Return literature reference for this method."""
        return "Alsop DC et al. MRM 2015;73(1):102-116."

    def calibrate(
        self,
        asl_data: "NDArray[np.floating[Any]]",
        m0_image: "NDArray[np.floating[Any]]",
        params: Any,
        mask: "NDArray[np.bool_] | None" = None,
    ) -> tuple["NDArray[np.floating[Any]]", "NDArray[np.floating[Any]]"]:
        """Calibrate ASL data using the mean M0 across masked voxels.

        Parameters
        ----------
        asl_data : NDArray
            ASL difference images.
        m0_image : NDArray
            M0 calibration image.
        params : Any
            M0 calibration parameters.
        mask : NDArray[np.bool_] | None
            Brain mask.

        Returns
        -------
        calibrated : NDArray
            Calibrated ASL data (delta-M / mean(M0)).
        m0_corrected : NDArray
            M0 after T1 recovery and T2* corrections.
        """
        xp = get_array_module(asl_data, m0_image)
        if mask is None:
            mask = xp.ones(asl_data.shape, dtype=bool)
        m0_corrected = _prepare_m0(m0_image, params)
        m0_mean = xp.mean(m0_corrected[mask])
        calibrated = asl_data / m0_mean
        calibrated = xp.nan_to_num(calibrated, nan=0.0, posinf=0.0, neginf=0.0)
        return calibrated, m0_corrected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_m0_calibration(
    asl_data: "NDArray[np.floating[Any]]",
    m0_image: "NDArray[np.floating[Any]]",
    params: M0CalibrationParams | None = None,
    mask: "NDArray[np.bool_] | None" = None,
) -> tuple["NDArray[np.floating[Any]]", "NDArray[np.floating[Any]]"]:
    """Apply M0 calibration to ASL data.

    Parameters
    ----------
    asl_data : NDArray[np.floating]
        ASL difference images (ΔM), shape (...).
    m0_image : NDArray[np.floating]
        M0 calibration image, shape (...).
    params : M0CalibrationParams | None
        Calibration parameters.
    mask : NDArray[np.bool_] | None
        Brain mask.

    Returns
    -------
    calibrated_data : NDArray[np.floating]
        Calibrated ASL data (ΔM/M0 ratio scaled appropriately).
    m0_corrected : NDArray[np.floating]
        M0 values after T1/T2* corrections.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.asl.calibration import apply_m0_calibration
    >>> asl_data = np.random.rand(64, 64, 20) * 100
    >>> m0_image = np.random.rand(64, 64, 20) * 1000 + 500
    >>> calibrated, m0 = apply_m0_calibration(asl_data, m0_image)
    """
    params = params or M0CalibrationParams()

    if asl_data.shape != m0_image.shape:
        msg = f"Shape mismatch: ASL {asl_data.shape} vs M0 {m0_image.shape}"
        raise DataValidationError(msg)

    # Dispatch via registry
    strategy = get_m0_calibration(params.method)
    return strategy.calibrate(asl_data, m0_image, params, mask)


def compute_m0_from_pd(
    pd_image: "NDArray[np.floating[Any]]",
    t1_tissue: float = 1330.0,
    t2_tissue: float = 80.0,
    tr: float = 6000.0,
    te: float = 13.0,
) -> "NDArray[np.floating[Any]]":
    """Compute M0 from proton density image.

    Parameters
    ----------
    pd_image : NDArray[np.floating]
        Proton density weighted image.
    t1_tissue : float
        T1 of tissue in milliseconds.
    t2_tissue : float
        T2 of tissue in milliseconds.
    tr : float
        TR of PD acquisition in milliseconds.
    te : float
        TE of PD acquisition in milliseconds.

    Returns
    -------
    NDArray[np.floating]
        Estimated M0 values.
    """
    xp = get_array_module(pd_image)

    # Correct for T1 saturation
    t1_s = t1_tissue / 1000.0
    tr_s = tr / 1000.0
    t1_factor = 1.0 / (1.0 - xp.exp(-tr_s / t1_s))

    # Correct for T2 decay
    t2_s = t2_tissue / 1000.0
    te_s = te / 1000.0
    t2_factor = xp.exp(te_s / t2_s)

    m0 = pd_image * t1_factor * t2_factor

    return m0
