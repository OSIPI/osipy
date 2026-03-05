"""Multi-PLD ASL quantification with ATT estimation.

This module implements CBF and ATT (arterial transit time) estimation
from multi-PLD ASL data using the Buxton general kinetic model.
Parameter naming follows the OSIPI ASL Lexicon conventions: CBF in
mL/100g/min and ATT in ms.

GPU/CPU agnostic using the xp array module pattern.
NO scipy dependency - uses custom Levenberg-Marquardt implementation.

References
----------
.. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
.. [2] Buxton RB et al. (1998). A general kinetic model for quantitative
   perfusion imaging with arterial spin labeling.
   Magn Reson Med 40(3):383-396.
.. [3] Dai W et al. (2012). Reduced resolution transit delay prescan for
   quantitative continuous arterial spin labeling perfusion imaging.
   Magn Reson Med 67(5):1252-1265.
.. [4] Suzuki Y et al. (2024). MRM 91(4):1411-1421.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.asl.labeling import LabelingScheme
from osipy.asl.quantification.att_registry import register_att_model
from osipy.asl.quantification.base import BaseASLModel
from osipy.asl.quantification.binding import BoundASLModel
from osipy.asl.quantification.registry import register_quantification_model
from osipy.common.backend.array_module import get_array_module, to_numpy
from osipy.common.exceptions import DataValidationError
from osipy.common.fitting.least_squares import LevenbergMarquardtFitter
from osipy.common.parameter_map import ParameterMap

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class MultiPLDParams:
    """Parameters for multi-PLD ASL quantification.

    Attributes
    ----------
    labeling_scheme : LabelingScheme
        Type of labeling (PASL, CASL, pCASL).
    plds : NDArray[np.floating]
        Array of post-labeling delays in milliseconds.
    label_duration : float
        Labeling duration in milliseconds (for pCASL/CASL).
    t1_blood : float
        T1 of arterial blood in milliseconds.
    t1_tissue : float
        T1 of gray matter tissue in milliseconds.
    partition_coefficient : float
        Blood-brain partition coefficient (λ).
    labeling_efficiency : float
        Labeling efficiency (α).
    """

    labeling_scheme: LabelingScheme = LabelingScheme.PCASL
    plds: NDArray[np.floating[Any]] = None  # type: ignore
    label_duration: float = 1800.0  # ms
    t1_blood: float = 1650.0  # ms at 3T
    t1_tissue: float = 1330.0  # ms at 3T
    partition_coefficient: float = 0.9  # ml/g
    labeling_efficiency: float = 0.85

    def __post_init__(self) -> None:
        """Set default PLD schedule if none provided."""
        if self.plds is None:
            # Default multi-PLD scheme
            self.plds = np.array([200, 500, 1000, 1500, 2000, 2500], dtype=float)


@dataclass
class MultiPLDResult:
    """Result of multi-PLD ASL quantification.

    Attributes
    ----------
    cbf_map : ParameterMap
        CBF map in mL/100g/min (OSIPI ASL Lexicon).
    att_map : ParameterMap
        Arterial transit time (ATT) map in ms (OSIPI ASL Lexicon).
    quality_mask : NDArray[np.bool_]
        Mask of reliable voxels.
    r_squared : NDArray[np.floating] | None
        Goodness of fit (R-squared) map.
    """

    cbf_map: ParameterMap
    att_map: ParameterMap
    quality_mask: NDArray[np.bool_]
    r_squared: NDArray[np.floating[Any]] | None = None


# ---------------------------------------------------------------------------
# Vectorized Buxton model (used by BoundASLModel and BuxtonATTModel)
# ---------------------------------------------------------------------------


def _buxton_model_pcasl_batch(
    plds: NDArray[np.floating[Any]],
    cbf: NDArray[np.floating[Any]],
    att: NDArray[np.floating[Any]],
    m0: NDArray[np.floating[Any]],
    tau: float,
    t1_blood: float,
    t1_tissue: float,
    alpha: float,
    lam: float,
    xp: Any,
) -> NDArray[np.floating[Any]]:
    """Vectorized Buxton model for multiple voxels.

    Parameters
    ----------
    plds : NDArray
        PLDs in seconds, shape (n_plds,).
    cbf : NDArray
        CBF in ml/100g/min, shape (n_voxels,).
    att : NDArray
        ATT in seconds, shape (n_voxels,).
    m0 : NDArray
        M0 values, shape (n_voxels,).
    tau : float
        Labeling duration in seconds.
    t1_blood : float
        T1 of blood in seconds.
    t1_tissue : float
        T1 of tissue in seconds.
    alpha : float
        Labeling efficiency.
    lam : float
        Partition coefficient.
    xp : module
        Array module.

    Returns
    -------
    NDArray
        Predicted signals, shape (n_plds, n_voxels).
    """
    # Convert CBF to SI: ml/100g/min -> ml/g/s
    f = cbf / 6000.0  # (n_voxels,)

    # T1 apparent
    t1_app = 1.0 / (1.0 / t1_tissue + f / lam)  # (n_voxels,)

    # Broadcast: plds (n_plds, 1), att (1, n_voxels)
    pld_2d = plds[:, xp.newaxis]  # (n_plds, 1)
    att_2d = att[xp.newaxis, :]  # (1, n_voxels)
    m0_2d = m0[xp.newaxis, :]  # (1, n_voxels)
    f_2d = f[xp.newaxis, :]  # (1, n_voxels)
    t1_app_2d = t1_app[xp.newaxis, :]  # (1, n_voxels)

    # Three regions: pld < att, att <= pld < att+tau, pld >= att+tau
    region1 = pld_2d < att_2d
    region2 = (pld_2d >= att_2d) & (pld_2d < att_2d + tau)
    # Region 2: during labeling
    q2 = 1.0 - xp.exp(-(pld_2d - att_2d) / (t1_app_2d + 1e-10))
    dm2 = 2.0 * m0_2d * f_2d * t1_app_2d * alpha * xp.exp(-att_2d / t1_blood) * q2

    # Region 3: after labeling
    q3 = 1.0 - xp.exp(-tau / (t1_app_2d + 1e-10))
    dm3 = (
        2.0
        * m0_2d
        * f_2d
        * t1_app_2d
        * alpha
        * xp.exp(-att_2d / t1_blood)
        * xp.exp(-(pld_2d - tau - att_2d) / (t1_app_2d + 1e-10))
        * q3
    )

    # Combine regions
    delta_m = xp.where(region1, 0.0, xp.where(region2, dm2, dm3))

    return delta_m


# ---------------------------------------------------------------------------
# Registered ATT model (E4)
# ---------------------------------------------------------------------------


@register_quantification_model("buxton_multi_pld")
class BuxtonMultiPLDModel(BaseASLModel):
    """Buxton general kinetic model for multi-PLD ATT estimation.

    Estimates CBF (mL/100g/min) and ATT (seconds) from multi-PLD ASL.

    References
    ----------
    .. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
    .. [2] Buxton RB et al. MRM 1998;40(3):383-396.
    """

    @property
    def name(self) -> str:
        """Return model registry name."""
        return "buxton_multi_pld"

    @property
    def parameters(self) -> list[str]:
        """Return fitted parameter names."""
        return ["CBF", "ATT"]

    @property
    def parameter_units(self) -> dict[str, str]:
        """Return parameter units per OSIPI ASL Lexicon."""
        return {"CBF": "mL/100g/min", "ATT": "s"}

    @property
    def reference(self) -> str:
        """Return literature reference for this model."""
        return "Buxton RB et al. MRM 1998;40(3):383-396."

    @property
    def labeling_type(self) -> str:
        """Return ASL labeling type."""
        return "pcasl"

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Return parameter bounds for CBF and ATT fitting."""
        return {"CBF": (0.0, 200.0), "ATT": (0.1, 5.0)}


# Register in ATT registry under short name
register_att_model("buxton")(BuxtonMultiPLDModel)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def quantify_multi_pld(
    delta_m: NDArray[np.floating[Any]],
    m0: NDArray[np.floating[Any]] | float,
    params: MultiPLDParams | None = None,
    mask: NDArray[np.bool_] | None = None,
) -> MultiPLDResult:
    """Quantify CBF and ATT from multi-PLD ASL data.

    Uses voxel-wise fitting of the Buxton general kinetic model
    to simultaneously estimate CBF and ATT via ``fit_image()``.
    Output units follow the OSIPI ASL Lexicon: CBF in mL/100g/min,
    ATT in ms.

    Parameters
    ----------
    delta_m : NDArray[np.floating]
        Difference magnetization at each PLD, shape (..., n_plds).
        Last dimension corresponds to PLDs.
    m0 : NDArray[np.floating] | float
        Equilibrium magnetization (M0, a.u.) from calibration.
    params : MultiPLDParams | None
        Quantification parameters.
    mask : NDArray[np.bool_] | None
        Brain mask.

    Returns
    -------
    MultiPLDResult
        CBF map (mL/100g/min) and ATT map (ms) with quality metrics.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.asl.quantification.multi_pld import quantify_multi_pld, MultiPLDParams
    >>> delta_m = np.random.rand(64, 64, 20, 6) * 50
    >>> params = MultiPLDParams(plds=np.array([200, 500, 1000, 1500, 2000, 2500]))
    >>> result = quantify_multi_pld(delta_m, m0=1000.0, params=params)

    References
    ----------
    .. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
    """
    params = params or MultiPLDParams()

    xp = get_array_module(delta_m)

    # Validate input dimensions
    n_plds = len(params.plds)
    if delta_m.shape[-1] != n_plds:
        msg = (
            f"Last dimension of delta_m ({delta_m.shape[-1]}) "
            f"must match number of PLDs ({n_plds})"
        )
        raise DataValidationError(msg)

    spatial_shape = delta_m.shape[:-1]

    # Convert M0 to array if scalar
    if np.isscalar(m0):
        m0_array = xp.full(spatial_shape, m0, dtype=float)
    else:
        m0_array = xp.asarray(m0)

    # Normalize delta-M by M0 so the BoundASLModel (which predicts with
    # unit M0) can be compared directly to observed data.
    m0_safe = xp.maximum(m0_array, 1e-10)
    normalized = delta_m / m0_safe[..., xp.newaxis]

    # Ensure 4D for fit_image: (x, y, z, n_plds)
    ndim = len(spatial_shape)
    if ndim == 1:
        data_4d = normalized.reshape(spatial_shape[0], 1, 1, n_plds)
        spatial_3d = (spatial_shape[0], 1, 1)
    elif ndim == 2:
        data_4d = normalized[:, :, xp.newaxis, :]
        spatial_3d = (spatial_shape[0], spatial_shape[1], 1)
    elif ndim == 3:
        data_4d = normalized
        spatial_3d = spatial_shape
    else:
        msg = f"Invalid delta_m shape: {delta_m.shape}"
        raise DataValidationError(msg)

    # Ensure numpy for fit_image (it handles GPU transfer internally)
    data_4d_np = to_numpy(data_4d)

    # Build mask
    if mask is None:
        mask_3d = np.ones(spatial_3d, dtype=bool)
    else:
        mask_np = to_numpy(mask)
        if ndim == 1:
            mask_3d = mask_np.reshape(spatial_3d)
        elif ndim == 2:
            mask_3d = mask_np[:, :, np.newaxis]
            mask_3d = np.broadcast_to(mask_3d, spatial_3d).copy()
        else:
            mask_3d = mask_np

    # Create bound model and fitter, then delegate to fit_image
    bound_model = BoundASLModel(params)
    fitter = LevenbergMarquardtFitter()

    param_maps = fitter.fit_image(
        model=bound_model,
        data=data_4d_np,
        mask=mask_3d,
    )

    # Extract fitted maps (3D volumes from fit_image)
    cbf_data = param_maps["CBF"].values
    att_s_data = param_maps["ATT"].values
    r2_data = param_maps["r_squared"].values
    fit_quality = param_maps["CBF"].quality_mask

    # Convert ATT from seconds to ms for OSIPI compliance
    att_data = att_s_data * 1000.0

    # Apply domain-specific quality constraints
    quality_mask = (
        fit_quality
        & (cbf_data > 0)
        & (cbf_data < 200)
        & (att_data > 100)
        & (att_data < 4000)
    )

    # Collapse back to original spatial dims if needed
    if ndim == 1:
        cbf_data = cbf_data.ravel()[: spatial_shape[0]]
        att_data = att_data.ravel()[: spatial_shape[0]]
        r2_data = r2_data.ravel()[: spatial_shape[0]]
        quality_mask = quality_mask.ravel()[: spatial_shape[0]]
    elif ndim == 2:
        cbf_data = cbf_data[:, :, 0]
        att_data = att_data[:, :, 0]
        r2_data = r2_data[:, :, 0]
        quality_mask = quality_mask[:, :, 0]

    # Ensure 3D shape for ParameterMap when input is 2D spatial
    if cbf_data.ndim == 2:
        cbf_data = cbf_data[..., np.newaxis]
        att_data = att_data[..., np.newaxis]
        quality_mask = quality_mask[..., np.newaxis]

    affine = np.eye(4)

    cbf_map = ParameterMap(
        name="CBF",
        symbol="CBF",
        units="mL/100g/min",
        values=cbf_data,
        affine=affine,
        quality_mask=quality_mask,
    )

    att_map = ParameterMap(
        name="ATT",
        symbol="ATT",
        units="ms",
        values=att_data,
        affine=affine,
        quality_mask=quality_mask,
    )

    return MultiPLDResult(
        cbf_map=cbf_map,
        att_map=att_map,
        quality_mask=quality_mask,
        r_squared=r2_data,
    )
