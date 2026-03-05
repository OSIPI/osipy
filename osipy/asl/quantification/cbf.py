"""ASL CBF quantification.

This module implements absolute CBF quantification from ASL difference
images using the single-compartment kinetic model recommended by the
ISMRM Perfusion Study Group. Parameter naming follows the OSIPI ASL
Lexicon conventions.

References
----------
.. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
.. [2] Alsop DC et al. (2015). Recommended implementation of arterial
   spin-labeled perfusion MRI for clinical applications.
   Magn Reson Med 73(1):102-116.
.. [3] Buxton RB et al. (1998). A general kinetic model for quantitative
   perfusion imaging with arterial spin labeling.
   Magn Reson Med 40(3):383-396.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.asl.labeling import LabelingScheme
from osipy.asl.quantification.base import BaseASLModel
from osipy.asl.quantification.registry import (
    get_quantification_model,
    register_quantification_model,
)
from osipy.common.backend.array_module import get_array_module
from osipy.common.exceptions import DataValidationError
from osipy.common.parameter_map import ParameterMap

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Difference method registry (E7)
# ---------------------------------------------------------------------------

DIFFERENCE_REGISTRY: dict[str, Callable] = {}


def register_difference_method(name: str):
    """Decorator to register an ASL difference method.

    Parameters
    ----------
    name : str
        Registry key for the method (e.g. ``'pairwise'``).

    Returns
    -------
    Callable
        Function decorator.
    """

    def decorator(func: Callable) -> Callable:
        if name in DIFFERENCE_REGISTRY:
            logger.warning("Overwriting difference method '%s'", name)
        DIFFERENCE_REGISTRY[name] = func
        return func

    return decorator


def get_difference_method(name: str) -> Callable:
    """Get a difference method by name.

    Parameters
    ----------
    name : str
        Method name (e.g. ``'pairwise'``).

    Returns
    -------
    Callable
        Difference computation function.

    Raises
    ------
    DataValidationError
        If method name is not recognized.
    """
    if name not in DIFFERENCE_REGISTRY:
        valid = ", ".join(sorted(DIFFERENCE_REGISTRY.keys()))
        raise DataValidationError(f"Unknown difference method: {name}. Valid: {valid}")
    return DIFFERENCE_REGISTRY[name]


def list_difference_methods() -> list[str]:
    """Return names of all registered difference methods.

    Returns
    -------
    list[str]
        Sorted list of registered method names.
    """
    return sorted(DIFFERENCE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ASLQuantificationParams:
    """Parameters for ASL CBF quantification.

    Attributes
    ----------
    labeling_scheme : LabelingScheme
        Type of labeling (PASL, CASL, pCASL).
    t1_blood : float
        T1 of arterial blood in milliseconds.
        Default 1650 ms at 3T (Alsop et al., 2015).
    t1_tissue : float
        T1 of gray matter tissue in milliseconds.
        Default 1330 ms at 3T.
    partition_coefficient : float
        Blood-brain partition coefficient (λ).
        Default 0.9 ml/g for whole brain.
    labeling_efficiency : float
        Labeling efficiency (α). Default depends on scheme.
    pld : float
        Post-labeling delay in milliseconds.
    label_duration : float
        Labeling duration in milliseconds (for pCASL/CASL).
    bolus_duration : float | None
        Bolus duration for PASL in milliseconds.
    """

    labeling_scheme: LabelingScheme = LabelingScheme.PCASL
    t1_blood: float = 1650.0  # ms at 3T
    t1_tissue: float = 1330.0  # ms at 3T
    partition_coefficient: float = 0.9  # ml/g
    labeling_efficiency: float = 0.85
    pld: float = 1800.0  # ms
    label_duration: float = 1800.0  # ms
    bolus_duration: float | None = None  # For PASL


@dataclass
class ASLQuantificationResult:
    """Result of ASL CBF quantification.

    Attributes
    ----------
    cbf_map : ParameterMap
        CBF map in mL/100g/min (OSIPI ASL Lexicon).
    quality_mask : NDArray[np.bool_]
        Mask of reliable voxels.
    m0_used : NDArray[np.floating] | None
        M0 values (a.u.) used for calibration.
    scaling_factor : float
        Conversion factor used.
    """

    cbf_map: ParameterMap
    quality_mask: "NDArray[np.bool_]"
    m0_used: "NDArray[np.floating[Any]] | None" = None
    scaling_factor: float = 1.0


# ---------------------------------------------------------------------------
# Internal quantification helpers (used by strategy classes)
# ---------------------------------------------------------------------------


def _quantify_pcasl(
    delta_m: "NDArray[np.floating[Any]]",
    m0: "NDArray[np.floating[Any]]",
    params: ASLQuantificationParams,
) -> "NDArray[np.floating[Any]]":
    """Quantify CBF for pCASL/CASL.

    pCASL/CASL CBF equation:
        CBF = (6000 × λ × ΔM × exp(PLD/T1b)) / (2 × α × T1b × M0 × (1 - exp(-τ/T1b)))

    Parameters
    ----------
    delta_m : NDArray
        Difference magnetization.
    m0 : NDArray
        Equilibrium magnetization.
    params : ASLQuantificationParams
        Quantification parameters.

    Returns
    -------
    NDArray
        CBF in ml/100g/min.
    """
    xp = get_array_module(delta_m, m0)

    # Convert times to seconds
    pld_s = params.pld / 1000.0
    tau_s = params.label_duration / 1000.0
    t1b_s = params.t1_blood / 1000.0

    # Labeling efficiency
    alpha = params.labeling_efficiency

    # Blood-brain partition coefficient
    lam = params.partition_coefficient

    # Scaling factor (6000 for ml/100g/min)
    scale = 6000.0

    # CBF calculation (GPU-accelerated when available)
    numerator = scale * lam * delta_m * xp.exp(pld_s / t1b_s)
    denominator = 2 * alpha * t1b_s * m0 * (1 - xp.exp(-tau_s / t1b_s))

    cbf = numerator / (denominator + 1e-10)

    # Handle invalid values
    cbf = xp.where(~xp.isfinite(cbf), 0.0, cbf)

    return cbf


def _quantify_pasl(
    delta_m: "NDArray[np.floating[Any]]",
    m0: "NDArray[np.floating[Any]]",
    params: ASLQuantificationParams,
) -> "NDArray[np.floating[Any]]":
    """Quantify CBF for PASL.

    PASL CBF equation (with QUIPSS II):
        CBF = (6000 × λ × ΔM × exp(TI/T1b)) / (2 × α × TI1 × M0)

    Parameters
    ----------
    delta_m : NDArray
        Difference magnetization.
    m0 : NDArray
        Equilibrium magnetization.
    params : ASLQuantificationParams
        Quantification parameters.

    Returns
    -------
    NDArray
        CBF in ml/100g/min.
    """
    xp = get_array_module(delta_m, m0)

    # For PASL, use PLD as TI (time from labeling to acquisition)
    ti_s = params.pld / 1000.0
    t1b_s = params.t1_blood / 1000.0

    # Bolus duration (TI1 for QUIPSS)
    ti1_s = params.bolus_duration / 1000.0 if params.bolus_duration is not None else 0.7

    alpha = params.labeling_efficiency
    lam = params.partition_coefficient
    scale = 6000.0

    # CBF calculation (GPU-accelerated when available)
    numerator = scale * lam * delta_m * xp.exp(ti_s / t1b_s)
    denominator = 2 * alpha * ti1_s * m0

    cbf = numerator / (denominator + 1e-10)

    cbf = xp.where(~xp.isfinite(cbf), 0.0, cbf)

    return cbf


# ---------------------------------------------------------------------------
# Registered quantification model classes (E2)
# ---------------------------------------------------------------------------


@register_quantification_model("pcasl_single_pld")
class PCASLSinglePLDModel(BaseASLModel):
    """pCASL single-PLD CBF quantification model (OSIPI ASL Lexicon).

    Implements the ISMRM consensus pCASL equation. Returns CBF in
    mL/100g/min per the OSIPI ASL Lexicon.

    References
    ----------
    .. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
    .. [2] Alsop DC et al. MRM 2015;73(1):102-116.
    """

    @property
    def name(self) -> str:
        """Return model registry name."""
        return "pcasl_single_pld"

    @property
    def parameters(self) -> list[str]:
        """Return fitted parameter names."""
        return ["CBF"]

    @property
    def parameter_units(self) -> dict[str, str]:
        """Return parameter units per OSIPI ASL Lexicon."""
        return {"CBF": "mL/100g/min"}

    @property
    def reference(self) -> str:
        """Return literature reference for this model."""
        return "Alsop DC et al. MRM 2015;73(1):102-116."

    @property
    def labeling_type(self) -> str:
        """Return ASL labeling type."""
        return "pcasl"

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Return parameter bounds for CBF fitting."""
        return {"CBF": (0.0, 200.0)}

    def quantify(self, delta_m, m0, params):
        """Compute CBF from delta-M and M0 using the pCASL equation."""
        return _quantify_pcasl(delta_m, m0, params)


@register_quantification_model("pasl_single_pld")
class PASLSinglePLDModel(BaseASLModel):
    """PASL single-PLD CBF quantification model (OSIPI ASL Lexicon).

    Implements the QUIPSS II PASL equation. Returns CBF in
    mL/100g/min per the OSIPI ASL Lexicon.

    References
    ----------
    .. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
    .. [2] Alsop DC et al. MRM 2015;73(1):102-116.
    """

    @property
    def name(self) -> str:
        """Return model registry name."""
        return "pasl_single_pld"

    @property
    def parameters(self) -> list[str]:
        """Return fitted parameter names."""
        return ["CBF"]

    @property
    def parameter_units(self) -> dict[str, str]:
        """Return parameter units per OSIPI ASL Lexicon."""
        return {"CBF": "mL/100g/min"}

    @property
    def reference(self) -> str:
        """Return literature reference for this model."""
        return "Alsop DC et al. MRM 2015;73(1):102-116."

    @property
    def labeling_type(self) -> str:
        """Return ASL labeling type."""
        return "pasl"

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Return parameter bounds for CBF fitting."""
        return {"CBF": (0.0, 200.0)}

    def quantify(self, delta_m, m0, params):
        """Compute CBF from delta-M and M0 using the PASL equation."""
        return _quantify_pasl(delta_m, m0, params)


@register_quantification_model("casl_single_pld")
class CASLSinglePLDModel(BaseASLModel):
    """CASL single-PLD CBF quantification model (OSIPI ASL Lexicon).

    Uses same equation as pCASL. Returns CBF in mL/100g/min per
    the OSIPI ASL Lexicon.

    References
    ----------
    .. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
    .. [2] Alsop DC et al. MRM 2015;73(1):102-116.
    """

    @property
    def name(self) -> str:
        """Return model registry name."""
        return "casl_single_pld"

    @property
    def parameters(self) -> list[str]:
        """Return fitted parameter names."""
        return ["CBF"]

    @property
    def parameter_units(self) -> dict[str, str]:
        """Return parameter units per OSIPI ASL Lexicon."""
        return {"CBF": "mL/100g/min"}

    @property
    def reference(self) -> str:
        """Return literature reference for this model."""
        return "Alsop DC et al. MRM 2015;73(1):102-116."

    @property
    def labeling_type(self) -> str:
        """Return ASL labeling type."""
        return "casl"

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Return parameter bounds for CBF fitting."""
        return {"CBF": (0.0, 200.0)}

    def quantify(self, delta_m, m0, params):
        """Compute CBF from delta-M and M0 using the pCASL/CASL equation."""
        return _quantify_pcasl(delta_m, m0, params)


# Map LabelingScheme enum to registry name for dispatch
_SCHEME_TO_MODEL: dict[LabelingScheme, str] = {
    LabelingScheme.PCASL: "pcasl_single_pld",
    LabelingScheme.CASL: "casl_single_pld",
    LabelingScheme.PASL: "pasl_single_pld",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def quantify_cbf(
    delta_m: "NDArray[np.floating[Any]]",
    m0: "NDArray[np.floating[Any]] | float",
    params: ASLQuantificationParams | None = None,
    mask: "NDArray[np.bool_] | None" = None,
) -> ASLQuantificationResult:
    """Quantify absolute CBF from ASL difference images.

    Uses the single-compartment model from ISMRM consensus.
    CBF is returned in mL/100g/min per the OSIPI ASL Lexicon.

    For pCASL/CASL:
        CBF = (6000 * lambda * DeltaM * exp(PLD/T1b)) /
              (2 * alpha * T1b * M0 * (1 - exp(-tau/T1b)))

    For PASL:
        CBF = (6000 * lambda * DeltaM * exp(TI/T1b)) /
              (2 * alpha * TI1 * M0)

    Parameters
    ----------
    delta_m : NDArray[np.floating]
        Difference magnetization (label - control), shape (...).
    m0 : NDArray[np.floating] | float
        Equilibrium magnetization (M0, a.u.) from calibration.
        Either a single value or array matching delta_m spatial shape.
    params : ASLQuantificationParams | None
        Quantification parameters.
    mask : NDArray[np.bool_] | None
        Brain mask.

    Returns
    -------
    ASLQuantificationResult
        CBF map (mL/100g/min) and quality metrics.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.asl import quantify_cbf, ASLQuantificationParams, LabelingScheme
    >>> delta_m = np.random.rand(64, 64, 20) * 100
    >>> m0 = 1000.0
    >>> params = ASLQuantificationParams(
    ...     labeling_scheme=LabelingScheme.PCASL,
    ...     pld=1800.0,
    ...     label_duration=1800.0,
    ... )
    >>> result = quantify_cbf(delta_m, m0, params)

    References
    ----------
    .. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
    """
    params = params or ASLQuantificationParams()

    # Get array module for GPU/CPU agnostic computation
    xp = get_array_module(delta_m)

    spatial_shape = delta_m.shape
    if mask is None:
        mask = xp.ones(spatial_shape, dtype=bool)

    # Convert M0 to array if scalar
    if np.isscalar(m0):
        m0_array = xp.full(spatial_shape, m0)
    else:
        m0_array = xp.asarray(m0)
        if m0_array.shape != spatial_shape:
            # Try to broadcast
            m0_array = xp.broadcast_to(m0_array, spatial_shape).copy()

    # Validate M0
    if xp.any(m0_array[mask] <= 0):
        msg = "M0 must be positive for all brain voxels"
        raise DataValidationError(msg)

    # Dispatch via registry
    model_name = _SCHEME_TO_MODEL.get(params.labeling_scheme)
    if model_name is None:
        msg = f"Unsupported labeling scheme: {params.labeling_scheme}"
        raise DataValidationError(msg)
    model = get_quantification_model(model_name)
    cbf = model.quantify(delta_m, m0_array, params)

    # Apply mask
    cbf = xp.where(mask, cbf, 0)

    # Create quality mask (reasonable CBF range)
    quality_mask = mask & (cbf > 0) & (cbf < 200)

    # Build result
    # Ensure 3D shape
    if cbf.ndim == 2:
        cbf = cbf[..., np.newaxis]
        quality_mask = quality_mask[..., np.newaxis]

    cbf_map = ParameterMap(
        name="CBF",
        symbol="CBF",
        units="mL/100g/min",
        values=cbf,
        affine=np.eye(4),
        quality_mask=quality_mask,
    )

    return ASLQuantificationResult(
        cbf_map=cbf_map,
        quality_mask=quality_mask,
        m0_used=m0_array,
        scaling_factor=6000.0,  # Factor for ml/100g/min conversion
    )


# ---------------------------------------------------------------------------
# Registered difference methods (E7)
# ---------------------------------------------------------------------------


@register_difference_method("pairwise")
def _difference_pairwise(
    controls: "NDArray[np.floating[Any]]",
    labels: "NDArray[np.floating[Any]]",
    control_indices: list[int],
    label_indices: list[int],
    asl_data: "NDArray[np.floating[Any]]",
    xp: Any,
) -> "NDArray[np.floating[Any]]":
    """Pair-wise subtraction (assumes interleaved acquisition)."""
    n_pairs = min(len(control_indices), len(label_indices))
    differences = []

    for i in range(n_pairs):
        diff = controls[..., i] - labels[..., i]
        differences.append(diff)

    if differences:
        return xp.mean(xp.stack(differences, axis=-1), axis=-1)
    msg = "No control-label pairs found"
    raise DataValidationError(msg)


@register_difference_method("surround")
def _difference_surround(
    controls: "NDArray[np.floating[Any]]",
    labels: "NDArray[np.floating[Any]]",
    control_indices: list[int],
    label_indices: list[int],
    asl_data: "NDArray[np.floating[Any]]",
    xp: Any,
) -> "NDArray[np.floating[Any]]":
    """Average surrounding controls for each label."""
    differences = []

    for label_idx in label_indices:
        prev_controls = [c for c in control_indices if c < label_idx]
        next_controls = [c for c in control_indices if c > label_idx]

        adjacent = []
        if prev_controls:
            adjacent.append(prev_controls[-1])
        if next_controls:
            adjacent.append(next_controls[0])

        if adjacent:
            mean_adjacent_control = xp.mean(asl_data[..., adjacent], axis=-1)
            diff = mean_adjacent_control - asl_data[..., label_idx]
            differences.append(diff)

    if differences:
        return xp.mean(xp.stack(differences, axis=-1), axis=-1)
    msg = "Could not compute surround subtraction"
    raise DataValidationError(msg)


@register_difference_method("mean")
def _difference_mean(
    controls: "NDArray[np.floating[Any]]",
    labels: "NDArray[np.floating[Any]]",
    control_indices: list[int],
    label_indices: list[int],
    asl_data: "NDArray[np.floating[Any]]",
    xp: Any,
) -> "NDArray[np.floating[Any]]":
    """Simple mean subtraction."""
    mean_control = xp.mean(controls, axis=-1)
    mean_label = xp.mean(labels, axis=-1)
    return mean_control - mean_label


def compute_control_label_difference(
    asl_data: "NDArray[np.floating[Any]]",
    context: list[str],
    method: str = "pairwise",
    mask: "NDArray[np.bool_] | None" = None,
) -> "NDArray[np.floating[Any]]":
    """Compute ASL difference from interleaved control/label data.

    Extracts control and label volumes from raw 4D ASL data using the
    aslcontext volume type information and computes ΔM = control - label.

    Parameters
    ----------
    asl_data : NDArray[np.floating]
        Raw 4D ASL timeseries, shape (x, y, z, n_volumes).
    context : list[str]
        Volume types from aslcontext.tsv. Should contain 'control' and 'label'.
        Other types like 'm0scan', 'deltam', 'cbf' are filtered out.
    method : str, default='pairwise'
        Subtraction method:
        - 'pairwise': Subtract adjacent control-label pairs, then average.
        - 'surround': Average surrounding controls for each label.
        - 'mean': Average all controls and labels separately, then subtract.
    mask : NDArray[np.bool_] | None
        Brain mask. If provided, sets non-brain voxels to 0.

    Returns
    -------
    NDArray[np.floating]
        Difference magnetization (ΔM), shape (x, y, z) or (x, y, z, n_pairs)
        depending on method. For 'pairwise' and 'mean', returns averaged 3D.

    Raises
    ------
    DataValidationError
        If context length doesn't match data or no control/label volumes found.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.asl import compute_control_label_difference
    >>> # Simulated interleaved ASL data
    >>> data = np.random.rand(64, 64, 20, 80) * 1000
    >>> context = ['control', 'label'] * 40
    >>> delta_m = compute_control_label_difference(data, context)
    >>> print(delta_m.shape)  # (64, 64, 20)

    Notes
    -----
    The context list should match the aslcontext.tsv format:
    - 'control': Control (no labeling) images
    - 'label': Labeled blood images
    - 'm0scan': M0 calibration images (excluded from difference)
    - 'deltam': Already-computed difference images (returned as-is)
    - 'cbf': Already-quantified CBF maps (excluded)
    """
    xp = get_array_module(asl_data)

    # Validate inputs
    if asl_data.ndim != 4:
        msg = f"asl_data must be 4D, got {asl_data.ndim}D"
        raise DataValidationError(msg)

    n_volumes = asl_data.shape[3]
    if len(context) != n_volumes:
        msg = f"Context length ({len(context)}) must match data volumes ({n_volumes})"
        raise DataValidationError(msg)

    # Normalize context to lowercase
    context_lower = [c.lower() for c in context]

    # Check for pre-computed deltam
    if "deltam" in context_lower:
        deltam_indices = [i for i, c in enumerate(context_lower) if c == "deltam"]
        delta_m = asl_data[..., deltam_indices]
        if delta_m.shape[-1] == 1:
            delta_m = delta_m[..., 0]
        else:
            delta_m = xp.mean(delta_m, axis=-1)
        if mask is not None:
            delta_m = xp.where(mask, delta_m, 0)
        return delta_m

    # Find control and label indices
    control_indices = [i for i, c in enumerate(context_lower) if c == "control"]
    label_indices = [i for i, c in enumerate(context_lower) if c == "label"]

    if not control_indices:
        msg = "No 'control' volumes found in context"
        raise DataValidationError(msg)
    if not label_indices:
        msg = "No 'label' volumes found in context"
        raise DataValidationError(msg)

    # Extract control and label volumes
    controls = asl_data[..., control_indices]
    labels = asl_data[..., label_indices]

    # Dispatch via registry
    diff_func = get_difference_method(method)
    delta_m = diff_func(controls, labels, control_indices, label_indices, asl_data, xp)

    # Apply mask if provided
    if mask is not None:
        delta_m = xp.where(mask, delta_m, 0)

    return delta_m


def compute_pcasl_difference(
    label: "NDArray[np.floating[Any]]",
    control: "NDArray[np.floating[Any]]",
    mask: "NDArray[np.bool_] | None" = None,
) -> "NDArray[np.floating[Any]]":
    """Compute ASL difference image (control - label).

    For pCASL/CASL, the difference magnetization is ΔM = M_control - M_label,
    which is positive where there is blood flow (labeled blood reduces signal).

    Parameters
    ----------
    label : NDArray[np.floating]
        Label images, shape (..., n_pairs) or (...,).
    control : NDArray[np.floating]
        Control images, shape (..., n_pairs) or (...,).
    mask : NDArray[np.bool_] | None
        Brain mask for averaging (optional).

    Returns
    -------
    NDArray[np.floating]
        Difference magnetization (ΔM = control - label).
        If multiple pairs, returns mean across pairs.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.asl import compute_pcasl_difference
    >>> control = np.random.rand(64, 64, 20, 10) * 1000 + 100
    >>> label = control - 50  # Simulated signal reduction
    >>> delta_m = compute_pcasl_difference(label, control)
    >>> print(delta_m.shape)  # (64, 64, 20)
    """
    xp = get_array_module(label, control)

    label = xp.asarray(label, dtype=xp.float64)
    control = xp.asarray(control, dtype=xp.float64)

    if label.shape != control.shape:
        msg = f"Label shape {label.shape} must match control shape {control.shape}"
        raise DataValidationError(msg)

    # Compute difference (control - label is positive for perfusion)
    delta_m = control - label

    # If 4D (multiple acquisitions), average across last dimension
    if delta_m.ndim == 4:
        delta_m = xp.mean(delta_m, axis=-1)

    # Apply mask if provided
    if mask is not None:
        delta_m = xp.where(mask, delta_m, 0)

    return delta_m


def compute_cbf_uncertainty(
    delta_m: "NDArray[np.floating[Any]]",
    delta_m_std: "NDArray[np.floating[Any]]",
    m0: "NDArray[np.floating[Any]]",
    m0_std: "NDArray[np.floating[Any]] | None",
    params: ASLQuantificationParams,
) -> "NDArray[np.floating[Any]]":
    """Compute CBF uncertainty from error propagation.

    Parameters
    ----------
    delta_m : NDArray
        Mean difference magnetization.
    delta_m_std : NDArray
        Standard deviation of difference magnetization.
    m0 : NDArray
        M0 value.
    m0_std : NDArray | None
        M0 uncertainty (if available).
    params : ASLQuantificationParams
        Quantification parameters.

    Returns
    -------
    NDArray
        CBF standard error in ml/100g/min.
    """
    xp = get_array_module(delta_m, delta_m_std, m0)

    # Compute base CBF via registry
    model_name = _SCHEME_TO_MODEL.get(params.labeling_scheme)
    if model_name is None:
        msg = f"Unsupported labeling scheme: {params.labeling_scheme}"
        raise DataValidationError(msg)
    model = get_quantification_model(model_name)
    cbf = model.quantify(delta_m, m0, params)

    # Error propagation: σ_CBF/CBF ≈ sqrt((σ_ΔM/ΔM)² + (σ_M0/M0)²)
    rel_error_dm = delta_m_std / (xp.abs(delta_m) + 1e-10)
    rel_error_dm = xp.where(~xp.isfinite(rel_error_dm), 1.0, rel_error_dm)

    if m0_std is not None:
        rel_error_m0 = m0_std / (m0 + 1e-10)
        rel_error_m0 = xp.where(~xp.isfinite(rel_error_m0), 0.0, rel_error_m0)
    else:
        rel_error_m0 = xp.zeros_like(delta_m)

    total_rel_error = xp.sqrt(rel_error_dm**2 + rel_error_m0**2)
    cbf_std = xp.abs(cbf) * total_rel_error

    return cbf_std
