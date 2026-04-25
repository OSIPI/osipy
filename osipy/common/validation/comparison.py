"""Validation comparison against Digital Reference Objects.

This module provides functions for comparing osipy computed parameter
maps against OSIPI Digital Reference Objects (DROs) or other reference
datasets with known ground truth values.

References
----------
OSIPI Task Force 4.1: DCE-MRI Technical Validation
OSIPI Task Force 4.2: DSC-MRI Technical Validation
QIBA DCE Profile v2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.exceptions import DataValidationError
from osipy.common.parameter_map import ParameterMap
from osipy.common.validation.report import ValidationReport

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Default tolerances from OSIPI/QIBA guidelines
DEFAULT_TOLERANCES: dict[str, dict[str, float]] = {
    # DCE parameters (OSIPI CodeCollection values)
    "Ktrans": {"absolute": 0.005, "relative": 0.1},
    "ve": {"absolute": 0.05, "relative": 0.0},
    "vp": {"absolute": 0.025, "relative": 0.0},
    "kep": {"absolute": 0.05, "relative": 0.10},
    # DSC parameters
    "CBV": {"absolute": 1.0, "relative": 0.10},
    "CBF": {"absolute": 15.0, "relative": 0.10},
    "MTT": {"absolute": 0.5, "relative": 0.15},
    # ASL parameters
    "CBF_ASL": {"absolute": 10.0, "relative": 0.10},
    "ATT": {"absolute": 200.0, "relative": 0.15},
    # IVIM parameters
    "D": {"absolute": 0.2e-3, "relative": 0.15},
    "D*": {"absolute": 5.0e-3, "relative": 0.20},
    "f": {"absolute": 0.02, "relative": 0.15},
    # OSIPI CodeCollection compliance entries
    "PS": {"absolute": 0.005, "relative": 0.1},
    "Fp": {"absolute": 5.0, "relative": 0.1},
    "delay": {"absolute": 1.0, "relative": 0.0},
    "R1": {"absolute": 0.05, "relative": 0.05},
    "concentration": {"absolute": 0.01, "relative": 0.05},
    "aif": {"absolute": 0.01, "relative": 0.05},
}


@dataclass
class DROData:
    """Container for Digital Reference Object data.

    Attributes
    ----------
    name : str
        DRO identifier name.
    parameters : dict[str, NDArray[np.floating]]
        Ground truth parameter values.
    mask : NDArray[np.bool_] | None
        Mask of valid comparison voxels.
    metadata : dict[str, Any]
        Additional DRO metadata.
    """

    name: str
    parameters: dict[str, NDArray[np.floating[Any]]]
    mask: NDArray[np.bool_] | None = None
    metadata: dict[str, Any] = None  # type: ignore

    def __post_init__(self) -> None:
        """Initialize mutable defaults."""
        if self.metadata is None:
            self.metadata = {}


def load_dro(
    dro_path: str | Path,
    dro_type: str = "osipi",
) -> DROData:
    """Load Digital Reference Object data.

    Parameters
    ----------
    dro_path : str | Path
        Path to DRO directory or file.
    dro_type : str
        DRO format type: "osipi", "qiba", or "custom".

    Returns
    -------
    DROData
        Loaded reference data.

    Examples
    --------
    >>> dro = load_dro("tests/fixtures/osipi_dro/dce_dro_v10")
    """
    dro_path = Path(dro_path)

    if not dro_path.exists():
        msg = f"DRO path not found: {dro_path}"
        raise DataValidationError(msg)

    # Load based on type
    if dro_type == "osipi":
        return _load_osipi_dro(dro_path)
    elif dro_type == "qiba":
        return _load_qiba_dro(dro_path)
    else:
        return _load_custom_dro(dro_path)


def _load_osipi_dro(dro_path: Path) -> DROData:
    """Load OSIPI-format DRO data."""
    import json

    import nibabel as nib

    parameters: dict[str, np.ndarray] = {}
    mask = None
    metadata: dict[str, Any] = {}

    # Look for parameter files
    param_patterns = {
        "Ktrans": "*[Kk]trans*.nii*",
        "ve": "*ve*.nii*",
        "vp": "*vp*.nii*",
        "CBF": "*[Cc][Bb][Ff]*.nii*",
        "D": "*[Dd]iff*.nii*",
    }

    for param, pattern in param_patterns.items():
        matches = list(dro_path.glob(pattern))
        if matches:
            img = nib.load(matches[0])
            parameters[param] = np.asarray(img.get_fdata())

    # Look for mask
    mask_files = list(dro_path.glob("*mask*.nii*"))
    if mask_files:
        mask_img = nib.load(mask_files[0])
        mask = np.asarray(mask_img.get_fdata()) > 0

    # Look for metadata
    json_files = list(dro_path.glob("*.json"))
    if json_files:
        with json_files[0].open(encoding="utf-8") as f:
            metadata = json.load(f)

    return DROData(
        name=dro_path.name,
        parameters=parameters,
        mask=mask,
        metadata=metadata,
    )


def _load_qiba_dro(dro_path: Path) -> DROData:
    """Load QIBA-format DRO data."""
    # QIBA DROs use similar format to OSIPI
    return _load_osipi_dro(dro_path)


def _load_custom_dro(dro_path: Path) -> DROData:
    """Load custom DRO format (NumPy .npz files)."""
    if dro_path.suffix == ".npz":
        data = np.load(dro_path)
        parameters = {k: v for k, v in data.items() if k not in ("mask", "metadata")}
        mask = data.get("mask", None)
        return DROData(
            name=dro_path.stem,
            parameters=parameters,
            mask=mask,
        )
    else:
        return _load_osipi_dro(dro_path)


def validate_against_dro(
    computed: dict[str, ParameterMap] | dict[str, NDArray[np.floating[Any]]],
    reference: DROData | dict[str, NDArray[np.floating[Any]]],
    tolerances: dict[str, dict[str, float]] | None = None,
    mask: NDArray[np.bool_] | None = None,
    reference_name: str = "unknown",
) -> ValidationReport:
    """Compare computed parameters against DRO reference values.

    Parameters
    ----------
    computed : dict[str, ParameterMap] | dict[str, NDArray]
        Computed parameter maps from osipy.
    reference : DROData | dict[str, NDArray]
        Reference (ground truth) parameter values.
    tolerances : dict[str, dict[str, float]] | None
        Tolerance thresholds per parameter.
        Uses OSIPI/QIBA defaults if not specified.
    mask : NDArray[np.bool_] | None
        Mask of voxels to compare. Overrides DRO mask if provided.
    reference_name : str
        Name for the reference dataset.

    Returns
    -------
    ValidationReport
        Detailed comparison results.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.validation.comparison import validate_against_dro
    >>> computed = {"Ktrans": np.random.rand(10, 10, 5) * 0.1}
    >>> reference = {"Ktrans": np.random.rand(10, 10, 5) * 0.1}
    >>> report = validate_against_dro(computed, reference)
    >>> print(report.summary())

    """
    # Extract reference parameters
    if isinstance(reference, DROData):
        ref_params = reference.parameters
        ref_name = reference.name
        if mask is None and reference.mask is not None:
            mask = reference.mask
    else:
        ref_params = reference
        ref_name = reference_name

    # Extract computed values
    comp_params: dict[str, np.ndarray] = {}
    for key, value in computed.items():
        if isinstance(value, ParameterMap):
            comp_params[key] = value.values
        else:
            comp_params[key] = np.asarray(value)

    # Use default tolerances if not specified
    tolerances = tolerances or DEFAULT_TOLERANCES

    # Initialize result containers
    absolute_errors: dict[str, np.ndarray] = {}
    relative_errors: dict[str, np.ndarray] = {}
    within_tolerance: dict[str, np.ndarray] = {}
    pass_rate: dict[str, float] = {}

    # Compare each parameter
    for param in ref_params:
        if param not in comp_params:
            continue

        ref_vals = ref_params[param]
        comp_vals = comp_params[param]

        # Ensure same shape
        if ref_vals.shape != comp_vals.shape:
            # Try to reshape or skip
            continue

        # Apply mask
        if mask is not None:
            ref_flat = ref_vals[mask]
            comp_flat = comp_vals[mask]
        else:
            ref_flat = ref_vals.ravel()
            comp_flat = comp_vals.ravel()

        # Remove invalid values
        valid = np.isfinite(ref_flat) & np.isfinite(comp_flat) & (ref_flat > 0)
        ref_valid = ref_flat[valid]
        comp_valid = comp_flat[valid]

        if len(ref_valid) == 0:
            continue

        # Compute errors
        abs_err = np.abs(comp_valid - ref_valid)
        rel_err = abs_err / (ref_valid + 1e-10)

        # Store full arrays
        full_abs = np.zeros_like(ref_vals)
        full_rel = np.zeros_like(ref_vals)
        if mask is not None:
            temp_abs = np.zeros(mask.sum())
            temp_rel = np.zeros(mask.sum())
            temp_abs[valid] = abs_err
            temp_rel[valid] = rel_err
            full_abs[mask] = temp_abs
            full_rel[mask] = temp_rel
        else:
            full_abs.ravel()[valid] = abs_err
            full_rel.ravel()[valid] = rel_err

        absolute_errors[param] = full_abs
        relative_errors[param] = full_rel

        # Check tolerance
        param_tol = tolerances.get(param, {"relative": 0.10})
        rel_tol = param_tol.get("relative", 0.10)
        abs_tol = param_tol.get("absolute", np.inf)

        within = (abs_err <= abs_tol) | (rel_err <= rel_tol)
        full_within = np.zeros_like(ref_vals, dtype=bool)
        if mask is not None:
            temp_within = np.zeros(mask.sum(), dtype=bool)
            temp_within[valid] = within
            full_within[mask] = temp_within
        else:
            full_within.ravel()[valid] = within

        within_tolerance[param] = full_within
        pass_rate[param] = float(np.mean(within)) if len(within) > 0 else 0.0

    # Determine overall pass
    min_pass_rate = 0.95  # 95% of voxels must be within tolerance
    overall_pass = all(rate >= min_pass_rate for rate in pass_rate.values())

    return ValidationReport(
        reference_name=ref_name,
        reference_parameters=ref_params,
        computed_parameters=comp_params,
        absolute_errors=absolute_errors,
        relative_errors=relative_errors,
        tolerances=tolerances,
        within_tolerance=within_tolerance,
        pass_rate=pass_rate,
        overall_pass=overall_pass,
    )


def create_synthetic_dro(
    shape: tuple[int, ...] = (16, 16, 8),
    modality: str = "dce",
    noise_level: float = 0.01,
) -> DROData:
    """Create synthetic DRO for testing.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of parameter maps.
    modality : str
        Modality type: "dce", "dsc", "asl", or "ivim".
    noise_level : float
        Noise level to add to ground truth.

    Returns
    -------
    DROData
        Synthetic reference data.

    Examples
    --------
    >>> dro = create_synthetic_dro(shape=(8, 8, 4), modality="dce")
    """
    rng = np.random.default_rng(42)

    if modality == "dce":
        parameters = {
            "Ktrans": rng.uniform(0.01, 0.3, shape),
            "ve": rng.uniform(0.05, 0.5, shape),
            "vp": rng.uniform(0.01, 0.1, shape),
        }
    elif modality == "dsc":
        parameters = {
            "CBV": rng.uniform(1.0, 6.0, shape),
            "CBF": rng.uniform(20, 80, shape),
            "MTT": rng.uniform(3, 8, shape),
        }
    elif modality == "asl":
        parameters = {
            "CBF": rng.uniform(30, 80, shape),
            "ATT": rng.uniform(500, 2000, shape),
        }
    elif modality == "ivim":
        parameters = {
            "D": rng.uniform(0.5e-3, 2e-3, shape),
            "D*": rng.uniform(5e-3, 30e-3, shape),
            "f": rng.uniform(0.05, 0.3, shape),
        }
    else:
        msg = f"Unknown modality: {modality}"
        raise DataValidationError(msg)

    # Add noise
    for key in parameters:
        parameters[key] += rng.normal(0, noise_level * parameters[key].mean(), shape)
        parameters[key] = np.clip(parameters[key], 0, None)

    mask = np.ones(shape, dtype=bool)

    return DROData(
        name=f"synthetic_{modality}_dro",
        parameters=parameters,
        mask=mask,
        metadata={"modality": modality, "noise_level": noise_level},
    )
