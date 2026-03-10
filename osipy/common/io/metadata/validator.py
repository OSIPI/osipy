"""Parameter validation for perfusion imaging.

This module validates that required parameters are present
and within acceptable ranges for each modality.

"""

import logging
from dataclasses import dataclass, field
from typing import Any

from osipy.common.io.metadata.defaults import PARAM_RANGES
from osipy.common.types import Modality

logger = logging.getLogger(__name__)


# Required parameters by modality
REQUIRED_PARAMS: dict[Modality, list[str]] = {
    Modality.DCE: ["tr", "flip_angle"],
    Modality.DSC: ["tr", "te"],
    Modality.ASL: ["labeling_type", "post_labeling_delay"],
    Modality.IVIM: ["b_values"],
}

# Recommended (but not required) parameters
RECOMMENDED_PARAMS: dict[Modality, list[str]] = {
    Modality.DCE: ["te", "temporal_resolution", "baseline_frames", "relaxivity"],
    Modality.DSC: ["temporal_resolution", "baseline_frames"],
    Modality.ASL: ["labeling_duration", "tr", "background_suppression"],
    Modality.IVIM: ["tr", "te"],
}


@dataclass
class ValidationResult:
    """Result of parameter validation.

    Attributes
    ----------
    is_valid : bool
        Whether all required parameters are present and valid.
    missing_required : list[str]
        List of missing required parameters.
    missing_recommended : list[str]
        List of missing recommended parameters.
    out_of_range : dict[str, tuple[Any, tuple[float, float]]]
        Parameters with values outside expected ranges.
        Maps param name to (value, (min, max)).
    warnings : list[str]
        Warning messages about the validation.
    """

    is_valid: bool = True
    missing_required: list[str] = field(default_factory=list)
    missing_recommended: list[str] = field(default_factory=list)
    out_of_range: dict[str, tuple[Any, tuple[float, float]]] = field(
        default_factory=dict
    )
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Format validation result as string."""
        lines = []
        if self.is_valid:
            lines.append("Validation passed")
        else:
            lines.append("Validation failed")

        if self.missing_required:
            lines.append(f"Missing required: {', '.join(self.missing_required)}")
        if self.missing_recommended:
            lines.append(f"Missing recommended: {', '.join(self.missing_recommended)}")
        if self.out_of_range:
            for param, (value, (min_v, max_v)) in self.out_of_range.items():
                lines.append(f"{param}={value} outside range [{min_v}, {max_v}]")
        if self.warnings:
            for warning in self.warnings:
                lines.append(f"Warning: {warning}")

        return "\n".join(lines)


class ParameterValidator:
    """Validates acquisition parameters for perfusion analysis.

    This class checks that required parameters are present and
    within acceptable ranges for the specified modality.

    Examples
    --------
    >>> from osipy.common.types import Modality
    >>> validator = ParameterValidator(Modality.ASL)
    >>> params = {"labeling_type": "PCASL", "post_labeling_delay": 1800.0}
    >>> result = validator.validate(params)
    >>> print(result.is_valid)
    True
    """

    def __init__(self, modality: Modality):
        """Initialize validator for a modality.

        Parameters
        ----------
        modality : Modality
            Perfusion imaging modality.
        """
        self.modality = modality
        self.required = REQUIRED_PARAMS.get(modality, [])
        self.recommended = RECOMMENDED_PARAMS.get(modality, [])

    def validate(
        self,
        params: dict[str, Any],
        check_ranges: bool = True,
        check_recommended: bool = True,
    ) -> ValidationResult:
        """Validate parameters against modality requirements.

        Parameters
        ----------
        params : dict[str, Any]
            Dictionary of parameter names and values.
        check_ranges : bool, default=True
            Whether to check value ranges.
        check_recommended : bool, default=True
            Whether to check for recommended parameters.

        Returns
        -------
        ValidationResult
            Validation result with details.
        """
        result = ValidationResult()

        # Check required parameters
        for param in self.required:
            if param not in params or params[param] is None:
                result.missing_required.append(param)
                result.is_valid = False

        # Check recommended parameters
        if check_recommended:
            for param in self.recommended:
                if param not in params or params[param] is None:
                    result.missing_recommended.append(param)

        # Check value ranges
        if check_ranges:
            for param, value in params.items():
                if value is not None and param in PARAM_RANGES:
                    min_val, max_val = PARAM_RANGES[param]
                    try:
                        num_value = float(value)
                        if num_value < min_val or num_value > max_val:
                            result.out_of_range[param] = (value, (min_val, max_val))
                            result.warnings.append(
                                f"{param}={value} is outside typical range "
                                f"[{min_val}, {max_val}]"
                            )
                    except (ValueError, TypeError):
                        # Non-numeric parameters are not range-checked
                        pass

        # Modality-specific validation
        self._validate_modality_specific(params, result)

        return result

    def _validate_modality_specific(
        self, params: dict[str, Any], result: ValidationResult
    ) -> None:
        """Perform modality-specific validation checks.

        Parameters
        ----------
        params : dict[str, Any]
            Parameter dictionary.
        result : ValidationResult
            Result object to update.
        """
        if self.modality == Modality.ASL:
            self._validate_asl_specific(params, result)
        elif self.modality == Modality.IVIM:
            self._validate_ivim_specific(params, result)
        elif self.modality == Modality.DCE:
            self._validate_dce_specific(params, result)
        elif self.modality == Modality.DSC:
            self._validate_dsc_specific(params, result)

    def _validate_asl_specific(
        self, params: dict[str, Any], result: ValidationResult
    ) -> None:
        """ASL-specific validation."""
        labeling_type = params.get("labeling_type", "").upper()

        # PASL requires TI1 (bolus cutoff) for QUIPSS
        if (
            labeling_type in ["PASL", "FAIR"]
            and "bolus_cutoff_delay" not in params
            and "ti" not in params
        ):
            result.warnings.append(
                "PASL without bolus cutoff time - consider using QUIPSS II"
            )

        # Multi-PLD check
        pld = params.get("post_labeling_delay")
        if (
            isinstance(pld, (list, tuple))
            and len(pld) > 1
            and "labeling_duration" not in params
        ):
            # Multi-PLD acquisition
            result.warnings.append(
                "Multi-PLD acquisition detected but labeling_duration not specified"
            )

    def _validate_ivim_specific(
        self, params: dict[str, Any], result: ValidationResult
    ) -> None:
        """IVIM-specific validation."""
        b_values = params.get("b_values")
        if b_values is not None:
            try:
                b_list = list(b_values)
                if len(b_list) < 4:
                    result.warnings.append(
                        f"Only {len(b_list)} b-values - IVIM fitting typically "
                        "requires at least 4 b-values"
                    )
                if 0 not in b_list:
                    result.warnings.append(
                        "No b=0 acquisition - required for IVIM fitting"
                    )
                # Check for adequate sampling of low b-values (perfusion sensitive)
                low_b = [b for b in b_list if b < 200]
                if len(low_b) < 3:
                    result.warnings.append(
                        "Insufficient low b-values (<200) for reliable perfusion estimation"
                    )
            except (ValueError, TypeError):
                result.warnings.append("Could not parse b_values as list")

    def _validate_dce_specific(
        self, params: dict[str, Any], result: ValidationResult
    ) -> None:
        """DCE-specific validation."""
        flip_angle = params.get("flip_angle")
        if flip_angle is not None:
            try:
                fa = float(flip_angle)
                if fa < 5 or fa > 30:
                    result.warnings.append(
                        f"Flip angle {fa}° is unusual for DCE (typically 10-25°)"
                    )
            except (ValueError, TypeError):
                pass

        # Check temporal resolution for pharmacokinetic modeling
        temp_res = params.get("temporal_resolution")
        if temp_res is not None:
            try:
                tr_sec = float(temp_res)
                if tr_sec > 10:
                    result.warnings.append(
                        f"Temporal resolution {tr_sec}s may be too slow for "
                        "accurate pharmacokinetic modeling"
                    )
            except (ValueError, TypeError):
                pass

    def _validate_dsc_specific(
        self, params: dict[str, Any], result: ValidationResult
    ) -> None:
        """DSC-specific validation."""
        te = params.get("te")
        if te is not None:
            try:
                te_ms = float(te)
                if te_ms < 20:
                    result.warnings.append(
                        f"TE={te_ms}ms is short for DSC - may have reduced "
                        "T2* sensitivity"
                    )
                elif te_ms > 50:
                    result.warnings.append(
                        f"TE={te_ms}ms is long for DSC - may have signal dropout"
                    )
            except (ValueError, TypeError):
                pass
