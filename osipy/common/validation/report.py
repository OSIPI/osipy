"""ValidationReport dataclass for DRO comparison results.

This module provides the data container for storing validation
comparison results against OSIPI Digital Reference Objects or
other reference datasets.

References
----------
QIBA DCE Profile v2.0
OSIPI Task Force 4.1/4.2 publications
"""

import json as _json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import osipy

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ValidationReport:
    """Validation comparison against reference values.

    This dataclass contains the results of comparing computed parameter
    maps against reference (ground truth) values from OSIPI DROs or
    other validation datasets.

    Attributes
    ----------
    reference_name : str
        Name of reference dataset (e.g., 'OSIPI_DRO_v10').
    reference_parameters : dict[str, NDArray[np.floating]]
        Ground truth parameter values.
    computed_parameters : dict[str, NDArray[np.floating]]
        osipy computed parameter values.
    absolute_errors : dict[str, NDArray[np.floating]]
        Absolute errors |computed - reference|.
    relative_errors : dict[str, NDArray[np.floating]]
        Relative errors |computed - reference| / reference.
    tolerances : dict[str, dict[str, float]]
        Tolerance thresholds for each parameter.
        Format: {'Ktrans': {'absolute': 0.005, 'relative': 0.01}}
    within_tolerance : dict[str, NDArray[np.bool_]]
        Boolean mask indicating voxels within tolerance.
    pass_rate : dict[str, float]
        Fraction of voxels within tolerance per parameter.
    overall_pass : bool
        True if all parameters meet their tolerance criteria.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.validation.report import ValidationReport
    >>> ref = {"Ktrans": np.array([0.1, 0.2, 0.3])}
    >>> comp = {"Ktrans": np.array([0.105, 0.195, 0.31])}
    >>> report = ValidationReport(
    ...     reference_name="OSIPI_DRO_v10",
    ...     reference_parameters=ref,
    ...     computed_parameters=comp,
    ...     absolute_errors={"Ktrans": np.abs(comp["Ktrans"] - ref["Ktrans"])},
    ...     relative_errors={"Ktrans": np.abs(comp["Ktrans"] - ref["Ktrans"]) / ref["Ktrans"]},
    ...     tolerances={"Ktrans": {"absolute": 0.01, "relative": 0.05}},
    ...     within_tolerance={"Ktrans": np.array([True, True, True])},
    ...     pass_rate={"Ktrans": 1.0},
    ...     overall_pass=True,
    ... )
    """

    # Reference
    reference_name: str
    reference_parameters: dict[str, "NDArray[np.floating[Any]]"]

    # Computed
    computed_parameters: dict[str, "NDArray[np.floating[Any]]"]

    # Comparison metrics
    absolute_errors: dict[str, "NDArray[np.floating[Any]]"] = field(
        default_factory=dict
    )
    relative_errors: dict[str, "NDArray[np.floating[Any]]"] = field(
        default_factory=dict
    )

    # Tolerance check
    tolerances: dict[str, dict[str, float]] = field(default_factory=dict)
    within_tolerance: dict[str, "NDArray[np.bool_]"] = field(default_factory=dict)

    # Summary
    pass_rate: dict[str, float] = field(default_factory=dict)
    overall_pass: bool = False

    @property
    def parameters(self) -> list[str]:
        """Return list of validated parameter names."""
        return list(self.reference_parameters.keys())

    @property
    def n_parameters(self) -> int:
        """Return number of validated parameters."""
        return len(self.reference_parameters)

    def summary(self) -> str:
        """Generate human-readable summary of validation results.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = [
            f"Validation Report: {self.reference_name}",
            f"Overall: {'PASS' if self.overall_pass else 'FAIL'}",
            "",
            "Parameter Results:",
        ]
        for param in self.parameters:
            rate = self.pass_rate.get(param, 0.0)
            status = "PASS" if rate >= 0.95 else "FAIL"
            lines.append(f"  {param}: {rate * 100:.1f}% within tolerance [{status}]")

        return "\n".join(lines)

    def get_statistics(self, parameter: str) -> dict[str, float]:
        """Get error statistics for a specific parameter.

        Parameters
        ----------
        parameter : str
            Parameter name.

        Returns
        -------
        dict[str, float]
            Statistics including mean/std/max absolute and relative errors.
        """
        if parameter not in self.absolute_errors:
            return {}

        abs_err = self.absolute_errors[parameter]
        rel_err = self.relative_errors.get(parameter, np.array([]))

        stats = {
            "mean_absolute_error": float(np.mean(abs_err)),
            "std_absolute_error": float(np.std(abs_err)),
            "max_absolute_error": float(np.max(abs_err)),
        }

        if len(rel_err) > 0:
            # Exclude NaN/inf from relative error stats
            valid_rel = rel_err[np.isfinite(rel_err)]
            if len(valid_rel) > 0:
                stats["mean_relative_error"] = float(np.mean(valid_rel))
                stats["std_relative_error"] = float(np.std(valid_rel))
                stats["max_relative_error"] = float(np.max(valid_rel))

        return stats

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to a dictionary.

        Returns
        -------
        dict[str, Any]
            Dict with keys: reference_name, overall_pass, parameters
            (dict of param -> {pass_rate, statistics}), tolerances,
            timestamp, version.
        """
        param_results: dict[str, Any] = {}
        for param in self.parameters:
            param_results[param] = {
                "pass_rate": self.pass_rate.get(param, 0.0),
                "statistics": self.get_statistics(param),
            }

        return {
            "reference_name": self.reference_name,
            "overall_pass": self.overall_pass,
            "parameters": param_results,
            "tolerances": {
                k: v for k, v in self.tolerances.items() if k in self.parameters
            },
            "timestamp": datetime.now(UTC).isoformat(),
            "version": getattr(osipy, "__version__", "unknown"),
        }

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialize report to JSON string.

        Parameters
        ----------
        path : str | Path | None
            Optional file path. If given, writes JSON to file.

        Returns
        -------
        str
            JSON string.
        """
        data = self.to_dict()
        json_str = _json.dumps(data, indent=2, default=str)

        if path is not None:
            Path(path).write_text(json_str)

        return json_str
