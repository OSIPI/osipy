"""ParameterMap container for computed perfusion parameters.

This module provides the data container for storing computed parameter
maps with associated metadata, uncertainty estimates, and quality masks.

The ``ParameterMap.name`` and ``ParameterMap.symbol`` fields use the
OSIPI CAPLEX naming convention where applicable (e.g. ``"Ktrans"``,
``"ve"``, ``"CBF"``). Units are stored as ASCII strings using caret
notation for exponents (e.g. ``"mm^2/s"``, ``"min^-1"``).

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.30101
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.exceptions import DataValidationError

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ParameterMap:
    """Container for a single computed parameter map.

    This dataclass holds a 3D parameter map along with metadata about
    the parameter, uncertainty estimates, and quality indicators.

    Attributes
    ----------
    name : str
        OSIPI lexicon parameter name (e.g., 'Ktrans', 'CBF', 'D').
    symbol : str
        Display symbol for visualization (e.g., 'Kᵗʳᵃⁿˢ', 'CBF').
    units : str
        Units string (e.g., 'min⁻¹', 'ml/100g/min', 'mm²/s').
    values : NDArray[np.floating]
        3D parameter map array with shape (x, y, z).
    affine : NDArray[np.floating]
        4x4 affine transformation matrix.
    uncertainty : NDArray[np.floating] | None
        Uncertainty estimate (standard error or confidence interval width).
    uncertainty_type : str | None
        Type of uncertainty ('std_error', 'ci_95', 'ci_99').
    quality_mask : NDArray[np.bool_]
        Boolean mask where True indicates valid/reliable estimate.
    failure_reasons : NDArray[np.object_] | None
        String array with failure codes for invalid voxels.
    model_name : str
        Name of the model used to compute this parameter.
    fitting_method : str
        Fitting algorithm used (e.g., 'levenberg_marquardt', 'bayesian').
    literature_reference : str
        Citation for the algorithm/model.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.parameter_map import ParameterMap
    >>> values = np.random.rand(64, 64, 20) * 0.5  # Ktrans values
    >>> affine = np.eye(4)
    >>> quality_mask = values > 0.01  # Exclude very low values
    >>> ktrans_map = ParameterMap(
    ...     name="Ktrans",
    ...     symbol="Kᵗʳᵃⁿˢ",
    ...     units="min⁻¹",
    ...     values=values,
    ...     affine=affine,
    ...     quality_mask=quality_mask,
    ...     model_name="ExtendedTofts",
    ...     fitting_method="levenberg_marquardt",
    ...     literature_reference="Tofts PS et al. (1999). JMRI.",
    ... )

    References
    ----------
    OSIPI CAPLEX Lexicon: https://osipi.github.io/OSIPI_CAPLEX/
    """

    # Parameter identification
    name: str
    symbol: str
    units: str

    # Data
    values: "NDArray[np.floating[Any]]"
    affine: "NDArray[np.floating[Any]]"

    # Uncertainty (optional)
    uncertainty: "NDArray[np.floating[Any]] | None" = None
    uncertainty_type: str | None = None

    # Quality
    quality_mask: "NDArray[np.bool_]" = field(
        default_factory=lambda: np.array([], dtype=bool)
    )
    failure_reasons: "NDArray[np.object_] | None" = None

    # Provenance
    model_name: str = ""
    fitting_method: str = ""
    literature_reference: str = ""

    def __post_init__(self) -> None:
        """Validate parameter map after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate parameter map dimensions and consistency.

        Raises
        ------
        ValueError
            If validation fails.
        """
        # Check values dimensions
        if self.values.ndim != 3:
            msg = f"Values must be 3D, got {self.values.ndim}D"
            raise DataValidationError(msg)

        # Check affine shape
        if self.affine.shape != (4, 4):
            msg = f"Affine must be 4x4, got {self.affine.shape}"
            raise DataValidationError(msg)

        # Initialize quality_mask if empty
        if self.quality_mask.size == 0:
            object.__setattr__(
                self,
                "quality_mask",
                np.ones(self.values.shape, dtype=bool),
            )

        # Check quality_mask shape
        if self.quality_mask.shape != self.values.shape:
            msg = (
                f"quality_mask shape {self.quality_mask.shape} must match "
                f"values shape {self.values.shape}"
            )
            raise DataValidationError(msg)

        # Check uncertainty shape if provided
        if self.uncertainty is not None and self.uncertainty.shape != self.values.shape:
            msg = (
                f"uncertainty shape {self.uncertainty.shape} must match "
                f"values shape {self.values.shape}"
            )
            raise DataValidationError(msg)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the shape of the parameter map."""
        return (self.values.shape[0], self.values.shape[1], self.values.shape[2])

    @property
    def valid_fraction(self) -> float:
        """Return fraction of voxels with valid estimates."""
        return float(np.mean(self.quality_mask))

    @property
    def n_valid(self) -> int:
        """Return number of voxels with valid estimates."""
        return int(np.sum(self.quality_mask))

    @property
    def n_failed(self) -> int:
        """Return number of voxels with failed estimates."""
        return int(np.sum(~self.quality_mask))

    def masked_values(self) -> "NDArray[np.floating[Any]]":
        """Return values array with invalid voxels set to NaN.

        Returns
        -------
        NDArray[np.floating]
            Copy of values with NaN where quality_mask is False.
        """
        result = self.values.copy()
        result[~self.quality_mask] = np.nan
        return result

    def statistics(self) -> dict[str, float]:
        """Compute summary statistics for valid voxels.

        Returns
        -------
        dict[str, float]
            Dictionary with mean, std, min, max, median for valid voxels.
        """
        valid_values = self.values[self.quality_mask]
        if len(valid_values) == 0:
            return {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "median": np.nan,
            }
        return {
            "mean": float(np.mean(valid_values)),
            "std": float(np.std(valid_values)),
            "min": float(np.min(valid_values)),
            "max": float(np.max(valid_values)),
            "median": float(np.median(valid_values)),
        }

    @classmethod
    def from_uniform(
        cls,
        value: float,
        shape: tuple[int, int, int],
        affine: "NDArray[np.floating[Any]]",
        name: str,
        units: str,
        symbol: str = "",
    ) -> "ParameterMap":
        """Create a uniform parameter map with constant value.

        This is useful when a measured parameter map is unavailable
        and a literature value must be assumed.

        Parameters
        ----------
        value : float
            Uniform value to fill the map.
        shape : tuple[int, int, int]
            Shape of the 3D parameter map (x, y, z).
        affine : NDArray[np.floating]
            4x4 affine transformation matrix.
        name : str
            Parameter name (e.g., 'T1', 'T2').
        units : str
            Units string (e.g., 'ms', 's').
        symbol : str, optional
            Display symbol, defaults to name.

        Returns
        -------
        ParameterMap
            Parameter map with uniform value.

        Examples
        --------
        >>> import numpy as np
        >>> from osipy.common.parameter_map import ParameterMap
        >>> # Create uniform T1 map with assumed value
        >>> t1_map = ParameterMap.from_uniform(
        ...     value=1400.0,  # Breast tissue at 3T
        ...     shape=(320, 320, 128),
        ...     affine=np.eye(4),
        ...     name="T1",
        ...     units="ms",
        ... )
        """
        values = np.full(shape, value, dtype=np.float64)
        return cls(
            name=name,
            symbol=symbol or name,
            units=units,
            values=values,
            affine=affine,
            model_name="assumed",
            fitting_method="uniform",
            literature_reference="User-specified assumed value",
        )


def create_uniform_t1_map(
    t1_ms: float,
    shape: tuple[int, int, int],
    affine: "NDArray[np.floating[Any]]",
) -> ParameterMap:
    """Create a uniform T1 map with assumed value.

    This function creates a ParameterMap representing pre-contrast
    T1 relaxation times when measured T1 mapping data is unavailable.
    Common use case: datasets without VFA or Look-Locker T1 mapping.

    Parameters
    ----------
    t1_ms : float
        Assumed T1 value in milliseconds. Typical values at 3T:
        - Breast tissue: ~1400 ms
        - Breast cancer: ~1200-1500 ms
        - Brain white matter: ~800 ms
        - Brain gray matter: ~1200 ms
        - Blood: ~1600 ms
        - Muscle: ~1400 ms
    shape : tuple[int, int, int]
        Shape of the 3D T1 map (x, y, z), matching DCE signal dimensions.
    affine : NDArray[np.floating]
        4x4 affine transformation matrix from DCE dataset.

    Returns
    -------
    ParameterMap
        T1 parameter map with uniform assumed value.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.parameter_map import create_uniform_t1_map
    >>> # Assume T1 = 1400 ms for breast tissue at 3T
    >>> t1_map = create_uniform_t1_map(
    ...     t1_ms=1400.0,
    ...     shape=(320, 320, 128),
    ...     affine=np.eye(4),
    ... )
    >>> # Use with signal_to_concentration
    >>> from osipy.dce import signal_to_concentration
    >>> concentration = signal_to_concentration(
    ...     signal=dce_data,
    ...     t1_map=t1_map,
    ...     acquisition_params=params,
    ... )

    Notes
    -----
    Using an assumed uniform T1 value introduces quantification bias
    because T1 varies across tissues. This is acceptable when:
    - T1 mapping data is unavailable (common in clinical datasets)
    - Relative comparisons are more important than absolute values
    - The tissue of interest has relatively uniform T1

    For more accurate quantification, measure T1 using VFA or
    Look-Locker sequences when possible.

    References
    ----------
    de Bazelaire CM et al. (2004). MR imaging relaxation times of
    abdominal and pelvic tissues. Radiology.
    """
    return ParameterMap.from_uniform(
        value=t1_ms,
        shape=shape,
        affine=affine,
        name="T1",
        units="ms",
        symbol="T1",
    )
