"""Base classes and dataclasses for Arterial Input Functions.

This module provides the abstract base class for AIF implementations
and the ArterialInputFunction dataclass for storing AIF data.

Population AIF models registered with ``@register_aif`` follow the
OSIPI CAPLEX convention for model identification (e.g. M.IC2.001
for the Parker AIF, M.IC2.002 for the Georgiou AIF).

References
----------
.. [1] Parker GJM et al. (2006). Experimentally-derived functional form
   for a population-averaged AIF. Magn Reson Med 56(5):993-1000.
.. [2] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [3] Dickie BR et al. MRM 2024. doi:10.1002/mrm.29840
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.exceptions import AIFError
from osipy.common.models.base import BaseComponent
from osipy.common.types import AIFType

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ArterialInputFunction:
    """Container for arterial input function data.

    This dataclass holds the AIF concentration values, timing information,
    and metadata about how the AIF was obtained.

    Attributes
    ----------
    time : NDArray[np.floating]
        Time points in seconds.
    concentration : NDArray[np.floating]
        Concentration values at each time point in mM.
    aif_type : AIFType
        Type of AIF (MEASURED, POPULATION, AUTOMATIC).
    population_model : str | None
        Name of population model if aif_type is POPULATION.
    model_parameters : dict[str, float] | None
        Model-specific parameters if using a population AIF.
    source_roi : NDArray[np.bool_] | None
        ROI mask used for extraction if aif_type is MEASURED.
    extraction_method : str | None
        Method used for extraction ('manual', 'automatic').
    reference : str
        Literature citation for the AIF model or extraction method.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.aif.base import ArterialInputFunction
    >>> from osipy.common.types import AIFType
    >>> time = np.linspace(0, 300, 60)
    >>> conc = np.exp(-time / 100) * 5  # Simple exponential decay
    >>> aif = ArterialInputFunction(
    ...     time=time,
    ...     concentration=conc,
    ...     aif_type=AIFType.POPULATION,
    ...     population_model="parker",
    ...     reference="Parker GJM et al. (2006). MRM.",
    ... )
    """

    time: "NDArray[np.floating[Any]]"
    concentration: "NDArray[np.floating[Any]]"
    aif_type: AIFType
    population_model: str | None = None
    model_parameters: dict[str, float] | None = field(default_factory=dict)
    source_roi: "NDArray[np.bool_] | None" = None
    extraction_method: str | None = None
    reference: str = ""

    def __post_init__(self) -> None:
        """Validate AIF after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate AIF data consistency.

        Raises
        ------
        ValueError
            If validation fails.
        """
        if len(self.time) != len(self.concentration):
            msg = (
                f"time length ({len(self.time)}) must match "
                f"concentration length ({len(self.concentration)})"
            )
            raise AIFError(msg)

        if len(self.time) == 0:
            msg = "AIF must have at least one time point"
            raise AIFError(msg)

    @property
    def n_timepoints(self) -> int:
        """Return number of time points."""
        return len(self.time)

    @property
    def peak_concentration(self) -> float:
        """Return peak concentration value."""
        return float(np.max(self.concentration))

    @property
    def peak_time(self) -> float:
        """Return time of peak concentration."""
        return float(self.time[np.argmax(self.concentration)])


class BaseAIF(BaseComponent):
    """Abstract base class for arterial input function models.

    All AIF models (population-based or extraction algorithms) must
    inherit from this class and implement the required methods.

    This enables a consistent interface for using different AIF
    sources in pharmacokinetic modeling.

    Examples
    --------
    >>> class MyAIF(BaseAIF):
    ...     @property
    ...     def name(self) -> str:
    ...         return "MyAIF"
    ...     @property
    ...     def reference(self) -> str:
    ...         return "Author et al. (2024)"
    ...     def __call__(self, t):
    ...         return np.exp(-t / 100)
    ...     def get_parameters(self):
    ...         return {"tau": 100.0}
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable AIF model name.

        Returns
        -------
        str
            Name of the AIF model.
        """
        ...

    @property
    @abstractmethod
    def reference(self) -> str:
        """Return literature citation for this AIF model.

        Returns
        -------
        str
            Citation string.
        """
        ...

    @abstractmethod
    def __call__(self, t: "NDArray[np.floating[Any]]") -> "NDArray[np.floating[Any]]":
        """Evaluate AIF at given time points.

        Parameters
        ----------
        t : NDArray[np.floating]
            Time points in seconds.

        Returns
        -------
        NDArray[np.floating]
            Concentration values at each time point in mM.
        """
        ...

    @abstractmethod
    def get_parameters(self) -> dict[str, float]:
        """Return current AIF model parameters.

        Returns
        -------
        dict[str, float]
            Parameter names and values.
        """
        ...

    def to_arterial_input_function(
        self,
        time: "NDArray[np.floating[Any]]",
    ) -> ArterialInputFunction:
        """Convert to ArterialInputFunction dataclass.

        Parameters
        ----------
        time : NDArray[np.floating]
            Time points at which to evaluate the AIF.

        Returns
        -------
        ArterialInputFunction
            AIF data container.
        """
        return ArterialInputFunction(
            time=time,
            concentration=self(time),
            aif_type=AIFType.POPULATION,
            population_model=self.name,
            model_parameters=self.get_parameters(),
            reference=self.reference,
        )
